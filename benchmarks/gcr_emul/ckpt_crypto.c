/**
 * ckpt_crypto.c — Multi-threaded AES-256-GCM encrypt/decrypt
 *
 * Uses OpenSSL EVP API which leverages AES-NI on x86_64.
 * Parallelizes across pages using pthreads — each thread handles
 * a contiguous chunk of pages with its own EVP context.
 *
 * Build:
 *   gcc -O2 -c ckpt_crypto.c -lssl -lcrypto -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include "ckpt_crypto.h"

/* ------------------------------------------------------------------ */
/* Key generation                                                      */
/* ------------------------------------------------------------------ */
int ckpt_crypto_gen_key(uint8_t key_out[CKPT_CRYPTO_KEY_SIZE])
{
    if (RAND_bytes(key_out, CKPT_CRYPTO_KEY_SIZE) != 1) {
        fprintf(stderr, "ckpt_crypto: RAND_bytes failed\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Worker thread context                                               */
/* ------------------------------------------------------------------ */
typedef struct {
    const uint8_t      *key;
    const uint8_t      *src;         /* plaintext (encrypt) or ciphertext (decrypt) */
    uint8_t            *dst;         /* ciphertext (encrypt) or plaintext (decrypt) */
    ckpt_page_meta_t   *meta;        /* per-page IV + auth_tag */
    const ckpt_page_meta_t *meta_in; /* for decrypt: read-only metadata */
    uint32_t            page_start;  /* first page index for this thread */
    uint32_t            page_count;  /* number of pages this thread handles */
    size_t              page_size;   /* bytes per page (4KB, 2MB, ...) */
    uint64_t            base_iv;     /* starting IV counter (encrypt only) */
    int                 encrypt;     /* 1 = encrypt, 0 = decrypt */
    int                 status;      /* 0 = success, -1 = error */
} crypto_worker_t;

/* Build a 12-byte IV from a 64-bit counter */
static void build_iv(uint8_t iv[CKPT_CRYPTO_IV_SIZE], uint64_t counter)
{
    /* First 4 bytes: fixed nonce (can be session-specific) */
    iv[0] = 0x43; iv[1] = 0x4B; iv[2] = 0x50; iv[3] = 0x54; /* "CKPT" */
    /* Last 8 bytes: counter in little-endian */
    memcpy(iv + 4, &counter, 8);
}

static void *encrypt_worker(void *arg)
{
    crypto_worker_t *w = (crypto_worker_t *)arg;
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) { w->status = -1; return NULL; }

    const EVP_CIPHER *cipher = EVP_aes_256_gcm();
    const size_t     psz     = w->page_size;

    for (uint32_t i = 0; i < w->page_count; i++) {
        uint32_t page_idx = w->page_start + i;
        const uint8_t *src = w->src + (size_t)page_idx * psz;
        uint8_t *dst       = w->dst + (size_t)page_idx * psz;
        ckpt_page_meta_t *m = &w->meta[page_idx];
        int outlen = 0;

        /* Generate IV for this page */
        build_iv(m->iv, w->base_iv + page_idx);

        if (EVP_EncryptInit_ex(ctx, cipher, NULL, w->key, m->iv) != 1 ||
            EVP_EncryptUpdate(ctx, dst, &outlen, src, psz) != 1 ||
            EVP_EncryptFinal_ex(ctx, dst + outlen, &outlen) != 1 ||
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG,
                                CKPT_CRYPTO_AUTH_TAG_SIZE, m->auth_tag) != 1) {
            fprintf(stderr, "ckpt_crypto: encrypt failed at page %u\n", page_idx);
            w->status = -1;
            break;
        }
    }

    EVP_CIPHER_CTX_free(ctx);
    return NULL;
}

static void *decrypt_worker(void *arg)
{
    crypto_worker_t *w = (crypto_worker_t *)arg;
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) { w->status = -1; return NULL; }

    const EVP_CIPHER *cipher = EVP_aes_256_gcm();
    const size_t     psz     = w->page_size;

    for (uint32_t i = 0; i < w->page_count; i++) {
        uint32_t page_idx = w->page_start + i;
        const uint8_t *src       = w->src + (size_t)page_idx * psz;
        uint8_t *dst             = w->dst + (size_t)page_idx * psz;
        const ckpt_page_meta_t *m = &w->meta_in[page_idx];
        int outlen = 0;

        if (EVP_DecryptInit_ex(ctx, cipher, NULL, w->key, m->iv) != 1 ||
            EVP_DecryptUpdate(ctx, dst, &outlen, src, psz) != 1 ||
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG,
                                CKPT_CRYPTO_AUTH_TAG_SIZE,
                                (void *)m->auth_tag) != 1 ||
            EVP_DecryptFinal_ex(ctx, dst + outlen, &outlen) != 1) {
            fprintf(stderr, "ckpt_crypto: decrypt/auth failed at page %u\n", page_idx);
            w->status = -1;
            break;
        }
    }

    EVP_CIPHER_CTX_free(ctx);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

static int get_num_threads(int requested)
{
    if (requested > 0)
        return requested;
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > 64) n = 64;   /* cap to avoid overhead */
    return n;
}

int ckpt_crypto_encrypt_pages(const uint8_t key[CKPT_CRYPTO_KEY_SIZE],
                               const uint8_t *plaintext,
                               uint8_t *ciphertext,
                               ckpt_page_meta_t *meta,
                               uint32_t num_pages,
                               size_t page_size,
                               uint64_t base_iv_counter,
                               int num_threads)
{
    if (num_pages == 0) return 0;
    if (page_size == 0) return -1;

    num_threads = get_num_threads(num_threads);
    if ((uint32_t)num_threads > num_pages)
        num_threads = (int)num_pages;

    crypto_worker_t *workers = calloc(num_threads, sizeof(crypto_worker_t));
    pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
    if (!workers || !threads) {
        free(workers); free(threads);
        return -1;
    }

    /* Divide pages among threads */
    uint32_t pages_per_thread = num_pages / (uint32_t)num_threads;
    uint32_t remainder = num_pages % (uint32_t)num_threads;
    uint32_t offset = 0;

    for (int t = 0; t < num_threads; t++) {
        workers[t].key        = key;
        workers[t].src        = plaintext;
        workers[t].dst        = ciphertext;
        workers[t].meta       = meta;
        workers[t].page_start = offset;
        workers[t].page_count = pages_per_thread + (t < (int)remainder ? 1 : 0);
        workers[t].page_size  = page_size;
        workers[t].base_iv    = base_iv_counter;
        workers[t].encrypt    = 1;
        workers[t].status     = 0;
        offset += workers[t].page_count;
    }

    /* Launch threads */
    for (int t = 0; t < num_threads; t++)
        pthread_create(&threads[t], NULL, encrypt_worker, &workers[t]);

    /* Wait and collect status */
    int result = 0;
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        if (workers[t].status != 0)
            result = -1;
    }

    free(workers);
    free(threads);
    return result;
}

int ckpt_crypto_decrypt_pages(const uint8_t key[CKPT_CRYPTO_KEY_SIZE],
                               const uint8_t *ciphertext,
                               uint8_t *plaintext,
                               const ckpt_page_meta_t *meta,
                               uint32_t num_pages,
                               size_t page_size,
                               int num_threads)
{
    if (num_pages == 0) return 0;
    if (page_size == 0) return -1; 

    num_threads = get_num_threads(num_threads);
    if ((uint32_t)num_threads > num_pages)
        num_threads = (int)num_pages;

    crypto_worker_t *workers = calloc(num_threads, sizeof(crypto_worker_t));
    pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
    if (!workers || !threads) {
        free(workers); free(threads);
        return -1;
    }

    uint32_t pages_per_thread = num_pages / (uint32_t)num_threads;
    uint32_t remainder = num_pages % (uint32_t)num_threads;
    uint32_t offset = 0;

    for (int t = 0; t < num_threads; t++) {
        workers[t].key        = key;
        workers[t].src        = ciphertext;
        workers[t].dst        = plaintext;
        workers[t].meta_in    = meta;
        workers[t].page_start = offset;
        workers[t].page_count = pages_per_thread + (t < (int)remainder ? 1 : 0);
        workers[t].page_size  = page_size;
        workers[t].encrypt    = 0;
        workers[t].status     = 0;
        offset += workers[t].page_count;
    }

    for (int t = 0; t < num_threads; t++)
        pthread_create(&threads[t], NULL, decrypt_worker, &workers[t]);

    int result = 0;
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        if (workers[t].status != 0)
            result = -1;
    }

    free(workers);
    free(threads);
    return result;
}

