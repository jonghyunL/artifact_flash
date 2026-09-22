/**
 * ckpt_crypto.h — Multi-threaded AES-256-GCM encrypt/decrypt for checkpoint images
 *
 * Provides high-throughput CPU-side encryption using:
 *   - OpenSSL EVP API (leverages AES-NI hardware instructions)
 *   - pthreads for parallel per-page encryption
 *
 * Each 4KB page gets its own IV and 16-byte auth_tag.
 *
 * Build:
 *   gcc -O2 ... ckpt_crypto.c -lssl -lcrypto -lpthread
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#define CKPT_CRYPTO_KEY_SIZE      32   /* AES-256 */
#define CKPT_CRYPTO_IV_SIZE       12   /* GCM standard */
#define CKPT_CRYPTO_AUTH_TAG_SIZE  16   /* GCM auth tag */
#define CKPT_CRYPTO_PAGE_SIZE   4096

/* Per-page metadata stored alongside ciphertext in checkpoint image */
typedef struct {
    uint8_t  iv[CKPT_CRYPTO_IV_SIZE];
    uint8_t  auth_tag[CKPT_CRYPTO_AUTH_TAG_SIZE];
} ckpt_page_meta_t;   /* 28 bytes per page */

/**
 * ckpt_crypto_init — generate a random 256-bit key for checkpoint encryption.
 * Returns 0 on success.
 */
int ckpt_crypto_gen_key(uint8_t key_out[CKPT_CRYPTO_KEY_SIZE]);

/**
 * ckpt_crypto_encrypt_pages — encrypt N fixed-size pages in parallel.
 *
 * @param key           AES-256 key (32 bytes)
 * @param plaintext     Input buffer (num_pages * page_size bytes)
 * @param ciphertext    Output buffer (num_pages * page_size bytes)
 * @param meta          Output per-page metadata (num_pages entries)
 * @param num_pages     Number of pages to encrypt
 * @param page_size     Bytes per page. Typical values:
 *                        CKPT_CRYPTO_PAGE_SIZE (4 KB) — v3 encrypted + H2D save
 *                        2*1024*1024 (2 MB)            — baseline whole-image
 * @param base_iv_counter Starting IV counter (incremented per page)
 * @param num_threads   Number of worker threads (0 = auto-detect)
 * @return              0 on success, -1 on error
 */
int ckpt_crypto_encrypt_pages(const uint8_t key[CKPT_CRYPTO_KEY_SIZE],
                               const uint8_t *plaintext,
                               uint8_t *ciphertext,
                               ckpt_page_meta_t *meta,
                               uint32_t num_pages,
                               size_t page_size,
                               uint64_t base_iv_counter,
                               int num_threads);

/**
 * ckpt_crypto_decrypt_pages — decrypt N fixed-size pages in parallel.
 *
 * Same contract as the encrypt variant; caller supplies page_size.
 * Returns -1 on auth-tag mismatch.
 */
int ckpt_crypto_decrypt_pages(const uint8_t key[CKPT_CRYPTO_KEY_SIZE],
                               const uint8_t *ciphertext,
                               uint8_t *plaintext,
                               const ckpt_page_meta_t *meta,
                               uint32_t num_pages,
                               size_t page_size,
                               int num_threads);
