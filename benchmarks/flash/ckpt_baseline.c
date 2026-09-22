/*
 * ckpt_baseline.c — In-app cudaMemcpyAsync baseline checkpoint (P1)
 *
 * The "naive baseline" — represents what unmodified user-space CR systems
 * (CRIU-CUDA, DMTCP-CUDA) can achieve today. No kernel modifications, no
 * dirty tracking, no precopy. Single stop-and-copy phase:
 *
 *   1. cudaDeviceSynchronize  (quiesce GPU)
 *   2. For each tracked allocation:
 *        cudaMemcpyAsync(D2H) into a pinned host staging buffer
 *        (chunked, with a small ring of staging buffers for overlap)
 *   3. memcpy each staging buffer into the mmap'd output image at the
 *      pre-computed offset
 *   4. ftruncate to exact size, msync, done
 *
 * The output image is byte-format compatible with the v2 plaintext format
 * emitted by ckpt_core.c's encrypted path: ckpt_v2_file_hdr_t + N×
 * ckpt_v2_range_desc_t + per-range (resmap [all GPU-resident] + cpu_section
 * [empty] + gpu_section [raw bytes from cudaMemcpy]). The existing mmap
 * restore in libckpt_vllm.c reads it unchanged.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <pthread.h>

#include <cuda_runtime.h>

#include "ckpt_gate.h"
#include "ckpt_baseline.h"
#include "ckpt_crypto.h"

/* Owned by libckpt_vllm.so (the only binary that links this TU).
 * ckpt_core.c has its own independent definition for its CLI flag.
 * In the .so this is always 0 (no env var hooked up yet), so the in-app
 * baseline/delta paths always write their image — same as before. */
int g_no_image_write = 0;

/* On-disk v2/v4 format (must match ckpt_core.c / libckpt_vllm.c)
 *
 * Version 2: plaintext gpu_section, crypto_meta_bytes=0
 * Version 3: kernel ioctl 115 path, 41-byte page_crypto_meta_t entries
 * Version 4: in-app baseline, 28-byte ckpt_page_meta_t entries, fixed 2MB chunks
 *            gpu_section is ciphertext; crypto_meta holds (length/2MB) entries,
 *            one per 2 MB chunk of ciphertext.
 */
#define CKPT_V2_MAGIC       0xC2C2C2C2u
#define CKPT_V2_VERSION     2u
#define CKPT_V4_VERSION     4u

/* Delta-v4 format: agent → app → disk. Structurally parallel to the
 * baseline v4: magic + version=4, per-block ciphertext + 28-byte meta per
 * 2 MB chunk. No resmap (all blocks are known-dirty). */
#define CKPT_DELTA_V4_MAGIC   0xDE17A004u

typedef struct __attribute__((packed)) {
    uint32_t magic;        /* CKPT_DELTA_V4_MAGIC */
    uint32_t version;      /* 4 */
    uint32_t num_blocks;
    uint32_t page_size;    /* 2 MB */
    uint64_t total_bytes;
} delta_v4_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va;          /* 2 MB-aligned VA */
    uint64_t length;           /* multiple of 2 MB */
    uint64_t data_offset;      /* file offset to ciphertext for this block */
    uint64_t crypto_meta_bytes; /* = (length/2MB) * sizeof(ckpt_page_meta_t) */
} delta_v4_block_t;

#define CRYPTO_CHUNK_BYTES  (2ULL * 1024 * 1024)  /* 2 MB, THP-aligned */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t num_ranges;
    uint32_t page_size;
    uint64_t total_bytes;
} v2_file_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va;
    uint64_t length;
    uint64_t num_pages;
    uint64_t data_offset;
    uint64_t resmap_size;
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    uint64_t crypto_meta_bytes;
} v2_range_desc_t;

#define CHUNK_BYTES        (256ULL * 1024 * 1024)  /* 256 MB per cudaMemcpy */
#define STAGING_RING_SLOTS 8                       /* 8 × 256 MB = 2 GB pinned */
#define N_WRITERS          4                       /* parallel memcpy threads */

/* Delta path uses smaller staging — the whole delta is typically tens of MB,
 * so allocating 2 GB of pinned memory (baseline's size) is wasteful and may
 * hit pinned-memory pressure right after the baseline's free. */
#define DELTA_CHUNK_BYTES        (8ULL * 1024 * 1024)   /* 8 MB = 4 × 2 MB */
#define DELTA_STAGING_RING_SLOTS 4                      /* 4 × 8 MB = 32 MB pinned */

/* Thread-local reentrancy guard: set while this thread is running baseline
 * or delta dispatch. libckpt_vllm.c's cudaMemcpyAsync interceptor checks it
 * and skips its maybe_checkpoint_post_sync side effect when called by our
 * own dispatch, preventing recursive re-entry into the AT_BOUNDARY loop. */
__thread int g_in_ckpt_worker = 0;
#define ALIGN_UP(x, a)     (((x) + (a) - 1) & ~((a) - 1))

/* ============================================================== */
/* Writer pool: main thread launches cudaMemcpyAsync + enqueues;   */
/* worker threads wait on events and memcpy staging → mmap output. */
/* This decouples the CPU-side memcpy from the GPU-side D2H launch */
/* path, so they overlap instead of serializing on main thread.    */
/* ============================================================== */

typedef struct {
    int                slot;     /* which staging slot to read from / release */
    uint8_t           *src;      /* staging buffer (already pinned) */
    uint8_t           *dst;      /* out_map + ciphertext offset */
    size_t             bytes;    /* chunk size (multiple of CRYPTO_CHUNK_BYTES) */
    ckpt_page_meta_t  *meta;     /* points into out_map crypto_meta section */
    uint64_t           base_iv;  /* starting IV counter for chunk 0 */
} bl_work_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
    bl_work_t      *items;
    int             cap;
    int             head, tail, count;
    int             closed;
} bl_wq_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  cv;
    int            *free_stack;
    int             n_slots;
    int             n_free;
} bl_pool_t;

typedef struct {
    bl_wq_t       *wq;
    bl_pool_t     *pool;
    cudaEvent_t   *events;
    const uint8_t *k3_key;     /* AES-256 key for in-app baseline encryption */
    int            id;
    /* accumulators */
    double         wait_ms;
    double         encrypt_ms;
    uint64_t       bytes;
    int            err;
} bl_worker_ctx_t;

static double bl_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* -- Work queue -- */
static int bl_wq_init(bl_wq_t *q, int cap)
{
    memset(q, 0, sizeof(*q));
    q->items = calloc(cap, sizeof(bl_work_t));
    if (!q->items) return -1;
    q->cap = cap;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
    return 0;
}

static void bl_wq_destroy(bl_wq_t *q)
{
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    free(q->items);
}

static void bl_wq_push(bl_wq_t *q, bl_work_t item)
{
    pthread_mutex_lock(&q->lock);
    while (q->count == q->cap)
        pthread_cond_wait(&q->not_full, &q->lock);
    q->items[q->tail] = item;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

static int bl_wq_pop(bl_wq_t *q, bl_work_t *out)
{
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->closed)
        pthread_cond_wait(&q->not_empty, &q->lock);
    if (q->count == 0 && q->closed) {
        pthread_mutex_unlock(&q->lock);
        return 0;  /* done */
    }
    *out = q->items[q->head];
    q->head = (q->head + 1) % q->cap;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

static void bl_wq_close(bl_wq_t *q)
{
    pthread_mutex_lock(&q->lock);
    q->closed = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

/* -- Slot pool -- */
static int bl_pool_init(bl_pool_t *p, int n_slots)
{
    memset(p, 0, sizeof(*p));
    p->free_stack = calloc(n_slots, sizeof(int));
    if (!p->free_stack) return -1;
    for (int i = 0; i < n_slots; i++) p->free_stack[i] = i;
    p->n_slots = n_slots;
    p->n_free = n_slots;
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->cv, NULL);
    return 0;
}

static void bl_pool_destroy(bl_pool_t *p)
{
    pthread_mutex_destroy(&p->lock);
    pthread_cond_destroy(&p->cv);
    free(p->free_stack);
}

static int bl_pool_acquire(bl_pool_t *p)
{
    pthread_mutex_lock(&p->lock);
    while (p->n_free == 0)
        pthread_cond_wait(&p->cv, &p->lock);
    int slot = p->free_stack[--p->n_free];
    pthread_mutex_unlock(&p->lock);
    return slot;
}

static void bl_pool_release(bl_pool_t *p, int slot)
{
    pthread_mutex_lock(&p->lock);
    p->free_stack[p->n_free++] = slot;
    pthread_cond_signal(&p->cv);
    pthread_mutex_unlock(&p->lock);
}

/* -- Worker thread --
 * Waits for D2H on its slot, then AES-GCM encrypts the staging buffer
 * directly into the mmap'd output file at 2 MB chunk granularity. One
 * ckpt_page_meta_t (IV+tag, 28 B) per 2 MB chunk is written to the
 * crypto_meta section. Encryption runs single-threaded inside this call
 * — parallelism comes from N_WRITERS concurrent workers.
 */
static void *bl_writer_thread(void *arg)
{
    bl_worker_ctx_t *ctx = arg;
    bl_work_t item;

    while (bl_wq_pop(ctx->wq, &item)) {
        /* Safety: encryption is fixed 2 MB chunks. Reject anything smaller
         * than one chunk or not a multiple. In practice the dispatch code
         * already guarantees this (vLLM allocs are 2 MB aligned), but a
         * fail-fast check prevents silent data corruption if that ever
         * changes. */
        if (item.bytes < CRYPTO_CHUNK_BYTES ||
            (item.bytes % CRYPTO_CHUNK_BYTES) != 0) {
            fprintf(stderr, "[baseline:w%d] FATAL: chunk bytes=%zu not a "
                    "multiple of 2 MB (slot=%d)\n",
                    ctx->id, item.bytes, item.slot);
            ctx->err = 1;
            bl_pool_release(ctx->pool, item.slot);
            continue;
        }

        double tw = bl_now_ms();
        cudaError_t cerr = cudaEventSynchronize(ctx->events[item.slot]);
        ctx->wait_ms += bl_now_ms() - tw;
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[baseline:w%d] cudaEventSync slot %d: %s\n",
                    ctx->id, item.slot, cudaGetErrorString(cerr));
            ctx->err = 1;
            bl_pool_release(ctx->pool, item.slot);
            continue;
        }

        /* AES-GCM encrypt staging → out_map, write meta entries in place. */
        uint32_t n_2mb = (uint32_t)(item.bytes / CRYPTO_CHUNK_BYTES);
        double te = bl_now_ms();
        int rc = ckpt_crypto_encrypt_pages(ctx->k3_key,
                                           item.src, item.dst,
                                           item.meta,
                                           n_2mb,
                                           (size_t)CRYPTO_CHUNK_BYTES,
                                           item.base_iv,
                                           /*num_threads=*/1);
        ctx->encrypt_ms += bl_now_ms() - te;
        if (rc != 0) {
            fprintf(stderr, "[baseline:w%d] encrypt failed slot=%d bytes=%zu\n",
                    ctx->id, item.slot, item.bytes);
            ctx->err = 1;
            bl_pool_release(ctx->pool, item.slot);
            continue;
        }
        ctx->bytes += item.bytes;
        bl_pool_release(ctx->pool, item.slot);
    }
    return NULL;
}

static double now_ms_local(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================== */
/* Process-scoped k3 key + monotonic IV counter.                   */
/* Key lifetime = process lifetime. Generated lazily on first      */
/* checkpoint call; reused across all subsequent calls in this     */
/* process. On process restart (including after restore) the       */
/* statics reinitialize, giving a fresh key + IV counter — which   */
/* is exactly what we want: each CR session has its own key.       */
/* ============================================================== */
static pthread_mutex_t s_k3_lock  = PTHREAD_MUTEX_INITIALIZER;
static uint8_t         s_k3_key[CKPT_CRYPTO_KEY_SIZE];
static int             s_k3_ready = 0;
static uint64_t        s_iv_base  = 0;   /* next unused 2 MB IV counter */

/* Generate-if-needed and save to <out_path>.k3.key. Returns 0 on success.
 * Must be called with s_k3_lock held. */
static int baseline_k3_ensure_key(const char *out_path)
{
    if (!s_k3_ready) {
        if (ckpt_crypto_gen_key(s_k3_key) != 0) {
            fprintf(stderr, "[baseline] ckpt_crypto_gen_key failed\n");
            return -1;
        }
        s_k3_ready = 1;
        fprintf(stderr, "[baseline] generated new k3 key (process-scoped)\n");
    }
    char key_path[1024];
    snprintf(key_path, sizeof(key_path), "%s.k3.key", out_path);
    int kf = open(key_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (kf < 0) {
        fprintf(stderr, "[baseline] open(%s): %s\n", key_path, strerror(errno));
        return -1;
    }
    ssize_t wn = write(kf, s_k3_key, CKPT_CRYPTO_KEY_SIZE);
    close(kf);
    if (wn != CKPT_CRYPTO_KEY_SIZE) {
        fprintf(stderr, "[baseline] write key file failed\n");
        return -1;
    }
    return 0;
}

static int _ckpt_baseline_run_body(const char *out_path,
                                    const ckpt_alloc_entry_t *allocs,
                                    uint32_t n_allocs,
                                    int stop_the_world,
                                    int n_mstreams_in,
                                    ckpt_baseline_result_t *result_out)
{
    if (!out_path || !allocs || n_allocs == 0) {
        fprintf(stderr, "[baseline] invalid args\n");
        return -1;
    }

    double t_start = now_ms_local();
    fprintf(stderr, "[baseline] mode=%s\n",
            stop_the_world ? "stop-and-copy (P1d)" : "concurrent precopy (P1e)");

    /* Per-phase timing accumulators */
    double acc_wait_ms    = 0.0;  /* workers: cudaEventSynchronize (sum)    */
    double acc_encrypt_ms = 0.0;  /* workers: AES-GCM encrypt + write (sum) */
    double acc_launch_ms  = 0.0;  /* main:    cudaMemcpyAsync launch        */
    double t_hot_wall     = 0.0;  /* main: hot phase wall (launch→join)     */
    double t_quiesce_ms   = 0.0;
    double t_prefill_ms   = 0.0;
    double t_setup_ms     = 0.0;
    double t_keygen_ms    = 0.0;
    uint64_t bytes_d2h     = 0;
    uint64_t bytes_encrypt = 0;

    /* --- k3 key + IV range (process-scoped, lazy init) --- */
    uint64_t run_base_iv = 0;
    {
        double tk = now_ms_local();
        pthread_mutex_lock(&s_k3_lock);
        if (baseline_k3_ensure_key(out_path) != 0) {
            pthread_mutex_unlock(&s_k3_lock);
            return -1;
        }
        /* Reserve the IV range for this run; release lock before any slow
         * work so concurrent callers (unlikely but possible) don't block. */
        run_base_iv = s_iv_base;
        /* We'll advance s_iv_base below once we know total_2mb_chunks. */
        pthread_mutex_unlock(&s_k3_lock);
        t_keygen_ms = now_ms_local() - tk;
    }

    /* ---- Phase 1: quiesce GPU (stop-and-copy only) ---------------- */
    cudaError_t cerr = cudaSuccess;
    if (stop_the_world) {
        double t_q0 = now_ms_local();
        cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[baseline] cudaDeviceSynchronize failed: %s\n",
                    cudaGetErrorString(cerr));
            return -1;
        }
        t_quiesce_ms = now_ms_local() - t_q0;
    }

    /* ---- Phase 2: filter freed zombies + compute layout ----------- */
    ckpt_alloc_entry_t *live = calloc(n_allocs, sizeof(*live));
    if (!live) { fprintf(stderr, "[baseline] calloc live\n"); return -1; }
    uint32_t n_live = 0, n_freed = 0;
    for (uint32_t i = 0; i < n_allocs; i++) {
        if (allocs[i].flags & CKPT_ALLOC_FLAG_FREED) { n_freed++; continue; }
        live[n_live++] = allocs[i];
    }
    fprintf(stderr, "[baseline] %u entries: %u live, %u freed (skipped)\n",
            n_allocs, n_live, n_freed);
    if (n_live == 0) {
        fprintf(stderr, "[baseline] no live allocations\n");
        free(live);
        return -1;
    }

    v2_range_desc_t *descs = calloc(n_live, sizeof(v2_range_desc_t));
    if (!descs) {
        fprintf(stderr, "[baseline] calloc descs\n");
        free(live); return -1;
    }

    uint64_t hdr_off = sizeof(v2_file_hdr_t)
                     + (uint64_t)n_live * sizeof(v2_range_desc_t);
    uint64_t cur_off = hdr_off;
    uint64_t total_bytes       = 0;
    uint64_t total_2mb_chunks  = 0;

    for (uint32_t i = 0; i < n_live; i++) {
        uint64_t length   = live[i].size;
        uint64_t n_pages  = length / 4096;
        /* vLLM allocations are 2 MB aligned; enforce so crypto layout is exact. */
        if (length % CRYPTO_CHUNK_BYTES != 0) {
            fprintf(stderr, "[baseline] alloc[%u] size %llu not a 2MB multiple\n",
                    i, (unsigned long long)length);
            free(descs); free(live);
            return -1;
        }
        uint64_t n_2mb          = length / CRYPTO_CHUNK_BYTES;
        uint64_t crypto_meta_sz = n_2mb * sizeof(ckpt_page_meta_t);

        descs[i].base_va           = live[i].va;
        descs[i].length            = length;
        descs[i].num_pages         = n_pages;
        descs[i].data_offset       = cur_off;
        descs[i].resmap_size       = n_pages;
        descs[i].cpu_bytes         = 0;
        descs[i].gpu_bytes         = length;
        descs[i].crypto_meta_bytes = crypto_meta_sz;
        cur_off     += n_pages + length + crypto_meta_sz;
        total_bytes += length;
        total_2mb_chunks += n_2mb;
    }
    uint64_t image_size = cur_off;

    /* Reserve the IV range for this run so it's monotonic across all
     * baseline runs in this process. */
    pthread_mutex_lock(&s_k3_lock);
    run_base_iv = s_iv_base;
    s_iv_base  += total_2mb_chunks;
    pthread_mutex_unlock(&s_k3_lock);

    fprintf(stderr, "[baseline] %u live allocs, %.2f GB total, image=%.2f GB → %s\n",
            n_live, total_bytes / (1024.0*1024.0*1024.0),
            image_size / (1024.0*1024.0*1024.0), out_path);

    /* Open WITHOUT O_TRUNC so we can reuse warm tmpfs pages from a prior
     * ckpt_prealloc run (matches the persistent staging pattern used in
     * ckpt_core.c's P3-B coalesce path). Only ftruncate UP if the existing
     * file is smaller than image_size; never truncate down.
     * Mode 0666: ckpt_prealloc (often root) and this code (regular user
     * under LD_PRELOAD) both need write access to the same file. */
    uint8_t *out_map;
    if (g_no_image_write) {
        /* No-disk: anonymous mapping discarded on munmap. */
        out_map = mmap(NULL, image_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        int fd = open(out_path, O_RDWR | O_CREAT, 0666);
        if (fd < 0) {
            fprintf(stderr, "[baseline] open(%s): %s\n", out_path, strerror(errno));
            free(descs); free(live);
            return -1;
        }
        (void)fchmod(fd, 0666);
        {
            struct stat out_st;
            if (fstat(fd, &out_st) < 0) {
                fprintf(stderr, "[baseline] fstat(%s): %s\n", out_path, strerror(errno));
                close(fd);
                free(descs); free(live);
                return -1;
            }
            if ((uint64_t)out_st.st_size < image_size) {
                if (ftruncate(fd, (off_t)image_size) < 0) {
                    fprintf(stderr, "[baseline] ftruncate(%llu): %s\n",
                            (unsigned long long)image_size, strerror(errno));
                    close(fd);
                    free(descs); free(live);
                    return -1;
                }
            }
        }
        out_map = mmap(NULL, image_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
        close(fd);
    }
    if (out_map == MAP_FAILED) {
        fprintf(stderr, "[baseline] mmap output: %s\n", strerror(errno));
        free(descs); free(live);
        return -1;
    }
    (void)madvise(out_map, image_size, MADV_HUGEPAGE);

    /* Write file header + descriptor table. Version 4 = in-app baseline,
     * ciphertext gpu_section, 28-byte ckpt_page_meta_t per 2 MB chunk. */
    v2_file_hdr_t fhdr = {
        .magic       = CKPT_V2_MAGIC,
        .version     = CKPT_V4_VERSION,
        .num_ranges  = n_live,
        .page_size   = 4096,
        .total_bytes = total_bytes,
    };
    memcpy(out_map, &fhdr, sizeof(fhdr));
    memcpy(out_map + sizeof(fhdr), descs,
           (size_t)n_live * sizeof(v2_range_desc_t));

    /* Pre-fill all resmaps with value 1 (= GPU-resident).
     * Also zero cpu_section (none) and we'll fill gpu_section via cudaMemcpy. */
    double t_p0 = now_ms_local();
    for (uint32_t i = 0; i < n_live; i++) {
        memset(out_map + descs[i].data_offset, 1, (size_t)descs[i].resmap_size);
    }
    t_prefill_ms = now_ms_local() - t_p0;
    t_setup_ms = now_ms_local() - t_start;

    /* ---- Phase 3: pinned staging slots + copy stream(s) + writer pool --
     *
     * n_mstreams_in (from caller, sourced from gate->baseline_mstreams)
     * creates N CUDA streams. Slots in the shared ring are bound to streams:
     * slot i → stream (i * n_mstreams / STAGING_RING_SLOTS). Main thread
     * calls bl_pool_acquire() which returns the first free slot (LIFO). The
     * stream whose chunk finished first released its slot first, so the
     * next launch naturally targets the stream with available capacity —
     * no round-robin counter, no stalls waiting on a slow stream.
     *
     * Single launcher thread = no concurrent cudaMemcpyAsync calls = no
     * CC-mode bounce-buffer launch contention (bench_mstream Test 7 showed
     * that kills multi-thread-launch variants). The P1f hypothesis under
     * test here is: with multiple streams but a single launcher, does
     * the driver actually schedule D2H across streams concurrently? */
    int n_mstreams = n_mstreams_in;
    if (n_mstreams < 1)  n_mstreams = 1;
    if (n_mstreams > STAGING_RING_SLOTS) n_mstreams = STAGING_RING_SLOTS;

    uint8_t   *staging[STAGING_RING_SLOTS] = {0};
    cudaEvent_t events[STAGING_RING_SLOTS] = {0};
    int         slot_to_stream[STAGING_RING_SLOTS];
    for (int i = 0; i < STAGING_RING_SLOTS; i++)
        slot_to_stream[i] = (i * n_mstreams) / STAGING_RING_SLOTS;

    cudaStream_t copy_streams[STAGING_RING_SLOTS] = {0};
    for (int si = 0; si < n_mstreams; si++) {
        cerr = cudaStreamCreateWithFlags(&copy_streams[si], cudaStreamNonBlocking);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[baseline] cudaStreamCreate[%d]: %s\n",
                    si, cudaGetErrorString(cerr));
            for (int sj = 0; sj < si; sj++) cudaStreamDestroy(copy_streams[sj]);
            munmap(out_map, image_size);
            free(descs); free(live);
            return -1;
        }
    }
    if (n_mstreams > 1)
        fprintf(stderr, "[baseline] P1f multi-stream: %d streams, %d slots/stream (slot-bound)\n",
                n_mstreams, STAGING_RING_SLOTS / n_mstreams);

    for (int s = 0; s < STAGING_RING_SLOTS; s++) {
        cerr = cudaMallocHost((void **)&staging[s], CHUNK_BYTES);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[baseline] cudaMallocHost slot %d: %s\n",
                    s, cudaGetErrorString(cerr));
            for (int j = 0; j < s; j++) cudaFreeHost(staging[j]);
            cudaStreamDestroy(copy_streams);
            munmap(out_map, image_size);
            free(descs); free(live);
            return -1;
        }
        cudaEventCreateWithFlags(&events[s], cudaEventDisableTiming);
    }

    /* Initialize work queue, slot pool, and worker threads. */
    bl_wq_t   wq;
    bl_pool_t pool;
    if (bl_wq_init(&wq, STAGING_RING_SLOTS * 2) < 0 ||
        bl_pool_init(&pool, STAGING_RING_SLOTS) < 0) {
        fprintf(stderr, "[baseline] wq/pool init failed\n");
        for (int s = 0; s < STAGING_RING_SLOTS; s++) {
            cudaFreeHost(staging[s]);
            cudaEventDestroy(events[s]);
        }
        cudaStreamDestroy(copy_streams);
        munmap(out_map, image_size);
        free(descs); free(live);
        return -1;
    }

    pthread_t        workers[N_WRITERS];
    bl_worker_ctx_t  wctx[N_WRITERS];
    for (int w = 0; w < N_WRITERS; w++) {
        memset(&wctx[w], 0, sizeof(wctx[w]));
        wctx[w].wq     = &wq;
        wctx[w].pool   = &pool;
        wctx[w].events = events;
        wctx[w].k3_key = s_k3_key;
        wctx[w].id     = w;
        pthread_create(&workers[w], NULL, bl_writer_thread, &wctx[w]);
    }

    /* ---- Phase 4: main thread launches D2H + enqueues work --------- */
    int err = 0;
    double t_hot_start = now_ms_local();

    /* Track global 2 MB chunk index within this run for IV derivation. */
    uint64_t chunk_counter = 0;

    for (uint32_t i = 0; i < n_live && !err; i++) {
        uint64_t base_va         = descs[i].base_va;
        uint64_t length          = descs[i].length;
        uint64_t gpu_section_off = descs[i].data_offset + descs[i].resmap_size;
        uint64_t meta_section_off = gpu_section_off + descs[i].gpu_bytes;
        uint64_t copied = 0;

        while (copied < length) {
            uint64_t this_chunk = length - copied;
            if (this_chunk > CHUNK_BYTES) this_chunk = CHUNK_BYTES;
            /* Enforce 2 MB alignment (safe since length is 2 MB aligned). */
            this_chunk &= ~(CRYPTO_CHUNK_BYTES - 1);
            if (this_chunk == 0) this_chunk = CRYPTO_CHUNK_BYTES;

            /* Acquire a free pinned staging slot (may block if all 8 in-flight).
             * Each slot is bound to a fixed stream via slot_to_stream[] so the
             * first-freed-slot semantics naturally dispatch to the stream with
             * available capacity (no round-robin, no slow-stream stalls). */
            int slot = bl_pool_acquire(&pool);
            cudaStream_t use_stream = copy_streams[slot_to_stream[slot]];

            /* Issue D2H into the slot's staging buffer */
            double tl = now_ms_local();
            cerr = cudaMemcpyAsync(staging[slot],
                                   (const void *)(uintptr_t)(base_va + copied),
                                   (size_t)this_chunk,
                                   cudaMemcpyDeviceToHost,
                                   use_stream);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "[baseline] cudaMemcpyAsync alloc[%u] off=%llu: %s\n",
                        i, (unsigned long long)copied, cudaGetErrorString(cerr));
                bl_pool_release(&pool, slot);
                err = 1;
                break;
            }
            cudaEventRecord(events[slot], use_stream);
            acc_launch_ms += now_ms_local() - tl;
            bytes_d2h += (uint64_t)this_chunk;

            /* Compute per-chunk meta location and IV base. Meta for the
             * chunk starts at (range_meta_base + chunks_before_in_range). */
            uint64_t chunks_in_range_before = copied / CRYPTO_CHUNK_BYTES;
            ckpt_page_meta_t *meta_ptr =
                (ckpt_page_meta_t *)(out_map + meta_section_off) + chunks_in_range_before;

            bl_work_t item = {
                .slot    = slot,
                .src     = staging[slot],
                .dst     = out_map + gpu_section_off + copied,
                .bytes   = (size_t)this_chunk,
                .meta    = meta_ptr,
                .base_iv = run_base_iv + chunk_counter,
            };
            bl_wq_push(&wq, item);

            chunk_counter += this_chunk / CRYPTO_CHUNK_BYTES;
            copied += this_chunk;
        }

        /* Per-alloc progress chatter — gated behind CKPT_VERBOSE since
         * a vLLM checkpoint produces 200+ lines that mostly drown out
         * the summary banners. Read once per process (cheap getenv). */
        static int s_verbose = -1;
        if (s_verbose < 0) {
            const char *vv = getenv("CKPT_VERBOSE");
            s_verbose = (vv && atoi(vv) > 0) ? 1 : 0;
        }
        if (s_verbose && ((i & 0xF) == 0 || i == n_live - 1))
            fprintf(stderr, "[baseline] alloc[%u/%u] queued (%.2f MB)\n",
                    i + 1, n_live, length / (1024.0*1024.0));
    }

    /* ---- Phase 5: close queue, join workers, cleanup --------------- */
    bl_wq_close(&wq);
    for (int w = 0; w < N_WRITERS; w++) {
        pthread_join(workers[w], NULL);
        acc_wait_ms    += wctx[w].wait_ms;
        acc_encrypt_ms += wctx[w].encrypt_ms;
        bytes_encrypt  += wctx[w].bytes;
        if (wctx[w].err) err = 1;
    }
    t_hot_wall = now_ms_local() - t_hot_start;

    bl_pool_destroy(&pool);
    bl_wq_destroy(&wq);

    for (int s = 0; s < STAGING_RING_SLOTS; s++) {
        if (staging[s]) cudaFreeHost(staging[s]);
        if (events[s])  cudaEventDestroy(events[s]);
    }
    for (int si = 0; si < n_mstreams; si++)
        cudaStreamDestroy(copy_streams[si]);

    if (!err && !g_no_image_write) {
        msync(out_map, image_size, MS_ASYNC);
    }
    munmap(out_map, image_size);
    free(descs);
    free(live);

    if (err) {
        if (!g_no_image_write) unlink(out_path);
        return -1;
    }

    double elapsed = now_ms_local() - t_start;
    double gbps    = elapsed > 0
                     ? total_bytes / (1024.0*1024.0*1024.0) / (elapsed / 1000.0)
                     : 0.0;

    fprintf(stderr, "[baseline] DONE  %.2f GB in %.1f ms  (%.2f GB/s)\n",
            total_bytes / (1024.0*1024.0*1024.0), elapsed, gbps);

    /* Per-phase breakdown (diagnostic) ------------------------------ */
    {
        double gb_d2h     = bytes_d2h     / (1024.0*1024.0*1024.0);
        double gb_encrypt = bytes_encrypt / (1024.0*1024.0*1024.0);
        double bw_encrypt_per_thread =
            acc_encrypt_ms > 0 ? gb_encrypt / (acc_encrypt_ms / 1000.0) : 0.0;
        double bw_encrypt_agg =
            t_hot_wall     > 0 ? gb_encrypt / (t_hot_wall     / 1000.0) : 0.0;
        fprintf(stderr, "[baseline] phase breakdown (writer pool x%d, %d copy stream(s), k3 AES-GCM 2MB):\n",
                N_WRITERS, n_mstreams);
        fprintf(stderr, "  keygen + save        : %8.1f ms\n", t_keygen_ms);
        fprintf(stderr, "  quiesce              : %8.1f ms\n", t_quiesce_ms);
        fprintf(stderr, "  setup (mmap/hdr)     : %8.1f ms  (prefill resmap %.1f ms)\n",
                t_setup_ms, t_prefill_ms);
        fprintf(stderr, "  main: launch overhead: %8.1f ms  (D2H launched from main thread)\n",
                acc_launch_ms);
        fprintf(stderr, "  hot phase wall       : %8.1f ms  (launch + pool-wait + join)\n",
                t_hot_wall);
        fprintf(stderr, "  workers wait D2H (sum over %d): %8.1f ms\n",
                N_WRITERS, acc_wait_ms);
        fprintf(stderr, "  workers encrypt (sum over %d) : %8.1f ms  (%.2f GB @ %.2f GB/s/thread)\n",
                N_WRITERS, acc_encrypt_ms, gb_encrypt, bw_encrypt_per_thread);
        fprintf(stderr, "  aggregate encrypt BW : %.2f GB/s  (%.2f GB / %.1f ms hot wall)\n",
                bw_encrypt_agg, gb_encrypt, t_hot_wall);
        fprintf(stderr, "  d2h total            : %.2f GB\n", gb_d2h);
        fprintf(stderr, "  IV range used        : [%llu, %llu)\n",
                (unsigned long long)run_base_iv,
                (unsigned long long)(run_base_iv + total_2mb_chunks));
    }

    if (result_out) {
        result_out->elapsed_ms = elapsed;
        result_out->bytes      = total_bytes;
        result_out->gbps       = gbps;
    }
    return 0;
}

/* ============================================================== */
/* ckpt_baseline_delta — P1e delta pass                             */
/*                                                                  */
/* Iterates a list of dirty 2 MB-aligned ranges from the kernel     */
/* dirty tracker, runs the same D2H + encrypt pipeline as the       */
/* baseline, and writes a delta-v4 format image.                    */
/*                                                                  */
/* Assumes s_k3_key is already generated (baseline must run first   */
/* within the process). IV counter continues from where the         */
/* baseline left off so ciphertexts never reuse a (key, IV) pair.   */
/* ============================================================== */
static int _ckpt_baseline_delta_body(const char *delta_path,
                                      const ckpt_dirty_range_t *ranges,
                                      uint32_t n_ranges,
                                      ckpt_baseline_result_t *result_out)
{
    if (!delta_path || !ranges) {
        fprintf(stderr, "[delta] invalid args\n");
        return -1;
    }
    if (n_ranges == 0) {
        /* Empty delta — overwrite the header (num_blocks=0) at offset 0
         * without truncating the file. Restore reads num_blocks first and
         * skips the rest. Keeps any pre-warmed tmpfs pages intact. */
        int fd = open(delta_path, O_WRONLY | O_CREAT, 0666);
        if (fd < 0) return -1;
        (void)fchmod(fd, 0666);
        delta_v4_hdr_t hdr = { CKPT_DELTA_V4_MAGIC, CKPT_V4_VERSION, 0,
                               (uint32_t)CRYPTO_CHUNK_BYTES, 0 };
        ssize_t w = pwrite(fd, &hdr, sizeof(hdr), 0);
        close(fd);
        if (result_out) {
            result_out->elapsed_ms = 0;
            result_out->bytes      = 0;
            result_out->gbps       = 0;
        }
        return (w == sizeof(hdr)) ? 0 : -1;
    }

    double t_start = now_ms_local();

    /* Ensure the key is ready (baseline must have run first in this process). */
    pthread_mutex_lock(&s_k3_lock);
    if (!s_k3_ready) {
        pthread_mutex_unlock(&s_k3_lock);
        fprintf(stderr, "[delta] FATAL: k3 key not initialized — baseline must run first\n");
        return -1;
    }
    pthread_mutex_unlock(&s_k3_lock);

    /* Validate + compute layout. Each range must be 2 MB aligned/multiple. */
    delta_v4_block_t *blocks = calloc(n_ranges, sizeof(delta_v4_block_t));
    if (!blocks) { fprintf(stderr, "[delta] calloc blocks\n"); return -1; }

    uint64_t hdr_off = sizeof(delta_v4_hdr_t)
                     + (uint64_t)n_ranges * sizeof(delta_v4_block_t);
    uint64_t cur_off = hdr_off;
    uint64_t total_bytes       = 0;
    uint64_t total_2mb_chunks  = 0;

    for (uint32_t i = 0; i < n_ranges; i++) {
        uint64_t va   = ranges[i].va;
        uint64_t len  = ranges[i].size;
        if ((va % CRYPTO_CHUNK_BYTES) != 0 || (len % CRYPTO_CHUNK_BYTES) != 0 || len == 0) {
            fprintf(stderr, "[delta] range[%u] va=0x%llx size=%llu not 2MB aligned\n",
                    i, (unsigned long long)va, (unsigned long long)len);
            free(blocks);
            return -1;
        }
        uint64_t n_2mb      = len / CRYPTO_CHUNK_BYTES;
        uint64_t meta_bytes = n_2mb * sizeof(ckpt_page_meta_t);

        blocks[i].base_va           = va;
        blocks[i].length            = len;
        blocks[i].data_offset       = cur_off;
        blocks[i].crypto_meta_bytes = meta_bytes;
        cur_off     += len + meta_bytes;
        total_bytes += len;
        total_2mb_chunks += n_2mb;
    }
    uint64_t image_size = cur_off;

    fprintf(stderr, "[delta] %u ranges, %.2f MB total, image=%.2f MB → %s\n",
            n_ranges, total_bytes / (1024.0*1024.0),
            image_size / (1024.0*1024.0), delta_path);

    /* Reserve IV range continuing from s_iv_base. */
    pthread_mutex_lock(&s_k3_lock);
    uint64_t run_base_iv = s_iv_base;
    s_iv_base += total_2mb_chunks;
    pthread_mutex_unlock(&s_k3_lock);

    /* Open WITHOUT O_TRUNC for consistency with baseline path — reuse warm
     * tmpfs pages from a prior ckpt_prealloc run (delta is small ~10 MB so
     * the gain is minimal, but the pattern is the same).
     * Mode 0666: same reason as baseline path. */
    fprintf(stderr, "[delta] %s...\n",
            g_no_image_write ? "anon mmap (no-image-write)" : "open/ftruncate/mmap");
    uint8_t *out_map;
    if (g_no_image_write) {
        out_map = mmap(NULL, image_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        int fd = open(delta_path, O_RDWR | O_CREAT, 0666);
        if (fd < 0) {
            fprintf(stderr, "[delta] open(%s): %s\n", delta_path, strerror(errno));
            free(blocks); return -1;
        }
        (void)fchmod(fd, 0666);
        {
            struct stat dst;
            if (fstat(fd, &dst) < 0) {
                fprintf(stderr, "[delta] fstat: %s\n", strerror(errno));
                close(fd); free(blocks); return -1;
            }
            if ((uint64_t)dst.st_size < image_size) {
                if (ftruncate(fd, (off_t)image_size) < 0) {
                    fprintf(stderr, "[delta] ftruncate failed: %s\n", strerror(errno));
                    close(fd); free(blocks); return -1;
                }
            }
        }
        out_map = mmap(NULL, image_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
        close(fd);
    }
    if (out_map == MAP_FAILED) {
        fprintf(stderr, "[delta] mmap failed: %s\n", strerror(errno));
        free(blocks); return -1;
    }
    (void)madvise(out_map, image_size, MADV_HUGEPAGE);
    fprintf(stderr, "[delta] out_map ready (%.2f MB)\n", image_size/(1024.0*1024.0));

    /* Header + block table */
    delta_v4_hdr_t hdr = {
        .magic       = CKPT_DELTA_V4_MAGIC,
        .version     = CKPT_V4_VERSION,
        .num_blocks  = n_ranges,
        .page_size   = (uint32_t)CRYPTO_CHUNK_BYTES,
        .total_bytes = total_bytes,
    };
    memcpy(out_map, &hdr, sizeof(hdr));
    memcpy(out_map + sizeof(hdr), blocks,
           (size_t)n_ranges * sizeof(delta_v4_block_t));

    /* Delta uses smaller staging (8 MB × 4 slots = 32 MB pinned) to avoid
     * contending with the baseline's recently freed 2 GB of pinned memory. */
    uint8_t   *staging[DELTA_STAGING_RING_SLOTS] = {0};
    cudaEvent_t events[DELTA_STAGING_RING_SLOTS] = {0};

    fprintf(stderr, "[delta] cudaStreamCreate...\n");
    cudaStream_t copy_stream = NULL;
    cudaError_t cerr = cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[delta] cudaStreamCreate: %s\n", cudaGetErrorString(cerr));
        munmap(out_map, image_size);
        free(blocks);
        return -1;
    }
    fprintf(stderr, "[delta] cudaMallocHost x%d (%.0f MB each)...\n",
            DELTA_STAGING_RING_SLOTS, DELTA_CHUNK_BYTES/(1024.0*1024.0));
    for (int s = 0; s < DELTA_STAGING_RING_SLOTS; s++) {
        cerr = cudaMallocHost((void **)&staging[s], DELTA_CHUNK_BYTES);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[delta] cudaMallocHost slot %d: %s\n",
                    s, cudaGetErrorString(cerr));
            for (int j = 0; j < s; j++) cudaFreeHost(staging[j]);
            cudaStreamDestroy(copy_stream);
            munmap(out_map, image_size);
            free(blocks);
            return -1;
        }
        cudaEventCreateWithFlags(&events[s], cudaEventDisableTiming);
    }
    fprintf(stderr, "[delta] staging buffers ready, spawning workers...\n");

    bl_wq_t   wq;
    bl_pool_t pool;
    if (bl_wq_init(&wq, DELTA_STAGING_RING_SLOTS * 2) < 0 ||
        bl_pool_init(&pool, DELTA_STAGING_RING_SLOTS) < 0) {
        for (int s = 0; s < DELTA_STAGING_RING_SLOTS; s++) {
            cudaFreeHost(staging[s]);
            cudaEventDestroy(events[s]);
        }
        cudaStreamDestroy(copy_stream);
        munmap(out_map, image_size);
        free(blocks);
        return -1;
    }

    pthread_t        workers[N_WRITERS];
    bl_worker_ctx_t  wctx[N_WRITERS];
    for (int w = 0; w < N_WRITERS; w++) {
        memset(&wctx[w], 0, sizeof(wctx[w]));
        wctx[w].wq     = &wq;
        wctx[w].pool   = &pool;
        wctx[w].events = events;
        wctx[w].k3_key = s_k3_key;
        wctx[w].id     = w;
        pthread_create(&workers[w], NULL, bl_writer_thread, &wctx[w]);
    }
    fprintf(stderr, "[delta] dispatching %u ranges...\n", n_ranges);

    /* Dispatch: iterate dirty ranges, split into <=256 MB D2H chunks. */
    int err = 0;
    uint64_t chunk_counter = 0;
    double t_hot_start = now_ms_local();

    for (uint32_t i = 0; i < n_ranges && !err; i++) {
        uint64_t base_va   = blocks[i].base_va;
        uint64_t length    = blocks[i].length;
        uint64_t data_off  = blocks[i].data_offset;
        uint64_t meta_off  = data_off + length;
        uint64_t copied    = 0;

        while (copied < length) {
            uint64_t this_chunk = length - copied;
            if (this_chunk > DELTA_CHUNK_BYTES) this_chunk = DELTA_CHUNK_BYTES;
            this_chunk &= ~(CRYPTO_CHUNK_BYTES - 1);
            if (this_chunk == 0) this_chunk = CRYPTO_CHUNK_BYTES;

            int slot = bl_pool_acquire(&pool);
            cerr = cudaMemcpyAsync(staging[slot],
                                   (const void *)(uintptr_t)(base_va + copied),
                                   (size_t)this_chunk,
                                   cudaMemcpyDeviceToHost, copy_stream);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "[delta] cudaMemcpyAsync range[%u] off=%llu: %s\n",
                        i, (unsigned long long)copied, cudaGetErrorString(cerr));
                bl_pool_release(&pool, slot);
                err = 1; break;
            }
            cudaEventRecord(events[slot], copy_stream);

            uint64_t chunks_before = copied / CRYPTO_CHUNK_BYTES;
            ckpt_page_meta_t *meta_ptr =
                (ckpt_page_meta_t *)(out_map + meta_off) + chunks_before;

            bl_work_t item = {
                .slot    = slot,
                .src     = staging[slot],
                .dst     = out_map + data_off + copied,
                .bytes   = (size_t)this_chunk,
                .meta    = meta_ptr,
                .base_iv = run_base_iv + chunk_counter,
            };
            bl_wq_push(&wq, item);

            chunk_counter += this_chunk / CRYPTO_CHUNK_BYTES;
            copied += this_chunk;
        }
    }

    fprintf(stderr, "[delta] dispatch done, closing wq...\n");
    bl_wq_close(&wq);
    for (int w = 0; w < N_WRITERS; w++) {
        fprintf(stderr, "[delta] joining worker %d...\n", w);
        pthread_join(workers[w], NULL);
        if (wctx[w].err) err = 1;
    }
    double t_hot_wall = now_ms_local() - t_hot_start;
    fprintf(stderr, "[delta] all workers joined, wall=%.1f ms\n", t_hot_wall);

    bl_pool_destroy(&pool);
    bl_wq_destroy(&wq);
    for (int s = 0; s < DELTA_STAGING_RING_SLOTS; s++) {
        if (staging[s]) cudaFreeHost(staging[s]);
        if (events[s])  cudaEventDestroy(events[s]);
    }
    cudaStreamDestroy(copy_stream);

    if (!err && !g_no_image_write) msync(out_map, image_size, MS_ASYNC);
    munmap(out_map, image_size);
    free(blocks);

    if (err) {
        if (!g_no_image_write) unlink(delta_path);
        return -1;
    }

    double elapsed = now_ms_local() - t_start;
    double gbps    = elapsed > 0
                     ? total_bytes / (1024.0*1024.0*1024.0) / (elapsed / 1000.0)
                     : 0.0;

    fprintf(stderr, "[delta] DONE  %.2f MB in %.1f ms  (%.2f GB/s, hot=%.1f ms)\n",
            total_bytes / (1024.0*1024.0), elapsed, gbps, t_hot_wall);

    if (result_out) {
        result_out->elapsed_ms = elapsed;
        result_out->bytes      = total_bytes;
        result_out->gbps       = gbps;
    }
    return 0;
}

/* ============================================================== */
/* Public wrappers — set/clear the TLS reentrancy guard around    */
/* the dispatch bodies so libckpt_vllm.c's cudaMemcpyAsync         */
/* interceptor can detect "this call came from our own dispatch"  */
/* and skip its maybe_checkpoint_post_sync side effect, which     */
/* would otherwise recursively re-enter the AT_BOUNDARY loop      */
/* (deadlock on g_ckpt_mutex).                                     */
/* ============================================================== */
int ckpt_baseline_run(const char *out_path,
                      const ckpt_alloc_entry_t *allocs,
                      uint32_t n_allocs,
                      int stop_the_world,
                      int n_mstreams,
                      ckpt_baseline_result_t *result_out)
{
    g_in_ckpt_worker = 1;
    int rc = _ckpt_baseline_run_body(out_path, allocs, n_allocs,
                                      stop_the_world, n_mstreams, result_out);
    g_in_ckpt_worker = 0;
    return rc;
}

int ckpt_baseline_delta(const char *delta_path,
                        const ckpt_dirty_range_t *ranges,
                        uint32_t n_ranges,
                        ckpt_baseline_result_t *result_out)
{
    g_in_ckpt_worker = 1;
    int rc = _ckpt_baseline_delta_body(delta_path, ranges, n_ranges, result_out);
    g_in_ckpt_worker = 0;
    return rc;
}
