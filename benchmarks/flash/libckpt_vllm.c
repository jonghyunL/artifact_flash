/**
 * libckpt_inference.c — LLM inference-optimized GPU checkpoint via LD_PRELOAD
 *
 * Optimized for LLM inference (vLLM, nano-vllm) and LoRA fine-tuning:
 *   - Model weights captured at cudaMemcpy H2D time (zero checkpoint cost)
 *   - KV cache: no prefetch → only used blocks become GPU-resident
 *   - At checkpoint: agent only precopy resident KV pages (not all 63GB)
 *   - Selective prefetch: model weights prefetched, KV cache faults gradually
 *
 * Key insight: model weights are loaded via cudaMemcpy H2D and never change.
 * We capture them from the HOST buffer during the H2D copy — no GPU read needed.
 * KV cache is written by GPU kernels only, never via H2D.
 *
 * Usage:
 *   rm -f /tmp/ckpt_gate
 *   LD_PRELOAD=./libckpt_inference.so CKPT_WEIGHTS_DIR=/tmp/ckpt_weights \
 *       python app.py
 *
 * Env vars:
 *   CKPT_GATE_FILE      gate file path (default: /tmp/ckpt_gate)
 *   CKPT_WEIGHTS_DIR    directory for weight images (default: /tmp/ckpt_weights)
 *   CKPT_VERBOSE        set "1" for verbose logging
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "ckpt_gate.h"
#include "ckpt_crypto.h"
#include "librst_vllm.h"
#include "ckpt_baseline.h"

/* Defined in ckpt_baseline.c — set while we're inside our own dispatch
 * loops so the cudaMemcpyAsync interceptor below can skip its recursive
 * maybe_checkpoint_post_sync side effect (which would deadlock on
 * g_ckpt_mutex when called from inside the AT_BOUNDARY polling loop). */
extern __thread int g_in_ckpt_worker;

/* ------------------------------------------------------------------ */
/* Function pointer types                                              */
/* ------------------------------------------------------------------ */
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);
typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);
typedef cudaError_t (*cudaMallocManaged_fn)(void **, size_t, unsigned int);
typedef cudaError_t (*cudaMalloc_fn)(void **, size_t);
typedef cudaError_t (*cudaMallocAsync_fn)(void **, size_t, cudaStream_t);
typedef cudaError_t (*cudaFree_fn)(void *);
typedef cudaError_t (*cudaFreeAsync_fn)(void *, cudaStream_t);
typedef CUresult    (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult    (*cuMemFree_v2_fn)(CUdeviceptr);
typedef cudaError_t (*cudaMemcpyAsync_fn)(void *, const void *, size_t,
                                          enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaMemcpy_fn)(void *, const void *, size_t,
                                     enum cudaMemcpyKind);
typedef cudaError_t (*cudaMemAdvise_fn)(const void *, size_t,
                                        enum cudaMemoryAdvise, int);
typedef cudaError_t (*cudaMemPrefetchAsync_fn)(const void *, size_t, int, cudaStream_t);

/* ------------------------------------------------------------------ */
/* Global state                                                        */
/* ------------------------------------------------------------------ */
static ckpt_gate_t *g_gate = NULL;
static char g_gate_path[256] = GATE_FILE_DEFAULT;
static char g_weights_dir[256] = "/tmp/ckpt_weights";
static int g_verbose = 0;
static int g_is_owner = 0;

static pthread_mutex_t g_ckpt_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;
static volatile int g_init_signaled = 0;

/* Real CUDA function pointers */
static cudaStreamSynchronize_fn     real_cudaStreamSynchronize = NULL;
static cudaDeviceSynchronize_fn     real_cudaDeviceSynchronize = NULL;
static cudaMallocManaged_fn         real_cudaMallocManaged = NULL;
static cudaMalloc_fn                real_cudaMalloc = NULL;
static cudaMallocAsync_fn           real_cudaMallocAsync = NULL;
static cudaFree_fn                  real_cudaFree = NULL;
static cudaFreeAsync_fn             real_cudaFreeAsync = NULL;
static cuMemAlloc_v2_fn             real_cuMemAlloc_v2 = NULL;
static cuMemFree_v2_fn              real_cuMemFree_v2 = NULL;
static cudaMemcpyAsync_fn           real_cudaMemcpyAsync_fn_ptr = NULL;
static cudaMemcpy_fn                real_cudaMemcpy_fn_ptr = NULL;
static cudaMemAdvise_fn             real_cudaMemAdvise = NULL;
static cudaMemPrefetchAsync_fn      real_cudaMemPrefetchAsync = NULL;

/* k3 key for weight encryption */
static uint8_t g_k3_key[CKPT_CRYPTO_KEY_SIZE];
static int g_k3_ready = 0;
static uint64_t g_k3_iv_counter = 0;
static pthread_mutex_t g_k3_lock = PTHREAD_MUTEX_INITIALIZER;

/* Weight save mode: read from gate file config_flags at startup */
static int g_save_weights = 0;

/* Weight save tracking */
static uint32_t g_weights_saved = 0;
static uint64_t g_weights_bytes = 0;

/* ------------------------------------------------------------------ */
/* Restore mode state — mmap-based per-alloc streaming restore         */
/* ------------------------------------------------------------------ */

/* Minimal v2 format structs (must match ckpt_core.c / librst_vllm.c) */
typedef struct __attribute__((packed)) {
    uint32_t magic; uint32_t version; uint32_t num_ranges;
    uint32_t page_size; uint64_t total_bytes;
} rst_v2_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va; uint64_t length; uint64_t num_pages;
    uint64_t data_offset; uint64_t resmap_size;
    uint64_t cpu_bytes; uint64_t gpu_bytes; uint64_t crypto_meta_bytes;
} rst_v2_desc_t;

typedef struct __attribute__((packed)) {
    uint32_t magic; uint32_t version; uint32_t num_blocks;
    uint32_t page_size; uint64_t alloc_start_va; uint64_t alloc_size;
    uint64_t num_dirty_pages_64k;
} rst_delta_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va; uint64_t length; uint64_t num_pages;
    uint64_t data_offset; uint64_t resmap_size;
    uint64_t cpu_bytes; uint64_t gpu_bytes;
} rst_delta_block_t;

#define RST_V2_MAGIC       0xC2C2C2C2u
#define RST_DELTA_MAGIC    0xDE17A001u
#define RST_V2_VERSION     2u
#define RST_V3_VERSION     3u   /* kernel ioctl 115 path: k1-encrypted GPU data */
#define RST_V4_VERSION     4u   /* in-app baseline: AES-GCM 2MB chunks */
#define RST_CRYPTO_CHUNK_BYTES  (2ULL * 1024 * 1024)

/* ---- V3 k1 decrypt via ioctl 112 ----
 * Mirrors the definitions in ckpt_core.c / librst_vllm.c. These must match
 * kernel-open/nvidia-uvm/uvm_ioctl.h UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES
 * and UVM_LIVE_MIGRATION_PAGE_CRYPTO_META. Layout verified against librst. */
#define UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES  112
#define RST_UVM_INITIALIZE                          0x30000001

typedef struct __attribute__((packed)) {
    uint64_t size;         /* ciphertext bytes this entry covers */
    uint8_t  iv[12];
    uint8_t  iv_fresh;
    uint8_t  auth_tag[16];
    uint32_t key_version;
} rst_page_crypto_meta_t;   /* 41 bytes, must match checkpoint-side page_crypto_meta_t */

typedef struct { uint8_t uuid[16]; } rst_nv_uuid_t;

typedef struct {
    uint64_t cipher_buf    __attribute__((aligned(8)));
    uint64_t plain_buf     __attribute__((aligned(8)));
    uint64_t total_size    __attribute__((aligned(8)));
    uint64_t crypto_meta   __attribute__((aligned(8)));
    uint64_t num_transfers __attribute__((aligned(8)));
    rst_nv_uuid_t gpu_uuid;
    uint32_t rmStatus;
} rst_uvm_decrypt_params_t;

typedef struct {
    uint64_t flags __attribute__((aligned(8)));
    uint32_t rmStatus;
} rst_uvm_init_params_t;

static int                    g_restore_mode = 0;
static int                    g_restore_pending = 0;

/* Snap gate data (read in constructor, before CUDA) */
static uint32_t               g_restore_alloc_total = 0;
static uint64_t              *g_restore_alloc_va = NULL;
static uint64_t              *g_restore_alloc_size = NULL;

/* mmap'd checkpoint files (set in constructor, read per-alloc) */
static uint8_t               *g_restore_base_map = NULL;
static size_t                 g_restore_base_size = 0;
static rst_v2_hdr_t          *g_restore_base_hdr = NULL;
static rst_v2_desc_t         *g_restore_base_descs = NULL;

static uint8_t               *g_restore_delta_map = NULL;
static size_t                 g_restore_delta_size = 0;

/* Staging buffer for per-alloc H2D copy (allocated once, reused) */
static uint8_t               *g_restore_staging = NULL;
static size_t                 g_restore_staging_size = 0;

/* V4 (in-app baseline) decrypt state */
static int                    g_restore_v4 = 0;
static uint8_t                g_restore_k3_key[CKPT_CRYPTO_KEY_SIZE];

/* V3 (kernel ioctl 115 / k1) decrypt state — per-alloc ioctl 112 call */
static int                    g_restore_v3 = 0;
static int                    g_restore_uvm_fd = -1;

/* V4 delta overlay state — parallel to baseline structs, for the delta image */
typedef struct __attribute__((packed)) {
    uint32_t magic; uint32_t version; uint32_t num_blocks;
    uint32_t page_size; uint64_t total_bytes;
} rst_delta_v4_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va; uint64_t length;
    uint64_t data_offset; uint64_t crypto_meta_bytes;
} rst_delta_v4_block_t;

#define RST_DELTA_V4_MAGIC    0xDE17A004u
#define RST_CRYPTO_CHUNK_BYTES_LOCAL  (2ULL * 1024 * 1024)

static int                        g_restore_delta_v4 = 0;
static rst_delta_v4_hdr_t        *g_restore_delta_hdr    = NULL;
static rst_delta_v4_block_t      *g_restore_delta_blocks = NULL;

/* P6 — static.img (k3-encrypted at 2 MB chunks, same format as v4 delta).
 * Mmap'd at restore init when CKPT_RESTORE_STATIC env var is set.
 * In restore_apply_alloc we look up the alloc's VA in static descriptors
 * first; if matched, decrypt from static.img mmap and skip base+delta
 * processing for that alloc. */
static uint8_t               *g_restore_static_map = NULL;
static size_t                 g_restore_static_size = 0;
static rst_v2_hdr_t          *g_restore_static_hdr = NULL;
static rst_v2_desc_t         *g_restore_static_descs = NULL;
static uint32_t               g_restore_static_allocs_applied = 0;
static uint64_t               g_restore_static_bytes_applied  = 0;

/* Diagnostic counters — answer "where did the missing allocs go?"
 * after a run. Incremented by restore_apply_alloc paths. */
static uint32_t               g_restore_invocations           = 0;
static uint32_t               g_restore_base_applied          = 0;
static uint32_t               g_restore_passthrough_past_snap = 0;
static uint32_t               g_restore_passthrough_no_desc   = 0;
/* Size-alignment diagnostic — does snap[idx].size match the actual new_size?
 * If most allocs are exact-match, idx-based alignment is preserved (semantics
 * likely correct). If many are mismatch, cudaMalloc order shuffled and our
 * restore is putting data at wrong tensors. */
static uint32_t               g_restore_size_exact            = 0;
static uint32_t               g_restore_size_close_2mb        = 0;
static uint32_t               g_restore_size_mismatch         = 0;
/* Stats for destructor summary — so we can verify delta was applied
 * without needing CKPT_VERBOSE=1 in normal runs. */
static uint32_t                   g_restore_delta_allocs_touched = 0;
static uint32_t                   g_restore_delta_overlays       = 0;
static uint64_t                   g_restore_delta_bytes_applied  = 0;

/* Counters — in shared anonymous mmap so they survive fork().
 * vLLM's engine subprocess inherits g_restore_mode=1 and does the actual
 * cudaMallocs. The parent prints the summary in its destructor. Without
 * shared memory, parent's counters stay at 0.
 *
 * Layout in the shared page (all 8-byte aligned, use atomics for updates):
 *   [ 0.. 3]  done_count         (uint32_t, atomic)
 *   [ 8..15]  bytes_restored     (uint64_t, atomic)
 *   [16..23]  sum_us             (uint64_t, atomic) — sum of per-alloc CPU time
 *   [24..31]  first_start_us     (uint64_t, CAS-once) — earliest restore call
 *   [32..39]  last_end_us        (uint64_t, max-update) — latest restore call
 *   [40..47]  dec_us             (uint64_t, atomic) — sum of per-alloc decrypt us
 *   [48..55]  h2d_us             (uint64_t, atomic) — sum of per-alloc H2D us */
static void                  *g_restore_shared_page  = NULL;
static uint32_t              *g_restore_shared_done   = NULL;
static uint64_t              *g_restore_shared_bytes  = NULL;
static uint64_t              *g_restore_shared_sum_us = NULL;
static uint64_t              *g_restore_shared_first_us = NULL;
static uint64_t              *g_restore_shared_last_us  = NULL;
static uint64_t              *g_restore_shared_dec_us   = NULL;
static uint64_t              *g_restore_shared_h2d_us   = NULL;

/* ------------------------------------------------------------------ */
/* P4: Background decrypt worker pool (v3 path)                        */
/*                                                                      */
/* Chunked decrypt: each alloc split into CHUNK_SIZE chunks, workers    */
/* decrypt chunks into ring slots ahead of the app. cudaMalloc hook     */
/* waits for its slot to be READY then does cudaMemcpy H2D.             */
/*                                                                      */
/* Ordering: per-slot monotone sequence. Slot s processes jobs where    */
/* (j % N_RING) == s, always in strictly increasing j order. Workers    */
/* block until slot->next_seq == j AND slot state is FREE. Prevents     */
/* out-of-order slot reuse that would deadlock the app consumer.        */
/* ------------------------------------------------------------------ */
#define RST_CHUNK_SIZE_DEFAULT    (256ULL * 1024 * 1024)
#define RST_WORKERS_DEFAULT        4
#define RST_RING_SLOTS_DEFAULT     8

typedef struct {
    uint32_t alloc_idx;
    uint32_t chunk_idx;
    uint64_t byte_off;              /* offset within alloc */
    uint64_t byte_size;             /* ≤ CHUNK_SIZE */
    uint64_t cipher_file_off;       /* offset into g_restore_base_map */
    const rst_page_crypto_meta_t *meta_ptr;
    uint32_t meta_count;
} rst_chunk_job_t;

enum { RST_SLOT_FREE = 0, RST_SLOT_ASSIGNED = 1, RST_SLOT_READY = 2 };

typedef struct {
    uint8_t        *buf;            /* CHUNK_SIZE bytes */
    int32_t         state;
    uint32_t        next_seq;       /* next job index expected for this slot */
    int32_t         job_idx;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
} rst_ring_slot_t;

static size_t             g_rst_chunk_size       = RST_CHUNK_SIZE_DEFAULT;
static int                g_rst_n_workers        = RST_WORKERS_DEFAULT;
static int                g_rst_n_ring           = RST_RING_SLOTS_DEFAULT;

static rst_chunk_job_t   *g_rst_jobs             = NULL;
static uint32_t           g_rst_n_jobs           = 0;
static uint32_t          *g_rst_alloc_first_job  = NULL;
static uint32_t          *g_rst_alloc_n_jobs     = NULL;

static rst_ring_slot_t   *g_rst_ring             = NULL;
static pthread_t         *g_rst_worker_tids      = NULL;
static volatile uint32_t  g_rst_next_dispatch    = 0;   /* atomic fetch-add */
static volatile int       g_rst_workers_stop     = 0;

static int                g_rst_pool_active      = 0;   /* 1 once workers running */
static pthread_once_t     g_rst_pool_once        = PTHREAD_ONCE_INIT;

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static uint32_t               g_restore_alloc_idx = 0;
static double                 g_restore_t_start_ms = 0;
static pthread_mutex_t        g_restore_lock = PTHREAD_MUTEX_INITIALIZER;


/* ------------------------------------------------------------------ */
/* Python API for engine state dump                                    */
/* ------------------------------------------------------------------ */
static int g_python_available = 0;
typedef void *PyGILState_STATE;
typedef PyGILState_STATE (*PyGILState_Ensure_fn)(void);
typedef void (*PyGILState_Release_fn)(PyGILState_STATE);
typedef int (*PyRun_SimpleString_fn)(const char *);
typedef int (*Py_IsInitialized_fn)(void);

static PyGILState_Ensure_fn   fn_gil_ensure = NULL;
static PyGILState_Release_fn  fn_gil_release = NULL;
static PyRun_SimpleString_fn  fn_pyrun = NULL;
static Py_IsInitialized_fn    fn_pyinit = NULL;

static void resolve_python_api(void)
{
    fn_gil_ensure  = dlsym(RTLD_DEFAULT, "PyGILState_Ensure");
    fn_gil_release = dlsym(RTLD_DEFAULT, "PyGILState_Release");
    fn_pyrun       = dlsym(RTLD_DEFAULT, "PyRun_SimpleString");
    fn_pyinit      = dlsym(RTLD_DEFAULT, "Py_IsInitialized");

    if (fn_gil_ensure && fn_gil_release && fn_pyrun && fn_pyinit && fn_pyinit())
        g_python_available = 1;
}

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* Symbol resolution                                                   */
/* ------------------------------------------------------------------ */
static void resolve_symbols_impl(void)
{
    real_cudaStreamSynchronize  = dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    real_cudaDeviceSynchronize  = dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    real_cudaMallocManaged      = dlsym(RTLD_NEXT, "cudaMallocManaged");
    real_cudaMalloc             = dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaMallocAsync        = dlsym(RTLD_NEXT, "cudaMallocAsync");
    real_cudaFree               = dlsym(RTLD_NEXT, "cudaFree");
    real_cudaFreeAsync          = dlsym(RTLD_NEXT, "cudaFreeAsync");
    real_cuMemAlloc_v2          = dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    real_cuMemFree_v2           = dlsym(RTLD_NEXT, "cuMemFree_v2");
    real_cudaMemcpyAsync_fn_ptr = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    real_cudaMemcpy_fn_ptr      = dlsym(RTLD_NEXT, "cudaMemcpy");
    real_cudaMemAdvise          = dlsym(RTLD_NEXT, "cudaMemAdvise");
    real_cudaMemPrefetchAsync   = dlsym(RTLD_NEXT, "cudaMemPrefetchAsync");

}

#define ENSURE_SYMBOLS pthread_once(&g_resolve_once, resolve_symbols_impl)

/* ================================================================== */
/* Restore mode — mmap-based per-alloc streaming                       */
/*                                                                      */
/* Constructor: mmap base.img + delta.img (on tmpfs = zero extra RAM),  */
/*   parse headers + descriptors, read snap gate, alloc ONE staging buf.*/
/* Per-cudaMalloc: find alloc[i]'s data in the mmap'd file via the      */
/*   descriptor table, memcpy into staging buffer, cudaMemcpy H2D.      */
/* Peak memory: ~4 GB (one staging buffer) + 0 for the mmap.            */
/* ================================================================== */

static int restore_mmap_file(const char *path, uint8_t **out_map, size_t *out_size)
{
    struct stat st;
    if (stat(path, &st) < 0) {
        fprintf(stderr, "[restore] stat(%s): %s\n", path, strerror(errno));
        return -1;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[restore] open(%s): %s\n", path, strerror(errno));
        return -1;
    }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        fprintf(stderr, "[restore] mmap(%s): %s\n", path, strerror(errno));
        return -1;
    }
    *out_map  = (uint8_t *)m;
    *out_size = (size_t)st.st_size;
    return 0;
}

/* Forward decl — defined after restore_init, used by rst_precompute_jobs. */
static int restore_find_desc(uint64_t va, uint64_t ckpt_size);

/* ------------------------------------------------------------------ */
/* Chunk-job precomputation (v3 only for now)                          */
/*                                                                      */
/* For each alloc with a matching descriptor, walk the crypto_meta      */
/* section to find the meta entry range covering the alloc's bytes,    */
/* then split into CHUNK_SIZE chunks, each a contiguous slice of meta  */
/* entries. Stores per-alloc first_job / n_jobs for the app consumer.   */
/* ------------------------------------------------------------------ */
static int rst_precompute_jobs(void)
{
    uint32_t n = g_restore_alloc_total;
    g_rst_alloc_first_job = calloc(n, sizeof(uint32_t));
    g_rst_alloc_n_jobs    = calloc(n, sizeof(uint32_t));
    if (!g_rst_alloc_first_job || !g_rst_alloc_n_jobs) return -1;

    /* Upper bound on job count: ceil(total_bytes / chunk_size) + 1 per alloc. */
    uint64_t max_jobs_est = 0;
    for (uint32_t i = 0; i < n; i++)
        max_jobs_est += (g_restore_alloc_size[i] + g_rst_chunk_size - 1) / g_rst_chunk_size + 1;
    g_rst_jobs = calloc(max_jobs_est, sizeof(rst_chunk_job_t));
    if (!g_rst_jobs) return -1;

    uint32_t nj = 0;
    for (uint32_t i = 0; i < n; i++) {
        g_rst_alloc_first_job[i] = nj;
        uint64_t ckpt_size = g_restore_alloc_size[i];
        uint64_t alloc_va  = g_restore_alloc_va[i];

        int d = restore_find_desc(alloc_va, ckpt_size);
        if (d < 0) {
            g_rst_alloc_n_jobs[i] = 0;
            continue;   /* passthrough — hook will skip */
        }
        rst_v2_desc_t *desc = &g_restore_base_descs[d];
        uint64_t offset_in_range   = alloc_va - desc->base_va;
        uint64_t gpu_section_start = desc->data_offset + desc->resmap_size + desc->cpu_bytes;
        uint64_t meta_section_off  = gpu_section_start + desc->gpu_bytes;
        const rst_page_crypto_meta_t *meta_base =
            (const rst_page_crypto_meta_t *)(g_restore_base_map + meta_section_off);
        uint64_t total_meta = desc->crypto_meta_bytes / sizeof(rst_page_crypto_meta_t);

        /* Walk meta to find start_idx where accumulated size == offset_in_range */
        uint64_t acc = 0;
        uint32_t cur_meta = 0;
        while (cur_meta < total_meta && acc < offset_in_range) {
            acc += meta_base[cur_meta].size;
            cur_meta++;
        }
        if (acc != offset_in_range) {
            fprintf(stderr, "[restore][precompute] FATAL: alloc[%u] meta misalign "
                    "(acc=%llu offset=%llu)\n",
                    i, (unsigned long long)acc, (unsigned long long)offset_in_range);
            return -1;
        }

        uint64_t bytes_done = 0;
        uint32_t chunk_idx  = 0;
        uint64_t cur_acc    = offset_in_range;
        while (bytes_done < ckpt_size) {
            uint64_t target = g_rst_chunk_size;
            if (ckpt_size - bytes_done < target) target = ckpt_size - bytes_done;

            uint32_t meta_start = cur_meta;
            uint64_t chunk_bytes = 0;
            while (cur_meta < total_meta && chunk_bytes < target) {
                chunk_bytes += meta_base[cur_meta].size;
                cur_meta++;
            }
            if (chunk_bytes == 0) {
                fprintf(stderr, "[restore][precompute] FATAL: alloc[%u] chunk %u has zero bytes\n",
                        i, chunk_idx);
                return -1;
            }

            rst_chunk_job_t *job = &g_rst_jobs[nj++];
            job->alloc_idx       = i;
            job->chunk_idx       = chunk_idx++;
            job->byte_off        = bytes_done;
            job->byte_size       = chunk_bytes;
            job->cipher_file_off = gpu_section_start + cur_acc;
            job->meta_ptr        = &meta_base[meta_start];
            job->meta_count      = cur_meta - meta_start;

            cur_acc    += chunk_bytes;
            bytes_done += chunk_bytes;
        }
        g_rst_alloc_n_jobs[i] = chunk_idx;
    }

    g_rst_n_jobs = nj;
    fprintf(stderr, "[restore] precomputed %u chunk jobs across %u allocs "
                    "(chunk=%zu MB, ring=%d, workers=%d)\n",
            nj, n, g_rst_chunk_size / (1024*1024), g_rst_n_ring, g_rst_n_workers);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Worker thread                                                       */
/* ------------------------------------------------------------------ */
static void *rst_worker_thread(void *arg)
{
    (void)arg;
    while (!g_rst_workers_stop) {
        uint32_t j = __sync_fetch_and_add(&g_rst_next_dispatch, 1);
        if (j >= g_rst_n_jobs) break;

        int s = (int)(j % (uint32_t)g_rst_n_ring);
        rst_ring_slot_t *slot = &g_rst_ring[s];
        const rst_chunk_job_t *job = &g_rst_jobs[j];

        pthread_mutex_lock(&slot->mu);
        while (!g_rst_workers_stop &&
               (slot->state != RST_SLOT_FREE || slot->next_seq != j)) {
            pthread_cond_wait(&slot->cv, &slot->mu);
        }
        if (g_rst_workers_stop) {
            pthread_mutex_unlock(&slot->mu);
            break;
        }
        slot->state   = RST_SLOT_ASSIGNED;
        slot->job_idx = (int32_t)j;
        pthread_mutex_unlock(&slot->mu);

        /* Decrypt directly into the slot buffer, no intermediate copy. */
        rst_uvm_decrypt_params_t p = {0};
        p.cipher_buf    = (uint64_t)(uintptr_t)(g_restore_base_map + job->cipher_file_off);
        p.plain_buf     = (uint64_t)(uintptr_t)slot->buf;
        p.total_size    = job->byte_size;
        p.crypto_meta   = (uint64_t)(uintptr_t)job->meta_ptr;
        p.num_transfers = job->meta_count;

        int ret = ioctl(g_restore_uvm_fd,
                        UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES, &p);
        if (ret < 0 || p.rmStatus != 0) {
            fprintf(stderr, "[restore][worker] job %u (alloc %u chunk %u) ioctl 112 failed: "
                    "ret=%d errno=%s rmStatus=0x%x\n",
                    j, job->alloc_idx, job->chunk_idx, ret, strerror(errno), p.rmStatus);
            /* still mark READY so consumer doesn't hang; error surfaces on H2D stage */
        }

        pthread_mutex_lock(&slot->mu);
        slot->state = RST_SLOT_READY;
        pthread_cond_broadcast(&slot->cv);
        pthread_mutex_unlock(&slot->mu);
    }
    return NULL;
}

/* Lazy init — first app-thread call to the restore hook spawns the pool. */
static void rst_pool_init_once(void)
{
    /* Env overrides */
    const char *e;
    if ((e = getenv("CKPT_RESTORE_WORKERS"))  && atoi(e) > 0) g_rst_n_workers = atoi(e);
    if ((e = getenv("CKPT_RESTORE_RING"))     && atoi(e) > 0) g_rst_n_ring    = atoi(e);
    if ((e = getenv("CKPT_RESTORE_CHUNK_MB")) && atoi(e) > 0)
        g_rst_chunk_size = (size_t)atoi(e) * 1024ULL * 1024ULL;

    if (g_rst_n_ring < g_rst_n_workers + 2) g_rst_n_ring = g_rst_n_workers + 2;

    if (rst_precompute_jobs() < 0) {
        fprintf(stderr, "[restore] pool: precompute failed, falling back to serial path\n");
        return;
    }
    if (g_rst_n_jobs == 0) {
        fprintf(stderr, "[restore] pool: no jobs to run, skipping worker spawn\n");
        return;
    }

    g_rst_ring = calloc((size_t)g_rst_n_ring, sizeof(rst_ring_slot_t));
    if (!g_rst_ring) return;
    for (int s = 0; s < g_rst_n_ring; s++) {
        g_rst_ring[s].buf = malloc(g_rst_chunk_size);
        if (!g_rst_ring[s].buf) {
            fprintf(stderr, "[restore] pool: slot %d malloc(%zu) failed\n", s, g_rst_chunk_size);
            return;
        }
        (void)madvise(g_rst_ring[s].buf, g_rst_chunk_size, MADV_HUGEPAGE);
        /* Fault in once so workers don't pay first-touch cost during decrypt. */
        memset(g_rst_ring[s].buf, 0, g_rst_chunk_size);
        g_rst_ring[s].state    = RST_SLOT_FREE;
        g_rst_ring[s].next_seq = (uint32_t)s;
        g_rst_ring[s].job_idx  = -1;
        pthread_mutex_init(&g_rst_ring[s].mu, NULL);
        pthread_cond_init(&g_rst_ring[s].cv, NULL);
    }

    g_rst_worker_tids = calloc((size_t)g_rst_n_workers, sizeof(pthread_t));
    if (!g_rst_worker_tids) return;
    for (int t = 0; t < g_rst_n_workers; t++) {
        if (pthread_create(&g_rst_worker_tids[t], NULL, rst_worker_thread, NULL) != 0) {
            fprintf(stderr, "[restore] pool: pthread_create worker %d failed\n", t);
            g_rst_n_workers = t;
            break;
        }
    }

    g_rst_pool_active = 1;
    fprintf(stderr, "[restore] pool: %d workers running, %d ring slots × %zu MB = %.2f GB staging\n",
            g_rst_n_workers, g_rst_n_ring, g_rst_chunk_size / (1024*1024),
            (g_rst_n_ring * g_rst_chunk_size) / (1024.0*1024.0*1024.0));
}

static void rst_pool_shutdown(void)
{
    if (!g_rst_pool_active) return;
    g_rst_workers_stop = 1;
    for (int s = 0; s < g_rst_n_ring; s++) {
        pthread_mutex_lock(&g_rst_ring[s].mu);
        pthread_cond_broadcast(&g_rst_ring[s].cv);
        pthread_mutex_unlock(&g_rst_ring[s].mu);
    }
    for (int t = 0; t < g_rst_n_workers; t++)
        pthread_join(g_rst_worker_tids[t], NULL);
    g_rst_pool_active = 0;
}

/*
 * restore_init — called in the constructor (before CUDA).
 * Reads the snap gate, mmaps both image files, parses their headers,
 * and allocates the reusable staging buffer.
 */
static int restore_init(const char *snap_path,
                        const char *base_path,
                        const char *delta_path)
{
    g_restore_t_start_ms = now_ms();

    /* 1. Read snap gate → alloc VA/size table */
    int fd = open(snap_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[restore] open(%s): %s\n", snap_path, strerror(errno));
        return -1;
    }
    void *snap = mmap(NULL, GATE_FILE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (snap == MAP_FAILED) {
        fprintf(stderr, "[restore] mmap(%s) failed\n", snap_path);
        return -1;
    }
    ckpt_alloc_hdr_t   *ahdr  = (ckpt_alloc_hdr_t   *)((char *)snap + GATE_ALLOC_HDR_OFF);
    ckpt_alloc_entry_t *atable = (ckpt_alloc_entry_t *)((char *)snap + GATE_ALLOC_HDR_OFF
                                                         + sizeof(ckpt_alloc_hdr_t));
    uint32_t n = ahdr->count;
    fprintf(stderr, "[restore] snap gate has %u allocations\n", n);
    if (n == 0 || n > GATE_ALLOC_MAX) {
        munmap(snap, GATE_FILE_SIZE);
        return -1;
    }
    g_restore_alloc_total = n;
    g_restore_alloc_va    = calloc(n, sizeof(uint64_t));
    g_restore_alloc_size  = calloc(n, sizeof(uint64_t));
    if (!g_restore_alloc_va || !g_restore_alloc_size) {
        munmap(snap, GATE_FILE_SIZE);
        return -1;
    }
    uint64_t max_alloc = 0;
    for (uint32_t i = 0; i < n; i++) {
        g_restore_alloc_va[i]   = atable[i].va;
        g_restore_alloc_size[i] = atable[i].size;
        if (atable[i].size > max_alloc) max_alloc = atable[i].size;
    }
    munmap(snap, GATE_FILE_SIZE);

    /* 2. mmap base.img (on tmpfs = shares physical pages, zero extra RAM) */
    if (restore_mmap_file(base_path, &g_restore_base_map, &g_restore_base_size) < 0)
        return -1;

    /* Validate header */
    if (g_restore_base_size < sizeof(rst_v2_hdr_t)) {
        fprintf(stderr, "[restore] base.img too small\n");
        return -1;
    }
    g_restore_base_hdr = (rst_v2_hdr_t *)g_restore_base_map;
    if (g_restore_base_hdr->magic != RST_V2_MAGIC) {
        fprintf(stderr, "[restore] base.img bad magic: 0x%08x\n", g_restore_base_hdr->magic);
        return -1;
    }
    g_restore_base_descs = (rst_v2_desc_t *)(g_restore_base_map + sizeof(rst_v2_hdr_t));
    fprintf(stderr, "[restore] base.img: version=%u, %u ranges, %.2f GB\n",
            g_restore_base_hdr->version, g_restore_base_hdr->num_ranges,
            g_restore_base_hdr->total_bytes / (1024.0*1024.0*1024.0));

    /* V3 = kernel ioctl 115 path: GPU data is k1-encrypted, decrypted
     * per-alloc via ioctl 112. Open /dev/nvidia-uvm and UVM_INITIALIZE
     * once here so each restore_apply_alloc can call the decrypt ioctl
     * without per-call open overhead. Requires CAP_SYS_ADMIN. */
    if (g_restore_base_hdr->version == RST_V3_VERSION) {
        g_restore_uvm_fd = open("/dev/nvidia-uvm", O_RDWR);
        if (g_restore_uvm_fd < 0) {
            fprintf(stderr, "[restore] v3: open(/dev/nvidia-uvm) failed: %s\n",
                    strerror(errno));
            return -1;
        }
        rst_uvm_init_params_t init = {0};
        if (ioctl(g_restore_uvm_fd, RST_UVM_INITIALIZE, &init) < 0 ||
            init.rmStatus != 0) {
            fprintf(stderr, "[restore] v3: UVM_INITIALIZE failed status=0x%x errno=%s\n",
                    init.rmStatus, strerror(errno));
            close(g_restore_uvm_fd);
            g_restore_uvm_fd = -1;
            return -1;
        }
        g_restore_v3 = 1;
        fprintf(stderr, "[restore] v3: UVM device opened for per-alloc ioctl 112 decrypt\n");
    }

    /* V4 = in-app baseline (AES-GCM, 2 MB chunks). Load k3 key. */
    if (g_restore_base_hdr->version == RST_V4_VERSION) {
        char key_path[1024];
        const char *env = getenv("CKPT_K3_KEY_FILE");
        if (env) {
            snprintf(key_path, sizeof(key_path), "%s", env);
        } else {
            snprintf(key_path, sizeof(key_path), "%s.k3.key", base_path);
        }
        int kf = open(key_path, O_RDONLY);
        if (kf < 0) {
            fprintf(stderr, "[restore] v4 needs k3 key at %s: %s\n",
                    key_path, strerror(errno));
            return -1;
        }
        ssize_t rn = read(kf, g_restore_k3_key, CKPT_CRYPTO_KEY_SIZE);
        close(kf);
        if (rn != CKPT_CRYPTO_KEY_SIZE) {
            fprintf(stderr, "[restore] v4 key file too small: %zd bytes\n", rn);
            return -1;
        }
        g_restore_v4 = 1;
        fprintf(stderr, "[restore] v4 k3 key loaded from %s\n", key_path);
    }

    /* 3. mmap delta.img */
    if (restore_mmap_file(delta_path, &g_restore_delta_map, &g_restore_delta_size) < 0)
        return -1;

    /* Detect v4 delta header (for P1e overlay). Legacy v3 deltas start
     * with 0xDE17A001; our v4 delta starts with 0xDE17A004. Anything else
     * (including an empty/zero file) is treated as "no delta to apply". */
    if (g_restore_delta_size >= sizeof(rst_delta_v4_hdr_t)) {
        rst_delta_v4_hdr_t *dh = (rst_delta_v4_hdr_t *)g_restore_delta_map;
        if (dh->magic == RST_DELTA_V4_MAGIC && dh->version == 4 &&
            dh->num_blocks > 0) {
            g_restore_delta_hdr    = dh;
            g_restore_delta_blocks = (rst_delta_v4_block_t *)
                (g_restore_delta_map + sizeof(rst_delta_v4_hdr_t));
            g_restore_delta_v4     = 1;
            fprintf(stderr, "[restore] delta.img: v4, %u blocks, %.2f MB\n",
                    dh->num_blocks,
                    dh->total_bytes / (1024.0*1024.0));
        } else {
            fprintf(stderr, "[restore] delta.img: no v4 overlay (magic=0x%08x)\n",
                    dh->magic);
        }
    }

    /* 3b. P6 — mmap static.img if CKPT_RESTORE_STATIC is set. Same v2
     *      header format as base.img, but encrypted with k3 at 2 MB chunks
     *      (matching v4 delta layout). On per-alloc restore we try its
     *      descriptors first; if matched, decrypt+H2D from here and skip
     *      base+delta processing for that alloc. */
    {
        const char *static_path = getenv("CKPT_RESTORE_STATIC");
        if (static_path && static_path[0]) {
            if (restore_mmap_file(static_path, &g_restore_static_map,
                                  &g_restore_static_size) < 0) {
                fprintf(stderr, "[restore] static.img: mmap(%s) failed — proceeding without P6\n",
                        static_path);
            } else if (g_restore_static_size < sizeof(rst_v2_hdr_t)) {
                fprintf(stderr, "[restore] static.img: too small — proceeding without P6\n");
                munmap(g_restore_static_map, g_restore_static_size);
                g_restore_static_map = NULL;
                g_restore_static_size = 0;
            } else {
                g_restore_static_hdr = (rst_v2_hdr_t *)g_restore_static_map;
                if (g_restore_static_hdr->magic != RST_V2_MAGIC) {
                    fprintf(stderr, "[restore] static.img: bad magic 0x%08x — proceeding without P6\n",
                            g_restore_static_hdr->magic);
                    munmap(g_restore_static_map, g_restore_static_size);
                    g_restore_static_map = NULL;
                    g_restore_static_hdr = NULL;
                    g_restore_static_size = 0;
                } else {
                    g_restore_static_descs = (rst_v2_desc_t *)
                        (g_restore_static_map + sizeof(rst_v2_hdr_t));
                    fprintf(stderr, "[restore] static.img: %u ranges, %.2f GB (P6 enabled)\n",
                            g_restore_static_hdr->num_ranges,
                            g_restore_static_hdr->total_bytes / (1024.0*1024.0*1024.0));

                    /* k3 key only needed for v2 static.img (CPU AES-GCM).
                     * v3 static.img uses ioctl 112 with kernel-side k1, no
                     * userspace key load required. */
                    int static_needs_k3 = (g_restore_static_hdr->version != RST_V3_VERSION);
                    if (static_needs_k3 && !g_restore_v4) {
                        char key_path[1024];
                        const char *kenv = getenv("CKPT_K3_KEY_FILE");
                        if (kenv) {
                            snprintf(key_path, sizeof(key_path), "%s", kenv);
                        } else {
                            char dpath_copy[1024];
                            snprintf(dpath_copy, sizeof(dpath_copy), "%s", delta_path);
                            char *sl = strrchr(dpath_copy, '/');
                            if (sl) *sl = '\0'; else strcpy(dpath_copy, ".");
                            snprintf(key_path, sizeof(key_path),
                                     "%s/ckpt_k3.key", dpath_copy);
                        }
                        int kf = open(key_path, O_RDONLY);
                        if (kf < 0) {
                            fprintf(stderr, "[restore] static.img needs k3 key at %s: %s — disabling P6\n",
                                    key_path, strerror(errno));
                            munmap(g_restore_static_map, g_restore_static_size);
                            g_restore_static_map = NULL;
                            g_restore_static_hdr = NULL;
                            g_restore_static_descs = NULL;
                            g_restore_static_size = 0;
                        } else {
                            ssize_t rn = read(kf, g_restore_k3_key, CKPT_CRYPTO_KEY_SIZE);
                            close(kf);
                            if (rn != CKPT_CRYPTO_KEY_SIZE) {
                                fprintf(stderr, "[restore] static.img k3 key short read (%zd) — disabling P6\n", rn);
                                munmap(g_restore_static_map, g_restore_static_size);
                                g_restore_static_map = NULL;
                                g_restore_static_hdr = NULL;
                                g_restore_static_descs = NULL;
                                g_restore_static_size = 0;
                            } else {
                                fprintf(stderr, "[restore] static.img k3 key loaded from %s\n", key_path);
                            }
                        }
                    }
                }
            }
        }
    }

    /* 4. Allocate ONE reusable staging buffer sized to the largest alloc.
     *    With MADV_HUGEPAGE, first-touch cost is paid once here, not per-alloc. */
    g_restore_staging_size = (size_t)max_alloc;
    g_restore_staging = malloc(g_restore_staging_size);
    if (!g_restore_staging) {
        fprintf(stderr, "[restore] malloc staging (%zu bytes) failed\n", g_restore_staging_size);
        return -1;
    }
    (void)madvise(g_restore_staging, g_restore_staging_size, MADV_HUGEPAGE);

    /* 5. Shared-memory counters (survive fork) */
    g_restore_shared_page = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                  MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (g_restore_shared_page == MAP_FAILED) {
        g_restore_shared_page = NULL;
    } else {
        memset(g_restore_shared_page, 0, 4096);
        g_restore_shared_done     = (uint32_t *)g_restore_shared_page;
        g_restore_shared_bytes    = (uint64_t *)((char *)g_restore_shared_page + 8);
        g_restore_shared_sum_us   = (uint64_t *)((char *)g_restore_shared_page + 16);
        g_restore_shared_first_us = (uint64_t *)((char *)g_restore_shared_page + 24);
        g_restore_shared_last_us  = (uint64_t *)((char *)g_restore_shared_page + 32);
        g_restore_shared_dec_us   = (uint64_t *)((char *)g_restore_shared_page + 40);
        g_restore_shared_h2d_us   = (uint64_t *)((char *)g_restore_shared_page + 48);
    }

    fprintf(stderr, "[restore] init done in %.1f ms: %u allocs, staging=%.2f GB, base mmap=%.2f GB\n",
            now_ms() - g_restore_t_start_ms, n,
            g_restore_staging_size / (1024.0*1024.0*1024.0),
            g_restore_base_size / (1024.0*1024.0*1024.0));
    return 0;
}

/*
 * Find the base.img descriptor whose range covers [va, va+ckpt_size).
 * Uses the CHECKPOINT size (from snap gate), not the new alloc size,
 * because KV cache allocs may be slightly larger on restore.
 * Returns the descriptor index, or -1 if not found.
 */
static int restore_find_desc(uint64_t va, uint64_t ckpt_size)
{
    uint32_t nr = g_restore_base_hdr->num_ranges;
    for (uint32_t d = 0; d < nr; d++) {
        uint64_t d_start = g_restore_base_descs[d].base_va;
        uint64_t d_end   = d_start + g_restore_base_descs[d].length;
        if (va >= d_start && va + ckpt_size <= d_end)
            return (int)d;
    }
    return -1;
}

/* P6 — find static.img descriptor covering [va, va+ckpt_size). Same
 * shape as restore_find_desc but on the separate static.img mmap. */
static int restore_find_static_desc(uint64_t va, uint64_t ckpt_size)
{
    if (!g_restore_static_hdr) return -1;
    uint32_t nr = g_restore_static_hdr->num_ranges;
    for (uint32_t d = 0; d < nr; d++) {
        uint64_t d_start = g_restore_static_descs[d].base_va;
        uint64_t d_end   = d_start + g_restore_static_descs[d].length;
        if (va >= d_start && va + ckpt_size <= d_end)
            return (int)d;
    }
    return -1;
}

/* P6 — try to apply this alloc from static.img.
 * Returns 1 if applied (caller should skip base+delta), 0 if not in static,
 * -1 on error. */
static int restore_try_apply_static(uint32_t idx, void *new_ptr,
                                    uint64_t alloc_va, uint64_t ckpt_size,
                                    uint64_t copy_size)
{
    int d = restore_find_static_desc(alloc_va, ckpt_size);
    if (d < 0) return 0;

    rst_v2_desc_t *desc = &g_restore_static_descs[d];
    uint64_t offset_in_range   = alloc_va - desc->base_va;
    uint64_t gpu_section_start = desc->data_offset + desc->resmap_size + desc->cpu_bytes;
    uint64_t file_offset       = gpu_section_start + offset_in_range;

    if (file_offset + copy_size > g_restore_static_size) {
        fprintf(stderr, "[restore] static[%u] file offset 0x%llx + 0x%llx > static size 0x%llx\n",
                idx, (unsigned long long)file_offset,
                (unsigned long long)copy_size,
                (unsigned long long)g_restore_static_size);
        return -1;
    }

    /* Two decrypt paths depending on static.img format:
     *   v3 (kernel-encrypted, k1) → ioctl 112 (same as base.img v3 path)
     *   v2 (CPU k3 at 2 MB chunks) → ckpt_crypto_decrypt_pages
     * Determined by static.img header version. */
    if (g_restore_static_hdr->version == RST_V3_VERSION && g_restore_uvm_fd >= 0) {
        /* v3 path — mirror libckpt_vllm.c:1188 base.img decrypt, pointed at
         * static.img mmap. crypto_meta lives right after gpu_bytes; walk it
         * to find the slice covering [offset_in_range, +copy_size). */
        uint64_t meta_section_off = desc->data_offset + desc->resmap_size
                                  + desc->cpu_bytes + desc->gpu_bytes;
        const rst_page_crypto_meta_t *range_meta =
            (const rst_page_crypto_meta_t *)(g_restore_static_map + meta_section_off);
        uint64_t total_meta_entries = desc->crypto_meta_bytes
                                    / sizeof(rst_page_crypto_meta_t);

        uint64_t acc = 0, start_idx = 0;
        while (start_idx < total_meta_entries && acc < offset_in_range) {
            acc += range_meta[start_idx].size;
            start_idx++;
        }
        if (acc != offset_in_range) {
            fprintf(stderr, "[restore] FATAL: v3 static[%u] offset_in_range=%llu "
                    "does not align to crypto_meta entry boundary (acc=%llu)\n",
                    idx, (unsigned long long)offset_in_range,
                    (unsigned long long)acc);
            return -1;
        }
        uint64_t end_idx = start_idx, slice_bytes = 0;
        while (end_idx < total_meta_entries && slice_bytes < copy_size) {
            slice_bytes += range_meta[end_idx].size;
            end_idx++;
        }
        if (slice_bytes != copy_size) {
            fprintf(stderr, "[restore] FATAL: v3 static[%u] copy_size=%llu does "
                    "not align to crypto_meta entry boundary (slice=%llu)\n",
                    idx, (unsigned long long)copy_size,
                    (unsigned long long)slice_bytes);
            return -1;
        }

        rst_uvm_decrypt_params_t p = {0};
        p.cipher_buf    = (uint64_t)(uintptr_t)(g_restore_static_map + file_offset);
        p.plain_buf     = (uint64_t)(uintptr_t)g_restore_staging;
        p.total_size    = copy_size;
        p.crypto_meta   = (uint64_t)(uintptr_t)&range_meta[start_idx];
        p.num_transfers = end_idx - start_idx;

        int ret = ioctl(g_restore_uvm_fd,
                        UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES, &p);
        if (ret < 0 || p.rmStatus != 0) {
            fprintf(stderr, "[restore] FATAL: v3 static[%u] ioctl 112 failed: "
                    "ret=%d errno=%s rmStatus=0x%x\n",
                    idx, ret, strerror(errno), p.rmStatus);
            return -1;
        }
    } else {
        /* v2 path — k3 CPU AES-GCM at 2 MB chunks (legacy static.img format). */
        if ((offset_in_range % RST_CRYPTO_CHUNK_BYTES_LOCAL) != 0 ||
            (copy_size       % RST_CRYPTO_CHUNK_BYTES_LOCAL) != 0 ||
            copy_size < RST_CRYPTO_CHUNK_BYTES_LOCAL) {
            fprintf(stderr, "[restore] FATAL: v2 static[%u] not 2MB aligned "
                    "(offset=%llu copy_size=%llu)\n", idx,
                    (unsigned long long)offset_in_range,
                    (unsigned long long)copy_size);
            return -1;
        }

        memcpy(g_restore_staging,
               g_restore_static_map + file_offset,
               (size_t)copy_size);

        uint64_t meta_section_off = desc->data_offset + desc->resmap_size
                                  + desc->cpu_bytes + desc->gpu_bytes;
        const ckpt_page_meta_t *meta =
            (const ckpt_page_meta_t *)(g_restore_static_map + meta_section_off)
            + (offset_in_range / RST_CRYPTO_CHUNK_BYTES_LOCAL);
        uint32_t n_chunks = (uint32_t)(copy_size / RST_CRYPTO_CHUNK_BYTES_LOCAL);
        int rc = ckpt_crypto_decrypt_pages(g_restore_k3_key,
                                           g_restore_staging,
                                           g_restore_staging,
                                           meta,
                                           n_chunks,
                                           (size_t)RST_CRYPTO_CHUNK_BYTES_LOCAL,
                                           /*num_threads=*/4);
        if (rc != 0) {
            fprintf(stderr, "[restore] FATAL: v2 static[%u] decrypt failed (n_chunks=%u)\n",
                    idx, n_chunks);
            return -1;
        }
    }

    cudaError_t err = real_cudaMemcpy_fn_ptr(new_ptr, g_restore_staging,
                                              (size_t)copy_size,
                                              cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[restore] static[%u] cudaMemcpy H2D failed: %s\n",
                idx, cudaGetErrorString(err));
        return -1;
    }

    __sync_fetch_and_add(&g_restore_static_allocs_applied, 1);
    __sync_fetch_and_add(&g_restore_static_bytes_applied, copy_size);
    return 1;
}

/*
 * restore_apply_alloc — called per cudaMalloc.
 *
 * For alloc[idx]: find its data in the mmap'd base.img, memcpy into
 * the staging buffer, cudaMemcpy H2D into the GPU alloc.
 *
 * For plaintext v2 images (the current path): data is raw bytes in the
 * gpu_section of the range, at an offset determined by alloc_va's position
 * within the range. For vLLM where every page is GPU-resident (cpu_bytes=0),
 * the offset is simply (alloc_va - range_base_va).
 *
 * TODO: add ioctl 112 decrypt path for v3 images.
 * TODO: overlay delta.img dirty pages on top of base data.
 */
static int restore_apply_alloc(void *new_ptr, size_t new_size)
{
    if (!g_restore_mode || !real_cudaMemcpy_fn_ptr) return 0;

    /* Lazy-spawn the decrypt worker pool on first hook call in this process
     * (child after fork, or parent if no fork). Only active when v3. */
    if (g_restore_v3 && !g_rst_pool_active)
        pthread_once(&g_rst_pool_once, rst_pool_init_once);

    pthread_mutex_lock(&g_restore_lock);
    uint32_t idx = g_restore_alloc_idx++;
    pthread_mutex_unlock(&g_restore_lock);

    __sync_fetch_and_add(&g_restore_invocations, 1);

    if (idx >= g_restore_alloc_total) {
        __sync_fetch_and_add(&g_restore_passthrough_past_snap, 1);
        if (g_verbose)
            fprintf(stderr, "[restore] alloc[%u] past snap end (size=%zu) — passthrough\n",
                    idx, new_size);
        return 0;
    }

    uint64_t ckpt_size = g_restore_alloc_size[idx];
    uint64_t alloc_va  = g_restore_alloc_va[idx];

    /* Size-alignment diagnostic — does the new cudaMalloc match the snap? */
    {
        int64_t diff = (int64_t)new_size - (int64_t)ckpt_size;
        if (diff < 0) diff = -diff;
        if (diff == 0)
            __sync_fetch_and_add(&g_restore_size_exact, 1);
        else if (diff <= (2 * 1024 * 1024))
            __sync_fetch_and_add(&g_restore_size_close_2mb, 1);
        else {
            __sync_fetch_and_add(&g_restore_size_mismatch, 1);
            if (g_verbose || g_restore_size_mismatch <= 10)
                fprintf(stderr, "[restore] alloc[%u] SIZE MISMATCH: snap=%llu, new=%zu (diff=%lld)\n",
                        idx, (unsigned long long)ckpt_size, new_size,
                        (long long)((int64_t)new_size - (int64_t)ckpt_size));
        }
    }

    /* KV cache blocks vary ±2 MB between runs (vLLM sizes them based on
     * available GPU memory). We copy min(actual, ckpt_size) bytes:
     *   actual > ckpt: fill beginning, tail stays zero (unused capacity)
     *   actual < ckpt: truncate — drop the last few MB of checkpoint data
     *                  (least-recently-used KV entries, regenerated on next decode)
     *   actual == ckpt: exact match */
    uint64_t copy_size = ckpt_size < (uint64_t)new_size ? ckpt_size : (uint64_t)new_size;

    /* P6 — try static.img first. H2D-flagged allocs (model weights) live
     * here, captured once into static.img and excluded from base/delta.
     * If matched, decrypt + H2D from static.img and skip base+delta. */
    if (g_restore_static_hdr) {
        int sret = restore_try_apply_static(idx, new_ptr, alloc_va,
                                            ckpt_size, copy_size);
        if (sret < 0) return -1;
        if (sret > 0) return 0;  /* applied from static.img — done */
        /* sret == 0: not in static, fall through to base+delta */
    }

    /* Find the base.img descriptor covering this alloc's original VA */
    int d = restore_find_desc(alloc_va, ckpt_size);
    if (d < 0) {
        __sync_fetch_and_add(&g_restore_passthrough_no_desc, 1);
        if (g_verbose)
            fprintf(stderr, "[restore] alloc[%u] VA=0x%llx size=%zu ckpt_size=%llu not found in base/static — passthrough\n",
                    idx, (unsigned long long)alloc_va, new_size,
                    (unsigned long long)ckpt_size);
        return 0;
    }
    __sync_fetch_and_add(&g_restore_base_applied, 1);

    rst_v2_desc_t *desc = &g_restore_base_descs[d];

    /* For vLLM: all pages are GPU-resident (cpu_bytes == 0), so the
     * gpu_section starts right after resmap. Within the GPU section,
     * the alloc's data starts at (alloc_va - range_base_va). */
    uint64_t offset_in_range   = alloc_va - desc->base_va;
    uint64_t gpu_section_start = desc->data_offset + desc->resmap_size + desc->cpu_bytes;
    uint64_t file_offset       = gpu_section_start + offset_in_range;

    /* Bounds check — use copy_size (the min of actual vs ckpt) */
    if (file_offset + copy_size > g_restore_base_size) {
        fprintf(stderr, "[restore] alloc[%u] file offset 0x%llx + 0x%llx exceeds base size 0x%llx\n",
                idx, (unsigned long long)file_offset,
                (unsigned long long)copy_size,
                (unsigned long long)g_restore_base_size);
        return 0;
    }

    /* memcpy from mmap into staging buffer, then H2D into GPU alloc */
    uint64_t t_start_us = now_us();
    if (g_restore_shared_first_us) {
        uint64_t expected = 0;
        __sync_bool_compare_and_swap(g_restore_shared_first_us, expected, t_start_us);
    }
    double t0 = now_ms();
    double t_dec_start = t0;

    /* -------- P4: chunked ring path (v3 only) -------- */
    if (g_rst_pool_active && g_restore_v3 && !g_restore_delta_v4) {
        uint32_t first_j = g_rst_alloc_first_job[idx];
        uint32_t njobs   = g_rst_alloc_n_jobs[idx];
        uint64_t logical_off = 0;    /* bytes walked in ckpt (for truncation logic) */

        double wait_ms_sum = 0.0;
        double h2d_ms_sum  = 0.0;

        for (uint32_t k = 0; k < njobs; k++) {
            uint32_t j = first_j + k;
            int s = (int)(j % (uint32_t)g_rst_n_ring);
            rst_ring_slot_t *slot = &g_rst_ring[s];
            const rst_chunk_job_t *job = &g_rst_jobs[j];

            double tw0 = now_ms();
            pthread_mutex_lock(&slot->mu);
            while (!(slot->state == RST_SLOT_READY && slot->next_seq == j)) {
                pthread_cond_wait(&slot->cv, &slot->mu);
            }
            pthread_mutex_unlock(&slot->mu);
            wait_ms_sum += now_ms() - tw0;

            /* Compute per-chunk truncation against new_size */
            uint64_t chunk_copy = job->byte_size;
            if (logical_off + chunk_copy > (uint64_t)new_size) {
                chunk_copy = (logical_off >= (uint64_t)new_size)
                                 ? 0 : (uint64_t)new_size - logical_off;
            }

            if (chunk_copy > 0) {
                double th0 = now_ms();
                cudaError_t cerr = real_cudaMemcpy_fn_ptr(
                    (uint8_t *)new_ptr + job->byte_off,
                    slot->buf,
                    (size_t)chunk_copy,
                    cudaMemcpyHostToDevice);
                h2d_ms_sum += now_ms() - th0;
                if (cerr != cudaSuccess) {
                    fprintf(stderr, "[restore] FATAL: ring H2D failed alloc[%u] chunk %u: %d\n",
                            idx, k, cerr);
                    return -1;
                }
            }

            /* Release slot back to workers. */
            pthread_mutex_lock(&slot->mu);
            slot->state    = RST_SLOT_FREE;
            slot->job_idx  = -1;
            slot->next_seq = j + (uint32_t)g_rst_n_ring;
            pthread_cond_broadcast(&slot->cv);
            pthread_mutex_unlock(&slot->mu);

            logical_off += job->byte_size;
        }

        /* Stats — shared counters */
        uint64_t t_end_us = now_us();
        if (g_restore_shared_done)
            __sync_fetch_and_add(g_restore_shared_done, 1);
        if (g_restore_shared_bytes)
            __sync_fetch_and_add(g_restore_shared_bytes, copy_size);
        if (g_restore_shared_sum_us)
            __sync_fetch_and_add(g_restore_shared_sum_us, t_end_us - t_start_us);
        if (g_restore_shared_dec_us)
            __sync_fetch_and_add(g_restore_shared_dec_us, (uint64_t)(wait_ms_sum * 1000.0));
        if (g_restore_shared_h2d_us)
            __sync_fetch_and_add(g_restore_shared_h2d_us, (uint64_t)(h2d_ms_sum * 1000.0));
        if (g_restore_shared_last_us) {
            uint64_t cur;
            do {
                cur = *g_restore_shared_last_us;
                if (t_end_us <= cur) break;
            } while (!__sync_bool_compare_and_swap(g_restore_shared_last_us, cur, t_end_us));
        }

        if (g_verbose || (idx < 5) || ((idx % 20) == 0)) {
            double total_ms = now_ms() - t0;
            fprintf(stderr, "[restore] alloc[%u] ring %u chunks, wait=%.1f ms H2D=%.1f ms total=%.1f ms (%.1f GB/s)\n",
                    idx, njobs, wait_ms_sum, h2d_ms_sum, total_ms,
                    total_ms > 0 ? copy_size / (1024.0*1024.0*1024.0) / (total_ms/1000.0) : 0.0);
        }
        return 0;
    }
    /* -------- end P4 ring path -------- */


    /* V3 path: GPU section is k1-encrypted ciphertext, decrypted via ioctl
     * 112. Pass the mmap'd ciphertext directly as cipher_buf and the staging
     * buffer as plain_buf so there's no extra memcpy. Per-transfer metadata
     * lives in the crypto_meta section right after gpu_bytes; we walk it to
     * locate the slice covering this alloc's [offset_in_range, +copy_size)
     * byte range. */
    if (g_restore_v3) {
        uint64_t meta_section_off = desc->data_offset + desc->resmap_size
                                  + desc->cpu_bytes + desc->gpu_bytes;
        const rst_page_crypto_meta_t *range_meta =
            (const rst_page_crypto_meta_t *)(g_restore_base_map + meta_section_off);
        uint64_t total_meta_entries = desc->crypto_meta_bytes
                                    / sizeof(rst_page_crypto_meta_t);

        /* Walk meta entries accumulating `size` until we hit offset_in_range
         * (start of this alloc's slice) then again until copy_size is covered. */
        uint64_t acc = 0;
        uint64_t start_idx = 0;
        while (start_idx < total_meta_entries && acc < offset_in_range) {
            acc += range_meta[start_idx].size;
            start_idx++;
        }
        if (acc != offset_in_range) {
            fprintf(stderr, "[restore] FATAL: v3 alloc[%u] offset_in_range=%llu "
                    "does not align to crypto_meta entry boundary (acc=%llu)\n",
                    idx, (unsigned long long)offset_in_range,
                    (unsigned long long)acc);
            return -1;
        }
        uint64_t end_idx = start_idx;
        uint64_t slice_bytes = 0;
        while (end_idx < total_meta_entries && slice_bytes < copy_size) {
            slice_bytes += range_meta[end_idx].size;
            end_idx++;
        }
        if (slice_bytes != copy_size) {
            fprintf(stderr, "[restore] FATAL: v3 alloc[%u] copy_size=%llu does "
                    "not align to crypto_meta entry boundary (slice=%llu)\n",
                    idx, (unsigned long long)copy_size,
                    (unsigned long long)slice_bytes);
            return -1;
        }

        rst_uvm_decrypt_params_t p = {0};
        p.cipher_buf    = (uint64_t)(uintptr_t)(g_restore_base_map + file_offset);
        p.plain_buf     = (uint64_t)(uintptr_t)g_restore_staging;
        p.total_size    = copy_size;
        p.crypto_meta   = (uint64_t)(uintptr_t)&range_meta[start_idx];
        p.num_transfers = end_idx - start_idx;
        /* gpu_uuid left as all-zero = first GPU */

        int ret = ioctl(g_restore_uvm_fd,
                        UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES, &p);
        if (ret < 0 || p.rmStatus != 0) {
            fprintf(stderr, "[restore] FATAL: v3 alloc[%u] ioctl 112 failed: "
                    "ret=%d errno=%s rmStatus=0x%x (%llu transfers, %llu bytes)\n",
                    idx, ret, strerror(errno), p.rmStatus,
                    (unsigned long long)p.num_transfers,
                    (unsigned long long)p.total_size);
            return -1;
        }
        /* Staging buffer now holds plaintext — fall through to the H2D
         * path. Skip the v2/v4 memcpy below since we already wrote staging. */
    } else {
        memcpy(g_restore_staging, g_restore_base_map + file_offset, (size_t)copy_size);
    }

    /* V4 baseline: ciphertext is in the gpu_section at 2 MB chunk granularity.
     * Decrypt the staging buffer in place using per-chunk meta entries that
     * live in the crypto_meta section right after gpu_bytes. The alloc may
     * start mid-range, so compute the meta offset by dividing offset_in_range
     * by the chunk size. copy_size is always a multiple of 2 MB for vLLM. */
    if (g_restore_v4) {
        /* Safety: V4 is strictly 2 MB chunk aligned. Alloc offset + copy_size
         * must both be multiples of 2 MB, else the meta lookup is wrong and
         * decrypt would silently produce garbage. Fail loudly instead. */
        if ((offset_in_range % RST_CRYPTO_CHUNK_BYTES) != 0 ||
            (copy_size       % RST_CRYPTO_CHUNK_BYTES) != 0 ||
            copy_size < RST_CRYPTO_CHUNK_BYTES) {
            fprintf(stderr, "[restore] FATAL: v4 alloc[%u] not 2MB aligned "
                    "(offset=%llu copy_size=%llu)\n",
                    idx, (unsigned long long)offset_in_range,
                    (unsigned long long)copy_size);
            return -1;
        }
        uint64_t meta_section_off = desc->data_offset + desc->resmap_size
                                  + desc->cpu_bytes + desc->gpu_bytes;
        const ckpt_page_meta_t *meta =
            (const ckpt_page_meta_t *)(g_restore_base_map + meta_section_off)
            + (offset_in_range / RST_CRYPTO_CHUNK_BYTES);
        uint32_t n_chunks = (uint32_t)(copy_size / RST_CRYPTO_CHUNK_BYTES);
        if (n_chunks > 0) {
            int rc = ckpt_crypto_decrypt_pages(g_restore_k3_key,
                                               g_restore_staging,
                                               g_restore_staging,
                                               meta,
                                               n_chunks,
                                               (size_t)RST_CRYPTO_CHUNK_BYTES,
                                               /*num_threads=*/4);
            if (rc != 0) {
                fprintf(stderr, "[restore] FATAL: v4 decrypt failed alloc[%u] "
                                "(n_chunks=%u)\n", idx, n_chunks);
                return -1;
            }
        }
    }

    /* V4 delta overlay: scan delta blocks for any that overlap this alloc's
     * [alloc_va, alloc_va + copy_size) window. For each intersecting block,
     * decrypt the relevant 2 MB chunks from the delta's ciphertext section
     * and overlay them onto the staging buffer at the correct in-alloc
     * offset. The base bytes + delta overlay together form the final H2D
     * payload. */
    if (g_restore_delta_v4 && g_restore_delta_blocks) {
        uint64_t alloc_end = alloc_va + copy_size;
        uint32_t nb = g_restore_delta_hdr->num_blocks;
        uint32_t overlays = 0;

        for (uint32_t b = 0; b < nb; b++) {
            rst_delta_v4_block_t *blk = &g_restore_delta_blocks[b];
            uint64_t blk_start = blk->base_va;
            uint64_t blk_end   = blk_start + blk->length;

            /* intersection */
            uint64_t lo = alloc_va  > blk_start ? alloc_va  : blk_start;
            uint64_t hi = alloc_end < blk_end   ? alloc_end : blk_end;
            if (lo >= hi) continue;  /* no overlap */

            /* Must be 2 MB aligned on both sides — else the meta lookup breaks. */
            if (((lo - blk_start) % RST_CRYPTO_CHUNK_BYTES_LOCAL) != 0 ||
                ((hi - lo)         % RST_CRYPTO_CHUNK_BYTES_LOCAL) != 0) {
                fprintf(stderr, "[restore] FATAL: delta block[%u] overlap not "
                        "2MB aligned (lo=0x%llx hi=0x%llx blk_start=0x%llx)\n",
                        b, (unsigned long long)lo, (unsigned long long)hi,
                        (unsigned long long)blk_start);
                return -1;
            }

            uint64_t chunks_before_in_blk = (lo - blk_start) / RST_CRYPTO_CHUNK_BYTES_LOCAL;
            uint32_t n_chunks_overlay     = (uint32_t)((hi - lo) / RST_CRYPTO_CHUNK_BYTES_LOCAL);

            /* Read ciphertext slice from delta image into staging (at the
             * right in-staging offset), decrypt in place, then the
             * subsequent cudaMemcpy will pick it up along with the base. */
            uint64_t src_off_in_delta = blk->data_offset + (lo - blk_start);
            uint64_t dst_off_in_stage = lo - alloc_va;
            if (src_off_in_delta + (hi - lo) > g_restore_delta_size) {
                fprintf(stderr, "[restore] FATAL: delta block[%u] exceeds delta size\n", b);
                return -1;
            }
            memcpy(g_restore_staging + dst_off_in_stage,
                   g_restore_delta_map + src_off_in_delta,
                   (size_t)(hi - lo));

            /* Meta for this block lives right after its ciphertext. */
            uint64_t meta_off_in_delta = blk->data_offset + blk->length;
            const ckpt_page_meta_t *meta =
                (const ckpt_page_meta_t *)(g_restore_delta_map + meta_off_in_delta)
                + chunks_before_in_blk;

            int rc = ckpt_crypto_decrypt_pages(g_restore_k3_key,
                                                g_restore_staging + dst_off_in_stage,
                                                g_restore_staging + dst_off_in_stage,
                                                meta,
                                                n_chunks_overlay,
                                                (size_t)RST_CRYPTO_CHUNK_BYTES_LOCAL,
                                                /*num_threads=*/4);
            if (rc != 0) {
                fprintf(stderr, "[restore] FATAL: delta block[%u] decrypt failed "
                        "(%u chunks)\n", b, n_chunks_overlay);
                return -1;
            }
            overlays++;
            __sync_fetch_and_add(&g_restore_delta_overlays, 1);
            __sync_fetch_and_add(&g_restore_delta_bytes_applied, (hi - lo));
        }

        if (overlays > 0) {
            __sync_fetch_and_add(&g_restore_delta_allocs_touched, 1);
            fprintf(stderr, "[restore] alloc[%u] overlaid %u delta block(s)\n",
                    idx, overlays);
        }
    }

    double dec_ms = now_ms() - t_dec_start;
    double t_h2d_start = now_ms();
    cudaError_t cerr = real_cudaMemcpy_fn_ptr(new_ptr, g_restore_staging,
                                               (size_t)copy_size,
                                               cudaMemcpyHostToDevice);
    double h2d_ms = now_ms() - t_h2d_start;
    double ms = now_ms() - t0;

    /* Aggregate decrypt/H2D breakdown across all allocs for end-of-restore summary */
    if (g_restore_shared_dec_us)
        __sync_fetch_and_add(g_restore_shared_dec_us, (uint64_t)(dec_ms * 1000.0));
    if (g_restore_shared_h2d_us)
        __sync_fetch_and_add(g_restore_shared_h2d_us, (uint64_t)(h2d_ms * 1000.0));

    if (cerr != cudaSuccess) {
        fprintf(stderr, "[restore] FATAL: cudaMemcpy H2D failed at alloc[%u]: %d\n", idx, cerr);
        return -1;
    }

    /* Shared counters — visible across fork boundaries */
    uint64_t t_end_us = now_us();
    if (g_restore_shared_done)
        __sync_fetch_and_add(g_restore_shared_done, 1);
    if (g_restore_shared_bytes)
        __sync_fetch_and_add(g_restore_shared_bytes, copy_size);
    if (g_restore_shared_sum_us)
        __sync_fetch_and_add(g_restore_shared_sum_us, t_end_us - t_start_us);
    if (g_restore_shared_last_us) {
        /* monotonic max-update; workers may race but we accept last-writer-wins
         * since we're only tracking a wall-clock end bound */
        uint64_t cur;
        do {
            cur = *g_restore_shared_last_us;
            if (t_end_us <= cur) break;
        } while (!__sync_bool_compare_and_swap(g_restore_shared_last_us, cur, t_end_us));
    }

    const char *tag = "";
    if ((uint64_t)new_size > ckpt_size)      tag = "  [alloc bigger, tail zero]";
    else if ((uint64_t)new_size < ckpt_size) tag = "  [alloc smaller, ckpt truncated]";

    if (g_verbose || (idx < 5) || ((idx % 20) == 0))
        fprintf(stderr, "[restore] alloc[%u] restored %.2f MB in %.1f ms (%.1f GB/s)%s\n",
                idx, copy_size / (1024.0*1024.0), ms,
                ms > 0 ? copy_size / (1024.0*1024.0*1024.0) / (ms/1000.0) : 0.0,
                tag);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Allocation recording                                                */
/* ------------------------------------------------------------------ */
static void gate_record_alloc(void *ptr, size_t size, const char *src)
{
    if (!g_gate || !ptr) return;

    uint64_t va = (uint64_t)(uintptr_t)ptr;

    if ((uint64_t)size > g_gate->user_size) {
        __sync_synchronize();
        g_gate->user_va = va;
        g_gate->user_size = (uint64_t)size;
        __sync_synchronize();
    }

    ckpt_alloc_hdr_t *hdr = GATE_ALLOC_HDR(g_gate);
    ckpt_alloc_entry_t *table = GATE_ALLOC_TABLE(g_gate);
    uint32_t idx = hdr->count;
    if (idx < GATE_ALLOC_MAX) {
        table[idx].va = va;
        table[idx].size = (uint64_t)size;
        table[idx].flags = 0;
        table[idx]._pad = 0;
        __sync_synchronize();
        hdr->count = idx + 1;
        __sync_synchronize();
    } else {
        fprintf(stderr, "[inference] WARNING: alloc table full (%u)\n", GATE_ALLOC_MAX);
        return;
    }

    if (g_verbose)
        fprintf(stderr, "[inference] %s: alloc[%u] va=0x%lx size=%zu\n",
                src, idx, (unsigned long)va, size);
}

/* ------------------------------------------------------------------ */
/* Mark an allocation as freed.                                        */
/*                                                                      */
/* Called from cudaFree / cudaFreeAsync / cuMemFree_v2 interceptors.   */
/* Walks the gate table from the END (most recent first — vLLM tends   */
/* to free in LIFO-ish order) and ORs in CKPT_ALLOC_FLAG_FREED on the  */
/* most recent live entry whose va matches.                            */
/*                                                                      */
/* The entry stays in the table (count is append-only) but the freed   */
/* flag tells the baseline / restore paths to skip it.                 */
/* ------------------------------------------------------------------ */
static void gate_mark_freed(void *ptr)
{
    if (!g_gate || !ptr) return;

    uint64_t va = (uint64_t)(uintptr_t)ptr;
    ckpt_alloc_hdr_t   *hdr   = GATE_ALLOC_HDR(g_gate);
    ckpt_alloc_entry_t *table = GATE_ALLOC_TABLE(g_gate);
    uint32_t n = hdr->count;

    for (int32_t i = (int32_t)n - 1; i >= 0; i--) {
        if (table[i].va == va && !(table[i].flags & CKPT_ALLOC_FLAG_FREED)) {
            table[i].flags |= CKPT_ALLOC_FLAG_FREED;
            __sync_synchronize();
            if (g_verbose)
                fprintf(stderr, "[inference] free: alloc[%u] va=0x%lx size=%llu marked freed\n",
                        i, (unsigned long)va,
                        (unsigned long long)table[i].size);
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Save H2D data to disk with k3 encryption (model weight capture)     */
/* ------------------------------------------------------------------ */
static void save_h2d_weight(void *gpu_dst, const void *host_src, size_t count)
{
    if (!g_gate) return;

    /* Skip tiny copies — these are PyTorch internal transfers, not model weights */
    if (count < 4096) return;

    /* Lazy k3 key generation on first H2D copy */
    if (!g_k3_ready) {
        pthread_mutex_lock(&g_k3_lock);
        if (!g_k3_ready) {
            if (ckpt_crypto_gen_key(g_k3_key) == 0) {
                g_k3_ready = 1;
                mkdir(g_weights_dir, 0755);
                char k3_path[512];
                snprintf(k3_path, sizeof(k3_path), "%s/weights_k3.key", g_weights_dir);
                FILE *kf = fopen(k3_path, "wb");
                if (kf) { fwrite(g_k3_key, 1, CKPT_CRYPTO_KEY_SIZE, kf); fclose(kf); }
                fprintf(stderr, "[llm_ckpt] k3 key generated → %s\n", k3_path);
            }
        }
        pthread_mutex_unlock(&g_k3_lock);
        if (!g_k3_ready) return;
    }

    /* Find the allocation this H2D copy targets */
    uint64_t addr = (uint64_t)(uintptr_t)gpu_dst;
    ckpt_alloc_hdr_t *hdr = GATE_ALLOC_HDR(g_gate);
    ckpt_alloc_entry_t *tbl = GATE_ALLOC_TABLE(g_gate);
    int alloc_idx = -1;

    for (uint32_t i = 0; i < hdr->count; i++) {
        if (addr >= tbl[i].va && addr < tbl[i].va + tbl[i].size) {
            tbl[i].flags |= CKPT_ALLOC_FLAG_H2D;
            alloc_idx = i;
            break;
        }
    }

    if (alloc_idx < 0) return;

    /* Save the host buffer with k3 encryption */
    mkdir(g_weights_dir, 0755);

    char path[512];
    snprintf(path, sizeof(path), "%s/weight_%04d_0x%lx.bin",
             g_weights_dir, alloc_idx, (unsigned long)addr);

    /* Encrypt the host data with k3 */
    pthread_mutex_lock(&g_k3_lock);

    uint32_t num_pages = (count + CKPT_CRYPTO_PAGE_SIZE - 1) / CKPT_CRYPTO_PAGE_SIZE;
    uint64_t padded_data_size = (uint64_t)num_pages * CKPT_CRYPTO_PAGE_SIZE;
    /* Header: VA(8) + offset_in_alloc(8) + data_size(8) + num_pages(4) + pad(4) = 32 bytes */
    #define WEIGHT_HDR_SIZE 32
    uint64_t meta_size = num_pages * sizeof(ckpt_page_meta_t);
    uint64_t total_size = WEIGHT_HDR_SIZE + padded_data_size + meta_size;
    uint64_t aligned_size = (total_size + 4095) & ~4095ULL;

    uint8_t *img_buf = NULL;
    if (posix_memalign((void **)&img_buf, 4096, aligned_size) != 0) {
        pthread_mutex_unlock(&g_k3_lock);
        return;
    }
    memset(img_buf, 0, aligned_size);  /* zero entire buffer */

    uint8_t *enc_data = img_buf + WEIGHT_HDR_SIZE;
    ckpt_page_meta_t *meta = (ckpt_page_meta_t *)(img_buf + WEIGHT_HDR_SIZE + padded_data_size);

    {
        double t0 = now_ms();

        ckpt_crypto_encrypt_pages(g_k3_key, (const uint8_t *)host_src, enc_data,
                                  meta, num_pages,
                                  CKPT_CRYPTO_PAGE_SIZE,
                                  g_k3_iv_counter, 0);
        g_k3_iv_counter += num_pages;

        /* Pack header at start of aligned buffer */
        uint64_t offset_in_alloc = addr - tbl[alloc_idx].va;
        uint32_t np = num_pages;
        uint32_t pad = 0;
        memcpy(img_buf,      &addr, 8);
        memcpy(img_buf + 8,  &offset_in_alloc, 8);
        memcpy(img_buf + 16, &count, 8);
        memcpy(img_buf + 24, &np, 4);
        memcpy(img_buf + 28, &pad, 4);

        /* Write with O_DIRECT */
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (fd < 0)
            fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644); /* fallback */
        if (fd >= 0) {
            uint64_t written = 0;
            while (written < aligned_size) {
                uint64_t chunk = aligned_size - written;
                if (chunk > (256ULL * 1024 * 1024)) chunk = 256ULL * 1024 * 1024;
                ssize_t ret = write(fd, img_buf + written, chunk);
                if (ret <= 0) break;
                written += ret;
            }
            ftruncate(fd, total_size);
            close(fd);

            double ms = now_ms() - t0;
            g_weights_saved++;
            g_weights_bytes += count;

            if (g_verbose || g_weights_saved <= 5 || g_weights_saved % 50 == 0)
                fprintf(stderr, "[inference] weight[%u] saved: alloc[%d] offset=0x%lx "
                        "size=%.2fMB enc=%.1fms → %s\n",
                        g_weights_saved, alloc_idx,
                        (unsigned long)offset_in_alloc,
                        count / (1024.0 * 1024.0), ms, path);
        }
    }

    free(img_buf);
    pthread_mutex_unlock(&g_k3_lock);
    #undef WEIGHT_HDR_SIZE
}

/* Just mark the H2D flag without saving data */
static void mark_h2d_flag(void *dst)
{
    if (!g_gate) return;
    uint64_t addr = (uint64_t)(uintptr_t)dst;
    ckpt_alloc_hdr_t *hdr = GATE_ALLOC_HDR(g_gate);
    ckpt_alloc_entry_t *tbl = GATE_ALLOC_TABLE(g_gate);
    for (uint32_t i = 0; i < hdr->count; i++) {
        if (addr >= tbl[i].va && addr < tbl[i].va + tbl[i].size) {
            tbl[i].flags |= CKPT_ALLOC_FLAG_H2D;
            break;
        }
    }
}

/* Prefetch only model weight pages (H2D destinations) */
static void prefetch_h2d(void *dst, size_t count)
{
    if (real_cudaMemPrefetchAsync)
        real_cudaMemPrefetchAsync(dst, count, 0, 0);
}

/* ------------------------------------------------------------------ */
/* Engine state dump                                                   */
/* ------------------------------------------------------------------ */
/* P5: dump vLLM V1 engine state (prefix cache + scheduler queues) by
 * delegating to vllm_ckpt/ckpt_engine_state.py. That file is editable
 * without rebuilding the .so and lives next to this source so both
 * checkpoint and restore sides share the same V1-aware logic.
 *
 * Env:
 *   CKPT_ENGINE_STATE   output JSON path (default /tmp/ckpt_engine_state.json)
 *   CKPT_PY_HELPER_DIR  directory containing ckpt_engine_state.py
 *                       (default /home/leejongh/migration/open-gpu-kernel-modules/vllm_ckpt)
 */
static void dump_engine_state(void)
{
    if (!g_python_available)
        resolve_python_api();
    if (!g_python_available) return;

    const char *out_path = getenv("CKPT_ENGINE_STATE");
    if (!out_path) out_path = "/tmp/ckpt_engine_state.json";
    const char *helper_dir = getenv("CKPT_PY_HELPER_DIR");
    if (!helper_dir)
        helper_dir = "/path/to/artifact_flash/benchmarks/flash/";

    static char pybuf[2048];
    snprintf(pybuf, sizeof(pybuf),
        "import sys\n"
        "try:\n"
        "    if '%s' not in sys.path: sys.path.insert(0, '%s')\n"
        "    import ckpt_engine_state as _ces\n"
        "    _ces.dump_state('%s')\n"
        "except Exception as _e:\n"
        "    import traceback\n"
        "    print('[inference] engine state dump failed: %%s' %% _e, file=sys.stderr)\n"
        "    traceback.print_exc()\n",
        helper_dir, helper_dir, out_path);

    PyGILState_STATE gstate = fn_gil_ensure();
    fn_pyrun(pybuf);
    fn_gil_release(gstate);
}

/* ------------------------------------------------------------------ */
/* Checkpoint handler                                                  */
/* ------------------------------------------------------------------ */
/* In-app cudaMemcpyAsync baseline handler (P1).                      */
/* Called from every sync hook in any process that LD_PRELOAD'd us.   */
/* CAS on baseline_req claims the work — only one process executes.  */
/* ------------------------------------------------------------------ */
/* Background baseline thread args — heap-allocated, freed by thread. */
typedef struct {
    char                 path[256];
    ckpt_alloc_entry_t  *tbl;
    uint32_t             count;
    uint32_t             n_mstreams;
} bl_thread_args_t;

static void *baseline_bg_thread(void *arg)
{
    bl_thread_args_t *a = (bl_thread_args_t *)arg;
    ckpt_baseline_result_t r = {0};
    /* Copy the alloc table into our own buffer so the background thread
     * doesn't race with app-side cudaMalloc/cudaFree mutations of the gate
     * alloc table during the concurrent baseline. */
    int rc = ckpt_baseline_run(a->path, a->tbl, a->count,
                               /*stop_the_world=*/0,
                               /*n_mstreams=*/(int)a->n_mstreams, &r);
    if (g_gate) {
        g_gate->baseline_ms    = r.elapsed_ms;
        g_gate->baseline_bytes = r.bytes;
        __sync_synchronize();
        if (rc == 0) {
            g_gate->baseline_done = 1;
        } else {
            g_gate->baseline_err  = 1;
            g_gate->baseline_done = 1;
        }
        __sync_synchronize();
    }
    free(a->tbl);
    free(a);
    return NULL;
}

static void maybe_baseline_post_sync(void)
{
    if (!g_gate || g_gate->baseline_req != 1) return;

    /* Atomically claim: 1 -> 2 means "in progress". */
    if (!__sync_bool_compare_and_swap(&g_gate->baseline_req, 1, 2))
        return;

    int concurrent = (int)g_gate->baseline_concurrent;
    fprintf(stderr, "[baseline] claimed baseline_req in pid=%d (mode=%s)\n",
            getpid(), concurrent ? "concurrent" : "stop-and-copy");

    const char *path = g_gate->baseline_path[0]
                       ? g_gate->baseline_path
                       : "/dev/shm/ckpt_base.img";

    ckpt_alloc_hdr_t   *hdr = GATE_ALLOC_HDR(g_gate);
    ckpt_alloc_entry_t *tbl = GATE_ALLOC_TABLE(g_gate);

    if (concurrent) {
        /* P1e: spawn a detached background thread so the main app thread
         * returns from the sync hook immediately and keeps running
         * (launching kernels, writing KV cache, etc). The background
         * thread runs the worker pool; the app continues in parallel so
         * the kernel dirty tracker sees real KV writes. */
        bl_thread_args_t *a = calloc(1, sizeof(*a));
        if (!a) { g_gate->baseline_err = 1; g_gate->baseline_done = 1; return; }
        snprintf(a->path, sizeof(a->path), "%s", path);
        /* Snapshot the current alloc table (it's append-only but can grow
         * during the baseline — we want the set as of the trigger). */
        uint32_t n = hdr->count;
        a->count = n;
        a->tbl = calloc(n, sizeof(ckpt_alloc_entry_t));
        if (!a->tbl) {
            free(a);
            g_gate->baseline_err = 1; g_gate->baseline_done = 1; return;
        }
        memcpy(a->tbl, tbl, (size_t)n * sizeof(ckpt_alloc_entry_t));
        a->n_mstreams = g_gate->baseline_mstreams;

        pthread_t th;
        if (pthread_create(&th, NULL, baseline_bg_thread, a) != 0) {
            free(a->tbl); free(a);
            g_gate->baseline_err = 1; g_gate->baseline_done = 1; return;
        }
        pthread_detach(th);
        fprintf(stderr, "[baseline] concurrent thread spawned; sync hook returning\n");
        return;
    }

    /* P1d: stop-and-copy — run inline on the main thread (app is quiesced). */
    ckpt_baseline_result_t r = {0};
    int rc = ckpt_baseline_run(path, tbl, hdr->count,
                               /*stop_the_world=*/1,
                               /*n_mstreams=*/(int)g_gate->baseline_mstreams, &r);

    g_gate->baseline_ms    = r.elapsed_ms;
    g_gate->baseline_bytes = r.bytes;
    __sync_synchronize();
    if (rc == 0) {
        g_gate->baseline_done = 1;
    } else {
        g_gate->baseline_err  = 1;
        g_gate->baseline_done = 1;
    }
    __sync_synchronize();
}

/* ------------------------------------------------------------------ */
/* P1e delta handler: agent signals after baseline + stun + GET_DIRTY  */
/* Reads dirty range list from gate's dirty table, runs                */
/* ckpt_baseline_delta, reports back via gate fields.                  */
/* ------------------------------------------------------------------ */
static void maybe_delta_post_sync(void)
{
    if (!g_gate || g_gate->delta_req != 1) return;
    if (!__sync_bool_compare_and_swap(&g_gate->delta_req, 1, 2))
        return;

    fprintf(stderr, "[delta] claimed delta_req in pid=%d\n", getpid());

    const char *path = g_gate->delta_path[0]
                       ? g_gate->delta_path
                       : "/dev/shm/ckpt_delta.img";

    ckpt_dirty_hdr_t   *dhdr = GATE_DIRTY_HDR(g_gate);
    ckpt_dirty_range_t *dtbl = GATE_DIRTY_TABLE(g_gate);
    uint32_t n = dhdr->count;

    fprintf(stderr, "[delta] %u dirty ranges\n", n);
    ckpt_baseline_result_t r = {0};
    int rc = ckpt_baseline_delta(path, dtbl, n, &r);

    g_gate->delta_ms    = r.elapsed_ms;
    g_gate->delta_bytes = r.bytes;
    __sync_synchronize();
    if (rc == 0) {
        g_gate->delta_done = 1;
    } else {
        g_gate->delta_err  = 1;
        g_gate->delta_done = 1;
    }
    __sync_synchronize();
}

static void maybe_checkpoint_post_sync(void)
{
    if (!g_gate || !g_is_owner || !g_gate->ckpt_req)
        return;

    pthread_mutex_lock(&g_ckpt_mutex);

    if (g_gate->ckpt_req) {
        fprintf(stderr, "[inference] GPU idle — checkpoint boundary\n");
        fprintf(stderr, "[inference] Weights already saved: %u files, %.2f GB\n",
                g_weights_saved, g_weights_bytes / (1024.0 * 1024.0 * 1024.0));

        dump_engine_state();

        g_gate->ckpt_seq++;
        __sync_synchronize();
        g_gate->phase = GATE_PHASE_AT_BOUNDARY;
        __sync_synchronize();

        fprintf(stderr, "[inference] AT_BOUNDARY (seq=%llu) — agent handles KV cache\n",
                (unsigned long long)g_gate->ckpt_seq);

        /* While parked at AT_BOUNDARY, service P1e delta requests in place.
         * This keeps the app fully quiesced (no new kernel launches, no
         * KV writes) between GET_DIRTY_PAGES and the delta copy, so the
         * snapshot stays consistent. */
        while (g_gate->phase != GATE_PHASE_RESUME) {
            maybe_delta_post_sync();
            usleep(100);
        }
        __sync_synchronize();

        fprintf(stderr, "[inference] RESUME\n");
    }

    pthread_mutex_unlock(&g_ckpt_mutex);
}

/* ================================================================== */
/* Intercepted CUDA functions                                          */
/* ================================================================== */

/* --- Memory allocation: cudaMallocManaged + advise + prefetch --- */

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    ENSURE_SYMBOLS;
    if (!real_cudaMallocManaged) {
        if (!real_cudaMalloc) return cudaErrorUnknown;
        return real_cudaMalloc(devPtr, size);
    }

    cudaError_t err = real_cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);
    if (err == cudaSuccess) {
        if (real_cudaMemAdvise)
            real_cudaMemAdvise(*devPtr, size, cudaMemAdviseSetPreferredLocation, 0);
        if (real_cudaMemPrefetchAsync)
            real_cudaMemPrefetchAsync(*devPtr, size, 0, 0);
        gate_record_alloc(*devPtr, size, "cudaMalloc→Managed");
        if (g_restore_mode && restore_apply_alloc(*devPtr, size) != 0)
            abort();
    }
    return err;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    (void)stream;
    if (!real_cudaMallocManaged) {
        if (!real_cudaMallocAsync) return cudaErrorUnknown;
        return real_cudaMallocAsync(devPtr, size, stream);
    }

    cudaError_t err = real_cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);
    if (err == cudaSuccess) {
        if (real_cudaMemAdvise)
            real_cudaMemAdvise(*devPtr, size, cudaMemAdviseSetPreferredLocation, 0);
        if (real_cudaMemPrefetchAsync)
            real_cudaMemPrefetchAsync(*devPtr, size, 0, 0);
        gate_record_alloc(*devPtr, size, "cudaMallocAsync→Managed");
        if (g_restore_mode && restore_apply_alloc(*devPtr, size) != 0)
            abort();
    }
    return err;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    ENSURE_SYMBOLS;
    typedef CUresult (*cuMemAllocManaged_fn)(CUdeviceptr *, size_t, unsigned int);
    static cuMemAllocManaged_fn real_cuMemAllocManaged = NULL;
    if (!real_cuMemAllocManaged)
        real_cuMemAllocManaged = dlsym(RTLD_NEXT, "cuMemAllocManaged");

    if (!real_cuMemAllocManaged) {
        if (!real_cuMemAlloc_v2) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemAlloc_v2(dptr, bytesize);
    }

    CUresult res = real_cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
    if (res == CUDA_SUCCESS) {
        if (real_cudaMemAdvise)
            real_cudaMemAdvise((void *)(uintptr_t)*dptr, bytesize,
                              cudaMemAdviseSetPreferredLocation, 0);
        if (real_cudaMemPrefetchAsync)
            real_cudaMemPrefetchAsync((void *)(uintptr_t)*dptr, bytesize, 0, 0);
        gate_record_alloc((void *)(uintptr_t)*dptr, bytesize, "cuMemAlloc_v2→Managed");
        if (g_restore_mode &&
            restore_apply_alloc((void *)(uintptr_t)*dptr, bytesize) != 0)
            abort();
    }
    return res;
}

/* --- Memory free: mark gate alloc table entries as stale --- */

cudaError_t cudaFree(void *devPtr)
{
    ENSURE_SYMBOLS;
    if (devPtr) gate_mark_freed(devPtr);
    if (!real_cudaFree) return cudaErrorUnknown;
    return real_cudaFree(devPtr);
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    if (devPtr) gate_mark_freed(devPtr);
    if (!real_cudaFreeAsync) {
        if (!real_cudaFree) return cudaErrorUnknown;
        return real_cudaFree(devPtr);
    }
    return real_cudaFreeAsync(devPtr, stream);
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    ENSURE_SYMBOLS;
    if (dptr) gate_mark_freed((void *)(uintptr_t)dptr);
    if (!real_cuMemFree_v2) return CUDA_ERROR_NOT_INITIALIZED;
    return real_cuMemFree_v2(dptr);
}

/* --- Memory copy: capture H2D weights + selective prefetch --- */

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind)
{
    ENSURE_SYMBOLS;
    if (!real_cudaMemcpy_fn_ptr) return cudaErrorUnknown;

    cudaError_t err = real_cudaMemcpy_fn_ptr(dst, src, count, kind);

    if (err == cudaSuccess && kind == cudaMemcpyHostToDevice) {
        if (g_save_weights)
            save_h2d_weight(dst, src, count);
        else
            mark_h2d_flag(dst);
        prefetch_h2d(dst, count);
    }

    return err;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    if (!real_cudaMemcpyAsync_fn_ptr) return cudaErrorUnknown;

    cudaError_t err = real_cudaMemcpyAsync_fn_ptr(dst, src, count, kind, stream);

    if (err == cudaSuccess && kind == cudaMemcpyHostToDevice) {
        if (g_save_weights)
            save_h2d_weight(dst, src, count);
        else
            mark_h2d_flag(dst);
        prefetch_h2d(dst, count);
    }

    if (!g_in_ckpt_worker && g_gate && g_is_owner && g_gate->ckpt_req) {
        if (real_cudaDeviceSynchronize)
            real_cudaDeviceSynchronize();

        if (g_gate && g_is_owner && __sync_bool_compare_and_swap(&g_init_signaled, 0, 1)) {
            __sync_synchronize();
            g_gate->phase = GATE_PHASE_INIT;
            __sync_synchronize();
            fprintf(stderr, "[inference] first sync — GATE_PHASE_INIT\n");
        }

        maybe_checkpoint_post_sync();
    }

    return err;
}

/* --- cudaMallocManaged passthrough --- */

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    ENSURE_SYMBOLS;
    if (!real_cudaMallocManaged) return cudaErrorUnknown;

    cudaError_t err = real_cudaMallocManaged(devPtr, size, flags);
    if (err == cudaSuccess)
        gate_record_alloc(*devPtr, size, "cudaMallocManaged");
    return err;
}

/* --- Synchronization --- */

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    if (!real_cudaStreamSynchronize) return cudaErrorUnknown;

    cudaError_t err = real_cudaStreamSynchronize(stream);

    if (g_gate && g_is_owner && __sync_bool_compare_and_swap(&g_init_signaled, 0, 1)) {
        __sync_synchronize();
        g_gate->phase = GATE_PHASE_INIT;
        __sync_synchronize();
        fprintf(stderr, "[inference] first sync — GATE_PHASE_INIT\n");
    }

    maybe_checkpoint_post_sync();
    maybe_baseline_post_sync();
    maybe_delta_post_sync();
    return err;
}

cudaError_t cudaDeviceSynchronize(void)
{
    ENSURE_SYMBOLS;
    if (!real_cudaDeviceSynchronize) return cudaErrorUnknown;

    cudaError_t err = real_cudaDeviceSynchronize();

    if (g_gate && g_is_owner && __sync_bool_compare_and_swap(&g_init_signaled, 0, 1)) {
        __sync_synchronize();
        g_gate->phase = GATE_PHASE_INIT;
        __sync_synchronize();
        fprintf(stderr, "[inference] first sync — GATE_PHASE_INIT\n");
    }

    maybe_checkpoint_post_sync();
    maybe_baseline_post_sync();
    maybe_delta_post_sync();
    return err;
}

/* ================================================================== */
/* Constructor / Destructor                                            */
/* ================================================================== */

__attribute__((constructor))
static void ckpt_inference_init(void)
{
    const char *env;

    env = getenv("CKPT_GATE_FILE");
    if (env) strncpy(g_gate_path, env, sizeof(g_gate_path) - 1);

    env = getenv("CKPT_WEIGHTS_DIR");
    if (env) strncpy(g_weights_dir, env, sizeof(g_weights_dir) - 1);

    g_verbose = (getenv("CKPT_VERBOSE") && strcmp(getenv("CKPT_VERBOSE"), "1") == 0);

    /* Try to open existing gate file (agent started first) */
    int fd = open(g_gate_path, O_RDWR);
    fprintf(stderr, "[llm_ckpt] open(%s) = %d (errno=%d)\n", g_gate_path, fd, errno);
    if (fd >= 0) {
        /* Gate exists — agent created it. Read config flags. */
        g_gate = mmap(NULL, GATE_FILE_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
        close(fd);
        if (g_gate == MAP_FAILED) { g_gate = NULL; return; }

        /* Read agent's config */
        g_save_weights = (g_gate->config_flags & CKPT_CFG_SAVE_WEIGHTS) != 0;
        fprintf(stderr, "[llm_ckpt] config_flags=0x%x save_weights=%d app_pid=%d\n",
                g_gate->config_flags, g_save_weights, g_gate->app_pid);

        /* First process to set app_pid becomes owner */
        int pid = getpid();
        int cas_ok = __sync_bool_compare_and_swap(&g_gate->app_pid, 0, pid);
        fprintf(stderr, "[llm_ckpt] CAS(app_pid, 0, %d) = %d, app_pid now = %d\n",
                pid, cas_ok, g_gate->app_pid);
        if (cas_ok) {
            g_is_owner = 1;
            fprintf(stderr, "[llm_ckpt] gate opened (owner, pid=%d, agent-created)\n", pid);
        } else {
            g_is_owner = 0;
            fprintf(stderr, "[llm_ckpt] gate opened (child, pid=%d)\n", pid);
        }
    } else {
        /* No agent — create gate file ourselves */
        fd = open(g_gate_path, O_CREAT | O_RDWR | O_EXCL, 0666);
        if (fd < 0 && errno == EEXIST) {
            /* Race: another process created it */
            fd = open(g_gate_path, O_RDWR);
            if (fd < 0) return;
            g_gate = mmap(NULL, GATE_FILE_SIZE, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, 0);
            close(fd);
            if (g_gate == MAP_FAILED) { g_gate = NULL; return; }
            g_is_owner = 0;
            g_save_weights = (g_gate->config_flags & CKPT_CFG_SAVE_WEIGHTS) != 0;
            fprintf(stderr, "[llm_ckpt] gate opened (child, pid=%d)\n", getpid());
        } else if (fd >= 0) {
            if (ftruncate(fd, GATE_FILE_SIZE) < 0) { close(fd); return; }
            g_gate = mmap(NULL, GATE_FILE_SIZE, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, 0);
            close(fd);
            if (g_gate == MAP_FAILED) { g_gate = NULL; return; }
            memset(g_gate, 0, GATE_FILE_SIZE);
            g_gate->app_pid = getpid();
            g_is_owner = 1;
            g_save_weights = 0;  /* no agent config — default off */
            fprintf(stderr, "[llm_ckpt] gate created (owner, pid=%d, no agent)\n", getpid());
        }
    }

    /* k3 key generated lazily on first H2D copy, not here */

    /* Restore mode: mmap-based per-alloc streaming restore.
       Init mmaps the checkpoint files and parses headers (no CUDA needed).
       Per-alloc restore (memcpy from mmap + H2D) happens inside each
       cudaMalloc interceptor — CUDA is up by then.
       Only the owner process restores. */
    if (g_gate && g_is_owner && (g_gate->config_flags & CKPT_CFG_RESTORE)) {
        const char *snap  = getenv("CKPT_RESTORE_SNAP");
        const char *base  = getenv("CKPT_RESTORE_BASE");
        const char *delta = getenv("CKPT_RESTORE_DELTA");
        if (!snap)  snap  = "/tmp/snap_ckpt_gate";
        if (!base)  base  = "/tmp/ckpt_inc_base.img";
        if (!delta) delta = "/tmp/ckpt_inc_delta.img";

        fprintf(stderr, "[restore] CKPT_CFG_RESTORE set; initializing mmap restore\n");
        if (restore_init(snap, base, delta) == 0) {
            g_restore_mode = 1;
        } else {
            fprintf(stderr, "[restore] FATAL: restore_init failed — aborting\n");
            abort();
        }
    }

    fprintf(stderr, "[llm_ckpt] LLM checkpoint loaded\n");
    fprintf(stderr, "[llm_ckpt]   gate=%s  weights=%s\n", g_gate_path, g_weights_dir);
    fprintf(stderr, "[llm_ckpt]   save_weights=%s  restore=%s\n",
            g_save_weights ? "ON" : "OFF", g_restore_mode ? "ON" : "OFF");
}

__attribute__((destructor))
static void ckpt_inference_fini(void)
{
    if (g_restore_mode) {
        uint32_t done   = g_restore_shared_done   ? *g_restore_shared_done   : 0;
        uint64_t bytes  = g_restore_shared_bytes  ? *g_restore_shared_bytes  : 0;
        uint64_t sum_us = g_restore_shared_sum_us ? *g_restore_shared_sum_us : 0;
        uint64_t first_us = g_restore_shared_first_us ? *g_restore_shared_first_us : 0;
        uint64_t last_us  = g_restore_shared_last_us  ? *g_restore_shared_last_us  : 0;

        double gb      = bytes / (1024.0 * 1024.0 * 1024.0);
        double sum_ms  = sum_us / 1000.0;
        double wall_ms = (first_us && last_us >= first_us)
                         ? (last_us - first_us) / 1000.0 : 0.0;
        double bw_sum  = sum_ms  > 0 ? gb / (sum_ms  / 1000.0) : 0.0;
        double bw_wall = wall_ms > 0 ? gb / (wall_ms / 1000.0) : 0.0;

        uint64_t dec_us = g_restore_shared_dec_us ? *g_restore_shared_dec_us : 0;
        uint64_t h2d_us = g_restore_shared_h2d_us ? *g_restore_shared_h2d_us : 0;
        double dec_ms   = dec_us / 1000.0;
        double h2d_ms   = h2d_us / 1000.0;
        double bw_dec   = dec_ms > 0 ? gb / (dec_ms / 1000.0) : 0.0;
        double bw_h2d   = h2d_ms > 0 ? gb / (h2d_ms / 1000.0) : 0.0;

        fprintf(stderr, "[restore] summary: %u allocations restored, %.2f GB H2D\n",
                done, gb);
        fprintf(stderr, "[restore]   decrypt sum      : %.1f ms  (%.2f GB/s)\n",
                dec_ms, bw_dec);
        fprintf(stderr, "[restore]   H2D sum          : %.1f ms  (%.2f GB/s)\n",
                h2d_ms, bw_h2d);
        fprintf(stderr, "[restore]   per-alloc sum    : %.1f ms  (%.2f GB/s throughput)\n",
                sum_ms, bw_sum);
        fprintf(stderr, "[restore]   first→last wall  : %.1f ms  (%.2f GB/s effective, including vLLM init gaps)\n",
                wall_ms, bw_wall);
        fprintf(stderr, "[restore]   alloc breakdown  : %u cudaMalloc hooks fired\n",
                g_restore_invocations);
        fprintf(stderr, "[restore]                      = %u static.img matched + %u base.img matched\n",
                g_restore_static_allocs_applied, g_restore_base_applied);
        fprintf(stderr, "[restore]                      + %u no-descriptor passthrough (VA in snap but not in base/static)\n",
                g_restore_passthrough_no_desc);
        fprintf(stderr, "[restore]                      + %u past-snap-end passthrough (cudaMalloc count > snap)\n",
                g_restore_passthrough_past_snap);
        fprintf(stderr, "[restore]   snap had %u allocs; static.img has %u descs; base.img has %u descs\n",
                g_restore_alloc_total,
                g_restore_static_hdr ? g_restore_static_hdr->num_ranges : 0,
                g_restore_base_hdr   ? g_restore_base_hdr->num_ranges   : 0);
        fprintf(stderr, "[restore]   size alignment   : %u exact + %u close(<=2MB) + %u MISMATCH (>2MB diff)\n",
                g_restore_size_exact, g_restore_size_close_2mb, g_restore_size_mismatch);
        fprintf(stderr, "[restore]                      (mismatch ~ cudaMalloc order shuffled — restore writes data to wrong tensors)\n");
        rst_pool_shutdown();
        if (g_restore_base_map)
            munmap(g_restore_base_map, g_restore_base_size);
        if (g_restore_delta_map)
            munmap(g_restore_delta_map, g_restore_delta_size);
        if (g_restore_shared_page)
            munmap(g_restore_shared_page, 4096);
        free(g_restore_staging);
        free(g_restore_alloc_va);
        free(g_restore_alloc_size);
        if (g_restore_uvm_fd >= 0) {
            close(g_restore_uvm_fd);
            g_restore_uvm_fd = -1;
        }
    }
    if (g_gate && g_is_owner) {
        fprintf(stderr, "[inference] Total weights saved: %u files, %.2f GB\n",
                g_weights_saved, g_weights_bytes / (1024.0 * 1024.0 * 1024.0));
        __sync_synchronize();
        g_gate->phase = GATE_PHASE_DONE;
        __sync_synchronize();
        usleep(200000);
    }
    if (g_gate) {
        munmap(g_gate, GATE_FILE_SIZE);
        g_gate = NULL;
    }
}
