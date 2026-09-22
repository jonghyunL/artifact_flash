/*
 * libgcr_intercept.cpp — GCR-style checkpoint emulator with byte-level
 *                        dirty tracking via CPU shadow execution.
 *
 * Emulation faithful to FAST'26 GCR (Zeng et al., Tsinghua). Reuses their
 * published per-kernel "dirty templates" (dirty_templates_kernels.cpp)
 * verbatim; implements the runtime plumbing (name resolution, dispatcher,
 * stop-and-copy incremental protocol, AES-GCM encrypt) ourselves,
 * cribbing the LD_PRELOAD scaffold from phos_emul/libphos_intercept.c.
 *
 * Dirty-marking sources:
 *   cudaMemcpy/cudaMemcpyAsync    — insert [dst, dst+count) into dirty set
 *   cudaMemset/cudaMemsetAsync    — insert [dst, dst+count) into dirty set
 *   cublasGemmEx                  — insert C output tile [C, C+n*ldc*bpe)
 *   cublasLtMatmul                — mark whole alloc containing D (fallback;
 *                                   descriptor-parsing TODO)
 *   cuLaunchKernel / cuLaunchKernelEx:
 *     (a) resolve CUfunction -> mangled name via cuModuleGetFunction or
 *         cuFuncGetName hook
 *     (b) look up template by name; if found, run synchronously on this
 *         thread right after real_cuLaunchKernel returns (shadow execution)
 *     (c) else fall back to pessimistic scan (all non-const arg slots that
 *         point into known allocs -> whole alloc dirty)
 *   cudaLaunchKernel:
 *     (a) resolve host_stub -> mangled name via __cudaRegisterFunction hook
 *     (b) same template lookup / pessimistic fallback as above
 *
 * Checkpoint protocol (stop-and-copy, byte-level incremental):
 *   At trigger time, with the app stalled:
 *     1. snapshot dirty set -> ranges, clear global set
 *     2. D2H + AES-GCM encrypt of every range, in place, into image slots
 *     3. resume app
 *   Round 0 captures the full working set (everything kernels have touched
 *   since process start); rounds 1+ capture only ranges dirtied since the
 *   previous round. Matches GCR/cuda.cpp:incremental_ckpt() semantics —
 *   one synchronous pass per checkpoint, no background precopy thread.
 *
 * Trigger: SIGUSR2 or GCR_CKPT_DELAY=<seconds> from first sync.
 *
 * Build: make -C gcr_emul libgcr_intercept.so
 */

#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <atomic>
#include <cstdarg>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/syscall.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

extern "C" {
#include "ckpt_crypto.h"
}
#include "gcr_runtime.h"

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
#define GCR_MAX_ALLOCS           4096
#define GCR_MAX_KERNELS          8192
#define GCR_MAX_HOSTFNS         65536
#define GCR_MAX_KNAME_LEN         512
#define GCR_MAX_KERNEL_PARAMS      64
#define GCR_MAX_RANGES         (1u << 20)   /* 1M range snapshot ceiling */
#define GCR_ENC_THREADS             0       /* 0 = all cores */

/* ------------------------------------------------------------------ */
/* Driver-API pointers shared with templates (declared in gcr_runtime.h) */
/* ------------------------------------------------------------------ */
extern "C" {
cuStreamCreate_fn          real_cuStreamCreate          = nullptr;
cuStreamSynchronize_fn     real_cuStreamSynchronize     = nullptr;
cuMemcpyDtoHAsync_v2_fn    real_cuMemcpyDtoHAsync_v2    = nullptr;
}

/* ------------------------------------------------------------------ */
/* Runtime-API + driver-API function pointers for our own hooks        */
/* ------------------------------------------------------------------ */
#define DECL_REAL(name, rty, args)                                     \
    typedef rty (*name##_fn) args;                                     \
    static name##_fn real_##name = nullptr

DECL_REAL(cudaMalloc,             cudaError_t,   (void **, size_t));
DECL_REAL(cudaMallocAsync,        cudaError_t,   (void **, size_t, cudaStream_t));
DECL_REAL(cudaFree,               cudaError_t,   (void *));
DECL_REAL(cudaFreeAsync,          cudaError_t,   (void *, cudaStream_t));
DECL_REAL(cudaMemcpy,             cudaError_t,   (void *, const void *, size_t, enum cudaMemcpyKind));
DECL_REAL(cudaMemcpyAsync,        cudaError_t,   (void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t));
DECL_REAL(cudaMemset,             cudaError_t,   (void *, int, size_t));
DECL_REAL(cudaMemsetAsync,        cudaError_t,   (void *, int, size_t, cudaStream_t));
DECL_REAL(cudaStreamSynchronize,  cudaError_t,   (cudaStream_t));
DECL_REAL(cudaDeviceSynchronize,  cudaError_t,   (void));
DECL_REAL(cudaStreamCreate,       cudaError_t,   (cudaStream_t *));
DECL_REAL(cudaStreamDestroy,      cudaError_t,   (cudaStream_t));
DECL_REAL(cudaMallocHost,         cudaError_t,   (void **, size_t));
DECL_REAL(cudaFreeHost,           cudaError_t,   (void *));
DECL_REAL(cudaEventCreate,        cudaError_t,   (cudaEvent_t *));
DECL_REAL(cudaEventDestroy,       cudaError_t,   (cudaEvent_t));
DECL_REAL(cudaEventRecord,        cudaError_t,   (cudaEvent_t, cudaStream_t));
DECL_REAL(cudaEventSynchronize,   cudaError_t,   (cudaEvent_t));
DECL_REAL(cudaEventElapsedTime,   cudaError_t,   (float *, cudaEvent_t, cudaEvent_t));
DECL_REAL(cudaLaunchKernel,       cudaError_t,   (const void *, dim3, dim3, void **, size_t, cudaStream_t));

DECL_REAL(cuMemAddressReserve,    CUresult, (CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long));
DECL_REAL(cuMemAddressFree,       CUresult, (CUdeviceptr, size_t));
DECL_REAL(cuMemMap,               CUresult, (CUdeviceptr, size_t, size_t,
                                             CUmemGenericAllocationHandle, unsigned long long));
DECL_REAL(cuMemUnmap,             CUresult, (CUdeviceptr, size_t));
DECL_REAL(cuMemAlloc_v2,          CUresult, (CUdeviceptr *, size_t));
DECL_REAL(cuMemFree_v2,           CUresult, (CUdeviceptr));

DECL_REAL(cuLaunchKernel,         CUresult, (CUfunction, unsigned, unsigned, unsigned,
                                             unsigned, unsigned, unsigned,
                                             unsigned, CUstream, void **, void **));
DECL_REAL(cuLaunchKernelEx,       CUresult, (const CUlaunchConfig *, CUfunction, void **, void **));
DECL_REAL(cuModuleGetFunction,    CUresult, (CUfunction *, CUmodule, const char *));
DECL_REAL(cuFuncGetName,          CUresult, (const char **, CUfunction));
DECL_REAL(cuFuncGetParamInfo,     CUresult, (CUfunction, size_t, size_t *, size_t *));

DECL_REAL(cublasGemmEx,           cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                                   int, int, int, const void *,
                                                   const void *, cudaDataType_t, int,
                                                   const void *, cudaDataType_t, int,
                                                   const void *, void *, cudaDataType_t, int,
                                                   cublasComputeType_t, cublasGemmAlgo_t));
DECL_REAL(cublasLtMatmul,         cublasStatus_t, (cublasLtHandle_t, cublasLtMatmulDesc_t,
                                                   const void *, const void *, cublasLtMatrixLayout_t,
                                                   const void *, cublasLtMatrixLayout_t,
                                                   const void *, const void *, cublasLtMatrixLayout_t,
                                                   void *, cublasLtMatrixLayout_t,
                                                   const cublasLtMatmulAlgo_t *,
                                                   void *, size_t, cudaStream_t));

typedef void (*reg_fn_t)(void **, const char *, char *, const char *, int,
                         uint3 *, uint3 *, dim3 *, dim3 *, int *);
static reg_fn_t real___cudaRegisterFunction = nullptr;

static pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;

static void resolve_symbols(void)
{
#define LOAD(name)   real_##name = (name##_fn)dlsym(RTLD_NEXT, #name)
    LOAD(cudaMalloc);
    LOAD(cudaMallocAsync);
    LOAD(cudaFree);
    LOAD(cudaFreeAsync);
    LOAD(cudaMemcpy);
    LOAD(cudaMemcpyAsync);
    LOAD(cudaMemset);
    LOAD(cudaMemsetAsync);
    LOAD(cudaStreamSynchronize);
    LOAD(cudaDeviceSynchronize);
    LOAD(cudaStreamCreate);
    LOAD(cudaStreamDestroy);
    LOAD(cudaMallocHost);
    LOAD(cudaFreeHost);
    LOAD(cudaEventCreate);
    LOAD(cudaEventDestroy);
    LOAD(cudaEventRecord);
    LOAD(cudaEventSynchronize);
    LOAD(cudaEventElapsedTime);
    LOAD(cudaLaunchKernel);
    LOAD(cuMemAddressReserve);
    LOAD(cuMemAddressFree);
    LOAD(cuMemMap);
    LOAD(cuMemUnmap);
    LOAD(cuMemAlloc_v2);
    LOAD(cuMemFree_v2);
    LOAD(cuLaunchKernel);
    LOAD(cuLaunchKernelEx);
    LOAD(cuModuleGetFunction);
    LOAD(cuFuncGetName);
    LOAD(cuFuncGetParamInfo);
    LOAD(cublasGemmEx);
    LOAD(cublasLtMatmul);
#undef LOAD
    real___cudaRegisterFunction = (reg_fn_t)dlsym(RTLD_NEXT, "__cudaRegisterFunction");

    /* Driver-API pointers exposed to templates. */
    real_cuStreamCreate          = (cuStreamCreate_fn)         dlsym(RTLD_NEXT, "cuStreamCreate");
    real_cuStreamSynchronize     = (cuStreamSynchronize_fn)    dlsym(RTLD_NEXT, "cuStreamSynchronize");
    real_cuMemcpyDtoHAsync_v2    = (cuMemcpyDtoHAsync_v2_fn)   dlsym(RTLD_NEXT, "cuMemcpyDtoHAsync_v2");
    /* Fallback name in some CUDA builds */
    if (!real_cuMemcpyDtoHAsync_v2)
        real_cuMemcpyDtoHAsync_v2 = (cuMemcpyDtoHAsync_v2_fn)  dlsym(RTLD_NEXT, "cuMemcpyDtoHAsync");
}
#define ENSURE_SYMBOLS pthread_once(&g_resolve_once, resolve_symbols)

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
/* Image backing store (F1): one anonymous-mmap region; per-alloc slot  */
/* assigned at record_alloc time. Round N's Phase 2 writes encrypted    */
/* dirty-range bytes into the corresponding slot, in-place — matches    */
/* GCR paper §5: "merge incremental into previous full checkpoint."     */
/* No history kept; the image at any moment IS the latest checkpoint.   */
/* ------------------------------------------------------------------ */
static uint8_t  *g_image_mem  = nullptr;
static uint64_t  g_image_size = 0;
static uint64_t  g_image_used = 0;     /* bump pointer */
static pthread_mutex_t g_image_mu = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Per-round metrics for CSV dump at exit                              */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t round;
    uint64_t p2_bytes;
    double   p2_copy_ms;
    double   p2_enc_ms;
    uint64_t p4_bytes;
    double   p4_copy_ms;
    double   p4_enc_ms;
    double   stall_ms;
} gcr_metric_t;
#define GCR_MAX_METRIC_ROUNDS 1024
static gcr_metric_t  g_metrics[GCR_MAX_METRIC_ROUNDS];
static int           g_n_metrics = 0;
static pthread_mutex_t g_metrics_mu = PTHREAD_MUTEX_INITIALIZER;
/* One-shot guard: dumped explicitly when the final round completes
 * (vLLM v1 SIGKILLs the engine-core worker at shutdown, so the atexit()
 * fallback never runs there). Makes explicit + atexit call idempotent. */
static std::atomic<bool> g_metrics_dumped{false};

static void metrics_record_round(uint64_t round,
                                 uint64_t p2_b, double p2_c, double p2_e,
                                 uint64_t p4_b, double p4_c, double p4_e,
                                 double stall_ms)
{
    pthread_mutex_lock(&g_metrics_mu);
    if (g_n_metrics < GCR_MAX_METRIC_ROUNDS) {
        gcr_metric_t *m = &g_metrics[g_n_metrics++];
        m->round      = round;
        m->p2_bytes   = p2_b;
        m->p2_copy_ms = p2_c;
        m->p2_enc_ms  = p2_e;
        m->p4_bytes   = p4_b;
        m->p4_copy_ms = p4_c;
        m->p4_enc_ms  = p4_e;
        m->stall_ms   = stall_ms;
    }
    pthread_mutex_unlock(&g_metrics_mu);
}

/* atexit handler — fires when the host app terminates (normal exit).
 * Doesn't fire on SIGKILL/segfault, but does on SIGINT/SIGTERM via the
 * default Python interpreter path. Writes CSV to stderr. */
static void metrics_dump_csv(void)
{
    if (g_n_metrics == 0) return;
    if (g_metrics_dumped.exchange(true)) return;  /* already dumped */
    fprintf(stderr, "\n=== gcr per-round metrics (CSV) ===\n");
    fprintf(stderr, "shim,round,p2_bytes,p2_mb,p2_copy_ms,p2_enc_ms,p2_gbps,"
                    "p4_bytes,p4_mb,p4_copy_ms,p4_enc_ms,stall_ms\n");
    double sum_p2_mb = 0, sum_p2_copy = 0, sum_p4_mb = 0, sum_stall = 0;
    for (int i = 0; i < g_n_metrics; i++) {
        gcr_metric_t *m = &g_metrics[i];
        double p2_mb = m->p2_bytes / (1024.0 * 1024.0);
        double p4_mb = m->p4_bytes / (1024.0 * 1024.0);
        double p2_gbps = (m->p2_copy_ms > 0)
            ? (m->p2_bytes / (1024.0*1024.0*1024.0)) / (m->p2_copy_ms / 1000.0)
            : 0.0;
        fprintf(stderr,
                "gcr,%lu,%lu,%.2f,%.1f,%.1f,%.2f,%lu,%.2f,%.1f,%.1f,%.1f\n",
                (unsigned long)m->round,
                (unsigned long)m->p2_bytes, p2_mb,
                m->p2_copy_ms, m->p2_enc_ms, p2_gbps,
                (unsigned long)m->p4_bytes, p4_mb,
                m->p4_copy_ms, m->p4_enc_ms, m->stall_ms);
        sum_p2_mb   += p2_mb;
        sum_p2_copy += m->p2_copy_ms;
        sum_p4_mb   += p4_mb;
        sum_stall   += m->stall_ms;
    }
    fprintf(stderr, "\n=== gcr aggregate (n=%d round(s)) ===\n", g_n_metrics);
    fprintf(stderr, "  avg p2 size  : %.2f MB\n", sum_p2_mb   / g_n_metrics);
    fprintf(stderr, "  avg p2 copy  : %.1f ms\n", sum_p2_copy / g_n_metrics);
    fprintf(stderr, "  avg p4 delta : %.2f MB\n", sum_p4_mb   / g_n_metrics);
    fprintf(stderr, "  avg stall    : %.1f ms\n", sum_stall   / g_n_metrics);
    fflush(stderr);
}

/* ------------------------------------------------------------------ */
/* Alloc table (for pessimistic fallback + cuBLAS alloc lookup)         */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t va;
    uint64_t size;
    uint64_t img_offset;        /* slot offset in g_image_mem; UINT64_MAX if none */
    uint8_t  alive;
    uint8_t  pending_drop;      /* set by retire_alloc during ckpt; swept at end */
} gcr_alloc_t;

static gcr_alloc_t      g_allocs[GCR_MAX_ALLOCS];
static uint32_t         g_num_allocs = 0;
static pthread_rwlock_t g_alloc_lock = PTHREAD_RWLOCK_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Round state (F4) + deferred-free coordination (F3)                   */
/* ------------------------------------------------------------------ */
static std::atomic<uint64_t> g_round_id        {0};
static std::atomic<int>      g_ckpt_in_flight  {0};
static pthread_mutex_t       g_ckpt_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t        g_ckpt_cv = PTHREAD_COND_INITIALIZER;

static int find_alloc_idx_unlocked(uint64_t addr)
{
    uint32_t n = g_num_allocs;
    for (uint32_t i = 0; i < n; i++) {
        if (!g_allocs[i].alive) continue;
        uint64_t base = g_allocs[i].va;
        if (addr >= base && addr < base + g_allocs[i].size)
            return (int)i;
    }
    return -1;
}

static int find_alloc_idx(uint64_t addr)
{
    pthread_rwlock_rdlock(&g_alloc_lock);
    int r = find_alloc_idx_unlocked(addr);
    pthread_rwlock_unlock(&g_alloc_lock);
    return r;
}

/* Reserve an image slot for a new alloc. Bump-pointer; no freelist (PyTorch's
 * caching allocator means real frees are rare on the hot path). Returns
 * UINT64_MAX if image is exhausted — alloc still recorded but won't be
 * checkpointable (logged once). */
static uint64_t image_reserve_slot(uint64_t size)
{
    if (!g_image_mem || size == 0) return UINT64_MAX;
    pthread_mutex_lock(&g_image_mu);
    if (g_image_used + size > g_image_size) {
        pthread_mutex_unlock(&g_image_mu);
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[gcr] image backing exhausted (%lu used / %lu total)"
                            " — increase CKPT_IMAGE_GB\n",
                    (unsigned long)g_image_used, (unsigned long)g_image_size);
        }
        return UINT64_MAX;
    }
    uint64_t off = g_image_used;
    g_image_used += size;
    pthread_mutex_unlock(&g_image_mu);
    return off;
}

static void record_alloc(void *ptr, size_t size)
{
    if (!ptr || size == 0) return;
    uint64_t slot = image_reserve_slot((uint64_t)size);
    pthread_rwlock_wrlock(&g_alloc_lock);
    if (g_num_allocs < GCR_MAX_ALLOCS) {
        uint32_t idx = g_num_allocs++;
        g_allocs[idx].va           = (uint64_t)(uintptr_t)ptr;
        g_allocs[idx].size         = (uint64_t)size;
        g_allocs[idx].img_offset   = slot;
        g_allocs[idx].alive        = 1;
        g_allocs[idx].pending_drop = 0;
    }
    pthread_rwlock_unlock(&g_alloc_lock);
}

/* Retire — but if a checkpoint is in flight, defer the actual teardown so
 * Phase 2/4 can finish reading from the GPU buffer + image slot. The slot
 * itself is intentionally NOT reclaimed in v1 (no freelist). */
static void retire_alloc(void *ptr)
{
    if (!ptr) return;
    uint64_t addr = (uint64_t)(uintptr_t)ptr;

    if (g_ckpt_in_flight.load(std::memory_order_acquire)) {
        pthread_rwlock_wrlock(&g_alloc_lock);
        int idx = find_alloc_idx_unlocked(addr);
        if (idx >= 0) g_allocs[idx].pending_drop = 1;
        pthread_rwlock_unlock(&g_alloc_lock);
        return;
    }

    pthread_rwlock_wrlock(&g_alloc_lock);
    int idx = find_alloc_idx_unlocked(addr);
    if (idx >= 0) g_allocs[idx].alive = 0;
    pthread_rwlock_unlock(&g_alloc_lock);
}

/* Called once Phase 4 completes. Walks the alloc table, retiring anything
 * marked pending_drop (the app called cudaFree mid-checkpoint). */
static void sweep_pending_drops(void)
{
    uint32_t swept = 0;
    pthread_rwlock_wrlock(&g_alloc_lock);
    for (uint32_t i = 0; i < g_num_allocs; i++) {
        if (g_allocs[i].pending_drop && g_allocs[i].alive) {
            g_allocs[i].alive = 0;
            g_allocs[i].pending_drop = 0;
            swept++;
        }
    }
    pthread_rwlock_unlock(&g_alloc_lock);
    if (swept) {
        fprintf(stderr, "[gcr] sweep_pending_drops: retired %u allocs\n", swept);
    }
}

/* Sum of (size) over all live allocations and the count. Used to compare
 * against the dirty-set total — if dirty << tracked, the workload really
 * only wrote a small subset of the allocated memory; if tracked << real
 * GPU footprint, we're missing alloc hooks somewhere. */
static void tracked_total(uint64_t *bytes_out, uint32_t *count_out)
{
    uint64_t b = 0;
    uint32_t c = 0;
    pthread_rwlock_rdlock(&g_alloc_lock);
    uint32_t n = g_num_allocs;
    for (uint32_t i = 0; i < n; i++) {
        if (!g_allocs[i].alive) continue;
        b += g_allocs[i].size;
        c++;
    }
    pthread_rwlock_unlock(&g_alloc_lock);
    if (bytes_out) *bytes_out = b;
    if (count_out) *count_out = c;
}

/* Mark the entire alloc containing addr as dirty (pessimistic). */
static inline void mark_alloc_dirty(uint64_t addr)
{
    pthread_rwlock_rdlock(&g_alloc_lock);
    int idx = find_alloc_idx_unlocked(addr);
    if (idx >= 0) {
        uint64_t s = g_allocs[idx].va;
        uint64_t e = s + g_allocs[idx].size;
        pthread_rwlock_unlock(&g_alloc_lock);
        add_and_merge_dirty_address((void *)s, (void *)e);
        return;
    }
    pthread_rwlock_unlock(&g_alloc_lock);
}

/* Mark byte range [va, va+size). Clipped to the containing alloc if any,
 * to avoid leaking past allocation boundaries on bad inputs. */
static inline void mark_range(uint64_t va, uint64_t size)
{
    if (size == 0) return;
    pthread_rwlock_rdlock(&g_alloc_lock);
    int idx = find_alloc_idx_unlocked(va);
    if (idx >= 0) {
        uint64_t s = g_allocs[idx].va;
        uint64_t e = s + g_allocs[idx].size;
        uint64_t lo = (va > s) ? va : s;
        uint64_t hi = (va + size < e) ? va + size : e;
        pthread_rwlock_unlock(&g_alloc_lock);
        if (hi > lo) add_and_merge_dirty_address((void *)lo, (void *)hi);
        return;
    }
    pthread_rwlock_unlock(&g_alloc_lock);
    /* Not in any known alloc — still insert raw; harmless. */
    add_and_merge_dirty_address((void *)va, (void *)(va + size));
}

/* Diagnostic flags + counters used by the cache helpers below.
 * (The bulk of the counters live in the Counters block further down,
 * but these two are referenced here so they must be in scope.) */
static int g_dump_names = 0;    /* GCR_DUMP_NAMES=1: print every unique name installed */
static std::atomic<uint64_t> g_n_lazy_resolve {0};

/* ------------------------------------------------------------------ */
/* CUfunction -> mangled name cache (for cuLaunchKernel path)          */
/* ------------------------------------------------------------------ */
typedef struct {
    std::atomic<uintptr_t>    f;                              /* key; 0 = empty */
    char                      name[GCR_MAX_KNAME_LEN];
    gcr_template_fn_t         tmpl;
    std::atomic<int>          nparams;                        /* -1 until resolved */
    std::atomic<uint8_t>      resolved;                       /* 0/1 */
} gcr_kernel_ent_t;

static gcr_kernel_ent_t g_kernel_cache[GCR_MAX_KERNELS];

static inline uint32_t hash_ptr(uintptr_t p)
{
    return (uint32_t)((p * 2654435769ULL) & (GCR_MAX_KERNELS - 1));
}

/* Install a (CUfunction, name) mapping; looks up and caches template. */
static gcr_kernel_ent_t *register_kernel_name(CUfunction f, const char *name)
{
    if (!f || !name) return nullptr;
    uintptr_t key = (uintptr_t)f;
    uint32_t h = hash_ptr(key);
    for (int probe = 0; probe < 64; probe++) {
        uint32_t idx = (h + probe) & (GCR_MAX_KERNELS - 1);
        uintptr_t cur = g_kernel_cache[idx].f.load(std::memory_order_acquire);
        if (cur == key) return &g_kernel_cache[idx];
        if (cur == 0) {
            uintptr_t expect = 0;
            if (g_kernel_cache[idx].f.compare_exchange_strong(
                    expect, key, std::memory_order_release, std::memory_order_acquire)) {
                std::strncpy(g_kernel_cache[idx].name, name, GCR_MAX_KNAME_LEN - 1);
                g_kernel_cache[idx].name[GCR_MAX_KNAME_LEN - 1] = '\0';
                g_kernel_cache[idx].tmpl = gcr_lookup_template(name);
                g_kernel_cache[idx].nparams.store(-1, std::memory_order_relaxed);
                g_kernel_cache[idx].resolved.store(1, std::memory_order_release);
                if (g_dump_names) {
                    fprintf(stderr, "[gcr] kname[install]: tmpl=%s %s\n",
                            g_kernel_cache[idx].tmpl ? "HIT " : "miss",
                            g_kernel_cache[idx].name);
                }
                return &g_kernel_cache[idx];
            }
            /* Lost race; retry */
        }
    }
    return nullptr;
}

static gcr_kernel_ent_t *find_kernel_ent(CUfunction f)
{
    if (!f) return nullptr;
    uintptr_t key = (uintptr_t)f;
    uint32_t h = hash_ptr(key);
    for (int probe = 0; probe < 64; probe++) {
        uint32_t idx = (h + probe) & (GCR_MAX_KERNELS - 1);
        uintptr_t cur = g_kernel_cache[idx].f.load(std::memory_order_acquire);
        if (cur == key) return &g_kernel_cache[idx];
        if (cur == 0) return nullptr;
    }
    return nullptr;
}

/* Lazy resolution: if we don't have a name for this CUfunction yet, try
 * cuFuncGetName. Then install the mapping and return the entry. */
static gcr_kernel_ent_t *resolve_cufunction(CUfunction f)
{
    gcr_kernel_ent_t *e = find_kernel_ent(f);
    if (e) return e;
    if (!real_cuFuncGetName) return nullptr;
    const char *name = nullptr;
    if (real_cuFuncGetName(&name, f) == CUDA_SUCCESS && name) {
        g_n_lazy_resolve.fetch_add(1, std::memory_order_relaxed);
        return register_kernel_name(f, name);
    }
    return nullptr;
}

/* Cached nparams lookup via cuFuncGetParamInfo, for pessimistic fallback. */
static int get_nparams_cached(gcr_kernel_ent_t *e, CUfunction f)
{
    if (!e || !real_cuFuncGetParamInfo) return -1;
    int cur = e->nparams.load(std::memory_order_relaxed);
    if (cur >= 0) return cur;
    int n = 0;
    size_t off, sz;
    while (real_cuFuncGetParamInfo(f, n, &off, &sz) == CUDA_SUCCESS) {
        n++;
        if (n > GCR_MAX_KERNEL_PARAMS) break;
    }
    e->nparams.store(n, std::memory_order_relaxed);
    return n;
}

/* ------------------------------------------------------------------ */
/* host_stub -> mangled name cache (for cudaLaunchKernel path)         */
/* ------------------------------------------------------------------ */
typedef struct {
    std::atomic<uintptr_t>    stub;
    char                      name[GCR_MAX_KNAME_LEN];
    gcr_template_fn_t         tmpl;
} gcr_hostfn_ent_t;

static gcr_hostfn_ent_t g_hostfn_cache[GCR_MAX_HOSTFNS];

static void register_hostfn_name(const void *stub, const char *name)
{
    if (!stub || !name) return;
    uintptr_t key = (uintptr_t)stub;
    uint32_t h = (uint32_t)((key * 2654435769ULL) & (GCR_MAX_HOSTFNS - 1));
    for (int probe = 0; probe < 64; probe++) {
        uint32_t idx = (h + probe) & (GCR_MAX_HOSTFNS - 1);
        uintptr_t cur = g_hostfn_cache[idx].stub.load(std::memory_order_acquire);
        if (cur == key) return;
        if (cur == 0) {
            uintptr_t expect = 0;
            if (g_hostfn_cache[idx].stub.compare_exchange_strong(
                    expect, key, std::memory_order_release, std::memory_order_acquire)) {
                std::strncpy(g_hostfn_cache[idx].name, name, GCR_MAX_KNAME_LEN - 1);
                g_hostfn_cache[idx].name[GCR_MAX_KNAME_LEN - 1] = '\0';
                g_hostfn_cache[idx].tmpl = gcr_lookup_template(name);
                if (g_dump_names) {
                    fprintf(stderr, "[gcr] hname[install]: tmpl=%s %s\n",
                            g_hostfn_cache[idx].tmpl ? "HIT " : "miss",
                            g_hostfn_cache[idx].name);
                }
                return;
            }
        }
    }
}

static gcr_hostfn_ent_t *find_hostfn_ent(const void *stub)
{
    if (!stub) return nullptr;
    uintptr_t key = (uintptr_t)stub;
    uint32_t h = (uint32_t)((key * 2654435769ULL) & (GCR_MAX_HOSTFNS - 1));
    for (int probe = 0; probe < 64; probe++) {
        uint32_t idx = (h + probe) & (GCR_MAX_HOSTFNS - 1);
        uintptr_t cur = g_hostfn_cache[idx].stub.load(std::memory_order_acquire);
        if (cur == key) return &g_hostfn_cache[idx];
        if (cur == 0) return nullptr;
    }
    return nullptr;
}

/* ------------------------------------------------------------------ */
/* Pessimistic pointer scan (fallback when no template is registered)  */
/*                                                                     */
/* For each kernel-arg slot, if the value looks like a VA pointing into */
/* a known alloc, mark that entire alloc dirty. This matches what GCR   */
/* does for closed-source kernels lacking dirty-template coverage.     */
/* ------------------------------------------------------------------ */
static pid_t g_self_pid = 0;
static __thread uintptr_t tls_safe_page = 0;

static inline int slot_looks_safe(const void *p)
{
    uint64_t v = (uint64_t)p;
    if (v == 0) return 0;
    if (v & 0x7) return 0;
    if (v < 0x0000400000000000ULL) return 0;
    if (v > 0x00007fffffffffffULL) return 0;
    return 1;
}

static int safe_read_qword(const void *src, uint64_t *out)
{
    uintptr_t pg = (uintptr_t)src & ~(uintptr_t)0xFFFUL;
    if (pg == tls_safe_page) {
        *out = *(const uint64_t *)src;
        return 1;
    }
    struct iovec local  = { .iov_base = out,         .iov_len = 8 };
    struct iovec remote = { .iov_base = (void *)src, .iov_len = 8 };
    if (process_vm_readv(g_self_pid, &local, 1, &remote, 1, 0) == 8) {
        tls_safe_page = pg;
        return 1;
    }
    return 0;
}

static void pessimistic_scan_exact(void **params, int nparams)
{
    if (!params || nparams <= 0) return;
    for (int i = 0; i < nparams; i++) {
        void *slot = params[i];
        if (!slot_looks_safe(slot)) continue;
        uint64_t cand = *(uint64_t *)slot;
        mark_alloc_dirty(cand);
    }
}

static void pessimistic_scan_safe(void **params, int max_depth)
{
    if (!params) return;
    uint64_t probe;
    if (!safe_read_qword(params, &probe)) return;
    for (int i = 0; i < max_depth; i++) {
        void *slot = params[i];
        if (!slot_looks_safe(slot)) break;
        uint64_t cand;
        if (!safe_read_qword(slot, &cand)) break;
        mark_alloc_dirty(cand);
    }
}

/* ------------------------------------------------------------------ */
/* Counters                                                            */
/* ------------------------------------------------------------------ */
static std::atomic<uint64_t> g_n_memcpy         {0};
static std::atomic<uint64_t> g_n_memset         {0};
static std::atomic<uint64_t> g_n_cublas_gemm    {0};
static std::atomic<uint64_t> g_n_cublas_lt      {0};
static std::atomic<uint64_t> g_n_kernel_total   {0};
static std::atomic<uint64_t> g_n_template_hits  {0};
static std::atomic<uint64_t> g_n_template_miss  {0};
static std::atomic<uint64_t> g_n_fallback_pessi {0};

/* Diagnostic counters — which name-resolution path saw any traffic.
 * (g_n_lazy_resolve is declared earlier so the cache helpers can see it.) */
static std::atomic<uint64_t> g_n_cumod_getfunc {0};
static std::atomic<uint64_t> g_n_cudareg_func  {0};

static int g_scan_depth = 16;   /* used only by cudaLaunchKernel fallback */

/* ------------------------------------------------------------------ */
/* Alloc hooks                                                         */
/* ------------------------------------------------------------------ */
extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMalloc(devPtr, size);
    if (err == cudaSuccess && devPtr) record_alloc(*devPtr, size);
    return err;
}

extern "C" cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t s)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMallocAsync(devPtr, size, s);
    if (err == cudaSuccess && devPtr) record_alloc(*devPtr, size);
    return err;
}

/* Block until the current checkpoint round finishes (or return immediately
 * if none is in flight). Called from every free hook so the GPU memory
 * isn't actually returned while Phase 2/4 is reading it.
 *
 * In practice rare on the vLLM hot path — PyTorch's caching allocator
 * intercepts most frees long before they reach our hook. Mostly fires
 * when a request finishes mid-round or vLLM rebalances its block pool. */
static void wait_until_ckpt_idle(void)
{
    if (!g_ckpt_in_flight.load(std::memory_order_acquire)) return;
    pthread_mutex_lock(&g_ckpt_mu);
    while (g_ckpt_in_flight.load(std::memory_order_acquire))
        pthread_cond_wait(&g_ckpt_cv, &g_ckpt_mu);
    pthread_mutex_unlock(&g_ckpt_mu);
}

extern "C" cudaError_t cudaFree(void *devPtr)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc(devPtr);
    return real_cudaFree(devPtr);
}

extern "C" cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t s)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc(devPtr);
    return real_cudaFreeAsync(devPtr, s);
}

/* Driver-API allocators. vLLM v1 paged KV uses cuMemAddressReserve +
 * cuMemMap directly (NOT cudaMalloc), so without these hooks the KV slabs
 * are invisible to the alloc table and every kernel touching them falls
 * through find_alloc_idx_unlocked == -1 (silent miss). */
extern "C" CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    ENSURE_SYMBOLS;
    CUresult r = real_cuMemAlloc_v2(dptr, bytesize);
    if (r == CUDA_SUCCESS && dptr) record_alloc((void *)(uintptr_t)*dptr, bytesize);
    return r;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc((void *)(uintptr_t)dptr);
    return real_cuMemFree_v2(dptr);
}

/* cuMemMap is the moment a VA range becomes backed by physical memory.
 * Treat (ptr, size) as an allocation record. cuMemAddressReserve alone
 * doesn't make memory writable, so we don't track reservations. */
extern "C" CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                             CUmemGenericAllocationHandle handle,
                             unsigned long long flags)
{
    ENSURE_SYMBOLS;
    CUresult r = real_cuMemMap(ptr, size, offset, handle, flags);
    if (r == CUDA_SUCCESS) record_alloc((void *)(uintptr_t)ptr, size);
    return r;
}

extern "C" CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc((void *)(uintptr_t)ptr);
    return real_cuMemUnmap(ptr, size);
}

/* ------------------------------------------------------------------ */
/* Memcpy / memset hooks — insert byte ranges                          */
/* ------------------------------------------------------------------ */
extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  enum cudaMemcpyKind kind)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemcpy(dst, src, count, kind);
    if (err == cudaSuccess && count > 0 &&
        (kind == cudaMemcpyHostToDevice ||
         kind == cudaMemcpyDeviceToDevice ||
         kind == cudaMemcpyDefault)) {
        mark_range((uint64_t)(uintptr_t)dst, count);
        g_n_memcpy.fetch_add(1, std::memory_order_relaxed);
    }
    return err;
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                       enum cudaMemcpyKind kind, cudaStream_t s)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemcpyAsync(dst, src, count, kind, s);
    if (err == cudaSuccess && count > 0 &&
        (kind == cudaMemcpyHostToDevice ||
         kind == cudaMemcpyDeviceToDevice ||
         kind == cudaMemcpyDefault)) {
        mark_range((uint64_t)(uintptr_t)dst, count);
        g_n_memcpy.fetch_add(1, std::memory_order_relaxed);
    }
    return err;
}

extern "C" cudaError_t cudaMemset(void *dst, int v, size_t count)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemset(dst, v, count);
    if (err == cudaSuccess && count > 0) {
        mark_range((uint64_t)(uintptr_t)dst, count);
        g_n_memset.fetch_add(1, std::memory_order_relaxed);
    }
    return err;
}

extern "C" cudaError_t cudaMemsetAsync(void *dst, int v, size_t count, cudaStream_t s)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemsetAsync(dst, v, count, s);
    if (err == cudaSuccess && count > 0) {
        mark_range((uint64_t)(uintptr_t)dst, count);
        g_n_memset.fetch_add(1, std::memory_order_relaxed);
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* cuBLAS — mark output tile as byte range                             */
/* ------------------------------------------------------------------ */
static int cuda_dtype_bytes(cudaDataType_t t)
{
    switch (t) {
        case CUDA_R_8I:    case CUDA_R_8U:   case CUDA_R_8F_E4M3: case CUDA_R_8F_E5M2: return 1;
        case CUDA_R_16F:   case CUDA_R_16BF: case CUDA_R_16I:     case CUDA_R_16U:     return 2;
        case CUDA_R_32F:   case CUDA_R_32I:  case CUDA_R_32U:                          return 4;
        case CUDA_R_64F:   case CUDA_R_64I:  case CUDA_R_64U:                          return 8;
        case CUDA_C_16F:   case CUDA_C_16BF:                                           return 4;
        case CUDA_C_32F:                                                                return 8;
        case CUDA_C_64F:                                                                return 16;
        default:                                                                        return 0;
    }
}

extern "C" cublasStatus_t
cublasGemmEx(cublasHandle_t h,
             cublasOperation_t ta, cublasOperation_t tb,
             int m, int n, int k,
             const void *alpha,
             const void *A, cudaDataType_t At, int lda,
             const void *B, cudaDataType_t Bt, int ldb,
             const void *beta,
             void *C, cudaDataType_t Ct, int ldc,
             cublasComputeType_t cty, cublasGemmAlgo_t algo)
{
    ENSURE_SYMBOLS;
    cublasStatus_t s = real_cublasGemmEx(h, ta, tb, m, n, k, alpha,
                                         A, At, lda, B, Bt, ldb,
                                         beta, C, Ct, ldc, cty, algo);
    if (s == CUBLAS_STATUS_SUCCESS && C) {
        int bpe = cuda_dtype_bytes(Ct);
        uint64_t tile_bytes = (bpe > 0) ? ((uint64_t)n * (uint64_t)ldc * (uint64_t)bpe) : 0;
        if (tile_bytes > 0) {
            mark_range((uint64_t)(uintptr_t)C, tile_bytes);
        } else {
            mark_alloc_dirty((uint64_t)(uintptr_t)C);
        }
        g_n_cublas_gemm.fetch_add(1, std::memory_order_relaxed);
    }
    return s;
}

extern "C" cublasStatus_t
cublasLtMatmul(cublasLtHandle_t lh, cublasLtMatmulDesc_t desc,
               const void *alpha,
               const void *A, cublasLtMatrixLayout_t Ad,
               const void *B, cublasLtMatrixLayout_t Bd,
               const void *beta,
               const void *C, cublasLtMatrixLayout_t Cd,
               void *D, cublasLtMatrixLayout_t Dd,
               const cublasLtMatmulAlgo_t *algo,
               void *ws, size_t ws_bytes, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    cublasStatus_t s = real_cublasLtMatmul(lh, desc, alpha, A, Ad, B, Bd, beta,
                                           C, Cd, D, Dd, algo, ws, ws_bytes, stream);
    if (s == CUBLAS_STATUS_SUCCESS && D) {
        /* Descriptor parsing is non-trivial; mark the whole containing alloc
         * as a conservative byte range. Tighter output-tile extraction TBD. */
        mark_alloc_dirty((uint64_t)(uintptr_t)D);
        g_n_cublas_lt.fetch_add(1, std::memory_order_relaxed);
    }
    return s;
}

/* ------------------------------------------------------------------ */
/* Kernel-name resolution hooks                                        */
/* ------------------------------------------------------------------ */
extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule m, const char *name)
{
    ENSURE_SYMBOLS;
    g_n_cumod_getfunc.fetch_add(1, std::memory_order_relaxed);
    CUresult r = real_cuModuleGetFunction(hfunc, m, name);
    if (r == CUDA_SUCCESS && hfunc && *hfunc && name) {
        register_kernel_name(*hfunc, name);
    }
    return r;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                                       char *deviceFun, const char *deviceName,
                                       int thread_limit, uint3 *tid, uint3 *bid,
                                       dim3 *bDim, dim3 *gDim, int *wSize)
{
    ENSURE_SYMBOLS;
    g_n_cudareg_func.fetch_add(1, std::memory_order_relaxed);
    if (real___cudaRegisterFunction) {
        real___cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                                    deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    }
    /* deviceFun is typically the Itanium-mangled name (same string GCR's
     * symbol_exec keys on). deviceName is sometimes unmangled. */
    if (hostFun && deviceFun) {
        register_hostfn_name((const void *)hostFun, deviceFun);
    }
}

/* ------------------------------------------------------------------ */
/* Kernel launch hooks — the heart of GCR emulation                    */
/* ------------------------------------------------------------------ */
static void shadow_exec_driver(CUfunction f,
                               unsigned gdx, unsigned gdy, unsigned gdz,
                               unsigned bdx, unsigned bdy, unsigned bdz,
                               unsigned smem, CUstream stream,
                               void **params, void **extra)
{
    g_n_kernel_total.fetch_add(1, std::memory_order_relaxed);
    gcr_kernel_ent_t *e = resolve_cufunction(f);
    if (e && e->tmpl) {
        /* GCR shadow execution: run template on CPU right after real launch.
         * Real kernel is already in flight on the GPU; template runs in
         * parallel on this thread. */
        e->tmpl(f, gdx, gdy, gdz, bdx, bdy, bdz, smem, stream, params, extra);
        g_n_template_hits.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    /* Fallback: pessimistic pointer scan */
    g_n_template_miss.fetch_add(1, std::memory_order_relaxed);
    int n = e ? get_nparams_cached(e, f) : -1;
    if (n > 0) pessimistic_scan_exact(params, n);
    else       pessimistic_scan_safe(params, g_scan_depth);
    g_n_fallback_pessi.fetch_add(1, std::memory_order_relaxed);
}

extern "C" CUresult cuLaunchKernel(CUfunction f,
                                   unsigned gdx, unsigned gdy, unsigned gdz,
                                   unsigned bdx, unsigned bdy, unsigned bdz,
                                   unsigned smem, CUstream stream,
                                   void **params, void **extra)
{
    ENSURE_SYMBOLS;
    CUresult r = real_cuLaunchKernel(f, gdx, gdy, gdz, bdx, bdy, bdz,
                                     smem, stream, params, extra);
    if (r == CUDA_SUCCESS) {
        shadow_exec_driver(f, gdx, gdy, gdz, bdx, bdy, bdz, smem, stream, params, extra);
    }
    return r;
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig *cfg, CUfunction f,
                                     void **params, void **extra)
{
    ENSURE_SYMBOLS;
    CUresult r = real_cuLaunchKernelEx(cfg, f, params, extra);
    if (r == CUDA_SUCCESS && cfg) {
        shadow_exec_driver(f,
                           cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
                           cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
                           cfg->sharedMemBytes, cfg->hStream, params, extra);
    }
    return r;
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block,
                                        void **args, size_t smem, cudaStream_t s)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaLaunchKernel(func, grid, block, args, smem, s);
    if (err == cudaSuccess) {
        g_n_kernel_total.fetch_add(1, std::memory_order_relaxed);
        gcr_hostfn_ent_t *e = find_hostfn_ent(func);
        if (e && e->tmpl) {
            /* Templates take a CUfunction handle as their first arg but only
             * forward it to real_cu* calls inside — for the runtime-API path
             * we don't have one. The kernels that actually use `f` internally
             * don't; they all just read params[i]. We pass nullptr, which is
             * fine for every template in dirty_templates_kernels.cpp. */
            e->tmpl(nullptr,
                    grid.x, grid.y, grid.z,
                    block.x, block.y, block.z,
                    (unsigned)smem, s, args, nullptr);
            g_n_template_hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_n_template_miss.fetch_add(1, std::memory_order_relaxed);
            pessimistic_scan_safe(args, g_scan_depth);
            g_n_fallback_pessi.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return err;
}

/* ================================================================== */
/* Checkpoint — stop-and-copy incremental over byte ranges             */
/* ================================================================== */
static volatile int g_ckpt_requested = 0;
static volatile int g_ckpt_phase     = 0;   /* 0 idle, 1 in-progress  */
static int          g_ckpt_delay_s   = 0;   /* seconds before round 0 */
static int          g_ckpt_interval_ms = 0; /* ms between consecutive rounds (0 = single-shot) */
static int          g_ckpt_rounds_max  = 1; /* stop after this many rounds */
static int                 g_wait_signal    = 0; /* 1 = defer round 0 until SIGUSR1 */
static std::atomic<int>    g_start_signaled {0}; /* set by sigusr1_start when SIGUSR1 received */
static double       g_first_sync_ms  = 0;
static volatile int g_first_sync_seen= 0;

static uint8_t  g_k3_key[CKPT_CRYPTO_KEY_SIZE];

/* F2: two pinned staging slots (ping-pong), allocated lazily on first
 * checkpoint (NOT in the constructor — cudaMallocHost in a preload-
 * constructor breaks vLLM's fork of EngineCore). Sized via CKPT_STAGING_MB
 * (default 256 MB) — each slot is g_staging_size bytes. The two slots
 * back the producer (cudaMemcpyAsync D2H) ↔ consumer (encrypt + memcpy
 * to image_mem) pipeline: while the producer fills one slot the consumer
 * drains the other. Mirrors phos_emul's run_pingpong. */
static uint8_t *g_staging[2]      = { nullptr, nullptr };
static uint64_t g_staging_size    = 0;
static uint64_t g_staging_want    = 256ULL * 1024 * 1024;  /* set in gcr_init from env */
static int      g_staging_pinned[2] = { 0, 0 };

/* Reusable encrypt scratch (ciphertext sink before memcpy into image
 * slot) and per-page metadata array — one pair per staging slot. */
static uint8_t          *g_enc_buf[2]    = { nullptr, nullptr };
static ckpt_page_meta_t *g_enc_meta[2]   = { nullptr, nullptr };
static uint32_t          g_enc_max_pg    = 0;

static void staging_init(uint64_t want_bytes)
{
    if (g_staging[0]) return;
    for (int s = 0; s < 2; s++) {
        void *p = nullptr;
        if (real_cudaMallocHost && real_cudaMallocHost(&p, want_bytes) == cudaSuccess) {
            g_staging[s] = (uint8_t *)p;
            g_staging_pinned[s] = 1;
        } else {
            g_staging[s] = (uint8_t *)malloc(want_bytes);
            g_staging_pinned[s] = 0;
            fprintf(stderr, "[gcr] staging[%d]: pinned alloc failed, using pageable\n", s);
        }
        g_enc_buf[s]  = (uint8_t *)malloc(want_bytes);
    }
    g_staging_size = want_bytes;
    g_enc_max_pg   = (want_bytes + CKPT_CRYPTO_PAGE_SIZE - 1) / CKPT_CRYPTO_PAGE_SIZE;
    g_enc_meta[0]  = (ckpt_page_meta_t *)malloc(g_enc_max_pg * sizeof(ckpt_page_meta_t));
    g_enc_meta[1]  = (ckpt_page_meta_t *)malloc(g_enc_max_pg * sizeof(ckpt_page_meta_t));
    fprintf(stderr, "[gcr] staging: 2x %.0f MB %s/%s, enc_buf 2x %.0f MB\n",
            want_bytes / (1024.0 * 1024.0),
            g_staging_pinned[0] ? "pinned" : "pageable",
            g_staging_pinned[1] ? "pinned" : "pageable",
            want_bytes / (1024.0 * 1024.0));
}

/* Pre-resolved sub-chunk: one cudaMemcpyAsync source + its scatter target.
 * Many of these get packed back-to-back into one staging slot so the slot
 * can be encrypted as a single multi-MB AES-GCM call (one EVP_CIPHER_CTX,
 * 64-thread parallelism) instead of one tiny encrypt per range. */
typedef struct {
    uint64_t va;       /* source GPU VA */
    uint64_t size;     /* sub-chunk size (≤ g_staging_size) */
    uint64_t img_abs;  /* absolute byte offset in g_image_mem (natural pos) */
    uint64_t pack_off; /* offset within staging[idx] where this sub-chunk lives */
} gcr_chunk_t;

/* A packed slot = N gcr_chunk_t entries that fit in one g_staging_size
 * region. The producer issues N cudaMemcpyAsync (one per sub-chunk) into
 * staging[idx] at pack_off; the consumer encrypts the whole packed region
 * once and scatters each sub-chunk back to its natural img_abs. */
typedef struct {
    uint32_t first;       /* index of first gcr_chunk_t in slot */
    uint32_t count;       /* number of sub-chunks in slot */
    uint64_t total_bytes; /* sum of sub-chunk sizes (= bytes encrypted) */
} gcr_slot_t;

/* Walk ranges, resolve each to its alloc/image slot, emit sub-chunks AND  *
 * pack them into staging-sized slots. The slot list drives the per-slot   *
 * AES-GCM call so encryption amortizes the EVP setup + uses 64-thread     *
 * parallelism instead of running 1-thread per tiny range.                 *
 *                                                                          *
 * skipped_out counts ranges with no slot or no containing alloc.           */
static gcr_chunk_t *build_chunks_gcr(const gcr_range_t *ranges, uint32_t n,
                                      uint64_t staging_size,
                                      uint32_t *n_chunks_out,
                                      gcr_slot_t **slots_out,
                                      uint32_t *n_slots_out,
                                      uint64_t *skipped_out)
{
    *n_chunks_out = 0;
    *n_slots_out  = 0;
    *slots_out    = nullptr;
    if (skipped_out) *skipped_out = 0;
    if (n == 0) return nullptr;

    /* Worst case: every range splits into ⌈size/staging⌉ chunks.
     * Upper bound the alloc count with a single pass under the lock. */
    pthread_rwlock_rdlock(&g_alloc_lock);

    uint32_t cap = n;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t sz = ranges[i].end - ranges[i].start;
        if (sz > staging_size)
            cap += (uint32_t)((sz + staging_size - 1) / staging_size);
    }
    gcr_chunk_t *chunks = (gcr_chunk_t *)malloc(cap * sizeof(gcr_chunk_t));
    if (!chunks) { pthread_rwlock_unlock(&g_alloc_lock); return nullptr; }

    /* Slot cap: at worst one slot per range plus one for the tail. */
    uint32_t slot_cap = n + 1;
    gcr_slot_t *slots = (gcr_slot_t *)malloc(slot_cap * sizeof(gcr_slot_t));
    if (!slots) { free(chunks); pthread_rwlock_unlock(&g_alloc_lock); return nullptr; }

    uint32_t ci = 0;
    uint32_t si = 0;
    uint64_t skipped = 0;

    /* Initialize current slot. */
    slots[si].first       = 0;
    slots[si].count       = 0;
    slots[si].total_bytes = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint64_t va = ranges[i].start;
        uint64_t sz = ranges[i].end - ranges[i].start;
        if (sz == 0) continue;

        int idx = find_alloc_idx_unlocked(va);
        if (idx < 0) { skipped++; continue; }
        uint64_t img_off  = g_allocs[idx].img_offset;
        uint64_t alloc_va = g_allocs[idx].va;
        uint64_t alloc_sz = g_allocs[idx].size;
        if (img_off == UINT64_MAX) { skipped++; continue; }

        uint64_t off_in_alloc = va - alloc_va;
        if (off_in_alloc + sz > alloc_sz) sz = alloc_sz - off_in_alloc;
        if (sz == 0) continue;

        uint64_t copied = 0;
        while (copied < sz) {
            uint64_t chunk = sz - copied;
            if (chunk > staging_size) chunk = staging_size;

            /* If this sub-chunk would overflow the current slot, close
             * the slot and start a new one. Empty slots never close
             * (they accept any size up to staging_size). */
            if (slots[si].count > 0 &&
                slots[si].total_bytes + chunk > staging_size) {
                si++;
                slots[si].first       = ci;
                slots[si].count       = 0;
                slots[si].total_bytes = 0;
            }

            chunks[ci].va       = va + copied;
            chunks[ci].size     = chunk;
            chunks[ci].img_abs  = img_off + off_in_alloc + copied;
            chunks[ci].pack_off = slots[si].total_bytes;

            slots[si].count++;
            slots[si].total_bytes += chunk;
            ci++;
            copied += chunk;
        }
    }

    pthread_rwlock_unlock(&g_alloc_lock);

    /* Drop trailing empty slot if loop never pushed into it. */
    uint32_t n_slots = (slots[si].count > 0) ? (si + 1) : si;

    *n_chunks_out = ci;
    *slots_out    = slots;
    *n_slots_out  = n_slots;
    if (skipped_out) *skipped_out = skipped;
    return chunks;
}

/* Producer-consumer pipeline with PACKED slots.                           *
 *                                                                          *
 * Producer (this thread): for each slot's sub-chunks, issue one             *
 *   cudaMemcpyAsync per sub-chunk into staging[idx] at pack_off. When the   *
 *   slot is fully launched, record end-event and hand off to consumer.     *
 * Consumer thread: one AES-GCM encrypt over the entire packed slot         *
 *   (np ≈ slot_total / 2 MB pages, 64-thread parallelism) → enc_buf[idx],  *
 *   then scatter each sub-chunk's encrypted bytes back to g_image_mem at   *
 *   its natural img_abs (so per-byte BW model is faithful).                *
 *                                                                          *
 * This batches encrypt across thousands of small ranges so we don't        *
 * pay an EVP_CIPHER_CTX_new + AES key schedule + 2 MB single-thread        *
 * encrypt PER range (the 50× penalty we saw at 55610 ranges).              */
typedef struct {
    cudaEvent_t        start_evt[2];
    cudaEvent_t        end_evt[2];
    /* Per-slot plan: which gcr_chunk_t entries this slot packs. */
    const gcr_chunk_t *chunks_base;       /* shared base ptr (all slots) */
    uint32_t           slot_first[2];     /* first chunk index in slot */
    uint32_t           slot_count[2];     /* sub-chunks in slot */
    uint64_t           slot_total[2];     /* total bytes packed in slot */
    uint32_t           slot_np[2];        /* AES-GCM pages over packed slot */
    uint64_t           slot_iv[2];        /* base IV counter for slot */

    int                slot_ready[2];
    int                producer_done;

    pthread_mutex_t    mu;
    pthread_cond_t     cv_ready;
    pthread_cond_t     cv_empty;

    double             tot_enc_ms;
    double             tot_wb_ms;
    double             tot_d2h_gpu_ms;
    uint64_t           tot_bytes;
} gcr_pingpong_ctx_t;

static void *gcr_pingpong_consumer(void *arg)
{
    gcr_pingpong_ctx_t *ctx = (gcr_pingpong_ctx_t *)arg;
    uint32_t consumed = 0;

    while (1) {
        int idx = (int)(consumed & 1);

        pthread_mutex_lock(&ctx->mu);
        while (!ctx->slot_ready[idx] && !ctx->producer_done) {
            pthread_cond_wait(&ctx->cv_ready, &ctx->mu);
        }
        int has_work = ctx->slot_ready[idx];
        pthread_mutex_unlock(&ctx->mu);

        if (!has_work) break;

        float gpu_ms_f = 0.0f;
        if (real_cudaEventElapsedTime &&
            real_cudaEventElapsedTime(&gpu_ms_f,
                                       ctx->start_evt[idx],
                                       ctx->end_evt[idx]) == cudaSuccess) {
            ctx->tot_d2h_gpu_ms += (double)gpu_ms_f;
        }

        /* One batched encrypt over the whole packed slot. */
        double t0 = now_ms();
        ckpt_crypto_encrypt_pages(g_k3_key,
                                   g_staging[idx], g_enc_buf[idx],
                                   g_enc_meta[idx], ctx->slot_np[idx],
                                   CKPT_CRYPTO_PAGE_SIZE,
                                   ctx->slot_iv[idx], GCR_ENC_THREADS);
        double t1 = now_ms();

        /* Scatter encrypted bytes per sub-chunk back to natural img_abs.  *
         * Note: AES-GCM page boundaries don't align with sub-chunk        *
         * boundaries in packed staging, so the scattered ciphertext is    *
         * not page-decryptable in place — fine for cost-model emulation.  */
        if (g_image_mem) {
            const gcr_chunk_t *cks = ctx->chunks_base + ctx->slot_first[idx];
            uint32_t cn = ctx->slot_count[idx];
            for (uint32_t k = 0; k < cn; k++) {
                std::memcpy(g_image_mem + cks[k].img_abs,
                            g_enc_buf[idx] + cks[k].pack_off,
                            (size_t)cks[k].size);
            }
        }
        double t2 = now_ms();

        ctx->tot_enc_ms += t1 - t0;
        ctx->tot_wb_ms  += t2 - t1;
        ctx->tot_bytes  += ctx->slot_total[idx];

        pthread_mutex_lock(&ctx->mu);
        ctx->slot_ready[idx] = 0;
        pthread_cond_signal(&ctx->cv_empty);
        pthread_mutex_unlock(&ctx->mu);

        consumed++;
    }
    return nullptr;
}

/* Copy + encrypt a set of byte ranges into the persistent image backing.
 * Implementation: producer/consumer with 2 staging slots (ping-pong).
 *   - Producer (this thread): cudaMemcpyAsync D2H, hand slot off
 *   - Consumer thread:        AES-GCM encrypt, memcpy into image slot
 * Each consumed chunk overwrites prior round's bytes in place (GCR §5
 * merge semantics).
 *
 * Counters: ranges skipped at chunk-build (no slot, no containing alloc)
 * increment *skipped_out so caller can correlate snapshot vs persisted.
 */
static void copy_and_encrypt_ranges(const gcr_range_t *ranges, uint32_t n,
                                    uint64_t /*max_sz_unused*/,
                                    double *copy_ms, double *enc_ms,
                                    double *wb_ms,
                                    uint64_t *bytes_out,
                                    uint64_t *skipped_out,
                                    uint64_t iv_base)
{
    *copy_ms = 0; *enc_ms = 0;
    if (wb_ms) *wb_ms = 0;
    *bytes_out = 0;
    if (skipped_out) *skipped_out = 0;
    if (n == 0 || !g_staging[0] || !g_image_mem) return;

    uint32_t n_chunks = 0;
    uint32_t n_slots  = 0;
    uint64_t range_skipped = 0;
    gcr_slot_t *slots = nullptr;
    gcr_chunk_t *chunks = build_chunks_gcr(ranges, n, g_staging_size,
                                            &n_chunks, &slots, &n_slots,
                                            &range_skipped);
    if (skipped_out) *skipped_out = range_skipped;
    if (!chunks || n_chunks == 0 || !slots || n_slots == 0) {
        free(chunks); free(slots); return;
    }

    cudaStream_t stream = nullptr;
    if (real_cudaStreamCreate) real_cudaStreamCreate(&stream);
    if (!stream) {
        fprintf(stderr, "[gcr] pipeline: cudaStreamCreate failed\n");
        free(chunks); free(slots);
        return;
    }

    gcr_pingpong_ctx_t ctx;
    std::memset(&ctx, 0, sizeof(ctx));
    ctx.chunks_base = chunks;
    if (real_cudaEventCreate(&ctx.start_evt[0]) != cudaSuccess ||
        real_cudaEventCreate(&ctx.start_evt[1]) != cudaSuccess ||
        real_cudaEventCreate(&ctx.end_evt[0])   != cudaSuccess ||
        real_cudaEventCreate(&ctx.end_evt[1])   != cudaSuccess) {
        fprintf(stderr, "[gcr] pipeline: cudaEventCreate failed\n");
        if (ctx.start_evt[0]) real_cudaEventDestroy(ctx.start_evt[0]);
        if (ctx.start_evt[1]) real_cudaEventDestroy(ctx.start_evt[1]);
        if (ctx.end_evt[0])   real_cudaEventDestroy(ctx.end_evt[0]);
        if (ctx.end_evt[1])   real_cudaEventDestroy(ctx.end_evt[1]);
        if (stream && real_cudaStreamDestroy) real_cudaStreamDestroy(stream);
        free(chunks); free(slots);
        return;
    }
    pthread_mutex_init(&ctx.mu, nullptr);
    pthread_cond_init(&ctx.cv_ready, nullptr);
    pthread_cond_init(&ctx.cv_empty, nullptr);

    pthread_t consumer_tid;
    if (pthread_create(&consumer_tid, nullptr, gcr_pingpong_consumer, &ctx) != 0) {
        fprintf(stderr, "[gcr] pipeline: failed to spawn consumer thread\n");
        real_cudaEventDestroy(ctx.start_evt[0]); real_cudaEventDestroy(ctx.start_evt[1]);
        real_cudaEventDestroy(ctx.end_evt[0]);   real_cudaEventDestroy(ctx.end_evt[1]);
        pthread_mutex_destroy(&ctx.mu);
        pthread_cond_destroy(&ctx.cv_ready);
        pthread_cond_destroy(&ctx.cv_empty);
        if (stream && real_cudaStreamDestroy) real_cudaStreamDestroy(stream);
        free(chunks); free(slots);
        return;
    }

    uint64_t iv = iv_base;
    double tot_launch = 0;
    uint32_t skipped_d2h = 0;

    for (uint32_t s = 0; s < n_slots; s++) {
        int idx = (int)(s & 1);

        pthread_mutex_lock(&ctx.mu);
        while (ctx.slot_ready[idx]) {
            pthread_cond_wait(&ctx.cv_empty, &ctx.mu);
        }
        pthread_mutex_unlock(&ctx.mu);

        /* Issue all sub-chunk D2Hs into the slot's packed buffer, one      *
         * cudaMemcpyAsync per sub-chunk (the launch tax is unavoidable     *
         * since each range lives at a different GPU VA). Record start on   *
         * first, end after last so the consumer can read GPU-side D2H ms.  */
        const gcr_slot_t  *sl = &slots[s];
        const gcr_chunk_t *cks = &chunks[sl->first];
        real_cudaEventRecord(ctx.start_evt[idx], stream);
        for (uint32_t k = 0; k < sl->count; k++) {
            double l0 = now_ms();
            cudaError_t err = real_cudaMemcpyAsync(
                                  g_staging[idx] + cks[k].pack_off,
                                  (void *)(uintptr_t)cks[k].va,
                                  (size_t)cks[k].size,
                                  cudaMemcpyDeviceToHost, stream);
            double l1 = now_ms();
            tot_launch += l1 - l0;
            if (err != cudaSuccess) {
                fprintf(stderr, "[gcr] pipeline: D2H slot=%u sub=%u failed: %d\n",
                        s, k, (int)err);
                skipped_d2h++;
            }
        }
        real_cudaEventRecord(ctx.end_evt[idx], stream);

        real_cudaEventSynchronize(ctx.end_evt[idx]);

        uint32_t np = (uint32_t)((sl->total_bytes + CKPT_CRYPTO_PAGE_SIZE - 1)
                                  / CKPT_CRYPTO_PAGE_SIZE);
        pthread_mutex_lock(&ctx.mu);
        ctx.slot_first[idx] = sl->first;
        ctx.slot_count[idx] = sl->count;
        ctx.slot_total[idx] = sl->total_bytes;
        ctx.slot_np[idx]    = np;
        ctx.slot_iv[idx]    = iv;
        ctx.slot_ready[idx] = 1;
        pthread_cond_signal(&ctx.cv_ready);
        pthread_mutex_unlock(&ctx.mu);

        iv += np;
    }

    pthread_mutex_lock(&ctx.mu);
    ctx.producer_done = 1;
    pthread_cond_broadcast(&ctx.cv_ready);
    pthread_mutex_unlock(&ctx.mu);

    pthread_join(consumer_tid, nullptr);

    real_cudaEventDestroy(ctx.start_evt[0]); real_cudaEventDestroy(ctx.start_evt[1]);
    real_cudaEventDestroy(ctx.end_evt[0]);   real_cudaEventDestroy(ctx.end_evt[1]);
    pthread_mutex_destroy(&ctx.mu);
    pthread_cond_destroy(&ctx.cv_ready);
    pthread_cond_destroy(&ctx.cv_empty);
    if (stream && real_cudaStreamDestroy) real_cudaStreamDestroy(stream);
    free(chunks); free(slots);

    *copy_ms   = tot_launch;
    *enc_ms    = ctx.tot_enc_ms;
    if (wb_ms) *wb_ms = ctx.tot_wb_ms;
    *bytes_out = ctx.tot_bytes;
    if (skipped_out) *skipped_out += skipped_d2h;
}

static void snapshot_to(const char *label,
                        gcr_range_t **out_ranges, uint32_t *out_n,
                        uint64_t *out_bytes, uint64_t *out_max_sz)
{
    static gcr_range_t *s_buf = nullptr;
    if (!s_buf) s_buf = (gcr_range_t *)malloc(GCR_MAX_RANGES * sizeof(*s_buf));

    uint64_t tot = 0;
    uint32_t n = gcr_snapshot_dirty_set(s_buf, GCR_MAX_RANGES, &tot);
    /* Reset the global set so new marks accumulate into the next phase. */
    gcr_clear_dirty_set();

    gcr_range_t *buf = (gcr_range_t *)malloc(n * sizeof(*buf));
    std::memcpy(buf, s_buf, n * sizeof(*buf));

    uint64_t max_sz = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t sz = buf[i].end - buf[i].start;
        if (sz > max_sz) max_sz = sz;
    }

    *out_ranges  = buf;
    *out_n       = n;
    *out_bytes   = tot;
    *out_max_sz  = max_sz;

    fprintf(stderr, "[gcr] %s: snapshot — %u ranges, %.2f GB, max range %.2f MB\n",
            label, n,
            tot / (1024.0 * 1024.0 * 1024.0),
            max_sz / (1024.0 * 1024.0));
}

/* Reset per-round dirty-marking counters so Phase 4's breakdown is per-round. */
static void reset_round_counters(void)
{
    g_n_memcpy.store(0);
    g_n_memset.store(0);
    g_n_cublas_gemm.store(0);
    g_n_cublas_lt.store(0);
    g_n_kernel_total.store(0);
    g_n_template_hits.store(0);
    g_n_template_miss.store(0);
    g_n_fallback_pessi.store(0);
}

/* Wall time of the last completed round — used to schedule the next one
 * after GCR_CKPT_INTERVAL_MS. */
static double   g_last_round_done_ms = 0;

/* ------------------------------------------------------------------ */
/* Diagnostic: namespace histogram of cached kernel names              */
/*                                                                     */
/* The hostfn cache holds every unique mangled name __cudaRegisterFunc */
/* gave us. For diagnosing template misses we want a top-level view —  */
/* "how many are vllm vs at vs cublas vs ..." — plus a short list of   */
/* the actual vllm:: names (the universe templates target) so we can   */
/* eyeball which ones drifted vs the 7 keys in dirty_templates.cpp.    */
/* ------------------------------------------------------------------ */
static void extract_first_ns(const char *mangled, char *out, size_t out_size)
{
    if (!mangled || out_size == 0) { if (out_size) out[0] = '\0'; return; }
    const char *p = mangled;
    /* Itanium nested-name: "_ZN<len><name>..." */
    if (p[0] == '_' && p[1] == 'Z' && p[2] == 'N') {
        p += 3;
        int len = 0;
        while (*p >= '0' && *p <= '9') { len = len * 10 + (*p - '0'); p++; }
        if (len > 0 && (size_t)len < out_size) {
            memcpy(out, p, len);
            out[len] = '\0';
            return;
        }
    }
    /* Fallback: copy identifier-ish leading chars up to out_size-1. */
    size_t i = 0;
    while (mangled[i] && i < out_size - 1) {
        char c = mangled[i];
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == ':';
        if (!ok) break;
        out[i] = c; i++;
    }
    out[i] = '\0';
    if (i == 0) snprintf(out, out_size, "<other>");
}

#define GCR_MAX_NS_BUCKETS 32
typedef struct {
    char     name[48];
    uint32_t total;
    uint32_t hits;
} ns_bucket_t;

static void bucket_add(ns_bucket_t *b, int *n, int max, const char *ns, int hit)
{
    for (int i = 0; i < *n; i++) {
        if (std::strcmp(b[i].name, ns) == 0) {
            b[i].total++;
            if (hit) b[i].hits++;
            return;
        }
    }
    if (*n < max) {
        std::strncpy(b[*n].name, ns, sizeof(b[*n].name) - 1);
        b[*n].name[sizeof(b[*n].name) - 1] = '\0';
        b[*n].total = 1;
        b[*n].hits  = hit ? 1 : 0;
        (*n)++;
    }
}

static void dump_name_summary(void)
{
    ns_bucket_t buckets[GCR_MAX_NS_BUCKETS];
    int nb = 0;
    char nsbuf[64];

    /* Walk the populated cache (kname + hname) and bin by first namespace. */
    for (uint32_t i = 0; i < GCR_MAX_KERNELS; i++) {
        if (g_kernel_cache[i].f.load(std::memory_order_relaxed) == 0) continue;
        extract_first_ns(g_kernel_cache[i].name, nsbuf, sizeof(nsbuf));
        bucket_add(buckets, &nb, GCR_MAX_NS_BUCKETS, nsbuf,
                   g_kernel_cache[i].tmpl != nullptr);
    }
    for (uint32_t i = 0; i < GCR_MAX_HOSTFNS; i++) {
        if (g_hostfn_cache[i].stub.load(std::memory_order_relaxed) == 0) continue;
        extract_first_ns(g_hostfn_cache[i].name, nsbuf, sizeof(nsbuf));
        bucket_add(buckets, &nb, GCR_MAX_NS_BUCKETS, nsbuf,
                   g_hostfn_cache[i].tmpl != nullptr);
    }

    /* Sort buckets by total desc (small array, plain insertion sort). */
    for (int i = 1; i < nb; i++) {
        ns_bucket_t cur = buckets[i];
        int j = i - 1;
        while (j >= 0 && buckets[j].total < cur.total) {
            buckets[j + 1] = buckets[j]; j--;
        }
        buckets[j + 1] = cur;
    }

    fprintf(stderr, "[gcr]   ns histogram  : (top %d, sorted by count)\n",
            nb < 15 ? nb : 15);
    for (int i = 0; i < nb && i < 15; i++) {
        fprintf(stderr, "[gcr]     %-20s : %5u  (tmpl=%u)\n",
                buckets[i].name, buckets[i].total, buckets[i].hits);
    }

    /* Focused dump of vllm:: kernels (the universe templates target).
     * Cap at 40 to keep output bounded; if more exist, indicate truncation. */
    fprintf(stderr, "[gcr]   vllm:: kernels seen (mangled, dedup by cache):\n");
    int vllm_shown = 0, vllm_total = 0;
    for (uint32_t i = 0; i < GCR_MAX_HOSTFNS; i++) {
        if (g_hostfn_cache[i].stub.load(std::memory_order_relaxed) == 0) continue;
        extract_first_ns(g_hostfn_cache[i].name, nsbuf, sizeof(nsbuf));
        if (std::strcmp(nsbuf, "vllm") != 0) continue;
        vllm_total++;
        if (vllm_shown < 40) {
            fprintf(stderr, "[gcr]     %s %s\n",
                    g_hostfn_cache[i].tmpl ? "HIT " : "miss",
                    g_hostfn_cache[i].name);
            vllm_shown++;
        }
    }
    if (vllm_total == 0) {
        fprintf(stderr, "[gcr]     (none — vLLM custom kernels never registered "
                "via __cudaRegisterFunction)\n");
    } else if (vllm_total > vllm_shown) {
        fprintf(stderr, "[gcr]     ... +%d more vllm:: entries truncated\n",
                vllm_total - vllm_shown);
    }
}

/* Faithful GCR: snapshot dirty set + copy/encrypt under app stall, in one
 * synchronous pass. Mirrors GCR/cuda.cpp's signal-handler path where
 * incremental_ckpt() walks `dirty_addresses` once with the app blocked.
 * No background thread, no precopy overlap. Round 0 captures everything
 * shadow execution has marked since process start; later rounds capture
 * only ranges dirtied since the previous round (we clear the set after
 * each snapshot). */
static void do_stop_and_copy(void)
{
    uint64_t rid = g_round_id.load();
    fprintf(stderr, "\n[gcr] === Round %lu — STOP-AND-COPY ===\n",
            (unsigned long)rid);

    /* Lazy staging init: CUDA is alive in this process by now (we got here
     * via cudaStream/DeviceSynchronize). Doing this in the LD_PRELOAD
     * constructor would break vLLM's EngineCore fork. */
    staging_init(g_staging_want);

    g_ckpt_in_flight.store(1, std::memory_order_release);
    g_ckpt_phase = 1;

    uint64_t tracked_b = 0;
    uint32_t tracked_n = 0;
    tracked_total(&tracked_b, &tracked_n);
    fprintf(stderr, "[gcr] tracked allocs: %u live, %.2f GB total"
            " (image used %.2f / %.2f GB)\n",
            tracked_n, tracked_b / (1024.0 * 1024.0 * 1024.0),
            g_image_used / (1024.0 * 1024.0 * 1024.0),
            g_image_size / (1024.0 * 1024.0 * 1024.0));

    /* Round 0: fresh K3 key. Subsequent rounds: reuse for now (real systems
     * would rotate per round; rotation-during-restore is downstream work). */
    if (rid == 0) ckpt_crypto_gen_key(g_k3_key);

    /* Snapshot dirty set + clear. App is stalled — caller is on the
     * cudaStreamSynchronize/cudaDeviceSynchronize fast path and won't
     * launch new work until we return. */
    gcr_range_t *ranges = nullptr;
    uint32_t     n      = 0;
    uint64_t     bytes  = 0;
    uint64_t     max_sz = 0;
    snapshot_to("snapshot", &ranges, &n, &bytes, &max_sz);

    fprintf(stderr, "[gcr] copy+encrypt %u ranges, %.2f GB (APP STALLED)\n",
            n, bytes / (1024.0 * 1024.0 * 1024.0));
    double   copy_ms = 0, enc_ms = 0, wb_ms = 0;
    uint64_t copied  = 0, skipped = 0;
    /* IV space per round: bias by round_id * 2^32 pages so rounds never
     * collide; one phase only, so no sub-phase offset needed. */
    uint64_t iv_base = rid * (uint64_t)(1ULL << 32);
    double   stall0  = now_ms();
    copy_and_encrypt_ranges(ranges, n, /*unused*/0,
                            &copy_ms, &enc_ms, &wb_ms, &copied, &skipped,
                            iv_base);
    double stall_ms = now_ms() - stall0;

    fprintf(stderr, "\n[gcr] === Round %lu results ===\n", (unsigned long)rid);
    fprintf(stderr, "[gcr]   ranges      : %u  (skipped %lu)\n",
            n, (unsigned long)skipped);
    if (rid == 0) {
        fprintf(stderr, "[gcr]   bytes       : %.2f GB (full base)\n",
                copied / (1024.0 * 1024.0 * 1024.0));
    } else {
        fprintf(stderr, "[gcr]   delta       : %.2f MB (incremental)\n",
                copied / (1024.0 * 1024.0));
    }
    fprintf(stderr, "[gcr]   D2H         : %8.1f ms  (%.2f GB/s)\n",
            copy_ms,
            copy_ms > 0 ? (copied / (1024.0 * 1024.0 * 1024.0))
                            / (copy_ms / 1000.0) : 0);
    fprintf(stderr, "[gcr]   k3 encrypt  : %8.1f ms\n", enc_ms);
    fprintf(stderr, "[gcr]   writeback   : %8.1f ms\n", wb_ms);
    fprintf(stderr, "[gcr]   STALL TIME  : %8.1f ms (producer/consumer ping-pong)\n",
            stall_ms);

    /* CSV: keep existing 7-field schema. There's only one phase under
     * stop-and-copy, so the "p2" (concurrent) columns are zero and the
     * full copy lands in the "p4" (stall) columns. The stall_ms field
     * is the authoritative end-to-end pause time. */
    metrics_record_round(rid,
                         /*p2_bytes*/0, /*p2_copy_ms*/0, /*p2_enc_ms*/0,
                         copied,        copy_ms,         enc_ms,
                         stall_ms);

    fprintf(stderr, "\n[gcr] Dirty-marking breakdown (this round):\n");
    fprintf(stderr, "[gcr]   memcpy        : %lu\n",
            (unsigned long)g_n_memcpy.load());
    fprintf(stderr, "[gcr]   memset        : %lu\n",
            (unsigned long)g_n_memset.load());
    fprintf(stderr, "[gcr]   cuBLAS gemmEx : %lu\n",
            (unsigned long)g_n_cublas_gemm.load());
    fprintf(stderr, "[gcr]   cuBLASLt      : %lu\n",
            (unsigned long)g_n_cublas_lt.load());
    fprintf(stderr, "[gcr]   kernels total : %lu\n",
            (unsigned long)g_n_kernel_total.load());
    fprintf(stderr, "[gcr]     tmpl hits   : %lu\n",
            (unsigned long)g_n_template_hits.load());
    fprintf(stderr, "[gcr]     tmpl misses : %lu  (pessimistic fallback: %lu)\n",
            (unsigned long)g_n_template_miss.load(),
            (unsigned long)g_n_fallback_pessi.load());

    /* Diagnostic — which name-resolution paths actually fired, and a
     * namespace-binned summary of what landed in the cache. Use this to
     * tell apart (a) hook never seeing names, (b) names registered but
     * none match templates, (c) cache saturated. */
    {
        uint32_t kname_n = 0, kname_hits = 0;
        for (uint32_t i = 0; i < GCR_MAX_KERNELS; i++) {
            if (g_kernel_cache[i].f.load(std::memory_order_relaxed) != 0) {
                kname_n++;
                if (g_kernel_cache[i].tmpl) kname_hits++;
            }
        }
        uint32_t hname_n = 0, hname_hits = 0;
        for (uint32_t i = 0; i < GCR_MAX_HOSTFNS; i++) {
            if (g_hostfn_cache[i].stub.load(std::memory_order_relaxed) != 0) {
                hname_n++;
                if (g_hostfn_cache[i].tmpl) hname_hits++;
            }
        }
        fprintf(stderr, "[gcr]   name paths    : cuModuleGetFunction=%lu "
                "__cudaRegisterFunction=%lu lazy(cuFuncGetName)=%lu\n",
                (unsigned long)g_n_cumod_getfunc.load(),
                (unsigned long)g_n_cudareg_func.load(),
                (unsigned long)g_n_lazy_resolve.load());
        fprintf(stderr, "[gcr]   kname cache   : %u entries, %u with template%s\n",
                kname_n, kname_hits,
                kname_n == GCR_MAX_KERNELS ? " (FULL — bump GCR_MAX_KERNELS)" : "");
        fprintf(stderr, "[gcr]   hname cache   : %u entries, %u with template%s\n",
                hname_n, hname_hits,
                hname_n == GCR_MAX_HOSTFNS ? " (FULL — bump GCR_MAX_HOSTFNS)" : "");
        dump_name_summary();
    }
    fflush(stderr);

    free(ranges);

    /* End-of-round housekeeping: drop deferred frees, advance round id,
     * release the cv so any cudaFree threads waiting on us can proceed,
     * reset per-round counters, mark idle. */
    sweep_pending_drops();
    g_round_id.fetch_add(1);
    reset_round_counters();

    /* Final round just finished — dump the summary now. vLLM v1 SIGKILLs
     * this engine-core worker at shutdown, so the atexit() fallback never
     * runs here; metrics_dump_csv() is one-shot guarded so this is safe. */
    if (g_round_id.load() >= (uint64_t)g_ckpt_rounds_max) {
        fprintf(stderr, "\n[gcr] final round (%d/%d) complete — metrics summary:\n",
                g_ckpt_rounds_max, g_ckpt_rounds_max);
        metrics_dump_csv();
    }

    g_ckpt_phase = 0;
    g_ckpt_requested = 0;
    g_last_round_done_ms = now_ms();

    pthread_mutex_lock(&g_ckpt_mu);
    g_ckpt_in_flight.store(0, std::memory_order_release);
    pthread_cond_broadcast(&g_ckpt_cv);
    pthread_mutex_unlock(&g_ckpt_mu);
}

static void maybe_checkpoint(void)
{
    /* Stop entirely once we've completed the configured number of rounds. */
    if (g_round_id.load() >= (uint64_t)g_ckpt_rounds_max) return;

    if (!g_first_sync_seen) {
        g_first_sync_seen = 1;
        g_first_sync_ms   = now_ms();
        fprintf(stderr, "[gcr] first sync at %.1f ms\n", g_first_sync_ms);
    }

    /* First-round trigger: SIGUSR1-driven (GCR_CKPT_WAIT_SIGNAL=1) OR
     * GCR_CKPT_DELAY seconds after first sync. */
    if (g_round_id.load() == 0 && !g_ckpt_requested) {
        if (g_wait_signal) {
            if (g_start_signaled.load()) {
                g_ckpt_requested = 1;
                fprintf(stderr, "[gcr] signal-trigger round 0\n");
            }
        } else if (g_ckpt_delay_s > 0) {
            double el = (now_ms() - g_first_sync_ms) / 1000.0;
            if (el >= g_ckpt_delay_s) {
                g_ckpt_requested = 1;
                fprintf(stderr, "[gcr] auto-trigger round 0 at %.1fs\n", el);
            }
        }
    }

    /* Subsequent rounds: GCR_CKPT_INTERVAL_MS after previous round done. */
    if (g_round_id.load() > 0 && g_ckpt_interval_ms > 0 && !g_ckpt_requested
        && g_ckpt_phase == 0) {
        double el_ms = now_ms() - g_last_round_done_ms;
        if (el_ms >= (double)g_ckpt_interval_ms) {
            g_ckpt_requested = 1;
            fprintf(stderr, "[gcr] auto-trigger round %lu (%.1f ms since prev)\n",
                    (unsigned long)g_round_id.load(), el_ms);
        }
    }

    if (g_ckpt_requested && g_ckpt_phase == 0) do_stop_and_copy();
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t s)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaStreamSynchronize(s);
    maybe_checkpoint();
    return err;
}

extern "C" cudaError_t cudaDeviceSynchronize(void)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaDeviceSynchronize();
    maybe_checkpoint();
    return err;
}

/* ================================================================== */
/* Init + SIGUSR2 (per-round trigger) + SIGUSR1 (start-loop trigger)   */
/* ================================================================== */
static void sigusr1_start(int sig)
{
    (void)sig;
    int expected = 0;
    if (g_start_signaled.compare_exchange_strong(expected, 1)) {
        fprintf(stderr, "[gcr] SIGUSR1 — start ckpt loop "
                "(round 0 fires at next sync)\n");
    }
}

static void sigusr2_trigger(int sig)
{
    (void)sig;
    if (!g_ckpt_requested && g_ckpt_phase == 0
        && g_round_id.load() < (uint64_t)g_ckpt_rounds_max) {
        g_ckpt_requested = 1;
        fprintf(stderr, "[gcr] SIGUSR2 — round %lu will fire at next sync\n",
                (unsigned long)g_round_id.load());
    }
}

static void preload_one(const char *label, const char *const names[])
{
    for (size_t i = 0; names[i]; i++) {
        if (dlopen(names[i], RTLD_LAZY | RTLD_GLOBAL)) {
            fprintf(stderr, "[gcr] preloaded %s via %s\n", label, names[i]);
            return;
        }
    }
    fprintf(stderr, "[gcr] warning: failed to preload %s\n", label);
}

/* Refresh g_self_pid after fork — see comment at the call site below. */
static void refresh_pid_after_fork(void)
{
    g_self_pid = getpid();
    fprintf(stderr, "[gcr] post-fork: refreshed g_self_pid = %d\n",
            (int)g_self_pid);
}

__attribute__((constructor))
static void gcr_init(void)
{
    static const char *const cuda_names[]     = { "libcuda.so.1", "libcuda.so", nullptr };
    static const char *const cudart_names[]   = { "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", nullptr };
    static const char *const cublas_names[]   = { "libcublas.so.12", "libcublas.so.11", "libcublas.so", nullptr };
    static const char *const cublaslt_names[] = { "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so", nullptr };
    preload_one("libcuda",     cuda_names);
    preload_one("libcudart",   cudart_names);
    preload_one("libcublas",   cublas_names);
    preload_one("libcublasLt", cublaslt_names);

    g_self_pid = getpid();
    /* vLLM v1 forks an engine subprocess after the LD_PRELOAD constructor
     * runs. Without this atfork handler, g_self_pid retains the parent's
     * pid, and process_vm_readv(parent_pid, ...) from the child fails with
     * EPERM (cross-process read needs PTRACE_MODE_ATTACH_REALCREDS).
     * Result: every safe_read_qword bails, pessimistic-scan no-ops silently. */
    pthread_atfork(nullptr, nullptr, refresh_pid_after_fork);
    ENSURE_SYMBOLS;

    const char *d = getenv("GCR_CKPT_DELAY");
    if (d) g_ckpt_delay_s = atoi(d);

    const char *sd = getenv("GCR_SCAN_DEPTH");
    if (sd) {
        int v = atoi(sd);
        if (v > 0 && v <= GCR_MAX_KERNEL_PARAMS) g_scan_depth = v;
    }

    const char *dn = getenv("GCR_DUMP_NAMES");
    if (dn) g_dump_names = atoi(dn);

    /* F5 pacing */
    const char *iv = getenv("GCR_CKPT_INTERVAL_MS");
    if (iv) g_ckpt_interval_ms = atoi(iv);
    const char *nr = getenv("GCR_CKPT_ROUNDS");
    if (nr) {
        int v = atoi(nr);
        if (v > 0) g_ckpt_rounds_max = v;
    }

    const char *ws = getenv("GCR_CKPT_WAIT_SIGNAL");
    g_wait_signal = (ws && atoi(ws) > 0) ? 1 : 0;
    if (g_wait_signal) {
        signal(SIGUSR1, sigusr1_start);
        fprintf(stderr, "[gcr] wait-signal mode — round 0 deferred until SIGUSR1\n");
    }

    /* F1 image backing.
     *
     * Preferred path: reuse the /dev/shm region that ckpt_prealloc already
     * allocated for v3 (default name "/ckpt_core_gpu_staging"). gcr_emul and
     * v3 don't run concurrently, so sharing that region avoids paying for
     * two duplicate ~90 GB host allocations across the eval matrix.
     *
     * Fallback: if no such shm exists, allocate our own anonymous mmap of
     * CKPT_IMAGE_GB (default 96 GB). MAP_PRIVATE so it's process-local;
     * lazy-faulted so unwritten pages cost zero physical RAM.
     */
    /* Default image backing is /dev/shm/ckpt_base.img (same destination
     * as v3/p1d/p1e for apples-to-apples comparison). The old name
     * /ckpt_core_gpu_staging is the v3 transport buffer — semantically
     * different (kernel transport vs final image). Set CKPT_IMAGE_SHM_NAME
     * to override. */
    const char *shm_name = getenv("CKPT_IMAGE_SHM_NAME");
    if (!shm_name) shm_name = "/ckpt_base.img";

    int shm_fd = shm_open(shm_name, O_RDWR, 0);
    if (shm_fd >= 0) {
        struct stat st;
        if (fstat(shm_fd, &st) == 0 && st.st_size > 0) {
            g_image_size = (uint64_t)st.st_size;
            g_image_mem  = (uint8_t *)mmap(nullptr, g_image_size,
                                           PROT_READ | PROT_WRITE,
                                           MAP_SHARED, shm_fd, 0);
            if (g_image_mem == MAP_FAILED) {
                fprintf(stderr, "[gcr] mmap(/dev/shm%s, %.1f GB) failed: %s\n",
                        shm_name,
                        g_image_size / (1024.0 * 1024.0 * 1024.0),
                        strerror(errno));
                g_image_mem = nullptr;
                g_image_size = 0;
            } else {
                fprintf(stderr, "[gcr] image backing: shared /dev/shm%s "
                        "(%.1f GB, MAP_SHARED) — reusing v3 prealloc\n",
                        shm_name,
                        g_image_size / (1024.0 * 1024.0 * 1024.0));
            }
        }
        close(shm_fd);
    }

    if (!g_image_mem) {
        uint64_t image_gb = 96;
        const char *ig = getenv("CKPT_IMAGE_GB");
        if (ig) {
            long v = atol(ig);
            if (v > 0) image_gb = (uint64_t)v;
        }
        g_image_size = image_gb * (1024ULL * 1024ULL * 1024ULL);
        g_image_mem = (uint8_t *)mmap(nullptr, g_image_size,
                                      PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_image_mem == MAP_FAILED) {
            fprintf(stderr, "[gcr] anon mmap(%lu GB) failed: %s — "
                    "checkpoint disabled\n",
                    (unsigned long)image_gb, strerror(errno));
            g_image_mem = nullptr;
            g_image_size = 0;
        } else {
            fprintf(stderr, "[gcr] image backing: %lu GB anonymous mmap "
                    "(no /dev/shm%s found — run ckpt_prealloc to share with v3)\n",
                    (unsigned long)image_gb, shm_name);
        }
    }

    /* F2 staging size — DEFERRED to first checkpoint. Calling cudaMallocHost
     * from an LD_PRELOAD constructor touches CUDA state in the parent process,
     * which then breaks when vLLM v1 forks its EngineCore subprocess (CUDA
     * driver state doesn't survive fork). Just stash the size; staging_init()
     * is called lazily by do_stop_and_copy once CUDA is alive in this process. */
    const char *sm = getenv("CKPT_STAGING_MB");
    if (sm) {
        long v = atol(sm);
        if (v > 0) g_staging_want = (uint64_t)v * 1024ULL * 1024ULL;
    }

    signal(SIGUSR2, sigusr2_trigger);
    atexit(metrics_dump_csv);

    fprintf(stderr, "[gcr] libgcr_intercept loaded "
            "(pid=%d, delay=%ds, interval=%dms, rounds=%d, scan_depth=%d, "
            "dump_names=%d, trigger=SIGUSR2)\n",
            (int)getpid(), g_ckpt_delay_s, g_ckpt_interval_ms,
            g_ckpt_rounds_max, g_scan_depth, g_dump_names);
}
