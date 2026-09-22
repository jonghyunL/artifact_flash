/*
 * libphos_intercept.c — PhoenixOS-style checkpoint emulator with
 *                       *real* per-buffer dirty tracking.
 *
 * PhoenixOS_emulate/libphos_intercept.c:
 *   the old shim flagged every non-H2D allocation dirty speculatively.
 *   This shim actually tracks which buffers an API call touches, matching
 *   the mechanism described in the PhoenixOS paper.
 *
 * Dirty-marking sources:
 *   cudaMemcpy / cudaMemcpyAsync            — mark dst alloc dirty
 *   cudaMemset / cudaMemsetAsync            — mark dst alloc dirty
 *   cublasGemmEx                            — mark C alloc dirty
 *   cublasLtMatmul                          — mark D alloc dirty
 *   cudaLaunchKernel / cuLaunchKernel       — scan the kernel-param array
 *                                             for pointers in tracked ranges
 *
 * Checkpoint pipeline (PhoenixOS "recopy protocol"):
 *   Phase 1: quiesce, clear dirtied-during-copy marks
 *   Phase 2: background cudaMemcpy D2H + k3 encrypt of all live allocs
 *            (app keeps running; new kernel launches mark buffers dirty)
 *   Phase 3: re-quiesce at next sync boundary after Phase 2 done
 *   Phase 4: recopy any alloc marked dirty during Phase 2 (STALL TIME)
 *
 * Trigger: SIGUSR2 or PHOS_CKPT_DELAY=<seconds>.
 *
 * Build: make -C phos_emul libphos_intercept.so
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
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

#include "ckpt_crypto.h"

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
#define PHOS_MAX_ALLOCS        4096
#define PHOS_MAX_KERNEL_PARAMS 64      /* upper bound for arg scan */
#define PHOS_CHUNK_SIZE        (4ULL * 1024 * 1024)  /* 4 MB PhoenixOS PCIe chunk */
#define PHOS_ENC_THREADS       0       /* 0 = all available cores */

/* ------------------------------------------------------------------ */
/* Allocation table                                                    */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t va;
    uint64_t size;
    uint64_t img_offset;  /* slot offset in g_image_*_mem; UINT64_MAX = no slot */
    uint8_t  is_h2d;      /* received at least one H2D memcpy */
    uint8_t  alive;       /* cleared on cudaFree */
    _Atomic uint8_t dirty_this_round; /* toggled by any touch since last reset */
} phos_alloc_t;

static phos_alloc_t    g_allocs[PHOS_MAX_ALLOCS];
static uint32_t        g_num_allocs = 0;
static pthread_rwlock_t g_alloc_lock = PTHREAD_RWLOCK_INITIALIZER;

/* ------------------------------------------------------------------ */
/* Image backing — two shm files. Every round: Phase 2 (full concurrent */
/* precopy) overwrites base.img; Phase 4 (stop-and-copy dirty recopy)   */
/* overwrites delta.img. Slot offsets are assigned once at record_alloc */
/* and reused across rounds; both files share the same layout so the    */
/* per-alloc bytes written each round land at a stable offset.          */
/* ------------------------------------------------------------------ */
static uint8_t  *g_image_base_mem   = NULL;
static uint64_t  g_image_base_size  = 0;
static uint8_t  *g_image_delta_mem  = NULL;
static uint64_t  g_image_delta_size = 0;
static uint64_t  g_image_used       = 0;       /* bump pointer */
static pthread_mutex_t g_image_mu   = PTHREAD_MUTEX_INITIALIZER;

/* Assign a slot in the image. The same offset is used in both base    */
/* and delta files. Returns UINT64_MAX if no image backing or out of   */
/* space — caller skips writeback for that alloc.                       */
static uint64_t assign_image_slot(uint64_t size)
{
    if (!g_image_base_mem || size == 0) return (uint64_t)-1;
    pthread_mutex_lock(&g_image_mu);
    if (g_image_used + size > g_image_base_size) {
        pthread_mutex_unlock(&g_image_mu);
        return (uint64_t)-1;
    }
    uint64_t off = g_image_used;
    g_image_used += size;
    pthread_mutex_unlock(&g_image_mu);
    return off;
}

/* Diagnostic counters */
static _Atomic uint64_t g_n_mark_memcpy  = 0;
static _Atomic uint64_t g_n_mark_memset  = 0;
static _Atomic uint64_t g_n_mark_cublas  = 0;
static _Atomic uint64_t g_n_mark_kernel  = 0;
static _Atomic uint64_t g_n_scan_hits    = 0;
static _Atomic uint64_t g_n_scan_misses  = 0;

/* Tunables controlled by environment variables */
static int g_scan_enabled      = 0;       /* PHOS_SCAN_KERNELS=1 to enable */
static int g_scan_depth        = 16;      /* PHOS_SCAN_DEPTH=<n>, default 16 */

/* ------------------------------------------------------------------ */
/* Real CUDA symbols                                                   */
/* ------------------------------------------------------------------ */
#define DECL_REAL(name, rty, args)                                     \
    typedef rty (*name##_fn) args;                                     \
    static name##_fn real_##name = NULL

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
DECL_REAL(cudaEventCreate,        cudaError_t,   (cudaEvent_t *));
DECL_REAL(cudaEventDestroy,       cudaError_t,   (cudaEvent_t));
DECL_REAL(cudaEventRecord,        cudaError_t,   (cudaEvent_t, cudaStream_t));
DECL_REAL(cudaEventSynchronize,   cudaError_t,   (cudaEvent_t));
DECL_REAL(cudaEventElapsedTime,   cudaError_t,   (float *, cudaEvent_t, cudaEvent_t));
DECL_REAL(cudaMallocHost,         cudaError_t,   (void **, size_t));
DECL_REAL(cudaFreeHost,           cudaError_t,   (void *));
DECL_REAL(cudaLaunchKernel,       cudaError_t,   (const void *, dim3, dim3, void **, size_t, cudaStream_t));
DECL_REAL(cuMemAddressReserve,    CUresult,      (CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long));
DECL_REAL(cuMemAddressFree,       CUresult,      (CUdeviceptr, size_t));
DECL_REAL(cuMemMap,               CUresult,      (CUdeviceptr, size_t, size_t,
                                                   CUmemGenericAllocationHandle, unsigned long long));
DECL_REAL(cuMemUnmap,             CUresult,      (CUdeviceptr, size_t));
DECL_REAL(cuMemAlloc_v2,          CUresult,      (CUdeviceptr *, size_t));
DECL_REAL(cuMemFree_v2,           CUresult,      (CUdeviceptr));
DECL_REAL(cuLaunchKernel,         CUresult,      (CUfunction, unsigned, unsigned, unsigned,
                                                   unsigned, unsigned, unsigned,
                                                   unsigned, CUstream, void **, void **));
DECL_REAL(cuLaunchKernelEx,       CUresult,      (const CUlaunchConfig *, CUfunction, void **, void **));
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

static pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;

static void resolve_symbols(void)
{
#define LOAD(name) real_##name = (name##_fn)dlsym(RTLD_NEXT, #name)
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
    LOAD(cudaEventCreate);
    LOAD(cudaEventDestroy);
    LOAD(cudaEventRecord);
    LOAD(cudaEventSynchronize);
    LOAD(cudaEventElapsedTime);
    LOAD(cudaMallocHost);
    LOAD(cudaFreeHost);
    LOAD(cudaLaunchKernel);
    LOAD(cuMemAddressReserve);
    LOAD(cuMemAddressFree);
    LOAD(cuMemMap);
    LOAD(cuMemUnmap);
    LOAD(cuMemAlloc_v2);
    LOAD(cuMemFree_v2);
    LOAD(cuLaunchKernel);
    LOAD(cuLaunchKernelEx);
    LOAD(cublasGemmEx);
    LOAD(cublasLtMatmul);
#undef LOAD
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
/* Alloc table — lookup, insert, retire                                */
/* ------------------------------------------------------------------ */
static int find_alloc_idx_unlocked(uint64_t addr)
{
    /* Linear scan — ~230 live allocs for vLLM, cache-friendly at this size. */
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

static void record_alloc(void *ptr, size_t size)
{
    if (!ptr || size == 0) return;
    uint64_t slot = assign_image_slot((uint64_t)size);
    pthread_rwlock_wrlock(&g_alloc_lock);
    if (g_num_allocs < PHOS_MAX_ALLOCS) {
        uint32_t idx = g_num_allocs++;
        g_allocs[idx].va         = (uint64_t)(uintptr_t)ptr;
        g_allocs[idx].size       = (uint64_t)size;
        g_allocs[idx].img_offset = slot;
        g_allocs[idx].is_h2d     = 0;
        g_allocs[idx].alive      = 1;
        atomic_store(&g_allocs[idx].dirty_this_round, 1); /* new alloc = dirty */
    }
    pthread_rwlock_unlock(&g_alloc_lock);
}

static void retire_alloc(void *ptr)
{
    if (!ptr) return;
    uint64_t addr = (uint64_t)(uintptr_t)ptr;
    pthread_rwlock_wrlock(&g_alloc_lock);
    int idx = find_alloc_idx_unlocked(addr);
    if (idx >= 0) g_allocs[idx].alive = 0;
    pthread_rwlock_unlock(&g_alloc_lock);
}

static inline void mark_dirty_idx(int idx)
{
    if (idx < 0) return;
    atomic_store_explicit(&g_allocs[idx].dirty_this_round, 1,
                          memory_order_relaxed);
}

static inline void mark_dirty_va(uint64_t addr)
{
    mark_dirty_idx(find_alloc_idx(addr));
}

/* ------------------------------------------------------------------ */
/* Kernel-param pointer scan                                           */
/*                                                                     */
/* kernelParams is void** — array of pointers to argument values.      */
/* For each slot, the pointed-to value may be a device pointer.        */
/* We scan up to PHOS_MAX_KERNEL_PARAMS slots, skipping NULLs.          */
/* False positives (scalars that look like valid VAs) only over-mark   */
/* dirty; they don't corrupt anything.                                 */
/* ------------------------------------------------------------------ */
/* Sanity-check a pointer slot before dereferencing.
 *   - Must be 8-byte aligned (real arg pointers always are)
 *   - Must lie in canonical userspace range on x86-64 Linux */
static inline int slot_looks_safe(void *p)
{
    uint64_t v = (uint64_t)p;
    if (v == 0) return 0;
    if (v & 0x7) return 0;                          /* unaligned */
    if (v < 0x0000400000000000ULL) return 0;        /* below typical mmap base */
    if (v > 0x00007fffffffffffULL) return 0;        /* above userspace canonical */
    return 1;
}

/* Safe 8-byte read via process_vm_readv(2).
 * Returns 1 and writes *out on success; 0 on fault (EFAULT / EPERM / ESRCH).
 * Works for same-process reads without special privileges on normal Linux.
 * Page-cache via TLS: most kernel params share a stack frame → one syscall
 * per frame, not per arg. */
static pid_t g_self_pid = 0;
static __thread uintptr_t tls_safe_page = 0;

static int safe_read_qword(const void *src, uint64_t *out)
{
    uintptr_t pg = (uintptr_t)src & ~(uintptr_t)0xFFFUL;
    if (pg == tls_safe_page) {
        *out = *(const uint64_t *)src;
        return 1;
    }
    struct iovec local  = { .iov_base = out,          .iov_len = 8 };
    struct iovec remote = { .iov_base = (void *)src,  .iov_len = 8 };
    if (process_vm_readv(g_self_pid, &local, 1, &remote, 1, 0) == 8) {
        tls_safe_page = pg;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-CUfunction parameter-count cache (for cuLaunchKernel path)      */
/*                                                                     */
/* On first encounter of a CUfunction, call cuFuncGetParamInfo in a    */
/* loop to discover the exact param count. Cache it. Subsequent scans  */
/* only touch the real args — no past-end garbage reads.                */
/* ------------------------------------------------------------------ */
#define FN_CACHE_SIZE 4096                        /* power of two */
#define FN_CACHE_MASK (FN_CACHE_SIZE - 1)

typedef struct {
    _Atomic uintptr_t f;         /* 0 = empty slot */
    _Atomic int       nparams;   /* -1 = unknown / query failed */
} fn_cache_ent_t;

static fn_cache_ent_t g_fn_cache[FN_CACHE_SIZE];

typedef CUresult (*cuFuncGetParamInfo_fn)(CUfunction, size_t, size_t *, size_t *);
static cuFuncGetParamInfo_fn real_cuFuncGetParamInfo = NULL;
static _Atomic int g_func_info_resolved = 0;

static void resolve_cuFuncGetParamInfo(void)
{
    if (atomic_exchange(&g_func_info_resolved, 1)) return;
    /* Symbol lives in libcuda.so (driver API). */
    real_cuFuncGetParamInfo = (cuFuncGetParamInfo_fn)
        dlsym(RTLD_DEFAULT, "cuFuncGetParamInfo");
    if (!real_cuFuncGetParamInfo)
        fprintf(stderr, "[phos] cuFuncGetParamInfo not found — "
                "cuLaunchKernel will fall back to safe-read scan\n");
}

/* Query driver for exact arg count. Returns -1 if unsupported / failed. */
static int query_param_count(CUfunction f)
{
    if (!real_cuFuncGetParamInfo) return -1;
    int n = 0;
    size_t off, sz;
    while (real_cuFuncGetParamInfo(f, n, &off, &sz) == CUDA_SUCCESS) {
        n++;
        if (n > PHOS_MAX_KERNEL_PARAMS) break;   /* sanity cap */
    }
    return n;
}

static int get_param_count_cached(CUfunction f)
{
    if (!f) return -1;
    uintptr_t key = (uintptr_t)f;
    uint64_t h = (key * 2654435769ULL) & FN_CACHE_MASK;
    for (int probe = 0; probe < 32; probe++) {
        uint32_t idx = (h + probe) & FN_CACHE_MASK;
        uintptr_t cur = atomic_load_explicit(&g_fn_cache[idx].f,
                                             memory_order_acquire);
        if (cur == key) {
            return atomic_load_explicit(&g_fn_cache[idx].nparams,
                                        memory_order_relaxed);
        }
        if (cur == 0) {
            int n = query_param_count(f);
            atomic_store_explicit(&g_fn_cache[idx].nparams, n,
                                  memory_order_relaxed);
            uintptr_t expect = 0;
            if (atomic_compare_exchange_strong_explicit(
                    &g_fn_cache[idx].f, &expect, key,
                    memory_order_release, memory_order_acquire)) {
                return n;
            }
            /* Lost the race — another thread claimed the slot. Retry. */
        }
    }
    return -1;   /* cache full — unlikely */
}

/* ------------------------------------------------------------------ */
/* Scanner                                                             */
/*                                                                     */
/* Two paths:                                                          */
/*   (a) CUfunction known → query exact nparams, scan only that many.   */
/*   (b) CUfunction unknown (runtime API) → safe-read scan up to depth.*/
/* ------------------------------------------------------------------ */
static void scan_params_exact(void **params, int nparams)
{
    pthread_rwlock_rdlock(&g_alloc_lock);
    int hits = 0, misses = 0;
    for (int i = 0; i < nparams; i++) {
        void *slot = params[i];
        if (!slot_looks_safe(slot)) { misses++; continue; }
        /* CUDA contract guarantees args[0..nparams-1] are caller-provided
         * valid pointers, so a direct deref is safe here. */
        uint64_t cand = *(uint64_t *)slot;
        int idx = find_alloc_idx_unlocked(cand);
        if (idx >= 0) {
            atomic_store_explicit(&g_allocs[idx].dirty_this_round, 1,
                                  memory_order_relaxed);
            hits++;
        } else {
            misses++;
        }
    }
    pthread_rwlock_unlock(&g_alloc_lock);
    atomic_fetch_add_explicit(&g_n_scan_hits,   hits,   memory_order_relaxed);
    atomic_fetch_add_explicit(&g_n_scan_misses, misses, memory_order_relaxed);
}

static void scan_params_safe(void **params)
{
    /* Reset the thread-local page cache for the params pointer itself. */
    uint64_t probe;
    if (!safe_read_qword(params, &probe)) return;

    pthread_rwlock_rdlock(&g_alloc_lock);
    int hits = 0, misses = 0;
    for (int i = 0; i < g_scan_depth; i++) {
        void *slot = params[i];
        if (!slot_looks_safe(slot)) break;
        uint64_t cand;
        if (!safe_read_qword(slot, &cand)) break;
        int idx = find_alloc_idx_unlocked(cand);
        if (idx >= 0) {
            atomic_store_explicit(&g_allocs[idx].dirty_this_round, 1,
                                  memory_order_relaxed);
            hits++;
        } else {
            misses++;
        }
    }
    pthread_rwlock_unlock(&g_alloc_lock);
    atomic_fetch_add_explicit(&g_n_scan_hits,   hits,   memory_order_relaxed);
    atomic_fetch_add_explicit(&g_n_scan_misses, misses, memory_order_relaxed);
}

/* Entry point when CUfunction is available (driver API path). */
static void scan_kernel_params_exact(CUfunction f, void **params)
{
    if (!g_scan_enabled || !params) return;
    if (!real_cuFuncGetParamInfo) resolve_cuFuncGetParamInfo();
    int n = get_param_count_cached(f);
    if (n > 0) {
        scan_params_exact(params, n);
    } else {
        /* Fallback if driver query fails. */
        scan_params_safe(params);
    }
}

/* Entry point for runtime API (no CUfunction). */
static void scan_kernel_params(void **params)
{
    if (!g_scan_enabled || !params) return;
    scan_params_safe(params);
}

/* ================================================================== */
/* Alloc hooks                                                         */
/* ================================================================== */
/* Forward decls for the multi-round state — actual definitions live with
 * the rest of the checkpoint state further down in the file. */
extern _Atomic int     g_ckpt_in_flight;
extern pthread_mutex_t g_ckpt_mu;
extern pthread_cond_t  g_ckpt_cv;

/* Block until the current checkpoint round finishes (or return immediately
 * if none is in flight). Called from every free hook so the GPU memory
 * isn't actually returned while Phase 2/4 is reading it. Rare on the
 * vLLM hot path (PyTorch's caching allocator usually intercepts frees
 * before they reach our hook). */
static void wait_until_ckpt_idle(void)
{
    if (!atomic_load_explicit(&g_ckpt_in_flight, memory_order_acquire)) return;
    pthread_mutex_lock(&g_ckpt_mu);
    while (atomic_load_explicit(&g_ckpt_in_flight, memory_order_acquire))
        pthread_cond_wait(&g_ckpt_cv, &g_ckpt_mu);
    pthread_mutex_unlock(&g_ckpt_mu);
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMalloc(devPtr, size);
    if (err == cudaSuccess) record_alloc(*devPtr, size);
    return err;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMallocAsync(devPtr, size, stream);
    if (err == cudaSuccess) record_alloc(*devPtr, size);
    return err;
}

cudaError_t cudaFree(void *devPtr)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc(devPtr);
    return real_cudaFree(devPtr);
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc(devPtr);
    return real_cudaFreeAsync(devPtr, stream);
}

/* Driver-API allocators. vLLM v1 paged KV uses cuMemAddressReserve +
 * cuMemMap directly (NOT cudaMalloc), so without these hooks the KV slabs
 * are invisible to record_alloc and the dirty-set undercounts. */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    ENSURE_SYMBOLS;
    CUresult r = real_cuMemAlloc_v2(dptr, bytesize);
    if (r == CUDA_SUCCESS && dptr) record_alloc((void *)(uintptr_t)*dptr, bytesize);
    return r;
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc((void *)(uintptr_t)dptr);
    return real_cuMemFree_v2(dptr);
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags)
{
    ENSURE_SYMBOLS;
    CUresult r = real_cuMemMap(ptr, size, offset, handle, flags);
    if (r == CUDA_SUCCESS) record_alloc((void *)(uintptr_t)ptr, size);
    return r;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    ENSURE_SYMBOLS;
    wait_until_ckpt_idle();
    retire_alloc((void *)(uintptr_t)ptr);
    return real_cuMemUnmap(ptr, size);
}

/* ================================================================== */
/* Memcpy / memset hooks                                               */
/* ================================================================== */
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemcpy(dst, src, count, kind);
    if (err == cudaSuccess && count > 0) {
        mark_dirty_va((uint64_t)(uintptr_t)dst);
        atomic_fetch_add_explicit(&g_n_mark_memcpy, 1, memory_order_relaxed);
        if (kind == cudaMemcpyHostToDevice) {
            int idx = find_alloc_idx((uint64_t)(uintptr_t)dst);
            if (idx >= 0) g_allocs[idx].is_h2d = 1;
        }
    }
    return err;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemcpyAsync(dst, src, count, kind, stream);
    if (err == cudaSuccess && count > 0) {
        mark_dirty_va((uint64_t)(uintptr_t)dst);
        atomic_fetch_add_explicit(&g_n_mark_memcpy, 1, memory_order_relaxed);
        if (kind == cudaMemcpyHostToDevice) {
            int idx = find_alloc_idx((uint64_t)(uintptr_t)dst);
            if (idx >= 0) g_allocs[idx].is_h2d = 1;
        }
    }
    return err;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemset(devPtr, value, count);
    if (err == cudaSuccess && count > 0) {
        mark_dirty_va((uint64_t)(uintptr_t)devPtr);
        atomic_fetch_add_explicit(&g_n_mark_memset, 1, memory_order_relaxed);
    }
    return err;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaMemsetAsync(devPtr, value, count, stream);
    if (err == cudaSuccess && count > 0) {
        mark_dirty_va((uint64_t)(uintptr_t)devPtr);
        atomic_fetch_add_explicit(&g_n_mark_memset, 1, memory_order_relaxed);
    }
    return err;
}

/* ================================================================== */
/* Kernel launch hooks                                                 */
/* ================================================================== */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    scan_kernel_params(args);
    atomic_fetch_add_explicit(&g_n_mark_kernel, 1, memory_order_relaxed);
    return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gdx, unsigned int gdy, unsigned int gdz,
                        unsigned int bdx, unsigned int bdy, unsigned int bdz,
                        unsigned int smem, CUstream hStream,
                        void **kernelParams, void **extra)
{
    ENSURE_SYMBOLS;
    scan_kernel_params_exact(f, kernelParams);
    atomic_fetch_add_explicit(&g_n_mark_kernel, 1, memory_order_relaxed);
    return real_cuLaunchKernel(f, gdx, gdy, gdz, bdx, bdy, bdz,
                               smem, hStream, kernelParams, extra);
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra)
{
    ENSURE_SYMBOLS;
    scan_kernel_params_exact(f, kernelParams);
    atomic_fetch_add_explicit(&g_n_mark_kernel, 1, memory_order_relaxed);
    return real_cuLaunchKernelEx(config, f, kernelParams, extra);
}

/* ================================================================== */
/* cuBLAS precise hooks                                                */
/* ================================================================== */
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void *alpha,
                            const void *A, cudaDataType_t Atype, int lda,
                            const void *B, cudaDataType_t Btype, int ldb,
                            const void *beta,
                            void *C, cudaDataType_t Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo)
{
    ENSURE_SYMBOLS;
    cublasStatus_t s = real_cublasGemmEx(handle, transa, transb, m, n, k,
                                         alpha, A, Atype, lda,
                                         B, Btype, ldb, beta,
                                         C, Ctype, ldc, computeType, algo);
    if (s == CUBLAS_STATUS_SUCCESS) {
        mark_dirty_va((uint64_t)(uintptr_t)C);
        atomic_fetch_add_explicit(&g_n_mark_cublas, 1, memory_order_relaxed);
    }
    return s;
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                              cublasLtMatmulDesc_t computeDesc,
                              const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc,
                              const void *B, cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              const void *C, cublasLtMatrixLayout_t Cdesc,
                              void *D, cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t *algo,
                              void *workspace, size_t workspaceSizeInBytes,
                              cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    cublasStatus_t s = real_cublasLtMatmul(lightHandle, computeDesc, alpha,
                                           A, Adesc, B, Bdesc, beta,
                                           C, Cdesc, D, Ddesc, algo,
                                           workspace, workspaceSizeInBytes, stream);
    if (s == CUBLAS_STATUS_SUCCESS) {
        mark_dirty_va((uint64_t)(uintptr_t)D);
        atomic_fetch_add_explicit(&g_n_mark_cublas, 1, memory_order_relaxed);
    }
    return s;
}

/* ================================================================== */
/* Checkpoint protocol (4-phase recopy)                                */
/* ================================================================== */
static volatile int    g_ckpt_requested = 0;
static volatile int    g_ckpt_phase     = 0;   /* 0=idle 1=bg running 2=bg-done */
static int             g_ckpt_delay_s   = 0;   /* seconds before round 0 */
static int             g_ckpt_interval_ms = 0; /* ms between rounds (0 = single-shot) */
static int             g_ckpt_rounds_max  = 1; /* stop after this many rounds */
static int             g_wait_signal      = 0; /* 1 = defer round 0 until SIGUSR1 */
static _Atomic int     g_start_signaled   = 0; /* set by sigusr1_start when SIGUSR1 received */
static double          g_first_sync_ms  = 0;
static volatile int    g_first_sync_seen= 0;
static pthread_t       g_bg_thread;

/* Multi-round state. g_ckpt_in_flight / g_ckpt_mu / g_ckpt_cv are
 * forward-declared near wait_until_ckpt_idle() above (not static) so the
 * free hooks earlier in the file can reach them. */
static _Atomic uint64_t  g_round_id        = 0;
       _Atomic int       g_ckpt_in_flight  = 0;
       pthread_mutex_t   g_ckpt_mu = PTHREAD_MUTEX_INITIALIZER;
       pthread_cond_t    g_ckpt_cv = PTHREAD_COND_INITIALIZER;
static double            g_last_phase4_done_ms = 0;

/* Snapshot of live allocs taken at Phase 1 (so we copy a consistent set). */
typedef struct { uint64_t va; uint64_t size; uint32_t idx; } snap_ent_t;
static snap_ent_t     *g_snap        = NULL;
static uint32_t        g_snap_n      = 0;
static uint64_t        g_snap_max_sz = 0;

/* Phase 2 metrics (populated by background thread) */
static double   g_p2_wait_ms    = 0, g_p2_enc_ms = 0, g_p2_wb_ms = 0;
static double   g_p2_d2h_gpu_ms = 0, g_p2_launch_ms = 0;
static uint64_t g_p2_bytes      = 0;
static double   g_p2_wall_ms    = 0;

/* Staging buffer size — read from PHOS_STAGING_MB in phos_init.
 * Default 4 GB matches gcr's CKPT_STAGING_MB. Two pinned buffers
 * of this size are allocated lazily by the phase functions for
 * ping-pong (one fills via cudaMemcpyAsync while the other is
 * encrypted + written back). */
static uint64_t g_staging_want = 4096ULL * 1024 * 1024;

static uint8_t  g_k3_key[CKPT_CRYPTO_KEY_SIZE];

/* Open one image-backing shm file (read-write, MAP_SHARED). Returns
 * pointer + sets *size_out on success, NULL on missing file. */
static uint8_t *open_image_shm(const char *shm_name, uint64_t *size_out,
                                const char *label)
{
    int fd = shm_open(shm_name, O_RDWR, 0);
    if (fd < 0) {
        fprintf(stderr, "[phos] image-%s: shm_open(/dev/shm%s) failed: %s\n"
                        "[phos] image-%s: writeback to this image will be SKIPPED\n",
                label, shm_name, strerror(errno), label);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size <= 0) {
        fprintf(stderr, "[phos] image-%s: fstat(%s) failed or empty: %s\n",
                label, shm_name, strerror(errno));
        close(fd);
        return NULL;
    }
    uint8_t *p = (uint8_t *)mmap(NULL, (size_t)st.st_size,
                                  PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[phos] image-%s: mmap(%s, %.1f GB) failed: %s\n",
                label, shm_name,
                st.st_size / (1024.0 * 1024.0 * 1024.0), strerror(errno));
        return NULL;
    }
    *size_out = (uint64_t)st.st_size;
    fprintf(stderr, "[phos] image-%s: /dev/shm%s (%.1f GB, MAP_SHARED)\n",
            label, shm_name, st.st_size / (1024.0 * 1024.0 * 1024.0));
    return p;
}

/* ------------------------------------------------------------------ */
/* Per-round metrics for end-of-run CSV                                */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t round;
    uint64_t p2_bytes;
    double   p2_wait_ms;      /* CPU blocked on event sync (was: copy_ms) */
    double   p2_d2h_gpu_ms;   /* GPU's own elapsed D2H time */
    double   p2_launch_ms;    /* CPU time inside cudaMemcpyAsync calls */
    double   p2_enc_ms;
    double   p2_wb_ms;        /* writeback into image_mem */
    double   p2_wall_ms;      /* pipelined wall */
    uint64_t p4_bytes;
    double   p4_wait_ms;
    double   p4_d2h_gpu_ms;
    double   p4_launch_ms;
    double   p4_enc_ms;
    double   p4_wb_ms;
    double   p4_wall_ms;
    double   stall_ms;
} phos_metric_t;
#define PHOS_MAX_METRIC_ROUNDS 1024
static phos_metric_t  g_metrics[PHOS_MAX_METRIC_ROUNDS];
static int            g_n_metrics = 0;
static pthread_mutex_t g_metrics_mu = PTHREAD_MUTEX_INITIALIZER;
/* One-shot guard: the summary is dumped explicitly when the final round
 * completes (vLLM v1 SIGKILLs the engine-core worker at shutdown, so the
 * atexit() fallback never runs in that process). This flag makes the
 * explicit call + the atexit fallback idempotent. */
static atomic_flag    g_metrics_dumped = ATOMIC_FLAG_INIT;

static void metrics_record_round(uint64_t round,
                                 uint64_t p2_b, double p2_wait, double p2_gpu,
                                 double p2_launch, double p2_e,
                                 double p2_wb, double p2_wall,
                                 uint64_t p4_b, double p4_wait, double p4_gpu,
                                 double p4_launch, double p4_e,
                                 double p4_wb, double p4_wall,
                                 double stall_ms)
{
    pthread_mutex_lock(&g_metrics_mu);
    if (g_n_metrics < PHOS_MAX_METRIC_ROUNDS) {
        phos_metric_t *m = &g_metrics[g_n_metrics++];
        m->round         = round;
        m->p2_bytes      = p2_b;
        m->p2_wait_ms    = p2_wait;
        m->p2_d2h_gpu_ms = p2_gpu;
        m->p2_launch_ms  = p2_launch;
        m->p2_enc_ms     = p2_e;
        m->p2_wb_ms      = p2_wb;
        m->p2_wall_ms    = p2_wall;
        m->p4_bytes      = p4_b;
        m->p4_wait_ms    = p4_wait;
        m->p4_d2h_gpu_ms = p4_gpu;
        m->p4_launch_ms  = p4_launch;
        m->p4_enc_ms     = p4_e;
        m->p4_wb_ms      = p4_wb;
        m->p4_wall_ms    = p4_wall;
        m->stall_ms      = stall_ms;
    }
    pthread_mutex_unlock(&g_metrics_mu);
}

static void metrics_dump_csv(void)
{
    if (g_n_metrics == 0) return;
    if (atomic_flag_test_and_set(&g_metrics_dumped)) return;  /* already dumped */
    fprintf(stderr, "\n=== phos per-round metrics (CSV) ===\n");
    fprintf(stderr,
        "shim,round,"
        "p2_bytes,p2_mb,p2_wait_ms,p2_d2h_gpu_ms,p2_launch_ms,"
        "p2_enc_ms,p2_wb_ms,p2_wall_ms,p2_wall_gbps,p2_d2h_gpu_gbps,"
        "p4_bytes,p4_mb,p4_wait_ms,p4_d2h_gpu_ms,p4_launch_ms,"
        "p4_enc_ms,p4_wb_ms,p4_wall_ms,"
        "stall_ms\n");
    double sum_p2_mb = 0, sum_p2_wall = 0, sum_p4_mb = 0, sum_stall = 0;
    for (int i = 0; i < g_n_metrics; i++) {
        phos_metric_t *m = &g_metrics[i];
        double p2_mb = m->p2_bytes / (1024.0 * 1024.0);
        double p4_mb = m->p4_bytes / (1024.0 * 1024.0);
        double p2_gbps_wall = (m->p2_wall_ms > 0)
            ? (m->p2_bytes / (1024.0*1024.0*1024.0)) / (m->p2_wall_ms / 1000.0)
            : 0.0;
        double p2_gbps_gpu = (m->p2_d2h_gpu_ms > 0)
            ? (m->p2_bytes / (1024.0*1024.0*1024.0)) / (m->p2_d2h_gpu_ms / 1000.0)
            : 0.0;
        fprintf(stderr,
                "phos,%lu,"
                "%lu,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,"
                "%lu,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
                "%.1f\n",
                (unsigned long)m->round,
                (unsigned long)m->p2_bytes, p2_mb,
                m->p2_wait_ms, m->p2_d2h_gpu_ms, m->p2_launch_ms,
                m->p2_enc_ms, m->p2_wb_ms, m->p2_wall_ms,
                p2_gbps_wall, p2_gbps_gpu,
                (unsigned long)m->p4_bytes, p4_mb,
                m->p4_wait_ms, m->p4_d2h_gpu_ms, m->p4_launch_ms,
                m->p4_enc_ms, m->p4_wb_ms, m->p4_wall_ms,
                m->stall_ms);
        sum_p2_mb   += p2_mb;
        sum_p2_wall += m->p2_wall_ms;
        sum_p4_mb   += p4_mb;
        sum_stall   += m->stall_ms;
    }
    fprintf(stderr, "\n=== phos aggregate (n=%d round(s)) ===\n", g_n_metrics);
    fprintf(stderr, "  avg p2 size  : %.2f MB\n", sum_p2_mb   / g_n_metrics);
    fprintf(stderr, "  avg p2 wall  : %.1f ms\n", sum_p2_wall / g_n_metrics);
    fprintf(stderr, "  avg p4 delta : %.2f MB\n", sum_p4_mb   / g_n_metrics);
    fprintf(stderr, "  avg stall    : %.1f ms\n", sum_stall   / g_n_metrics);
    fflush(stderr);
}

/* Allocate a host staging buffer. Prefers pinned memory (cudaMallocHost)
 * so D2H copies can DMA directly. Falls back to malloc if pinning fails
 * (large CUDA pinned allocations can be rejected by the kernel). */
static uint8_t *alloc_host_staging(size_t sz, int *was_pinned)
{
    void *p = NULL;
    if (real_cudaMallocHost && real_cudaMallocHost(&p, sz) == cudaSuccess) {
        *was_pinned = 1;
        return (uint8_t *)p;
    }
    *was_pinned = 0;
    return (uint8_t *)malloc(sz);
}

static void free_host_staging(uint8_t *p, int was_pinned)
{
    if (!p) return;
    if (was_pinned && real_cudaFreeHost) real_cudaFreeHost(p);
    else                                 free(p);
}

static void snapshot_live_allocs(void)
{
    pthread_rwlock_rdlock(&g_alloc_lock);
    uint32_t cap = g_num_allocs;
    g_snap = (snap_ent_t *)malloc(cap * sizeof(*g_snap));
    g_snap_n = 0;
    g_snap_max_sz = 0;
    for (uint32_t i = 0; i < cap; i++) {
        if (!g_allocs[i].alive) continue;
        g_snap[g_snap_n].va   = g_allocs[i].va;
        g_snap[g_snap_n].size = g_allocs[i].size;
        g_snap[g_snap_n].idx  = i;
        if (g_allocs[i].size > g_snap_max_sz) g_snap_max_sz = g_allocs[i].size;
        /* Clear dirty-this-round for buffers about to be pre-copied.
         * Any mark that comes in after this point means "dirtied during Phase 2". */
        atomic_store_explicit(&g_allocs[i].dirty_this_round, 0, memory_order_relaxed);
        g_snap_n++;
    }
    pthread_rwlock_unlock(&g_alloc_lock);
}

/* ------------------------------------------------------------------ */
/* Ping-pong pipeline helpers — used by both Phase 2 (concurrent) and  */
/* Phase 4 (stall recopy).                                              */
/*                                                                      */
/* Pattern: two staging buffer pairs ping-pong via CUDA events on a    */
/* single stream. While one slot is being encrypted + written back to  */
/* image_mem on the CPU, the other slot is being filled by GPU D2H.    */
/* Allocs larger than staging_size are chunked into staging-sized      */
/* pieces; chunk_t is the unit of work in the pipeline.                 */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t va;       /* device VA to D2H from                          */
    uint64_t size;     /* chunk bytes                                    */
    uint64_t img_off;  /* destination offset in image_mem (base or delta) */
} phos_chunk_t;

/* Build a chunk list from the snapshot. If dirty_only, skip allocs    */
/* whose dirty_this_round flag is 0 (Phase 4 filter). Allocs without a */
/* slot or already retired are skipped silently.                        */
static phos_chunk_t *build_chunks(const snap_ent_t *snaps, uint32_t n_snap,
                                   uint64_t staging_size, int dirty_only,
                                   uint32_t *n_out)
{
    *n_out = 0;
    uint32_t n_chunks = 0;
    for (uint32_t i = 0; i < n_snap; i++) {
        uint32_t ai = snaps[i].idx;
        if (dirty_only &&
            !atomic_load_explicit(&g_allocs[ai].dirty_this_round,
                                  memory_order_relaxed))
            continue;
        if (!g_allocs[ai].alive) continue;
        if (g_allocs[ai].img_offset == (uint64_t)-1) continue;
        uint64_t sz = snaps[i].size;
        n_chunks += (sz + staging_size - 1) / staging_size;
    }
    if (n_chunks == 0) return NULL;
    phos_chunk_t *chunks = (phos_chunk_t *)malloc(n_chunks * sizeof(*chunks));
    if (!chunks) return NULL;
    uint32_t ci = 0;
    for (uint32_t i = 0; i < n_snap; i++) {
        uint32_t ai = snaps[i].idx;
        if (dirty_only &&
            !atomic_load_explicit(&g_allocs[ai].dirty_this_round,
                                  memory_order_relaxed))
            continue;
        if (!g_allocs[ai].alive) continue;
        if (g_allocs[ai].img_offset == (uint64_t)-1) continue;
        uint64_t va = snaps[i].va;
        uint64_t sz = snaps[i].size;
        uint64_t img_off_base = g_allocs[ai].img_offset;
        uint64_t off = 0;
        while (off < sz) {
            uint64_t this_chunk = sz - off;
            if (this_chunk > staging_size) this_chunk = staging_size;
            chunks[ci].va      = va + off;
            chunks[ci].size    = this_chunk;
            chunks[ci].img_off = img_off_base + off;
            ci++;
            off += this_chunk;
        }
    }
    *n_out = ci;
    return chunks;
}

/* Producer-consumer pipeline: producer thread launches cudaMemcpyAsync   *
 * D2H, consumer thread does AES-GCM encrypt + writeback into image-mem.  *
 *                                                                        *
 * Why 2 threads: in CC mode cudaMemcpyAsync is effectively CPU-blocking  *
 * for ~160 ms/call (per-call CE setup tax), so the launch itself is the  *
 * bottleneck. A single-thread "ping-pong" loop serialises D2H + enc + wb *
 * because the next D2H launch holds the CPU. With a worker thread, the   *
 * consumer can encrypt+writeback chunk i during the producer's blocking  *
 * launch of chunk i+1, recovering the encrypt+wb time as wall savings.   */
typedef struct {
    uint8_t           *host_bufs[2];
    uint8_t           *enc_bufs[2];
    ckpt_page_meta_t  *metas[2];
    uint8_t           *target_image;

    cudaEvent_t        slot_start_evt[2];
    cudaEvent_t        slot_end_evt[2];
    uint64_t           slot_img_off[2];
    uint64_t           slot_size[2];
    uint32_t           slot_np[2];
    uint64_t           slot_iv[2];

    int                slot_ready[2];   /* 1 = filled, consumer may drain */
    int                producer_done;

    pthread_mutex_t    mu;
    pthread_cond_t     cv_ready;
    pthread_cond_t     cv_empty;

    double             tot_enc_ms;
    double             tot_wb_ms;
    double             tot_d2h_gpu_ms;
    uint64_t           tot_bytes;
} pingpong_ctx_t;

static void *pingpong_consumer(void *arg)
{
    pingpong_ctx_t *ctx = (pingpong_ctx_t *)arg;
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
                                       ctx->slot_start_evt[idx],
                                       ctx->slot_end_evt[idx]) == cudaSuccess) {
            ctx->tot_d2h_gpu_ms += (double)gpu_ms_f;
        }

        double t0 = now_ms();
        ckpt_crypto_encrypt_pages(g_k3_key,
                                   ctx->host_bufs[idx], ctx->enc_bufs[idx],
                                   ctx->metas[idx], ctx->slot_np[idx],
                                   CKPT_CRYPTO_PAGE_SIZE,
                                   ctx->slot_iv[idx], PHOS_ENC_THREADS);
        double t1 = now_ms();

        if (ctx->target_image) {
            memcpy(ctx->target_image + ctx->slot_img_off[idx],
                   ctx->enc_bufs[idx], (size_t)ctx->slot_size[idx]);
        }
        double t2 = now_ms();

        ctx->tot_enc_ms += t1 - t0;
        ctx->tot_wb_ms  += t2 - t1;
        ctx->tot_bytes  += ctx->slot_size[idx];

        pthread_mutex_lock(&ctx->mu);
        ctx->slot_ready[idx] = 0;
        pthread_cond_signal(&ctx->cv_empty);
        pthread_mutex_unlock(&ctx->mu);

        consumed++;
    }
    return NULL;
}

static int run_pingpong(const phos_chunk_t *chunks, uint32_t n_chunks,
                        uint64_t staging_size,
                        uint8_t  *host_bufs[2],
                        uint8_t  *enc_bufs[2],
                        ckpt_page_meta_t *metas[2],
                        cudaStream_t stream,
                        uint64_t iv_base,
                        uint8_t  *target_image,
                        double *wait_ms, double *enc_ms, double *wb_ms,
                        double *d2h_gpu_ms, double *launch_ms,
                        uint64_t *bytes_out)
{
    (void)staging_size;
    *wait_ms = 0; *enc_ms = 0; *wb_ms = 0;
    *d2h_gpu_ms = 0; *launch_ms = 0; *bytes_out = 0;
    if (n_chunks == 0 || !stream) return 0;

    cudaEvent_t starts[2] = { NULL, NULL };
    cudaEvent_t ends[2]   = { NULL, NULL };
    if (real_cudaEventCreate(&starts[0]) != cudaSuccess ||
        real_cudaEventCreate(&starts[1]) != cudaSuccess ||
        real_cudaEventCreate(&ends[0])   != cudaSuccess ||
        real_cudaEventCreate(&ends[1])   != cudaSuccess) {
        if (starts[0]) real_cudaEventDestroy(starts[0]);
        if (starts[1]) real_cudaEventDestroy(starts[1]);
        if (ends[0])   real_cudaEventDestroy(ends[0]);
        if (ends[1])   real_cudaEventDestroy(ends[1]);
        return -1;
    }

    pingpong_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.host_bufs[0] = host_bufs[0]; ctx.host_bufs[1] = host_bufs[1];
    ctx.enc_bufs[0]  = enc_bufs[0];  ctx.enc_bufs[1]  = enc_bufs[1];
    ctx.metas[0]     = metas[0];     ctx.metas[1]     = metas[1];
    ctx.target_image = target_image;
    ctx.slot_start_evt[0] = starts[0]; ctx.slot_start_evt[1] = starts[1];
    ctx.slot_end_evt[0]   = ends[0];   ctx.slot_end_evt[1]   = ends[1];
    pthread_mutex_init(&ctx.mu, NULL);
    pthread_cond_init(&ctx.cv_ready, NULL);
    pthread_cond_init(&ctx.cv_empty, NULL);

    pthread_t consumer_tid;
    if (pthread_create(&consumer_tid, NULL, pingpong_consumer, &ctx) != 0) {
        fprintf(stderr, "[phos] pipeline: failed to spawn consumer thread\n");
        real_cudaEventDestroy(starts[0]); real_cudaEventDestroy(starts[1]);
        real_cudaEventDestroy(ends[0]);   real_cudaEventDestroy(ends[1]);
        pthread_mutex_destroy(&ctx.mu);
        pthread_cond_destroy(&ctx.cv_ready);
        pthread_cond_destroy(&ctx.cv_empty);
        return -1;
    }

    uint64_t iv = iv_base;
    double tot_launch = 0, tot_wait = 0;

    for (uint32_t i = 0; i < n_chunks; i++) {
        int idx = (int)(i & 1);

        /* Wait for this slot to be released by the consumer. */
        pthread_mutex_lock(&ctx.mu);
        while (ctx.slot_ready[idx]) {
            pthread_cond_wait(&ctx.cv_empty, &ctx.mu);
        }
        pthread_mutex_unlock(&ctx.mu);

        real_cudaEventRecord(starts[idx], stream);
        double l0 = now_ms();
        cudaError_t err = real_cudaMemcpyAsync(host_bufs[idx],
                              (void *)(uintptr_t)chunks[i].va,
                              (size_t)chunks[i].size,
                              cudaMemcpyDeviceToHost, stream);
        double l1 = now_ms();
        if (err != cudaSuccess) {
            fprintf(stderr, "[phos] pipeline: D2H[%u] failed: %d\n", i, (int)err);
            break;
        }
        real_cudaEventRecord(ends[idx], stream);
        tot_launch += l1 - l0;

        /* Drain the event so the consumer can safely read elapsed time.
         * In CC the launch already blocked until the bytes were on host,
         * so this is near-immediate (sub-ms). */
        double w0 = now_ms();
        real_cudaEventSynchronize(ends[idx]);
        double w1 = now_ms();
        tot_wait += w1 - w0;

        uint32_t np = (uint32_t)((chunks[i].size + CKPT_CRYPTO_PAGE_SIZE - 1)
                                  / CKPT_CRYPTO_PAGE_SIZE);
        pthread_mutex_lock(&ctx.mu);
        ctx.slot_img_off[idx] = chunks[i].img_off;
        ctx.slot_size[idx]    = chunks[i].size;
        ctx.slot_np[idx]      = np;
        ctx.slot_iv[idx]      = iv;
        ctx.slot_ready[idx]   = 1;
        pthread_cond_signal(&ctx.cv_ready);
        pthread_mutex_unlock(&ctx.mu);

        iv += np;
    }

    pthread_mutex_lock(&ctx.mu);
    ctx.producer_done = 1;
    pthread_cond_broadcast(&ctx.cv_ready);
    pthread_mutex_unlock(&ctx.mu);

    pthread_join(consumer_tid, NULL);

    real_cudaEventDestroy(starts[0]); real_cudaEventDestroy(starts[1]);
    real_cudaEventDestroy(ends[0]);   real_cudaEventDestroy(ends[1]);
    pthread_mutex_destroy(&ctx.mu);
    pthread_cond_destroy(&ctx.cv_ready);
    pthread_cond_destroy(&ctx.cv_empty);

    *wait_ms    = tot_wait;
    *enc_ms     = ctx.tot_enc_ms;
    *wb_ms      = ctx.tot_wb_ms;
    *d2h_gpu_ms = ctx.tot_d2h_gpu_ms;
    *launch_ms  = tot_launch;
    *bytes_out  = ctx.tot_bytes;
    return 0;
}

/* Allocate two pinned host bufs + two enc bufs + two meta arrays      */
/* sized to staging_size. On failure, frees what was allocated and     */
/* returns -1. Caller owns the buffers and must call release_staging.  */
static int acquire_staging_pair(uint64_t staging_size,
                                uint8_t *host_bufs[2], int pinned[2],
                                uint8_t *enc_bufs[2],
                                ckpt_page_meta_t *metas[2])
{
    host_bufs[0] = alloc_host_staging((size_t)staging_size, &pinned[0]);
    host_bufs[1] = alloc_host_staging((size_t)staging_size, &pinned[1]);
    enc_bufs[0]  = (uint8_t *)malloc((size_t)staging_size);
    enc_bufs[1]  = (uint8_t *)malloc((size_t)staging_size);
    uint32_t max_pg = (uint32_t)((staging_size + CKPT_CRYPTO_PAGE_SIZE - 1)
                                  / CKPT_CRYPTO_PAGE_SIZE);
    metas[0] = (ckpt_page_meta_t *)malloc(max_pg * sizeof(ckpt_page_meta_t));
    metas[1] = (ckpt_page_meta_t *)malloc(max_pg * sizeof(ckpt_page_meta_t));
    if (!host_bufs[0] || !host_bufs[1] ||
        !enc_bufs[0]  || !enc_bufs[1]  ||
        !metas[0]     || !metas[1]) {
        free_host_staging(host_bufs[0], pinned[0]);
        free_host_staging(host_bufs[1], pinned[1]);
        free(enc_bufs[0]); free(enc_bufs[1]);
        free(metas[0]); free(metas[1]);
        return -1;
    }
    return 0;
}

static void release_staging_pair(uint8_t *host_bufs[2], int pinned[2],
                                  uint8_t *enc_bufs[2],
                                  ckpt_page_meta_t *metas[2])
{
    free_host_staging(host_bufs[0], pinned[0]);
    free_host_staging(host_bufs[1], pinned[1]);
    free(enc_bufs[0]); free(enc_bufs[1]);
    free(metas[0]); free(metas[1]);
}

static void *phase2_concurrent_copy(void *arg)
{
    (void)arg;

    uint64_t staging_size = g_staging_want;

    uint8_t  *host_bufs[2];
    uint8_t  *enc_bufs[2];
    ckpt_page_meta_t *metas[2];
    int       pinned[2];
    if (acquire_staging_pair(staging_size, host_bufs, pinned,
                              enc_bufs, metas) < 0) {
        fprintf(stderr, "[phos] Phase 2: staging alloc failed (want 2x %.1f GB)\n",
                staging_size / (1024.0*1024.0*1024.0));
        g_ckpt_phase = 2;
        return NULL;
    }

    cudaStream_t stream = NULL;
    if (real_cudaStreamCreate) real_cudaStreamCreate(&stream);

    /* Build chunk list (no dirty filter — Phase 2 captures the full
     * snapshot). */
    uint32_t n_chunks = 0;
    phos_chunk_t *chunks = build_chunks(g_snap, g_snap_n, staging_size,
                                         /*dirty_only=*/0, &n_chunks);

    /* Phase 2 is the full concurrent precopy — it ALWAYS targets
     * base.img, every round (base is overwritten each round). The
     * per-round stop-and-copy delta goes to delta.img in Phase 4.
     * MAP_SHARED so the bytes hit /dev/shm immediately. */
    uint8_t *target = g_image_base_mem;

    fprintf(stderr, "[phos] Phase 2: started — pinned=%s/%s, stream=%p, "
            "chunks=%u, target=%s\n",
            pinned[0] ? "yes" : "no", pinned[1] ? "yes" : "no",
            (void *)stream, n_chunks,
            target ? "base.img" : "(no image)");

    double t_wall0 = now_ms();
    double wait_ms = 0, ems = 0, wms = 0, gpu_ms = 0, launch_ms = 0;
    uint64_t bytes = 0;

    if (chunks && stream) {
        run_pingpong(chunks, n_chunks, staging_size,
                     host_bufs, enc_bufs, metas, stream,
                     /*iv_base=*/0, target,
                     &wait_ms, &ems, &wms,
                     &gpu_ms, &launch_ms,
                     &bytes);
    }

    double wall = now_ms() - t_wall0;

    g_p2_wait_ms    = wait_ms;
    g_p2_enc_ms     = ems;
    g_p2_wb_ms      = wms;
    g_p2_d2h_gpu_ms = gpu_ms;
    g_p2_launch_ms  = launch_ms;
    g_p2_bytes      = bytes;
    g_p2_wall_ms    = wall;

    free(chunks);
    if (stream && real_cudaStreamDestroy) real_cudaStreamDestroy(stream);
    release_staging_pair(host_bufs, pinned, enc_bufs, metas);

    __sync_synchronize();
    g_ckpt_phase = 2;
    fprintf(stderr, "[phos] Phase 2: done — d2h_gpu %.1f, launch %.1f, "
            "wait %.1f, enc %.1f, wb %.1f, wall %.1f ms, %.2f GB "
            "(%.2f GB/s effective, GPU %.2f GB/s)\n",
            gpu_ms, launch_ms, wait_ms, ems, wms, wall,
            bytes / (1024.0*1024.0*1024.0),
            wall    > 0 ? (bytes / (1024.0*1024.0*1024.0)) / (wall    / 1000.0) : 0.0,
            gpu_ms  > 0 ? (bytes / (1024.0*1024.0*1024.0)) / (gpu_ms  / 1000.0) : 0.0);
    return NULL;
}

/* Reset per-round dirty-marking counters so each round's breakdown is local. */
static void reset_round_counters(void)
{
    atomic_store(&g_n_mark_memcpy, 0);
    atomic_store(&g_n_mark_memset, 0);
    atomic_store(&g_n_mark_cublas, 0);
    atomic_store(&g_n_mark_kernel, 0);
    atomic_store(&g_n_scan_hits,   0);
    atomic_store(&g_n_scan_misses, 0);
}

static void do_phase1(void)
{
    uint64_t rid = atomic_load(&g_round_id);
    fprintf(stderr, "\n[phos] === Round %lu — PhoenixOS recopy protocol ===\n",
            (unsigned long)rid);
    fprintf(stderr, "[phos] Phase 1: quiesce + snapshot\n");

    atomic_store_explicit(&g_ckpt_in_flight, 1, memory_order_release);

    /* Round 0: fresh K3 key. Subsequent rounds reuse — rotation is downstream. */
    if (rid == 0) ckpt_crypto_gen_key(g_k3_key);

    snapshot_live_allocs();

    uint64_t tracked_b = 0;
    for (uint32_t i = 0; i < g_snap_n; i++) tracked_b += g_snap[i].size;
    fprintf(stderr, "[phos] tracked allocs: %u live, %.2f GB total\n",
            g_snap_n, tracked_b / (1024.0 * 1024.0 * 1024.0));

    g_ckpt_phase = 1;
    pthread_create(&g_bg_thread, NULL, phase2_concurrent_copy, NULL);
    fprintf(stderr, "[phos] Phase 1: done — app resumes, Phase 2 in background\n");
}

static void do_phase34(void)
{
    fprintf(stderr, "[phos] Phase 3: re-quiesce\n");
    pthread_join(g_bg_thread, NULL);

    /* Phase 4: STALL — recopy anything marked dirty during Phase 2 */
    fprintf(stderr, "[phos] Phase 4: recopy dirty buffers (APP STALLED)\n");

    uint64_t staging_size = g_staging_want;

    uint8_t  *host_bufs[2];
    uint8_t  *enc_bufs[2];
    ckpt_page_meta_t *metas[2];
    int       pinned[2];
    int       have_staging = (acquire_staging_pair(staging_size, host_bufs,
                                                    pinned, enc_bufs, metas) == 0);
    if (!have_staging) {
        fprintf(stderr, "[phos] Phase 4: staging alloc failed (want 2x %.1f GB)\n",
                staging_size / (1024.0*1024.0*1024.0));
    }

    cudaStream_t p4_stream = NULL;
    if (real_cudaStreamCreate) real_cudaStreamCreate(&p4_stream);

    /* Build chunk list — Phase 4 only re-copies allocs marked dirty
     * during Phase 2. */
    uint32_t n_chunks = 0;
    phos_chunk_t *chunks = NULL;
    if (have_staging) {
        chunks = build_chunks(g_snap, g_snap_n, staging_size,
                              /*dirty_only=*/1, &n_chunks);
    }

    /* Phase 4 is the stop-and-copy dirty recopy — it ALWAYS targets
     * delta.img, every round (including round 0). Phase 2 already
     * wrote the full precopy to base.img.                            */
    uint8_t *target = g_image_delta_mem;

    /* IV base offset to avoid reuse vs Phase 2's stream of pages.    */
    uint64_t iv_base = (uint64_t)g_snap_n * 4096;

    double stall0 = now_ms();
    double rc_wait = 0, rc_e = 0, rc_w = 0, rc_gpu = 0, rc_launch = 0;
    uint64_t rc_b = 0;

    if (chunks && p4_stream && have_staging) {
        run_pingpong(chunks, n_chunks, staging_size,
                     host_bufs, enc_bufs, metas, p4_stream,
                     iv_base, target,
                     &rc_wait, &rc_e, &rc_w,
                     &rc_gpu, &rc_launch,
                     &rc_b);
    }

    double stall_ms = now_ms() - stall0;
    double p4_wall_ms = stall_ms;

    if (p4_stream && real_cudaStreamDestroy) real_cudaStreamDestroy(p4_stream);

    /* Count dirty-alloc participation for the result print (one entry
     * per snap idx whose dirty bit was set; chunks may be many per
     * alloc, so n_chunks ≠ rc_n). */
    uint32_t rc_n = 0;
    for (uint32_t i = 0; i < g_snap_n; i++) {
        uint32_t ai = g_snap[i].idx;
        if (atomic_load_explicit(&g_allocs[ai].dirty_this_round,
                                 memory_order_relaxed) &&
            g_allocs[ai].alive) rc_n++;
    }

    fprintf(stderr, "\n[phos] === Checkpoint results ===\n");
    fprintf(stderr, "[phos] Phase 2 (concurrent, not stall):\n");
    fprintf(stderr, "[phos]   D2H (GPU)    : %8.1f ms  (%.2f GB/s on the GPU)\n",
            g_p2_d2h_gpu_ms,
            g_p2_d2h_gpu_ms > 0
                ? (g_p2_bytes / (1024.0*1024.0*1024.0)) / (g_p2_d2h_gpu_ms / 1000.0)
                : 0.0);
    fprintf(stderr, "[phos]   launch tax   : %8.1f ms  (CPU time inside cudaMemcpyAsync)\n",
            g_p2_launch_ms);
    fprintf(stderr, "[phos]   D2H wait     : %8.1f ms  (CPU blocked on event sync)\n",
            g_p2_wait_ms);
    fprintf(stderr, "[phos]   k3 encrypt   : %8.1f ms\n", g_p2_enc_ms);
    fprintf(stderr, "[phos]   writeback    : %8.1f ms\n", g_p2_wb_ms);
    fprintf(stderr, "[phos]   bytes        : %.2f GB\n",
            g_p2_bytes / (1024.0*1024.0*1024.0));
    fprintf(stderr, "[phos]   wall         : %8.1f ms  (%.2f GB/s effective)\n\n",
            g_p2_wall_ms,
            g_p2_wall_ms > 0
                ? (g_p2_bytes / (1024.0*1024.0*1024.0)) / (g_p2_wall_ms / 1000.0)
                : 0.0);

    fprintf(stderr, "[phos] Phase 4 (recopy, STALL):\n");
    fprintf(stderr, "[phos]   dirty bufs   : %u / %u (chunks=%u)\n",
            rc_n, g_snap_n, n_chunks);
    fprintf(stderr, "[phos]   D2H (GPU)    : %8.1f ms  (%.2f GB/s on the GPU)\n",
            rc_gpu,
            rc_gpu > 0 ? (rc_b / (1024.0*1024.0*1024.0)) / (rc_gpu / 1000.0) : 0.0);
    fprintf(stderr, "[phos]   launch tax   : %8.1f ms\n", rc_launch);
    fprintf(stderr, "[phos]   D2H wait     : %8.1f ms\n", rc_wait);
    fprintf(stderr, "[phos]   k3 encrypt   : %8.1f ms\n", rc_e);
    fprintf(stderr, "[phos]   writeback    : %8.1f ms\n", rc_w);
    fprintf(stderr, "[phos]   delta        : %.2f MB\n", rc_b / (1024.0*1024.0));
    fprintf(stderr, "[phos]   STALL TIME   : %8.1f ms  (delta total work)\n",
            stall_ms);

    /* Stash per-round metrics for end-of-run CSV (now includes GPU
     * D2H elapsed + launch tax columns alongside the CPU-side waits). */
    metrics_record_round(atomic_load(&g_round_id),
                         g_p2_bytes,
                         g_p2_wait_ms, g_p2_d2h_gpu_ms, g_p2_launch_ms,
                         g_p2_enc_ms, g_p2_wb_ms, g_p2_wall_ms,
                         rc_b,
                         rc_wait, rc_gpu, rc_launch,
                         rc_e, rc_w, p4_wall_ms,
                         stall_ms);

    fprintf(stderr, "\n[phos] Dirty-marking breakdown (this round):\n");
    fprintf(stderr, "[phos]   memcpy     : %lu\n", (unsigned long)atomic_load(&g_n_mark_memcpy));
    fprintf(stderr, "[phos]   memset     : %lu\n", (unsigned long)atomic_load(&g_n_mark_memset));
    fprintf(stderr, "[phos]   cuBLAS     : %lu\n", (unsigned long)atomic_load(&g_n_mark_cublas));
    fprintf(stderr, "[phos]   kernels    : %lu (scan hits %lu, misses %lu)\n",
            (unsigned long)atomic_load(&g_n_mark_kernel),
            (unsigned long)atomic_load(&g_n_scan_hits),
            (unsigned long)atomic_load(&g_n_scan_misses));
    fflush(stderr);

    free(chunks);
    if (have_staging) release_staging_pair(host_bufs, pinned, enc_bufs, metas);
    free(g_snap); g_snap = NULL;

    /* Advance to next round, release any cudaFree threads waiting on us,
     * reset per-round counters, mark idle so the next interval-trigger fires. */
    atomic_fetch_add(&g_round_id, 1);
    reset_round_counters();
    g_ckpt_phase     = 0;
    g_ckpt_requested = 0;
    g_last_phase4_done_ms = now_ms();

    /* Final round just finished — dump the summary now. vLLM v1 SIGKILLs
     * this engine-core worker at shutdown, so the atexit() fallback never
     * runs here; metrics_dump_csv() is one-shot guarded so this is safe. */
    if (atomic_load(&g_round_id) >= (uint64_t)g_ckpt_rounds_max) {
        fprintf(stderr, "\n[phos] final round (%d/%d) complete — metrics summary:\n",
                g_ckpt_rounds_max, g_ckpt_rounds_max);
        metrics_dump_csv();
    }

    pthread_mutex_lock(&g_ckpt_mu);
    atomic_store_explicit(&g_ckpt_in_flight, 0, memory_order_release);
    pthread_cond_broadcast(&g_ckpt_cv);
    pthread_mutex_unlock(&g_ckpt_mu);
}

/* Sync-boundary hook — drives the phase machine across multiple rounds. */
static void maybe_checkpoint(void)
{
    if (atomic_load(&g_round_id) >= (uint64_t)g_ckpt_rounds_max) return;

    if (!g_first_sync_seen) {
        g_first_sync_seen = 1;
        g_first_sync_ms   = now_ms();
        fprintf(stderr, "[phos] first sync at %.1f ms\n", g_first_sync_ms);
    }

    /* Round 0: SIGUSR1-driven (PHOS_CKPT_WAIT_SIGNAL=1) OR
     * PHOS_CKPT_DELAY seconds after first sync. */
    if (atomic_load(&g_round_id) == 0 && !g_ckpt_requested) {
        if (g_wait_signal) {
            if (atomic_load(&g_start_signaled)) {
                g_ckpt_requested = 1;
                fprintf(stderr, "[phos] signal-trigger round 0\n");
            }
        } else if (g_ckpt_delay_s > 0) {
            double el = (now_ms() - g_first_sync_ms) / 1000.0;
            if (el >= g_ckpt_delay_s) {
                g_ckpt_requested = 1;
                fprintf(stderr, "[phos] auto-trigger round 0 at %.1fs\n", el);
            }
        }
    }

    /* Subsequent rounds: PHOS_CKPT_INTERVAL_MS after previous Phase 4 done. */
    if (atomic_load(&g_round_id) > 0 && g_ckpt_interval_ms > 0
        && !g_ckpt_requested && g_ckpt_phase == 0) {
        double el_ms = now_ms() - g_last_phase4_done_ms;
        if (el_ms >= (double)g_ckpt_interval_ms) {
            g_ckpt_requested = 1;
            fprintf(stderr, "[phos] auto-trigger round %lu (%.1f ms since prev)\n",
                    (unsigned long)atomic_load(&g_round_id), el_ms);
        }
    }

    if (g_ckpt_requested && g_ckpt_phase == 0) {
        do_phase1();
    } else if (g_ckpt_phase == 2) {
        do_phase34();
    }
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    cudaError_t err = real_cudaStreamSynchronize(stream);
    maybe_checkpoint();
    return err;
}

cudaError_t cudaDeviceSynchronize(void)
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
    if (!atomic_load(&g_start_signaled)) {
        atomic_store(&g_start_signaled, 1);
        fprintf(stderr, "[phos] SIGUSR1 — start ckpt loop "
                "(round 0 fires at next sync)\n");
    }
}

static void sigusr2_trigger(int sig)
{
    (void)sig;
    if (!g_ckpt_requested && g_ckpt_phase == 0
        && atomic_load(&g_round_id) < (uint64_t)g_ckpt_rounds_max) {
        g_ckpt_requested = 1;
        fprintf(stderr, "[phos] SIGUSR2 — round %lu will fire at next sync\n",
                (unsigned long)atomic_load(&g_round_id));
    }
}

static void preload_one(const char *label, const char *const names[])
{
    for (size_t i = 0; names[i]; i++) {
        if (dlopen(names[i], RTLD_LAZY | RTLD_GLOBAL)) {
            fprintf(stderr, "[phos] preloaded %s via %s\n", label, names[i]);
            return;
        }
    }
    fprintf(stderr, "[phos] warning: failed to preload %s\n", label);
}

/* Refresh g_self_pid after fork — see comment at the call site below. */
static void refresh_pid_after_fork(void)
{
    g_self_pid = getpid();
    fprintf(stderr, "[phos] post-fork: refreshed g_self_pid = %d\n",
            (int)g_self_pid);
}

__attribute__((constructor))
static void phos_init(void)
{
    static const char *const cuda_names[]     = { "libcuda.so.1", "libcuda.so", NULL };
    static const char *const cudart_names[]   = { "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", NULL };
    static const char *const cublas_names[]   = { "libcublas.so.12", "libcublas.so.11", "libcublas.so", NULL };
    static const char *const cublaslt_names[] = { "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so", NULL };
    preload_one("libcuda",     cuda_names);
    preload_one("libcudart",   cudart_names);
    preload_one("libcublas",   cublas_names);
    preload_one("libcublasLt", cublaslt_names);

    g_self_pid = getpid();
    /* vLLM v1 forks an engine subprocess after the LD_PRELOAD constructor
     * runs. Without this atfork handler, g_self_pid retains the parent's
     * pid, and process_vm_readv(parent_pid, ...) from the child fails with
     * EPERM (cross-process read needs PTRACE_MODE_ATTACH_REALCREDS).
     * Result: every safe_read_qword bails, kernel-arg scan no-ops silently.
     * Symptom: scan=on but scan hits 0 / misses 0; phos misses KV cache. */
    pthread_atfork(NULL, NULL, refresh_pid_after_fork);
    resolve_cuFuncGetParamInfo();

    const char *d = getenv("PHOS_CKPT_DELAY");
    if (d) g_ckpt_delay_s = atoi(d);

    const char *iv = getenv("PHOS_CKPT_INTERVAL_MS");
    if (iv) g_ckpt_interval_ms = atoi(iv);

    const char *nr = getenv("PHOS_CKPT_ROUNDS");
    if (nr) {
        int v = atoi(nr);
        if (v > 0) g_ckpt_rounds_max = v;
    }

    const char *s = getenv("PHOS_SCAN_KERNELS");
    if (s) g_scan_enabled = atoi(s);

    const char *sd = getenv("PHOS_SCAN_DEPTH");
    if (sd) {
        int v = atoi(sd);
        if (v > 0 && v <= PHOS_MAX_KERNEL_PARAMS) g_scan_depth = v;
    }

    const char *ws = getenv("PHOS_CKPT_WAIT_SIGNAL");
    g_wait_signal = (ws && atoi(ws) > 0) ? 1 : 0;
    if (g_wait_signal) {
        signal(SIGUSR1, sigusr1_start);
        fprintf(stderr, "[phos] wait-signal mode — round 0 deferred until SIGUSR1\n");
    }

    /* Staging size for Phase 2 / Phase 4 ping-pong buffers. Default
     * 4 GB matches gcr's CKPT_STAGING_MB. Two pinned host bufs + two
     * enc bufs of this size are allocated lazily per phase. */
    const char *sm = getenv("PHOS_STAGING_MB");
    if (sm) {
        long v = atol(sm);
        if (v > 0) g_staging_want = (uint64_t)v * 1024ULL * 1024ULL;
    }

    /* Image backing: every round, Phase 2 overwrites base.img (full
     * precopy) and Phase 4 overwrites delta.img (stop-and-copy). Both
     * must be pre-allocated >= full footprint by ckpt_prealloc (shared
     * with v3 — delta.img must be large enough for the buffer-granular
     * dirty set, ~full footprint, not v3's page-granular delta size).
     * Override names via PHOS_IMAGE_BASE_SHM / PHOS_IMAGE_DELTA_SHM. */
    const char *base_name  = getenv("PHOS_IMAGE_BASE_SHM");
    if (!base_name)  base_name  = "/ckpt_base.img";
    const char *delta_name = getenv("PHOS_IMAGE_DELTA_SHM");
    if (!delta_name) delta_name = "/ckpt_delta.img";
    g_image_base_mem  = open_image_shm(base_name,  &g_image_base_size,  "base");
    g_image_delta_mem = open_image_shm(delta_name, &g_image_delta_size, "delta");

    signal(SIGUSR2, sigusr2_trigger);
    atexit(metrics_dump_csv);

    fprintf(stderr, "[phos] libphos_intercept loaded "
            "(pid=%d, delay=%ds, interval=%dms, rounds=%d, "
            "scan=%s depth=%d, trigger=SIGUSR2)\n",
            (int)getpid(), g_ckpt_delay_s, g_ckpt_interval_ms,
            g_ckpt_rounds_max,
            g_scan_enabled ? "on" : "off", g_scan_depth);
}
