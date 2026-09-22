/*
 * libbaseline_uvm.c — LD_PRELOAD shim that swaps cudaMalloc / cudaMallocAsync
 * / cuMemAlloc_v2 to their cudaMallocManaged / cuMemAllocManaged equivalents,
 * applies SetPreferredLocation(GPU) + prefetch-to-GPU. No checkpointing, no
 * gate, no alloc table, no restore.
 *
 * Purpose: isolate the cost of running vLLM on UVM substrate vs the cost of
 * checkpointing on top of it.
 *
 * Build:  gcc -shared -fPIC -O2 -o libbaseline_uvm.so libbaseline_uvm.c \
 *             -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
 *             -lcudart -lcuda -ldl
 * Use:    LD_PRELOAD=./libbaseline_uvm.so python -m vllm.entrypoints.openai.api_server ...
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

typedef cudaError_t (*cudaMallocManaged_fn)(void **, size_t, unsigned int);
typedef cudaError_t (*cudaMalloc_fn)(void **, size_t);
typedef cudaError_t (*cudaMallocAsync_fn)(void **, size_t, cudaStream_t);
typedef cudaError_t (*cudaFree_fn)(void *);
typedef cudaError_t (*cudaFreeAsync_fn)(void *, cudaStream_t);
typedef cudaError_t (*cudaMemAdvise_fn)(const void *, size_t,
                                        enum cudaMemoryAdvise, int);
typedef cudaError_t (*cudaMemPrefetchAsync_fn)(const void *, size_t, int,
                                               cudaStream_t);
typedef CUresult    (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult    (*cuMemAllocManaged_fn)(CUdeviceptr *, size_t, unsigned int);
typedef CUresult    (*cuMemFree_v2_fn)(CUdeviceptr);

static cudaMallocManaged_fn    real_cudaMallocManaged    = NULL;
static cudaMalloc_fn           real_cudaMalloc           = NULL;
static cudaMallocAsync_fn      real_cudaMallocAsync      = NULL;
static cudaFree_fn             real_cudaFree             = NULL;
static cudaFreeAsync_fn        real_cudaFreeAsync        = NULL;
static cudaMemAdvise_fn        real_cudaMemAdvise        = NULL;
static cudaMemPrefetchAsync_fn real_cudaMemPrefetchAsync = NULL;
static cuMemAlloc_v2_fn        real_cuMemAlloc_v2        = NULL;
static cuMemAllocManaged_fn    real_cuMemAllocManaged    = NULL;
static cuMemFree_v2_fn         real_cuMemFree_v2         = NULL;

static pthread_once_t           g_once     = PTHREAD_ONCE_INIT;
static __thread int             g_in_us    = 0;

static int g_verbose = 0;

static void resolve_symbols(void)
{
    real_cudaMallocManaged    = dlsym(RTLD_NEXT, "cudaMallocManaged");
    real_cudaMalloc           = dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaMallocAsync      = dlsym(RTLD_NEXT, "cudaMallocAsync");
    real_cudaFree             = dlsym(RTLD_NEXT, "cudaFree");
    real_cudaFreeAsync        = dlsym(RTLD_NEXT, "cudaFreeAsync");
    real_cudaMemAdvise        = dlsym(RTLD_NEXT, "cudaMemAdvise");
    real_cudaMemPrefetchAsync = dlsym(RTLD_NEXT, "cudaMemPrefetchAsync");
    real_cuMemAlloc_v2        = dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    real_cuMemAllocManaged    = dlsym(RTLD_NEXT, "cuMemAllocManaged");
    real_cuMemFree_v2         = dlsym(RTLD_NEXT, "cuMemFree_v2");

    const char *v = getenv("BASELINE_UVM_VERBOSE");
    g_verbose = (v && v[0] == '1');

    if (g_verbose)
        fprintf(stderr, "[baseline_uvm] symbols resolved (managed=%p)\n",
                (void *)real_cudaMallocManaged);
}

#define ENSURE_SYMBOLS pthread_once(&g_once, resolve_symbols)

static void advise_and_prefetch(void *p, size_t n)
{
    if (real_cudaMemAdvise)
        real_cudaMemAdvise(p, n, cudaMemAdviseSetPreferredLocation, 0);
    if (real_cudaMemPrefetchAsync)
        real_cudaMemPrefetchAsync(p, n, 0, 0);
}

/* ------------------------------------------------------------------ */
/* cudaMalloc -> cudaMallocManaged + advise + prefetch                 */
/* ------------------------------------------------------------------ */
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    ENSURE_SYMBOLS;
    if (g_in_us || !real_cudaMallocManaged) {
        if (!real_cudaMalloc) return cudaErrorUnknown;
        return real_cudaMalloc(devPtr, size);
    }
    g_in_us = 1;
    cudaError_t err = real_cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);
    if (err == cudaSuccess)
        advise_and_prefetch(*devPtr, size);
    g_in_us = 0;
    if (g_verbose)
        fprintf(stderr, "[baseline_uvm] cudaMalloc(%zu) -> managed %p err=%d\n",
                size, devPtr ? *devPtr : NULL, err);
    return err;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    (void)stream;
    if (g_in_us || !real_cudaMallocManaged) {
        if (!real_cudaMallocAsync) return cudaErrorUnknown;
        return real_cudaMallocAsync(devPtr, size, stream);
    }
    g_in_us = 1;
    cudaError_t err = real_cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);
    if (err == cudaSuccess)
        advise_and_prefetch(*devPtr, size);
    g_in_us = 0;
    if (g_verbose)
        fprintf(stderr, "[baseline_uvm] cudaMallocAsync(%zu) -> managed %p err=%d\n",
                size, devPtr ? *devPtr : NULL, err);
    return err;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    ENSURE_SYMBOLS;
    if (g_in_us || !real_cuMemAllocManaged) {
        if (!real_cuMemAlloc_v2) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemAlloc_v2(dptr, bytesize);
    }
    g_in_us = 1;
    CUresult res = real_cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
    if (res == CUDA_SUCCESS)
        advise_and_prefetch((void *)(uintptr_t)*dptr, bytesize);
    g_in_us = 0;
    if (g_verbose)
        fprintf(stderr, "[baseline_uvm] cuMemAlloc_v2(%zu) -> managed 0x%llx res=%d\n",
                bytesize, dptr ? (unsigned long long)*dptr : 0ULL, res);
    return res;
}

/* ------------------------------------------------------------------ */
/* Frees: pass through directly (no gate to update)                    */
/* ------------------------------------------------------------------ */
cudaError_t cudaFree(void *devPtr)
{
    ENSURE_SYMBOLS;
    if (!real_cudaFree) return cudaErrorUnknown;
    return real_cudaFree(devPtr);
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    ENSURE_SYMBOLS;
    if (!real_cudaFreeAsync) {
        if (!real_cudaFree) return cudaErrorUnknown;
        return real_cudaFree(devPtr);
    }
    return real_cudaFreeAsync(devPtr, stream);
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    ENSURE_SYMBOLS;
    if (!real_cuMemFree_v2) return CUDA_ERROR_NOT_INITIALIZED;
    return real_cuMemFree_v2(dptr);
}

__attribute__((constructor))
static void libbaseline_uvm_init(void)
{
    /* Force-load libcudart / libcuda so RTLD_NEXT can resolve their symbols
     * at constructor time. Without this, dlsym would return NULL and the
     * pthread_once below would lock in those NULLs. */
    static const char *const cudart_names[] = {
        "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", NULL
    };
    static const char *const cuda_names[] = {
        "libcuda.so.1", "libcuda.so", NULL
    };
    for (int i = 0; cudart_names[i]; i++)
        if (dlopen(cudart_names[i], RTLD_NOW | RTLD_GLOBAL)) break;
    for (int i = 0; cuda_names[i]; i++)
        if (dlopen(cuda_names[i], RTLD_NOW | RTLD_GLOBAL)) break;

    ENSURE_SYMBOLS;
    fprintf(stderr, "[baseline_uvm] loaded (cudaMalloc/cudaMallocAsync/"
                    "cuMemAlloc_v2 -> managed + SetPreferredLocation=GPU + "
                    "prefetch). No checkpointing.\n");
}
