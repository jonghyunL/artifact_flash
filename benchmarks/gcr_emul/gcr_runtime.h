/*
 * gcr_runtime.h — shared declarations between libgcr_intercept.cpp and
 *                 dirty_templates.cpp / dirty_templates_kernels.cpp.
 *
 * Provides:
 *   - real_cuStreamCreate / real_cuStreamSynchronize /
 *     real_cuMemcpyDtoHAsync_v2: function pointers populated once at shim
 *     init and consumed by the templates (they DtoH-copy small metadata
 *     arrays like block_mapping).
 *   - CU_CHECK macro: abort-on-error helper for the templates.
 *   - add_and_merge_dirty_address: the byte-range-set insert entry point
 *     used by every template (thread-safe wrapper in dirty_templates.cpp).
 *   - gcr_snapshot_dirty_set / gcr_clear_dirty_set / gcr_lookup_template:
 *     C-linkage handles for the intercept .cpp file to drive the
 *     checkpoint phases and dispatch the right template per kernel name.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Driver-API function-pointer typedefs (populated by intercept init) */
/* ------------------------------------------------------------------ */
typedef CUresult (*cuStreamCreate_fn)(CUstream *phStream, unsigned int Flags);
typedef CUresult (*cuStreamSynchronize_fn)(CUstream hStream);
typedef CUresult (*cuMemcpyDtoHAsync_v2_fn)(void *dstHost, CUdeviceptr srcDevice,
                                            size_t ByteCount, CUstream hStream);

#ifdef __cplusplus
extern "C" {
#endif

extern cuStreamCreate_fn          real_cuStreamCreate;
extern cuStreamSynchronize_fn     real_cuStreamSynchronize;
extern cuMemcpyDtoHAsync_v2_fn    real_cuMemcpyDtoHAsync_v2;

/* ------------------------------------------------------------------ */
/* Byte-range dirty-set API (defined in dirty_templates.cpp)          */
/* ------------------------------------------------------------------ */

/* Called by every template and by the hooks (memcpy/cublas/fallback). */
void add_and_merge_dirty_address(void *start_addr, void *end_addr);

typedef struct {
    uint64_t start;
    uint64_t end;      /* exclusive */
} gcr_range_t;

/* Copy the current dirty set into `out` (up to `max` entries).
 * On return: *total_bytes is the sum of (end-start) across all emitted ranges.
 * Returns the number of ranges written. Does not clear the set. */
uint32_t gcr_snapshot_dirty_set(gcr_range_t *out, uint32_t max, uint64_t *total_bytes);

/* Empty the dirty set. Called between rounds. */
void gcr_clear_dirty_set(void);

/* Number of ranges currently in the set. */
uint32_t gcr_dirty_set_count(void);

/* ------------------------------------------------------------------ */
/* Template dispatch                                                   */
/* ------------------------------------------------------------------ */

/* Signature matches the generated templates in dirty_templates_kernels.cpp. */
typedef void (*gcr_template_fn_t)(CUfunction f,
                                  unsigned int gridDimX, unsigned int gridDimY,
                                  unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY,
                                  unsigned int blockDimZ,
                                  unsigned int sharedMemBytes,
                                  CUstream hStream,
                                  void **kernelParams, void **extra);

/* Look up a template by the mangled kernel name (e.g. the Itanium-mangled
 * symbol returned by cuFuncGetName or __cudaRegisterFunction's deviceName).
 * Returns NULL if no template is registered for that kernel. */
gcr_template_fn_t gcr_lookup_template(const char *mangled_name);

#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ */
/* CU_CHECK — used by templates only; abort on any driver error.       */
/* Keep this header-only so the templates compile unchanged from GCR.  */
/* ------------------------------------------------------------------ */
#ifndef CU_CHECK
#define CU_CHECK(call)                                                        \
    do {                                                                      \
        CUresult _cu_r_ = (call);                                             \
        if (_cu_r_ != CUDA_SUCCESS) {                                         \
            fprintf(stderr,                                                   \
                    "[gcr] CU_CHECK failed: %s = %d at %s:%d\n",              \
                    #call, (int)_cu_r_, __FILE__, __LINE__);                  \
            abort();                                                          \
        }                                                                     \
    } while (0)
#endif
