/*
 * ckpt_baseline.h — In-app cudaMemcpyAsync baseline checkpoint (P1)
 */
#pragma once

#include <stdint.h>
#include "ckpt_gate.h"

typedef struct {
    double   elapsed_ms;
    uint64_t bytes;
    double   gbps;
} ckpt_baseline_result_t;

/*
 * Baseline checkpoint — in-app cudaMemcpyAsync with k3 AES-GCM 2 MB chunks.
 *
 *   stop_the_world = 1 (P1d): call cudaDeviceSynchronize up front; app must
 *     be quiesced. Used for stop-and-copy strawman.
 *   stop_the_world = 0 (P1e): skip cudaDeviceSynchronize; app keeps running
 *     concurrently. Dirty pages written during the copy are captured later
 *     via the delta pass.
 *   n_mstreams: P1f multi-stream count. 0 or 1 = current single-stream
 *     writer-pool design. 2..STAGING_RING_SLOTS = N cudaStreams with slot-
 *     to-stream binding (first-freed-slot semantics dispatch to the stream
 *     with available capacity). Single launcher thread regardless.
 *
 * Emits a v4-format image at out_path with version=CKPT_V4_VERSION.
 * Returns 0 on success, -1 on failure. Fills result_out if non-NULL.
 */
int ckpt_baseline_run(const char *out_path,
                      const ckpt_alloc_entry_t *allocs,
                      uint32_t n_allocs,
                      int stop_the_world,
                      int n_mstreams,
                      ckpt_baseline_result_t *result_out);

/*
 * Delta checkpoint — in-app cudaMemcpyAsync over a list of (va, size) dirty
 * ranges reported by kernel dirty tracking at 2 MB granularity. Same
 * pipeline as ckpt_baseline_run but over specific ranges; writes a v4
 * delta-format output image.
 *
 * Reuses the process-scoped k3 key and advances the same IV counter, so
 * ciphertext IVs never repeat across baseline + delta within one process.
 */
int ckpt_baseline_delta(const char *delta_path,
                        const ckpt_dirty_range_t *ranges,
                        uint32_t n_ranges,
                        ckpt_baseline_result_t *result_out);
