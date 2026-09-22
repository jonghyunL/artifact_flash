/**
 * libckpt_restore.h  –  Shared library for checkpoint memory reassembly
 *
 * Layer 1 of the two-layer restore architecture.
 * Reads base.img + delta.img (produced by test_inc_ckpt_agent) and returns
 * reassembled memory blocks as in-memory buffers.
 *
 * Used by:
 *   - ML Layer 2 adapters (Python via ctypes)
 *   - HPC Layer 2 adapters (C/C++ — link against libckpt_restore.so)
 *   - ckpt_raw_restore CLI tool (debugging/inspection)
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o libckpt_restore.so libckpt_restore.c \
 *       -lssl -lcrypto
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Public types                                                        */
/* ------------------------------------------------------------------ */

/* One reassembled memory block (one per contiguous allocation group) */
typedef struct {
    uint64_t  orig_va;   /* original virtual address from checkpoint   */
    uint64_t  size;      /* total bytes                                */
    uint8_t  *data;      /* heap-allocated buffer with reassembled content
                            caller must NOT free directly — use
                            ckpt_restore_free()                        */
} ckpt_mem_block_t;

/* Result from ckpt_restore_from_files() */
typedef struct {
    uint32_t         num_blocks;
    ckpt_mem_block_t *blocks;    /* array of num_blocks entries */
} ckpt_restore_result_t;

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/**
 * ckpt_restore_from_files — read base + delta images, return reassembled
 * memory blocks.
 *
 * @param base_path   Path to baseline image (ckpt_v2 format)
 * @param delta_path  Path to delta image (ckpt_inc_delta format)
 * @param result      Output: populated with reassembled blocks on success
 * @return            0 on success, -1 on error (messages printed to stderr)
 *
 * On success, result->blocks is heap-allocated and must be freed by calling
 * ckpt_restore_free().
 */
int ckpt_restore_from_files(const char *base_path,
                            const char *delta_path,
                            ckpt_restore_result_t *result);

/**
 * ckpt_restore_selective — read base + delta images, but only restore
 * specific VA ranges (e.g. individual KV cache blocks).
 *
 * Instead of allocating the full VA span, only allocates and fills
 * the requested ranges.  Pages outside the requested ranges are skipped.
 *
 * @param base_path       Path to baseline image
 * @param delta_path      Path to delta image
 * @param requested_vas   Array of VA addresses to restore
 * @param requested_sizes Array of sizes (bytes) for each VA
 * @param num_requests    Number of entries in requested_vas/requested_sizes
 * @param result          Output: one block per request, in same order
 * @return                0 on success, -1 on error
 */
int ckpt_restore_selective(const char *base_path,
                           const char *delta_path,
                           const uint64_t *requested_vas,
                           const uint64_t *requested_sizes,
                           uint32_t num_requests,
                           ckpt_restore_result_t *result);

/**
 * ckpt_restore_selective_encrypted — same as ckpt_restore_selective but
 * handles encrypted images (v2 with k3, v3 with k1+k3).
 */
int ckpt_restore_selective_encrypted(const char *base_path,
                                      const char *delta_path,
                                      const uint64_t *requested_vas,
                                      const uint64_t *requested_sizes,
                                      uint32_t num_requests,
                                      const char *k3_key_path,
                                      const char *uvm_dev,
                                      ckpt_restore_result_t *result);

/**
 * ckpt_restore_from_files_encrypted — read encrypted base + delta images,
 * decrypt, and return reassembled plaintext blocks.
 *
 * Automatically detects image version:
 *   v2 (no crypto_meta): plaintext — same as ckpt_restore_from_files
 *   v2 (with crypto_meta): k3-encrypted — decrypts with provided key
 *   v3: GPU pages k1-encrypted (decrypted via ioctl 112), CPU pages
 *       k3-encrypted (decrypted with provided key)
 *
 * @param base_path   Path to baseline image
 * @param delta_path  Path to delta image
 * @param k3_key_path Path to 32-byte k3 key file (NULL if no k3 encryption)
 * @param uvm_dev     Path to UVM device for ioctl 112 (NULL = "/dev/nvidia-uvm")
 * @param result      Output: plaintext blocks
 * @return            0 on success, -1 on error
 */
int ckpt_restore_from_files_encrypted(const char *base_path,
                                       const char *delta_path,
                                       const char *k3_key_path,
                                       const char *uvm_dev,
                                       ckpt_restore_result_t *result);

/**
 * ckpt_restore_free — release all memory owned by a restore result.
 */
void ckpt_restore_free(ckpt_restore_result_t *result);

#ifdef __cplusplus
}
#endif
