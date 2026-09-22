/*******************************************************************************
    Copyright (c) 2024 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM_LIVE_MIGRATION_H__
#define __UVM_LIVE_MIGRATION_H__

#include "uvm_types.h"
#include "uvm_forward_decl.h"

//
// Live Migration Internal Helper Functions
//
// These functions are called from within the UVM driver to support
// VM live migration functionality. They are not part of the public API.
//

// Record a write fault for dirty page tracking during live migration (64KB granularity).
// This function is called from the GPU fault handler when a write fault
// occurs on a page that has been marked read-only for migration tracking.
//
// Parameters:
//   va_block:   The VA block containing the faulted address
//   fault_addr: The virtual address that was written to
//
// This function is thread-safe and can be called from interrupt context.
void uvm_live_migration_record_write_fault(uvm_va_block_t *va_block, NvU64 fault_addr);

// Reset all migration tracking state.
// Called when starting a new migration phase.
void uvm_live_migration_reset_tracking(void);

// ---------------------------------------------------------------------------
// Checkpoint ioctls (106)
// ---------------------------------------------------------------------------

// Return {base, size} for every UVM_VA_RANGE_TYPE_MANAGED allocation across
// all VA spaces on the system.  Requires CAP_SYS_ADMIN.
NV_STATUS uvm_api_live_migration_get_va_ranges(
    UVM_LIVE_MIGRATION_GET_VA_RANGES_PARAMS *params,
    struct file *filp);


// 109 – READ_PAGES_RESIDENT
// Checkpoint a managed VA range with per-page residency separation.
// CPU pages captured via kmap(); GPU pages via copy engine.
// No migration performed – pages remain in place.
// Requires CAP_SYS_ADMIN; app must be quiesced at AT_BOUNDARY.
NV_STATUS uvm_api_live_migration_read_pages_resident(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_PARAMS *params,
    struct file *filp);


// 111 – READ_PAGES_RESIDENT_ENCRYPTED
// Same as 109 but returns GPU pages as CE-encrypted ciphertext with
// per-page crypto metadata (IV, auth_tag, key_version).
NV_STATUS uvm_api_live_migration_read_pages_resident_encrypted(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS *params,
    struct file *filp);

// 112 – DECRYPT_ENCRYPTED_PAGES
// Decryption service: decrypt k1-encrypted checkpoint pages, return plaintext.
NV_STATUS uvm_api_live_migration_decrypt_encrypted_pages(
    UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES_PARAMS *params,
    struct file *filp);

// 113 – READ_PAGES_RESIDENT_ENCRYPTED_V2
// Same as 111 but with double-buffered CE transfers for higher throughput.
// Two DMA buffers ping-pong: CE fills one while CPU copies the other to userspace.
NV_STATUS uvm_api_live_migration_read_pages_resident_encrypted_v2(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS *params,
    struct file *filp);

// 114 – READ_PAGES_RESIDENT_MT (multi-threaded, plaintext)
// Same semantics as ioctl 109 but parallelizes the GPU-page copy stage
// across N kthread workers, each owning a dedicated pair of DMA buffers,
// a CE channel (via uvm_push_begin), and a slice of the coalesced run list.
NV_STATUS uvm_api_live_migration_read_pages_resident_mt(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_MT_PARAMS *params,
    struct file *filp);

// 115 – READ_PAGES_RESIDENT_ENCRYPTED_MT (multi-threaded, ciphertext)
// Same semantics as ioctl 113 but with N kthread workers, each copying
// ciphertext + per-transfer crypto metadata directly to userspace.
NV_STATUS uvm_api_live_migration_read_pages_resident_encrypted_mt(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS *params,
    struct file *filp);

// 116 – READ_DIRTY_DELTA_ENCRYPTED_MT (Tier3-fused dirty-delta, ciphertext)
// Same MT ciphertext pipeline as 115 but internally filters to pages in
// dirty 2MB va_blocks, fusing GET_DIRTY_PAGES + the per-region 115 loop
// into one ioctl (setup tax paid once for the whole delta).
NV_STATUS uvm_api_live_migration_read_dirty_delta_encrypted_mt(
    UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT_PARAMS *params,
    struct file *filp);

// 117 - BENCH_DECOMPOSE (debug only)
// Instrumented fork of the 114/115 MT pipeline that reports per-stage timings
// (submit / IV+CSL / CE wait / CPU decrypt / output copy). Never used on the
// checkpoint path; exists so 114/115/116 stay uninstrumented.
NV_STATUS uvm_api_live_migration_bench_decompose(
    UVM_LIVE_MIGRATION_BENCH_DECOMPOSE_PARAMS *params,
    struct file *filp);

#endif // __UVM_LIVE_MIGRATION_H__
