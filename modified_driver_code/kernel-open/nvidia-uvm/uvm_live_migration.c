/*******************************************************************************
    Live Migration Support for UVM - MINIMAL STARTER VERSION
    
    This is a bare-bones implementation to test that the ioctl routing works.
    Once this compiles and the test passes, you can add the actual logic.
*******************************************************************************/

#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_types.h"
#include "uvm_api.h"
#include "uvm_global.h"
#include "uvm_va_space.h"
#include "uvm_va_range.h"
#include "uvm_va_block.h"
#include "uvm_live_migration.h"
#include "uvm_gpu.h"
#include "uvm_mem.h"
#include "uvm_push.h"
#include "uvm_channel.h"
#include "uvm_hal.h"
#include "uvm_conf_computing.h"

#include <linux/kthread.h>
#include <linux/completion.h>
#include <linux/sched/mm.h>

// Simple global counter for proof-of-concept dirty page tracking
static atomic64_t g_migration_write_fault_count = ATOMIC64_INIT(0);
static atomic64_t g_fault_log_count = ATOMIC64_INIT(0);
#define FAULT_LOG_DETAIL_LIMIT  200   /* log first N faults with full detail */
#define FAULT_LOG_SUMMARY_INTERVAL 1000 /* then log every Nth fault */

// Called from fault handler when a write fault occurs during migration
// Tracks dirty pages at 2MB granularity
void uvm_live_migration_record_write_fault(uvm_va_block_t *va_block, NvU64 fault_addr)
{
    NvU64 count, log_count;

    // Unconditional probe — fires even if tracking disabled
    {
        static atomic_t rwf_probe = ATOMIC_INIT(0);
        int p = atomic_inc_return(&rwf_probe);
        if (p <= 10)
            printk(KERN_ERR "UVM RWF_PROBE: #%d addr=0x%llx block=%p tracking=%d\n",
                   p, fault_addr, va_block,
                   (va_block ? va_block->live_migration.migration_tracking_enabled : -1));
    }

    if (!va_block || !va_block->live_migration.migration_tracking_enabled)
        return;

    count = atomic64_inc_return(&g_migration_write_fault_count);
    log_count = atomic64_inc_return(&g_fault_log_count);

    // FAULT PROFILING: Log individual write faults (rate-limited)
    if (log_count <= FAULT_LOG_DETAIL_LIMIT) {
        printk(KERN_INFO "UVM FAULT_PROF: write_fault #%llu addr=0x%llx "
               "block=[0x%llx-0x%llx] block_dirty_before=%d\n",
               count, fault_addr,
               va_block->start, va_block->end,
               va_block->live_migration.dirty ? 1 : 0);
    } else if ((log_count % FAULT_LOG_SUMMARY_INTERVAL) == 0) {
        printk(KERN_INFO "UVM FAULT_PROF: write_fault summary total=%llu "
               "latest_addr=0x%llx block=[0x%llx-0x%llx]\n",
               count, fault_addr, va_block->start, va_block->end);
    }

    // Mark this 2MB page as dirty
    va_block->live_migration.dirty = true;
}

// Reset counter (called when starting new migration round)
void uvm_live_migration_reset_tracking(void)
{
    atomic64_set(&g_migration_write_fault_count, 0);
    atomic64_set(&g_fault_log_count, 0);
    printk(KERN_DEBUG "UVM LivMig: Fault counter reset\n");
}

// Helper function to prepare a single VA space for migration
static NV_STATUS prepare_va_space_for_migration(uvm_va_space_t *va_space)
{
    uvm_va_range_t *va_range;
    NV_STATUS status = NV_OK;
    int total_blocks = 0;
    int blocks_with_residents = 0;

    printk(KERN_DEBUG "UVM LivMig:\tPreparing VA space %p\n", va_space);

    // Lock VA space for modification
    // NOTE: We're operating from a different process context (migration agent),
    // so we cannot use uvm_va_space_mm_retain_lock() which checks process ownership.
    // For cross-process migration, we only modify GPU PTEs, not CPU page tables.
    // CPU page table coordination will happen naturally on page faults.
    uvm_va_space_down_write(va_space);

    // Iterate over all VA ranges in this VA space
    uvm_for_each_va_range(va_range, va_space) {
        uvm_va_block_t *va_block;
        uvm_va_block_context_t *va_block_context;

        // Only process managed ranges (skip external, channel ranges, etc.)
        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
            continue;

        printk(KERN_DEBUG "UVM LivMig:\t  Processing VA range [0x%llx, 0x%llx]\n",
               va_range->node.start, va_range->node.end);

        // Allocate block context without mm_struct since we're cross-process
        // This limits us to GPU-only operations, which is fine for migration dirty tracking
        va_block_context = uvm_va_block_context_alloc(NULL);
        if (!va_block_context) {
            status = NV_ERR_NO_MEMORY;
            goto unlock;
        }

        // Revoke write permissions on all blocks in this range
        for_each_va_block_in_va_range(va_range, va_block) {
            uvm_va_block_region_t region;
            uvm_va_block_retry_t va_block_retry;

            total_blocks++;

            printk(KERN_DEBUG "UVM LivMig PREPARE:\t    Block %d [0x%llx-0x%llx]: resident_mask=%lx\n",
                   total_blocks, va_block->start, va_block->end,
                   *va_block->resident.bitmap);

            region = uvm_va_block_region_from_block(va_block);

            // Enable migration tracking for this 2MB page
            va_block->live_migration.dirty = false;
            va_block->live_migration.migration_tracking_enabled = true;

            // Skip blocks with no resident pages (nothing to revoke)
            if (uvm_processor_mask_empty(&va_block->resident)) {
                printk(KERN_DEBUG "UVM LivMig PREPARE:\t      Block %d: No residents, tracking enabled but no revoke\n", total_blocks);
                continue;
            }

            blocks_with_residents++;
            printk(KERN_DEBUG "UVM LivMig PREPARE:\t      Block %d: Has residents, will revoke write permissions\n", total_blocks);
            
            // LIVE MIGRATION: Keep 2M PTEs intact for dirty tracking.
            // Revoking write on a 2M PTE makes the entire 2MB read-only.
            // A write to any address within triggers a fault at 2MB granularity.
            // This avoids the TLB pressure of splitting to 64KB PTEs.
            // (disable_2m_ptes stays false — 2M PTE merging allowed)
            
            printk(KERN_DEBUG "UVM LivMig PREPARE:\t      Block %d [0x%llx-0x%llx]: Calling revoke_prot_mask...\n",
                   total_blocks, va_block->start, va_block->end);

            // Revoke write (and atomic) permissions from all processors with residency.
            // Must use UVM_PROT_READ_WRITE (not READ_WRITE_ATOMIC) because
            // block_revoke_prot_gpu_to sets new_prot = prot_to_revoke - 1.
            // Passing ATOMIC only strips the ATOMIC bit, leaving READ_WRITE intact.
            // Passing READ_WRITE strips both WRITE and ATOMIC, leaving READ_ONLY,
            // which forces K1 write faults through service_finish for dirty tracking.
            status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                uvm_va_block_revoke_prot_mask(va_block,
                                              va_block_context,
                                              &va_block->resident,
                                              region,
                                              NULL,
                                              UVM_PROT_READ_WRITE));

            if (status != NV_OK) {
                printk(KERN_ERR "UVM LivMig:\t  Failed to revoke write on block, status=%d\n", status);
                uvm_va_block_context_free(va_block_context);
                goto unlock;
            }
            
            printk(KERN_DEBUG "UVM LivMig PREPARE:\t      Block %d [0x%llx-0x%llx]: Revoke completed successfully\n",
                   total_blocks, va_block->start, va_block->end);
        }

        uvm_va_block_context_free(va_block_context);
    }

    printk(KERN_DEBUG "UVM LivMig:\tVA space %p prepared: %d blocks (%d with residents)\n",
           va_space, total_blocks, blocks_with_residents);

    // Mark this VA space as actively tracked so that blocks created after
    // PREPARE_ALL (by first-fault on new pages) also get tracking enabled.
    if (status == NV_OK)
        va_space->migration_tracking_active = true;

unlock:
    uvm_va_space_up_write(va_space);

    return status;
}

// System-wide migration prepare - affects ALL processes with GPU memory
// This is called by a privileged migration agent to prepare the entire system
NV_STATUS uvm_api_live_migration_prepare_all(
    UVM_LIVE_MIGRATION_PREPARE_ALL_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space;
    NV_STATUS status = NV_OK;
    int va_space_count = 0;
    int va_spaces_prepared = 0;

    printk(KERN_DEBUG "=== UVM: System-wide live migration prepare ===\n");
    printk(KERN_DEBUG "UVM:   flags = 0x%x\n", params->flags);

    // Security check - only root can do system-wide migration
    if (!capable(CAP_SYS_ADMIN)) {
        printk(KERN_WARNING "UVM LivMig: System-wide migration requires CAP_SYS_ADMIN\n");
        status = NV_ERR_INSUFFICIENT_PERMISSIONS;
        goto out;
    }

    printk(KERN_DEBUG "UVM LivMig: Starting system-wide migration preparation\n");

    // Snapshot VA space pointers under the global lock, then release the lock
    // BEFORE processing any VA space.
    //
    // Why: prepare_va_space_for_migration() acquires the va_space write lock
    // and calls uvm_va_block_revoke_prot_mask(), which pushes CE ops and GPU
    // TLB invalidations.  The GPU fault handler must acquire the va_space read
    // lock to service those faults/replays.  If we hold g_uvm_global.va_spaces
    // .lock across the revoke, the fault handler can never progress, and the
    // revoke waits forever for the GPU -> deadlock that stalls the CUDA app.
    {
        uvm_va_space_t **snapshot = NULL;
        int i;

        uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
        list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node)
            va_space_count++;

        if (va_space_count > 0) {
            snapshot = kvmalloc_array(va_space_count, sizeof(*snapshot), GFP_KERNEL);
            if (!snapshot) {
                uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
                status = NV_ERR_NO_MEMORY;
                goto out;
            }
            i = 0;
            list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node)
                snapshot[i++] = va_space;
        }
        // Global lock released here; CE ops / fault handler are unblocked.
        uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

        for (i = 0; i < va_space_count; i++) {
            status = prepare_va_space_for_migration(snapshot[i]);
            if (status != NV_OK) {
                printk(KERN_ERR "UVM LivMig: Failed to prepare VA space %p (status=%d)\n",
                       snapshot[i], status);
                // Continue with other VA spaces even if one fails
                continue;
            }
            va_spaces_prepared++;
        }

        kvfree(snapshot);
    }

    printk(KERN_DEBUG "UVM LivMig: System-wide migration prepare complete\n");
    printk(KERN_DEBUG "UVM LivMig:   Total VA spaces: %d\n", va_space_count);
    printk(KERN_DEBUG "UVM LivMig:   Successfully prepared: %d\n", va_spaces_prepared);

    // Report success if at least one VA space was prepared
    if (va_spaces_prepared > 0)
        status = NV_OK;
    else if (va_space_count == 0)
        status = NV_WARN_NOTHING_TO_DO;

out:
    params->rmStatus = status;
    return status;
}

// Get dirty page addresses (64KB granularity)
NV_STATUS uvm_api_live_migration_get_dirty_pages(
    UVM_LIVE_MIGRATION_GET_DIRTY_PAGES_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space;
    uvm_va_range_t *va_range;
    NvU64 *user_buffer = params->dirty_addresses;
    NvU64 count = 0;
    NvU64 max = params->max_pages;
    NV_STATUS status = NV_OK;
    
    printk(KERN_DEBUG "UVM LivMig: GET_DIRTY_PAGES called (max=%llu)\n", max);
    
    // System-wide operation - iterate all VA spaces
    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
    
    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node) {
        int range_count = 0;
        printk(KERN_DEBUG "UVM LivMig:   Checking VA space %p\n", va_space);
        uvm_va_space_down_read(va_space);
        
        uvm_for_each_va_range(va_range, va_space) {
            uvm_va_block_t *va_block;
            
            range_count++;
            
            // Only process managed ranges
            if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
                printk(KERN_DEBUG "UVM LivMig:     VA range [0x%llx-0x%llx] type=%d (skipping non-managed)\n",
                       va_range->node.start, va_range->node.end, va_range->type);
                continue;
            }
            
            printk(KERN_DEBUG "UVM LivMig:     VA range [0x%llx-0x%llx] (managed)\n", 
                   va_range->node.start, va_range->node.end);
            
            // Iterate all blocks in this range (only populated blocks)
            for_each_va_block_in_va_range(va_range, va_block) {
                uvm_mutex_lock(&va_block->lock);

                // Check if migration tracking is enabled
                if (!va_block->live_migration.migration_tracking_enabled) {
                    uvm_mutex_unlock(&va_block->lock);
                    continue;
                }

                // Check if this 2MB page is dirty
                if (!va_block->live_migration.dirty) {
                    uvm_mutex_unlock(&va_block->lock);
                    continue;
                }

                if (count >= max) {
                    uvm_mutex_unlock(&va_block->lock);
                    goto done;
                }

                // Report the 2MB-aligned page address
                {
                    static atomic_t dirty_found_probe = ATOMIC_INIT(0);
                    int dfp = atomic_inc_return(&dirty_found_probe);
                    if (dfp <= 20)
                        printk(KERN_ERR "UVM DIRTY_FOUND: #%d block=[0x%llx-0x%llx] dirty=1\n",
                               dfp, va_block->start, va_block->end);
                }
                {
                    NvU64 page_addr = va_block->start;
                    if (copy_to_user(&user_buffer[count], &page_addr, sizeof(NvU64))) {
                        status = NV_ERR_INVALID_ADDRESS;
                        uvm_mutex_unlock(&va_block->lock);
                        goto done;
                    }
                    count++;
                }

                uvm_mutex_unlock(&va_block->lock);
            }
        }
        
        printk(KERN_DEBUG "UVM LivMig:   VA space %p had %d VA ranges\n", va_space, range_count);
        uvm_va_space_up_read(va_space);
    }
    
done:
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
    
    params->num_pages = count;
    params->rmStatus = status;
    
    printk(KERN_DEBUG "UVM LivMig: GET_DIRTY_PAGES returning %llu dirty pages (64KB each)\n", count);
    
    return status;
}

//
// Phase 1: GPU Page Reading Implementation
//

// Helper: Read GPU physical memory as encrypted ciphertext (CC mode).
// Forward declaration (defined after this function)
static NV_STATUS read_gpu_physical_memory(uvm_gpu_t *gpu,
                                          uvm_gpu_phys_address_t phys_addr,
                                          NvU64 size,
                                          void *dest_buffer);

// Returns the CE-encrypted data + per-page metadata without decrypting.
// For non-CC mode, falls back to read_gpu_physical_memory (plaintext).
static NV_STATUS read_gpu_physical_memory_encrypted(uvm_gpu_t *gpu,
                                                     uvm_gpu_phys_address_t phys_addr,
                                                     NvU64 size,
                                                     void *cipher_buffer,
                                                     UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *meta_out)
{
    uvm_gpu_address_t src_addr;
    NV_STATUS status;
    NvU8 auth_tag[UVM_CONF_COMPUTING_AUTH_TAG_SIZE];
    UvmCslIv decrypt_iv;
    NvU32 key_version;

    UVM_ASSERT(gpu);
    UVM_ASSERT(cipher_buffer);
    UVM_ASSERT(meta_out);
    UVM_ASSERT(size > 0);
    UVM_ASSERT(size <= UVM_CONF_COMPUTING_DMA_BUFFER_SIZE);

    if (!g_uvm_global.conf_computing_enabled) {
        // Non-CC: fall back to plaintext read, zero metadata
        memset(meta_out, 0, sizeof(*meta_out));
        return read_gpu_physical_memory(gpu, phys_addr, size, cipher_buffer);
    }

    src_addr = uvm_gpu_address_copy(gpu, phys_addr);

    status = uvm_conf_computing_util_memcopy_gpu_to_cpu_encrypted(
                gpu,
                cipher_buffer,
                auth_tag,
                &decrypt_iv,
                &key_version,
                NULL,           // don't need channel for checkpoint
                src_addr,
                size,
                NULL,
                "LivMig CC encrypted PA=0x%llx",
                phys_addr.address);

    if (status != NV_OK)
        return status;

    // Pack metadata into the output struct
    memcpy(meta_out->iv, decrypt_iv.iv, 12);
    meta_out->iv_fresh = decrypt_iv.fresh;
    memcpy(meta_out->auth_tag, auth_tag, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
    meta_out->key_version = key_version;

    return NV_OK;
}

// Helper: Read GPU physical memory using copy engine.
// In Confidential Computing (CC/HCC) mode the CE cannot perform a plain
// physical write from protected vidmem to unprotected sysmem; doing so
// triggers FAULT_INFO_TYPE_REGION_VIOLATION (Xid 31) and corrupts the GPU
// context of the target application.  The correct CC path uses
// uvm_conf_computing_util_memcopy_gpu_to_cpu, which internally:
//   1. CE-encrypts the vidmem page into an unprotected DMA buffer
//   2. CPU-decrypts that DMA buffer into the supplied kernel pointer
// The non-CC path keeps the original plain CE physical copy.
static NV_STATUS read_gpu_physical_memory(uvm_gpu_t *gpu,
                                          uvm_gpu_phys_address_t phys_addr,
                                          NvU64 size,
                                          void *dest_buffer)
{
    uvm_gpu_address_t src_addr;

    UVM_ASSERT(gpu);
    UVM_ASSERT(dest_buffer);
    UVM_ASSERT(size > 0);
    UVM_ASSERT(size <= UVM_CONF_COMPUTING_DMA_BUFFER_SIZE);

    src_addr = uvm_gpu_address_copy(gpu, phys_addr);

    printk(KERN_DEBUG "UVM LivMig: Reading %llu bytes from GPU PA 0x%llx (CC=%s)\n",
           size, phys_addr.address,
           g_uvm_global.conf_computing_enabled ? "yes" : "no");

    if (g_uvm_global.conf_computing_enabled) {
        // CC path: CE encrypts vidmem → unprotected DMA buffer, then CPU
        // decrypts directly into dest_buffer.  No plain write to sysmem.
        return uvm_conf_computing_util_memcopy_gpu_to_cpu(gpu,
                                                          dest_buffer,
                                                          src_addr,
                                                          size,
                                                          NULL,
                                                          "LivMig CC GPU-to-CPU PA=0x%llx",
                                                          phys_addr.address);
    }

    // Non-CC path: plain CE physical copy vidmem → sysmem staging buffer.
    {
        NV_STATUS status;
        uvm_push_t push;
        uvm_mem_t *staging_mem = NULL;
        void *staging_cpu_va;
        uvm_gpu_address_t dst_addr;

        status = uvm_mem_alloc_sysmem_and_map_cpu_kernel(size, current->mm, &staging_mem);
        if (status != NV_OK) {
            printk(KERN_ERR "UVM LivMig: Failed to allocate staging buffer: %s\n",
                   nvstatusToString(status));
            return status;
        }

        staging_cpu_va = uvm_mem_get_cpu_addr_kernel(staging_mem);
        if (!staging_cpu_va) {
            printk(KERN_ERR "UVM LivMig: Failed to get staging CPU VA\n");
            status = NV_ERR_INVALID_ADDRESS;
            goto noncc_cleanup;
        }

        status = uvm_mem_map_gpu_phys(staging_mem, gpu);
        if (status != NV_OK) {
            printk(KERN_ERR "UVM LivMig: Failed to map staging buffer to GPU: %s\n",
                   nvstatusToString(status));
            goto noncc_cleanup;
        }

        status = uvm_push_begin(gpu->channel_manager,
                               UVM_CHANNEL_TYPE_GPU_TO_CPU,
                               &push,
                               "LivMig non-CC GPU-to-CPU PA=0x%llx",
                               phys_addr.address);
        if (status != NV_OK) {
            printk(KERN_ERR "UVM LivMig: Failed to begin push: %s\n",
                   nvstatusToString(status));
            uvm_mem_unmap_gpu_phys(staging_mem, gpu);
            goto noncc_cleanup;
        }

        dst_addr = uvm_mem_gpu_address_physical(staging_mem, gpu, 0, size);
        gpu->parent->ce_hal->memcopy(&push, dst_addr, src_addr, size);

        status = uvm_push_end_and_wait(&push);
        uvm_mem_unmap_gpu_phys(staging_mem, gpu);

        if (status != NV_OK) {
            printk(KERN_ERR "UVM LivMig: GPU CE copy failed: %s\n",
                   nvstatusToString(status));
            goto noncc_cleanup;
        }

        memcpy(dest_buffer, staging_cpu_va, size);
        printk(KERN_DEBUG "UVM LivMig: Successfully copied %llu bytes from GPU to host\n", size);

    noncc_cleanup:
        uvm_mem_free(staging_mem);
        return status;
    }
}

// ===========================================================================
// Checkpoint ioctl 106 – GET_VA_RANGES
// ===========================================================================
// Return {base, size} for every UVM_VA_RANGE_TYPE_MANAGED allocation on the
// system, collected while holding the global va_spaces list lock.
// Results are accumulated in kernel-side arrays and copied to user only after
// all locks are released to avoid copy_to_user under a mutex.
// ===========================================================================
NV_STATUS uvm_api_live_migration_get_va_ranges(
    UVM_LIVE_MIGRATION_GET_VA_RANGES_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space;
    uvm_va_range_t *va_range;
    NvU64          *k_bases   = NULL;
    NvU64          *k_sizes   = NULL;
    NvU64           count     = 0;
    NvU64           max       = params->max_ranges;
    NV_STATUS       status    = NV_OK;

    printk(KERN_DEBUG "UVM LivMig: GET_VA_RANGES called (max=%llu)\n", max);

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    if (!params->base_addrs || !params->range_sizes || max == 0)
        return NV_ERR_INVALID_ARGUMENT;

    /* Allocate kernel-side staging arrays */
    k_bases = kvmalloc_array(max, sizeof(NvU64), GFP_KERNEL);
    k_sizes = kvmalloc_array(max, sizeof(NvU64), GFP_KERNEL);
    if (!k_bases || !k_sizes) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }

    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);

    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node) {
        uvm_va_space_down_read(va_space);

        uvm_for_each_va_range(va_range, va_space) {
            if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
                continue;
            if (count >= max) {
                status = NV_ERR_BUFFER_TOO_SMALL;
                break;
            }
            k_bases[count] = va_range->node.start;
            k_sizes[count] = va_range->node.end - va_range->node.start + 1;
            printk(KERN_DEBUG "UVM LivMig:   range[%llu] base=0x%llx size=%llu\n",
                   count, k_bases[count], k_sizes[count]);
            count++;
        }

        uvm_va_space_up_read(va_space);

        if (status != NV_OK)
            break;
    }

    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    /* Copy results to user outside all locks */
    if (status == NV_OK || status == NV_ERR_BUFFER_TOO_SMALL) {
        if (copy_to_user((void __user *)params->base_addrs, k_bases,
                         count * sizeof(NvU64)) ||
            copy_to_user((void __user *)params->range_sizes, k_sizes,
                         count * sizeof(NvU64))) {
            status = NV_ERR_INVALID_ARGUMENT;
        }
    }

    params->num_ranges = count;
    printk(KERN_DEBUG "UVM LivMig: GET_VA_RANGES returning %llu ranges\n", count);

out_free:
    kvfree(k_bases);
    kvfree(k_sizes);
    params->rmStatus = status;
    return status;
}

// System-wide finalize: undo PREPARE_ALL.
// Clears migration tracking flags AND bulk-restores RWA permissions on all
// resident pages of every managed block. The bulk restore eliminates one
// GPU MMU fault per 2 MB page on first post-resume write — at H100 +
// multi-GB KV-cache scale that's hundreds of ms of cumulative TTFT/TBT tax
// that would otherwise be paid lazily by the fault handler's fast path.
//
// Hardcodes UVM_PROT_READ_WRITE_ATOMIC: H100 supports native atomics on
// managed memory so RWA is always honored. Callers running on hardware
// without native atomics should adopt the per-processor has_native_atomics
// check used by do_block_add_mappings_after_migration.
NV_STATUS uvm_api_live_migration_finalize(
    UVM_LIVE_MIGRATION_FINALIZE_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space;
    int va_space_count = 0;
    int blocks_finalized = 0;
    int blocks_restored = 0;

    printk(KERN_DEBUG "=== UVM: live_migration_finalize (system-wide) ===\n");

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);

    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node) {
        uvm_va_range_t *va_range;

        uvm_va_space_down_write(va_space);

        // Clear the VA space tracking flag
        va_space->migration_tracking_active = false;

        // Clear per-block tracking flags AND bulk-restore RWA so the app
        // doesn't fault on every 2M page on first post-resume write.
        uvm_for_each_va_range(va_range, va_space) {
            uvm_va_block_t *va_block;
            uvm_va_block_context_t *va_block_context;

            if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
                continue;

            // Cross-process call (migration agent), so mm is NULL — same
            // pattern as prepare_va_space_for_migration.
            va_block_context = uvm_va_block_context_alloc(NULL);
            if (!va_block_context)
                continue;

            for_each_va_block_in_va_range(va_range, va_block) {
                uvm_va_block_region_t region;
                uvm_va_block_retry_t va_block_retry;
                NV_STATUS map_status;

                va_block->live_migration.migration_tracking_enabled = false;
                va_block->live_migration.dirty = false;
                blocks_finalized++;

                // Skip blocks with no resident pages — nothing to map.
                if (uvm_processor_mask_empty(&va_block->resident))
                    continue;

                region = uvm_va_block_region_from_block(va_block);

                // Symmetric inverse of the revoke(READ_WRITE) in PREPARE_ALL:
                // map all resident pages back at the highest permission so
                // the next write hits valid PTEs instead of faulting.
                map_status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                    uvm_va_block_map_mask(va_block,
                                          va_block_context,
                                          &va_block->resident,
                                          region,
                                          NULL,
                                          UVM_PROT_READ_WRITE_ATOMIC,
                                          UvmEventMapRemoteCausePolicy));
                if (map_status != NV_OK) {
                    printk(KERN_ERR "UVM LivMig FINALIZE: map_mask failed on "
                           "block [0x%llx-0x%llx]: status=%d\n",
                           va_block->start, va_block->end, map_status);
                    // Don't abort — surviving pages will be faulted back to
                    // RWA on first access via the standard write-fault path.
                    continue;
                }

                blocks_restored++;
            }

            uvm_va_block_context_free(va_block_context);
        }

        uvm_va_space_up_write(va_space);
        va_space_count++;
    }

    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    printk(KERN_DEBUG "UVM LivMig FINALIZE: %d VA space(s), %d block(s) finalized "
           "(%d block(s) RWA-restored)\n",
           va_space_count, blocks_finalized, blocks_restored);

    params->rmStatus = NV_OK;
    return NV_OK;
}

// ===========================================================================
// Checkpoint ioctl 109 – READ_PAGES_RESIDENT
// ===========================================================================
// Capture a managed VA range without migrating any pages first.
// Walk every 4KB page in [base, base+length), query its residency from the
// live UVM block bitmaps, and separate the data into two output buffers:
//
//   cpu_buf  – CPU-resident pages (plaintext), densely packed in VA order
//   gpu_buf  – GPU-resident pages (plaintext, CE-decrypted in CC mode),
//              densely packed in VA order
//
// residency_map[i] = 0 (CPU), 1 (GPU), 2 (absent) for each 4KB page.
//
// Implementation:
//   Pass 1 (va_space + global lock held):
//     Per-4KB page classification and CPU page capture:
//       CPU: kmap/memcpy/kunmap → copy_to_user(cpu_buf)
//       GPU: record physical address → defer to pass 2
//       absent: mark resmap[i]=2, skip
//   Pass 2 (no locks, contiguous coalescing):
//     Groups physically contiguous GPU pages into bulk CE transfers
//     (up to 2MB per transfer in CC mode) to reduce per-page push overhead.
//     In CC mode: CE encrypts → CPU decrypts per coalesced chunk.
//   Final: copy resmap to userspace.
// ===========================================================================

/* Per-GPU-page descriptor collected in pass 1 for deferred CE copy */
struct live_migration_gpu_page {
    uvm_gpu_phys_address_t phys;           /* GPU physical address of the page  */
    NvU64                  gpu_buf_offset; /* byte offset into params->gpu_buf  */
};

NV_STATUS uvm_api_live_migration_read_pages_resident(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t                 *va_space     = NULL;
    uvm_va_range_t                 *va_range     = NULL;
    uvm_gpu_t                      *gpu          = NULL;
    void                           *page_buf     = NULL;
    NvU8                           *resmap       = NULL;
    struct live_migration_gpu_page *gpu_pages    = NULL;
    uvm_va_space_t                 *vs_iter;
    NvU64                           base, length, end, n_pages;
    NvU64                           cpu_off      = 0;
    NvU64                           gpu_off      = 0;
    NvU64                           gpu_page_cnt = 0;
    NvU64                           gpu_page_cap = 0;
    NvU64                           i;
    NV_STATUS                       status       = NV_ERR_INVALID_ADDRESS;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;

    end     = base + length - 1;
    n_pages = length / PAGE_SIZE;

    /* One reusable 4KB staging buffer (used for both CPU kmap and CE copies) */
    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf)
        return NV_ERR_NO_MEMORY;

    /* Kernel-side residency map; 0=CPU 1=GPU 2=absent */
    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }
    memset(resmap, 2, n_pages);   /* default: absent */

    /* Initial GPU page list (grows dynamically in pass 1) */
    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages    = kvmalloc_array(gpu_page_cap,
                                  sizeof(struct live_migration_gpu_page),
                                  GFP_KERNEL);
    if (!gpu_pages) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }

    /* ------------------------------------------------------------------ */
    /* PASS 1: hold global list lock + va_space read lock throughout.      */
    /* CPU pages: kmap → memcpy → copy_to_user.                           */
    /* GPU pages: collect (phys, offset) for deferred CE copy in pass 2.  */
    /* ------------------------------------------------------------------ */
    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);

    list_for_each_entry(vs_iter, &g_uvm_global.va_spaces.list, list_node) {
        uvm_va_space_down_read(vs_iter);
        va_range = uvm_va_range_find(vs_iter, base);
        if (va_range && va_range->type == UVM_VA_RANGE_TYPE_MANAGED) {
            va_space = vs_iter;
            break;          /* keep both locks held */
        }
        uvm_va_space_up_read(vs_iter);
    }

    if (!va_space) {
        uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
        status = NV_ERR_INVALID_ADDRESS;
        goto out_free;
    }

    /* Resolve GPU: explicit UUID or first registered GPU */
    {
        static const NvProcessorUuid zero_uuid = {{0}};
        if (memcmp(&params->gpu_uuid, &zero_uuid, sizeof(zero_uuid)) == 0)
            gpu = uvm_processor_mask_find_first_va_space_gpu(
                      &va_space->registered_gpus, va_space);
        else
            gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpu_uuid);
    }

    status = NV_OK;

    uvm_for_each_va_range_in(va_range, va_space, base, end) {
        uvm_va_block_t *va_block;

        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
            continue;

        for_each_va_block_in_va_range(va_range, va_block) {
            NvU64 blk_s = max(va_block->start, base);
            NvU64 blk_e = min(va_block->end,   end);
            NvU64 page_va;

            if (blk_s > blk_e)
                continue;

            for (page_va  = blk_s;
                 page_va <= blk_e;
                 page_va += PAGE_SIZE) {

                uvm_page_index_t           pi      = uvm_va_block_cpu_page_index(va_block, page_va);
                NvU64                      pidx    = (page_va - base) / PAGE_SIZE;
                uvm_va_block_gpu_state_t  *gpu_st;
                bool                       is_cpu, is_gpu;

                uvm_mutex_lock(&va_block->lock);

                /* CPU residency: block->cpu.resident is always valid */
                is_cpu = uvm_page_mask_test(&va_block->cpu.resident, pi);

                /* GPU residency: gpu_state may be NULL if GPU never touched block */
                is_gpu = false;
                if (!is_cpu && gpu) {
                    gpu_st = uvm_va_block_gpu_state_get(va_block, gpu->id);
                    if (gpu_st)
                        is_gpu = uvm_page_mask_test(&gpu_st->resident, pi);
                }

                if (is_cpu) {
                    struct page *p      = uvm_va_block_get_cpu_page(va_block, pi);
                    void        *mapped = kmap(p);
                    memcpy(page_buf, mapped, PAGE_SIZE);
                    kunmap(mapped);
                    uvm_mutex_unlock(&va_block->lock);

                    resmap[pidx] = 0;
                    if (params->cpu_buf &&
                        copy_to_user((void __user *)(params->cpu_buf + cpu_off),
                                     page_buf, PAGE_SIZE)) {
                        status = NV_ERR_INVALID_ARGUMENT;
                        goto pass1_done;
                    }
                    cpu_off += PAGE_SIZE;

                } else if (is_gpu) {
                    uvm_gpu_phys_address_t phys =
                        uvm_va_block_gpu_phys_page_address(va_block, pi, gpu);
                    uvm_mutex_unlock(&va_block->lock);

                    resmap[pidx] = 1;

                    /* Grow GPU page list if needed (doubling strategy) */
                    if (gpu_page_cnt >= gpu_page_cap) {
                        NvU64                           new_cap = gpu_page_cap * 2;
                        struct live_migration_gpu_page *new_arr =
                            kvmalloc_array(new_cap,
                                          sizeof(struct live_migration_gpu_page),
                                          GFP_KERNEL);
                        if (!new_arr) {
                            status = NV_ERR_NO_MEMORY;
                            goto pass1_done;
                        }
                        memcpy(new_arr, gpu_pages,
                               gpu_page_cnt * sizeof(struct live_migration_gpu_page));
                        kvfree(gpu_pages);
                        gpu_pages    = new_arr;
                        gpu_page_cap = new_cap;
                    }

                    gpu_pages[gpu_page_cnt].phys           = phys;
                    gpu_pages[gpu_page_cnt].gpu_buf_offset = gpu_off;
                    gpu_page_cnt++;
                    gpu_off += PAGE_SIZE;

                } else {
                    /* absent / evicted: resmap[pidx] stays 2 */
                    uvm_mutex_unlock(&va_block->lock);
                }
            }
        }
    }

pass1_done:
    uvm_va_space_up_read(va_space);
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    if (status != NV_OK)
        goto out_free;

    /* ------------------------------------------------------------------ */
    /* PASS 2: Double-buffered CE copy GPU-resident pages.                */
    /* In CC mode: CE encrypt → DMA buffer → CPU decrypt → copy_to_user  */
    /* Two DMA buffers ping-pong for overlap.                             */
    /* ------------------------------------------------------------------ */
    {
        NvU64 max_transfer = g_uvm_global.conf_computing_enabled
                             ? UVM_CONF_COMPUTING_DMA_BUFFER_SIZE
                             : (2ULL * 1024 * 1024);
        NvU64 num_transfers = 0;

        if (!g_uvm_global.conf_computing_enabled || !gpu) {
            /* Non-CC fallback: serial plain CE copy */
            void *bb = kmalloc(max_transfer, GFP_KERNEL);
            if (!bb) { bb = page_buf; max_transfer = PAGE_SIZE; }
            i = 0;
            while (i < gpu_page_cnt) {
                NvU64 rs = i, rb = PAGE_SIZE;
                while (i+1 < gpu_page_cnt && gpu_pages[i+1].phys.address == gpu_pages[i].phys.address + PAGE_SIZE &&
                       gpu_pages[i+1].phys.aperture == gpu_pages[rs].phys.aperture &&
                       gpu_pages[i+1].gpu_buf_offset == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
                       rb + PAGE_SIZE <= max_transfer) { i++; rb += PAGE_SIZE; }
                status = read_gpu_physical_memory(gpu, gpu_pages[rs].phys, rb, bb);
                if (status != NV_OK) { if (bb != page_buf) kfree(bb); goto out_free; }
                if (params->gpu_buf && copy_to_user((void __user *)(params->gpu_buf + gpu_pages[rs].gpu_buf_offset), bb, rb))
                { status = NV_ERR_INVALID_ARGUMENT; if (bb != page_buf) kfree(bb); goto out_free; }
                num_transfers++; i++;
            }
            if (bb != page_buf) kfree(bb);
        } else {
            /* CC mode: double-buffered CE encrypt + CPU decrypt */
            uvm_conf_computing_dma_buffer_t *db[2] = {NULL, NULL};
            uvm_push_t pu[2]; int act[2] = {0, 0};
            NvU64 s_rs[2], s_rb[2]; int c;
            void *plain_buf = NULL;

            plain_buf = kmalloc(max_transfer, GFP_KERNEL);
            if (!plain_buf) { status = NV_ERR_NO_MEMORY; goto out_free; }

            status = uvm_conf_computing_dma_buffer_alloc(&gpu->conf_computing.dma_buffer_pool, &db[0], NULL);
            if (status != NV_OK) { kfree(plain_buf); goto out_free; }
            status = uvm_conf_computing_dma_buffer_alloc(&gpu->conf_computing.dma_buffer_pool, &db[1], NULL);
            if (status != NV_OK) {
                uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[0], NULL);
                kfree(plain_buf); goto out_free;
            }

            /* Coalesce runs */
            NvU64 *rl_s = NULL, *rl_b = NULL, nr = 0, rc = 0;
            i = 0;
            while (i < gpu_page_cnt) {
                NvU64 rs = i, rb = PAGE_SIZE;
                while (i+1 < gpu_page_cnt && gpu_pages[i+1].phys.address == gpu_pages[i].phys.address + PAGE_SIZE &&
                       gpu_pages[i+1].phys.aperture == gpu_pages[rs].phys.aperture &&
                       gpu_pages[i+1].gpu_buf_offset == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
                       rb + PAGE_SIZE <= max_transfer) { i++; rb += PAGE_SIZE; }
                if (nr >= rc) {
                    NvU64 nc = rc ? rc*2 : 256;
                    NvU64 *a = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
                    NvU64 *b = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
                    if (!a || !b) { kvfree(a); kvfree(b); kvfree(rl_s); kvfree(rl_b);
                        status = NV_ERR_NO_MEMORY; goto r109_dfree; }
                    if (rl_s) { memcpy(a, rl_s, nr*sizeof(NvU64)); kvfree(rl_s); }
                    if (rl_b) { memcpy(b, rl_b, nr*sizeof(NvU64)); kvfree(rl_b); }
                    rl_s = a; rl_b = b; rc = nc;
                }
                rl_s[nr] = rs; rl_b[nr] = rb; nr++; i++;
            }

            /* Ping-pong loop: CE encrypt async, then decrypt + copy previous */
            for (i = 0; i < nr; i++) {
                uvm_gpu_address_t sa, da, ta;
                c = i & 1;

                /* Wait + decrypt + copy previous on this slot */
                if (act[c]) {
                    status = uvm_push_wait(&pu[c]);
                    if (status != NV_OK) goto r109_drain;
                    {
                        void *cipher = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                        void *atag   = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);

                        /* CPU decrypt CE ciphertext → plaintext */
                        status = uvm_conf_computing_cpu_decrypt(pu[c].channel,
                                    plain_buf, cipher, db[c]->decrypt_iv,
                                    db[c]->key_version[0], s_rb[c], atag);
                        if (status != NV_OK) goto r109_drain;

                        if (params->gpu_buf &&
                            copy_to_user((void __user *)(params->gpu_buf + gpu_pages[s_rs[c]].gpu_buf_offset),
                                         plain_buf, s_rb[c]))
                        { status = NV_ERR_INVALID_ARGUMENT; goto r109_drain; }
                    }
                    act[c] = 0;
                }

                /* Submit async CE encrypt */
                status = uvm_push_begin(gpu->channel_manager, UVM_CHANNEL_TYPE_GPU_TO_CPU, &pu[c], "LivMig r109[%d]", c);
                if (status != NV_OK) goto r109_drain;
                uvm_conf_computing_log_gpu_encryption(pu[c].channel, rl_b[i], db[c]->decrypt_iv);
                db[c]->key_version[0] = uvm_channel_pool_key_version(pu[c].channel->pool);
                sa = uvm_gpu_address_copy(gpu, gpu_pages[rl_s[i]].phys);
                da = uvm_mem_gpu_address_virtual_kernel(db[c]->alloc, gpu);
                ta = uvm_mem_gpu_address_virtual_kernel(db[c]->auth_tag, gpu);
                gpu->parent->ce_hal->encrypt(&pu[c], da, sa, rl_b[i], ta);
                uvm_push_end(&pu[c]);
                act[c] = 1; s_rs[c] = rl_s[i]; s_rb[c] = rl_b[i];
                num_transfers++;
            }

r109_drain:
            for (c = 0; c < 2; c++) {
                if (!act[c]) continue;
                { NV_STATUS ws = uvm_push_wait(&pu[c]);
                  if (ws != NV_OK && status == NV_OK) status = ws; }
                if (status == NV_OK) {
                    void *cipher = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                    void *atag   = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);
                    NV_STATUS ds = uvm_conf_computing_cpu_decrypt(pu[c].channel,
                                    plain_buf, cipher, db[c]->decrypt_iv,
                                    db[c]->key_version[0], s_rb[c], atag);
                    if (ds != NV_OK && status == NV_OK) status = ds;
                    if (status == NV_OK && params->gpu_buf &&
                        copy_to_user((void __user *)(params->gpu_buf + gpu_pages[s_rs[c]].gpu_buf_offset),
                                     plain_buf, s_rb[c]))
                        status = NV_ERR_INVALID_ARGUMENT;
                }
                act[c] = 0;
            }
            kvfree(rl_s); kvfree(rl_b);
r109_dfree:
            uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[0], NULL);
            uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[1], NULL);
            kfree(plain_buf);
            if (status != NV_OK) goto out_free;
        }

        printk(KERN_DEBUG "UVM LivMig: PASS 2 coalesced %llu GPU pages into %llu transfers (double-buf)\n",
               gpu_page_cnt, num_transfers);
    }

    /* Copy residency map to userspace */
    if (copy_to_user((void __user *)params->residency_map, resmap, n_pages)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out_free;
    }

    printk(KERN_DEBUG
           "UVM LivMig: READ_PAGES_RESIDENT done: %llu CPU pages (%llu B), "
           "%llu GPU pages (%llu B), %llu absent\n",
           cpu_off / PAGE_SIZE, cpu_off,
           gpu_page_cnt, gpu_off,
           n_pages - (cpu_off / PAGE_SIZE) - gpu_page_cnt);

out_free:
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);

    params->cpu_bytes_out = cpu_off;
    params->gpu_bytes_out = gpu_off;
    params->num_pages     = n_pages;
    params->rmStatus      = status;
    return status;
}



// ---------------------------------------------------------------------------
// 111 – READ_PAGES_RESIDENT_ENCRYPTED
// Same two-pass structure as ioctl 109, but GPU pages are returned as
// CE-encrypted ciphertext (no CPU decryption) with per-transfer crypto
// metadata (IV, auth_tag, key_version, transfer size).
//
//   Pass 1: identical to ioctl 109 — per-4KB page classification,
//           CPU pages copied as plaintext.
//   Pass 2: contiguous coalescing of GPU pages (up to 2MB per transfer).
//           CE encrypts each coalesced chunk; ciphertext + one crypto_meta
//           entry per chunk written to userspace.  No CPU decrypt.
//
// CPU pages are plaintext (agent encrypts them with k3 separately).
// num_transfers (OUT) = number of coalesced crypto_meta entries.
// ---------------------------------------------------------------------------
NV_STATUS uvm_api_live_migration_read_pages_resident_encrypted(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t                 *va_space     = NULL;
    uvm_va_range_t                 *va_range     = NULL;
    uvm_gpu_t                      *gpu          = NULL;
    void                           *page_buf     = NULL;
    NvU8                           *resmap       = NULL;
    struct live_migration_gpu_page *gpu_pages    = NULL;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf = NULL;
    uvm_va_space_t                 *vs_iter;
    NvU64                           base, length, end, n_pages;
    NvU64                           cpu_off      = 0;
    NvU64                           gpu_off      = 0;
    NvU64                           gpu_page_cnt = 0;
    NvU64                           gpu_page_cap = 0;
    NvU64                           i;
    NV_STATUS                       status       = NV_ERR_INVALID_ADDRESS;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;

    end     = base + length - 1;
    n_pages = length / PAGE_SIZE;

    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf)
        return NV_ERR_NO_MEMORY;

    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }
    memset(resmap, 2, n_pages);

    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages    = kvmalloc_array(gpu_page_cap,
                                  sizeof(struct live_migration_gpu_page),
                                  GFP_KERNEL);
    if (!gpu_pages) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }

    /* Per-GPU-page crypto metadata buffer */
    crypto_meta_buf = kvmalloc_array(gpu_page_cap,
                                      sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META),
                                      GFP_KERNEL);
    if (!crypto_meta_buf) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }

    /* PASS 1: same as ioctl 109 — classify pages, copy CPU pages */
    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);

    list_for_each_entry(vs_iter, &g_uvm_global.va_spaces.list, list_node) {
        uvm_va_space_down_read(vs_iter);
        va_range = uvm_va_range_find(vs_iter, base);
        if (va_range && va_range->type == UVM_VA_RANGE_TYPE_MANAGED) {
            va_space = vs_iter;
            break;
        }
        uvm_va_space_up_read(vs_iter);
    }

    if (!va_space) {
        uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
        status = NV_ERR_INVALID_ADDRESS;
        goto out_free;
    }

    {
        static const NvProcessorUuid zero_uuid = {{0}};
        if (memcmp(&params->gpu_uuid, &zero_uuid, sizeof(zero_uuid)) == 0)
            gpu = uvm_processor_mask_find_first_va_space_gpu(
                      &va_space->registered_gpus, va_space);
        else
            gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpu_uuid);
    }

    status = NV_OK;

    uvm_for_each_va_range_in(va_range, va_space, base, end) {
        uvm_va_block_t *va_block;

        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
            continue;

        for_each_va_block_in_va_range(va_range, va_block) {
            NvU64 blk_s = max(va_block->start, base);
            NvU64 blk_e = min(va_block->end,   end);
            NvU64 page_va;

            if (blk_s > blk_e)
                continue;

            for (page_va  = blk_s;
                 page_va <= blk_e;
                 page_va += PAGE_SIZE) {

                uvm_page_index_t           pi      = uvm_va_block_cpu_page_index(va_block, page_va);
                NvU64                      pidx    = (page_va - base) / PAGE_SIZE;
                uvm_va_block_gpu_state_t  *gpu_st;
                bool                       is_cpu, is_gpu;

                uvm_mutex_lock(&va_block->lock);

                is_cpu = uvm_page_mask_test(&va_block->cpu.resident, pi);

                is_gpu = false;
                if (!is_cpu && gpu) {
                    gpu_st = uvm_va_block_gpu_state_get(va_block, gpu->id);
                    if (gpu_st)
                        is_gpu = uvm_page_mask_test(&gpu_st->resident, pi);
                }

                if (is_cpu) {
                    struct page *p      = uvm_va_block_get_cpu_page(va_block, pi);
                    void        *mapped = kmap(p);
                    memcpy(page_buf, mapped, PAGE_SIZE);
                    kunmap(mapped);
                    uvm_mutex_unlock(&va_block->lock);

                    resmap[pidx] = 0;
                    if (params->cpu_buf &&
                        copy_to_user((void __user *)(params->cpu_buf + cpu_off),
                                     page_buf, PAGE_SIZE)) {
                        status = NV_ERR_INVALID_ARGUMENT;
                        goto pass1_done;
                    }
                    cpu_off += PAGE_SIZE;

                } else if (is_gpu) {
                    uvm_gpu_phys_address_t phys =
                        uvm_va_block_gpu_phys_page_address(va_block, pi, gpu);
                    uvm_mutex_unlock(&va_block->lock);

                    resmap[pidx] = 1;

                    /* Grow GPU page + crypto_meta lists if needed */
                    if (gpu_page_cnt >= gpu_page_cap) {
                        NvU64 new_cap = gpu_page_cap * 2;
                        struct live_migration_gpu_page *new_arr =
                            kvmalloc_array(new_cap,
                                          sizeof(struct live_migration_gpu_page),
                                          GFP_KERNEL);
                        UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *new_meta =
                            kvmalloc_array(new_cap,
                                          sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META),
                                          GFP_KERNEL);
                        if (!new_arr || !new_meta) {
                            kvfree(new_arr);
                            kvfree(new_meta);
                            status = NV_ERR_NO_MEMORY;
                            goto pass1_done;
                        }
                        memcpy(new_arr, gpu_pages,
                               gpu_page_cnt * sizeof(struct live_migration_gpu_page));
                        memcpy(new_meta, crypto_meta_buf,
                               gpu_page_cnt * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META));
                        kvfree(gpu_pages);
                        kvfree(crypto_meta_buf);
                        gpu_pages       = new_arr;
                        crypto_meta_buf = new_meta;
                        gpu_page_cap    = new_cap;
                    }

                    gpu_pages[gpu_page_cnt].phys           = phys;
                    gpu_pages[gpu_page_cnt].gpu_buf_offset = gpu_off;
                    gpu_page_cnt++;
                    gpu_off += PAGE_SIZE;

                } else {
                    uvm_mutex_unlock(&va_block->lock);
                }
            }
        }
    }

pass1_done:
    uvm_va_space_up_read(va_space);
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    if (status != NV_OK)
        goto out_free;

    /* ------------------------------------------------------------------ */
    /* PASS 2: Double-buffered CE encrypted copy with coalescing.         */
    /* Two pre-allocated DMA buffers ping-pong: CE fills one while CPU    */
    /* copies the other to userspace. Skips CE decrypt (v3 advantage).    */
    /* ------------------------------------------------------------------ */
    {
        NvU64 max_transfer = g_uvm_global.conf_computing_enabled
                             ? UVM_CONF_COMPUTING_DMA_BUFFER_SIZE
                             : (2ULL * 1024 * 1024);
        NvU64 num_transfers = 0;

        if (!g_uvm_global.conf_computing_enabled || !gpu) {
            /* Non-CC fallback: serial */
            void *bb = kmalloc(max_transfer, GFP_KERNEL);
            if (!bb) { bb = page_buf; max_transfer = PAGE_SIZE; }
            i = 0;
            while (i < gpu_page_cnt) {
                NvU64 rs = i, rb = PAGE_SIZE;
                while (i+1 < gpu_page_cnt && gpu_pages[i+1].phys.address == gpu_pages[i].phys.address + PAGE_SIZE &&
                       gpu_pages[i+1].phys.aperture == gpu_pages[rs].phys.aperture &&
                       gpu_pages[i+1].gpu_buf_offset == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
                       rb + PAGE_SIZE <= max_transfer) { i++; rb += PAGE_SIZE; }
                status = read_gpu_physical_memory_encrypted(gpu, gpu_pages[rs].phys, rb, bb, &crypto_meta_buf[num_transfers]);
                if (status != NV_OK) { if (bb != page_buf) kfree(bb); goto out_free; }
                crypto_meta_buf[num_transfers].size = rb;
                if (params->gpu_buf && copy_to_user((void __user *)(params->gpu_buf + gpu_pages[rs].gpu_buf_offset), bb, rb))
                { status = NV_ERR_INVALID_ARGUMENT; if (bb != page_buf) kfree(bb); goto out_free; }
                num_transfers++; i++;
            }
            if (bb != page_buf) kfree(bb);
        } else {
            /* CC mode: double-buffered CE transfer */
            uvm_conf_computing_dma_buffer_t *db[2] = {NULL, NULL};
            uvm_push_t pu[2]; int act[2] = {0, 0};
            NvU64 s_rs[2], s_rb[2], s_ti[2]; int c;

            status = uvm_conf_computing_dma_buffer_alloc(&gpu->conf_computing.dma_buffer_pool, &db[0], NULL);
            if (status != NV_OK) goto out_free;
            status = uvm_conf_computing_dma_buffer_alloc(&gpu->conf_computing.dma_buffer_pool, &db[1], NULL);
            if (status != NV_OK) { uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[0], NULL); goto out_free; }

            /* Coalesce runs */
            NvU64 *rl_s = NULL, *rl_b = NULL, nr = 0, rc = 0;
            i = 0;
            while (i < gpu_page_cnt) {
                NvU64 rs = i, rb = PAGE_SIZE;
                while (i+1 < gpu_page_cnt && gpu_pages[i+1].phys.address == gpu_pages[i].phys.address + PAGE_SIZE &&
                       gpu_pages[i+1].phys.aperture == gpu_pages[rs].phys.aperture &&
                       gpu_pages[i+1].gpu_buf_offset == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
                       rb + PAGE_SIZE <= max_transfer) { i++; rb += PAGE_SIZE; }
                if (nr >= rc) {
                    NvU64 nc = rc ? rc*2 : 256;
                    NvU64 *a = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
                    NvU64 *b = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
                    if (!a || !b) { kvfree(a); kvfree(b); kvfree(rl_s); kvfree(rl_b);
                        status = NV_ERR_NO_MEMORY; goto p2_dfree; }
                    if (rl_s) { memcpy(a, rl_s, nr*sizeof(NvU64)); kvfree(rl_s); }
                    if (rl_b) { memcpy(b, rl_b, nr*sizeof(NvU64)); kvfree(rl_b); }
                    rl_s = a; rl_b = b; rc = nc;
                }
                rl_s[nr] = rs; rl_b[nr] = rb; nr++; i++;
            }

            /* Ping-pong loop */
            for (i = 0; i < nr; i++) {
                uvm_gpu_address_t sa, da, ta;
                c = i & 1;
                if (act[c]) {
                    status = uvm_push_wait(&pu[c]);
                    if (status != NV_OK) goto p2_drain;
                    { void *ci = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                      void *tg = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);
                      NvU64 ti = s_ti[c];
                      crypto_meta_buf[ti].size = s_rb[c];
                      memcpy(crypto_meta_buf[ti].iv, db[c]->decrypt_iv[0].iv, 12);
                      crypto_meta_buf[ti].iv_fresh = db[c]->decrypt_iv[0].fresh;
                      memcpy(crypto_meta_buf[ti].auth_tag, tg, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
                      crypto_meta_buf[ti].key_version = db[c]->key_version[0];
                      if (params->gpu_buf &&
                          copy_to_user((void __user *)(params->gpu_buf + gpu_pages[s_rs[c]].gpu_buf_offset), ci, s_rb[c]))
                      { status = NV_ERR_INVALID_ARGUMENT; goto p2_drain; } }
                    act[c] = 0;
                }
                status = uvm_push_begin(gpu->channel_manager, UVM_CHANNEL_TYPE_GPU_TO_CPU, &pu[c], "LivMig enc[%d]", c);
                if (status != NV_OK) goto p2_drain;
                uvm_conf_computing_log_gpu_encryption(pu[c].channel, rl_b[i], db[c]->decrypt_iv);
                db[c]->key_version[0] = uvm_channel_pool_key_version(pu[c].channel->pool);
                sa = uvm_gpu_address_copy(gpu, gpu_pages[rl_s[i]].phys);
                da = uvm_mem_gpu_address_virtual_kernel(db[c]->alloc, gpu);
                ta = uvm_mem_gpu_address_virtual_kernel(db[c]->auth_tag, gpu);
                gpu->parent->ce_hal->encrypt(&pu[c], da, sa, rl_b[i], ta);
                uvm_push_end(&pu[c]);
                act[c] = 1; s_rs[c] = rl_s[i]; s_rb[c] = rl_b[i]; s_ti[c] = num_transfers;
                num_transfers++;
            }
p2_drain:
            for (c = 0; c < 2; c++) {
                if (!act[c]) continue;
                { NV_STATUS ws = uvm_push_wait(&pu[c]);
                  if (ws != NV_OK && status == NV_OK) status = ws; }
                if (status == NV_OK) {
                    void *ci = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                    void *tg = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);
                    NvU64 ti = s_ti[c];
                    crypto_meta_buf[ti].size = s_rb[c];
                    memcpy(crypto_meta_buf[ti].iv, db[c]->decrypt_iv[0].iv, 12);
                    crypto_meta_buf[ti].iv_fresh = db[c]->decrypt_iv[0].fresh;
                    memcpy(crypto_meta_buf[ti].auth_tag, tg, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
                    crypto_meta_buf[ti].key_version = db[c]->key_version[0];
                    if (params->gpu_buf &&
                        copy_to_user((void __user *)(params->gpu_buf + gpu_pages[s_rs[c]].gpu_buf_offset), ci, s_rb[c]))
                        status = NV_ERR_INVALID_ARGUMENT;
                }
                act[c] = 0;
            }
            kvfree(rl_s); kvfree(rl_b);
p2_dfree:
            uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[0], NULL);
            uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[1], NULL);
            if (status != NV_OK) goto out_free;
        }

        /* Copy residency map */
        if (copy_to_user((void __user *)params->residency_map, resmap, n_pages)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto out_free;
        }

        /* Copy coalesced crypto metadata */
        if (num_transfers > 0 && params->crypto_meta) {
            NvU64 meta_bytes = num_transfers * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META);
            if (meta_bytes > params->crypto_meta_size) {
                printk(KERN_ERR "UVM LivMig: crypto_meta buffer too small: need %llu, have %llu\n",
                       meta_bytes, params->crypto_meta_size);
                status = NV_ERR_INVALID_ARGUMENT;
                goto out_free;
            }
            if (copy_to_user((void __user *)params->crypto_meta,
                             crypto_meta_buf, meta_bytes)) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto out_free;
            }
        }

        printk(KERN_DEBUG
               "UVM LivMig: READ_PAGES_RESIDENT_ENCRYPTED done: %llu CPU pages, "
               "%llu GPU pages coalesced into %llu transfers, %llu absent\n",
               cpu_off / PAGE_SIZE, gpu_page_cnt, num_transfers,
               n_pages - (cpu_off / PAGE_SIZE) - gpu_page_cnt);

        params->num_transfers = num_transfers;
    }

out_free:
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);
    kvfree(crypto_meta_buf);

    params->cpu_bytes_out  = cpu_off;
    params->gpu_bytes_out  = gpu_off;
    params->num_pages      = n_pages;
    params->rmStatus       = status;
    return status;
}


// ===========================================================================
// Ioctl 112 – DECRYPT_ENCRYPTED_PAGES
// ===========================================================================
// Decryption service for restoring v3 checkpoint images.
// Takes k1-encrypted ciphertext + per-transfer crypto metadata (produced by
// ioctl 111), decrypts each coalesced chunk using the kernel's CSL context
// (GPU→CPU channel key k1), and returns plaintext to userspace.
//
// Each crypto_meta entry specifies the size of the chunk it covers
// (up to 2MB, matching the coalesced transfers from ioctl 111).
// The caller writes the plaintext to GPU via cudaMemcpy (normal CC path).
//
// Non-CC mode: copies data as-is (no decryption needed).

NV_STATUS uvm_api_live_migration_decrypt_encrypted_pages(
    UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES_PARAMS *params,
    struct file *filp)
{
    NV_STATUS status;
    uvm_gpu_t *gpu = NULL;
    void *cipher_buf = NULL;
    void *plain_buf = NULL;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *meta_buf = NULL;
    NvU64 num_transfers, total_size;
    NvU64 i;
    NvU64 max_chunk;
    uvm_channel_t *decrypt_channel = NULL;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    num_transfers = params->num_transfers;
    total_size    = params->total_size;

    if (num_transfers == 0 || total_size == 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->cipher_buf || !params->plain_buf || !params->crypto_meta)
        return NV_ERR_INVALID_ARGUMENT;

    if (!g_uvm_global.conf_computing_enabled) {
        // Non-CC: copy as-is
        void *tmp = kmalloc(PAGE_SIZE, GFP_KERNEL);
        if (!tmp) { params->rmStatus = NV_ERR_NO_MEMORY; return NV_ERR_NO_MEMORY; }
        NvU64 off;
        for (off = 0; off < total_size; off += PAGE_SIZE) {
            NvU64 chunk = min((NvU64)PAGE_SIZE, total_size - off);
            if (copy_from_user(tmp, (void __user *)(params->cipher_buf + off), chunk) ||
                copy_to_user((void __user *)(params->plain_buf + off), tmp, chunk)) {
                kfree(tmp);
                params->rmStatus = NV_ERR_INVALID_ARGUMENT;
                return NV_ERR_INVALID_ARGUMENT;
            }
        }
        kfree(tmp);
        params->rmStatus = NV_OK;
        return NV_OK;
    }

    printk(KERN_DEBUG "UVM LivMig DECRYPT: %llu transfers, %llu bytes\n",
           num_transfers, total_size);

    /* Allocate buffers sized for the largest possible transfer (2MB) */
    max_chunk = UVM_CONF_COMPUTING_DMA_BUFFER_SIZE;
    cipher_buf = kmalloc(max_chunk, GFP_KERNEL);
    plain_buf  = kmalloc(max_chunk, GFP_KERNEL);
    meta_buf   = kvmalloc(num_transfers * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META),
                           GFP_KERNEL);
    if (!cipher_buf || !plain_buf || !meta_buf) {
        status = NV_ERR_NO_MEMORY;
        goto out_free;
    }

    if (copy_from_user(meta_buf, (void __user *)params->crypto_meta,
                       num_transfers * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META))) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out_free;
    }

    // Find GPU
    {
        uvm_gpu_t *g;
        static const NvProcessorUuid zero_uuid = {{0}};

        uvm_mutex_lock(&g_uvm_global.global_lock);
        for_each_gpu(g) {
            if (memcmp(&params->gpu_uuid, &zero_uuid, sizeof(zero_uuid)) == 0 ||
                memcmp(&g->uuid, &params->gpu_uuid, sizeof(params->gpu_uuid)) == 0) {
                gpu = g;
                break;
            }
        }
        uvm_mutex_unlock(&g_uvm_global.global_lock);
    }

    if (!gpu) {
        printk(KERN_ERR "UVM LivMig DECRYPT: No GPU found\n");
        status = NV_ERR_INVALID_DEVICE;
        goto out_free;
    }

    // Get decrypt channel
    {
        uvm_push_t tmp_push;
        status = uvm_push_begin(gpu->channel_manager,
                               UVM_CHANNEL_TYPE_GPU_TO_CPU,
                               &tmp_push,
                               "LivMig decrypt channel");
        if (status == NV_OK) {
            decrypt_channel = tmp_push.channel;
            status = uvm_push_end_and_wait(&tmp_push);
        }
        if (status != NV_OK || !decrypt_channel) {
            printk(KERN_ERR "UVM LivMig DECRYPT: Could not get channel: %s\n",
                   nvstatusToString(status));
            goto out_free;
        }
    }

    // Decrypt each transfer
    {
        NvU64 cipher_off = 0;
        status = NV_OK;

        for (i = 0; i < num_transfers && status == NV_OK; i++) {
            UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *meta = &meta_buf[i];
            NvU64 chunk_size = meta->size;
            UvmCslIv iv;

            printk(KERN_DEBUG "UVM LivMig DECRYPT: transfer[%llu] size=%llu off=%llu "
                   "key_ver=%u iv=[%02x%02x%02x%02x] tag=[%02x%02x%02x%02x]\n",
                   i, chunk_size, cipher_off, meta->key_version,
                   meta->iv[0], meta->iv[1], meta->iv[2], meta->iv[3],
                   meta->auth_tag[0], meta->auth_tag[1],
                   meta->auth_tag[2], meta->auth_tag[3]);

            if (chunk_size == 0 || chunk_size > max_chunk ||
                cipher_off + chunk_size > total_size) {
                printk(KERN_ERR "UVM LivMig DECRYPT: invalid transfer %llu size=%llu\n",
                       i, chunk_size);
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }

            if (copy_from_user(cipher_buf,
                               (void __user *)(params->cipher_buf + cipher_off),
                               chunk_size)) {
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }

            memcpy(iv.iv, meta->iv, 12);
            iv.fresh = meta->iv_fresh;

            status = uvm_conf_computing_cpu_decrypt(decrypt_channel,
                                                     plain_buf,
                                                     cipher_buf,
                                                     &iv,
                                                     meta->key_version,
                                                     chunk_size,
                                                     meta->auth_tag);
            if (status != NV_OK) {
                printk(KERN_ERR "UVM LivMig DECRYPT: transfer %llu (%llu bytes) failed: %s\n",
                       i, chunk_size, nvstatusToString(status));
                break;
            }

            if (copy_to_user((void __user *)(params->plain_buf + cipher_off),
                             plain_buf, chunk_size)) {
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }

            cipher_off += chunk_size;
        }

        if (status == NV_OK)
            printk(KERN_DEBUG "UVM LivMig DECRYPT: %llu transfers decrypted (%llu bytes)\n",
                   num_transfers, total_size);
    }

out_free:
    kfree(cipher_buf);
    kfree(plain_buf);
    kvfree(meta_buf);

    params->rmStatus = status;
    return status;
}


// ===========================================================================
// Ioctl 113 – READ_PAGES_RESIDENT_ENCRYPTED_V2
// ===========================================================================
// Same as ioctl 111 but with double-buffered CE transfers.
// Two pre-allocated DMA buffers ping-pong: while CE encrypts into buffer A,
// CPU copies buffer B to userspace.
// Pass 1 is identical to ioctl 111.
// Pass 2 uses double-buffered pipelining for GPU page CE transfers.
// ---------------------------------------------------------------------------
NV_STATUS uvm_api_live_migration_read_pages_resident_encrypted_v2(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t                 *va_space     = NULL;
    uvm_va_range_t                 *va_range     = NULL;
    uvm_gpu_t                      *gpu          = NULL;
    void                           *page_buf     = NULL;
    NvU8                           *resmap       = NULL;
    struct live_migration_gpu_page *gpu_pages    = NULL;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf = NULL;
    uvm_va_space_t                 *vs_iter;
    NvU64                           base, length, end, n_pages;
    NvU64                           cpu_off      = 0;
    NvU64                           gpu_off      = 0;
    NvU64                           gpu_page_cnt = 0;
    NvU64                           gpu_page_cap = 0;
    NvU64                           i;
    NV_STATUS                       status       = NV_ERR_INVALID_ADDRESS;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;

    end     = base + length - 1;
    n_pages = length / PAGE_SIZE;

    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf) return NV_ERR_NO_MEMORY;

    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) { status = NV_ERR_NO_MEMORY; goto v2_out; }
    memset(resmap, 2, n_pages);

    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages = kvmalloc_array(gpu_page_cap, sizeof(struct live_migration_gpu_page), GFP_KERNEL);
    if (!gpu_pages) { status = NV_ERR_NO_MEMORY; goto v2_out; }

    crypto_meta_buf = kvmalloc_array(gpu_page_cap, sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META), GFP_KERNEL);
    if (!crypto_meta_buf) { status = NV_ERR_NO_MEMORY; goto v2_out; }

    /* ---- PASS 1: classify pages, copy CPU pages (identical to ioctl 111) ---- */
    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
    list_for_each_entry(vs_iter, &g_uvm_global.va_spaces.list, list_node) {
        uvm_va_space_down_read(vs_iter);
        va_range = uvm_va_range_find(vs_iter, base);
        if (va_range && va_range->type == UVM_VA_RANGE_TYPE_MANAGED) { va_space = vs_iter; break; }
        uvm_va_space_up_read(vs_iter);
    }
    if (!va_space) { uvm_mutex_unlock(&g_uvm_global.va_spaces.lock); status = NV_ERR_INVALID_ADDRESS; goto v2_out; }

    { static const NvProcessorUuid z = {{0}};
      gpu = (memcmp(&params->gpu_uuid, &z, sizeof(z)) == 0)
          ? uvm_processor_mask_find_first_va_space_gpu(&va_space->registered_gpus, va_space)
          : uvm_va_space_get_gpu_by_uuid(va_space, &params->gpu_uuid); }

    status = NV_OK;
    uvm_for_each_va_range_in(va_range, va_space, base, end) {
        uvm_va_block_t *vb;
        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED) continue;
        for_each_va_block_in_va_range(va_range, vb) {
            NvU64 bs = max(vb->start, base), be = min(vb->end, end), pv;
            if (bs > be) continue;
            for (pv = bs; pv <= be; pv += PAGE_SIZE) {
                uvm_page_index_t pi = uvm_va_block_cpu_page_index(vb, pv);
                NvU64 px = (pv - base) / PAGE_SIZE;
                uvm_va_block_gpu_state_t *gs; bool ic, ig;
                uvm_mutex_lock(&vb->lock);
                ic = uvm_page_mask_test(&vb->cpu.resident, pi);
                ig = false;
                if (!ic && gpu) { gs = uvm_va_block_gpu_state_get(vb, gpu->id); if (gs) ig = uvm_page_mask_test(&gs->resident, pi); }
                if (ic) {
                    struct page *p = uvm_va_block_get_cpu_page(vb, pi); void *m = kmap(p);
                    memcpy(page_buf, m, PAGE_SIZE); kunmap(m); uvm_mutex_unlock(&vb->lock);
                    resmap[px] = 0;
                    if (params->cpu_buf && copy_to_user((void __user *)(params->cpu_buf + cpu_off), page_buf, PAGE_SIZE))
                    { status = NV_ERR_INVALID_ARGUMENT; goto v2_p1done; }
                    cpu_off += PAGE_SIZE;
                } else if (ig) {
                    uvm_gpu_phys_address_t ph = uvm_va_block_gpu_phys_page_address(vb, pi, gpu);
                    uvm_mutex_unlock(&vb->lock); resmap[px] = 1;
                    if (gpu_page_cnt >= gpu_page_cap) {
                        NvU64 nc = gpu_page_cap * 2;
                        struct live_migration_gpu_page *na = kvmalloc_array(nc, sizeof(*na), GFP_KERNEL);
                        UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *nm = kvmalloc_array(nc, sizeof(*nm), GFP_KERNEL);
                        if (!na || !nm) { kvfree(na); kvfree(nm); status = NV_ERR_NO_MEMORY; goto v2_p1done; }
                        memcpy(na, gpu_pages, gpu_page_cnt * sizeof(*na));
                        memcpy(nm, crypto_meta_buf, gpu_page_cnt * sizeof(*nm));
                        kvfree(gpu_pages); kvfree(crypto_meta_buf);
                        gpu_pages = na; crypto_meta_buf = nm; gpu_page_cap = nc;
                    }
                    gpu_pages[gpu_page_cnt].phys = ph; gpu_pages[gpu_page_cnt].gpu_buf_offset = gpu_off;
                    gpu_page_cnt++; gpu_off += PAGE_SIZE;
                } else { uvm_mutex_unlock(&vb->lock); }
            }
        }
    }
v2_p1done:
    uvm_va_space_up_read(va_space);
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
    if (status != NV_OK) goto v2_out;

    /* ---- PASS 2: Double-buffered CE encrypted copy ---- */
    {
        NvU64 max_xfer = UVM_CONF_COMPUTING_DMA_BUFFER_SIZE;
        NvU64 num_transfers = 0;

        if (!g_uvm_global.conf_computing_enabled || !gpu) {
            /* Non-CC fallback */
            void *bb = kmalloc(max_xfer, GFP_KERNEL);
            if (!bb) { bb = page_buf; max_xfer = PAGE_SIZE; }
            i = 0;
            while (i < gpu_page_cnt) {
                NvU64 rs = i, rb = PAGE_SIZE;
                while (i+1 < gpu_page_cnt && gpu_pages[i+1].phys.address == gpu_pages[i].phys.address + PAGE_SIZE &&
                       gpu_pages[i+1].phys.aperture == gpu_pages[rs].phys.aperture &&
                       gpu_pages[i+1].gpu_buf_offset == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
                       rb + PAGE_SIZE <= max_xfer) { i++; rb += PAGE_SIZE; }
                status = read_gpu_physical_memory_encrypted(gpu, gpu_pages[rs].phys, rb, bb, &crypto_meta_buf[num_transfers]);
                if (status != NV_OK) { if (bb != page_buf) kfree(bb); goto v2_out; }
                crypto_meta_buf[num_transfers].size = rb;
                if (params->gpu_buf && copy_to_user((void __user *)(params->gpu_buf + gpu_pages[rs].gpu_buf_offset), bb, rb))
                { status = NV_ERR_INVALID_ARGUMENT; if (bb != page_buf) kfree(bb); goto v2_out; }
                num_transfers++; i++;
            }
            if (bb != page_buf) kfree(bb);
        } else {
            /* Double-buffered CE transfer */
            uvm_conf_computing_dma_buffer_t *db[2] = {NULL, NULL};
            uvm_push_t pu[2]; int act[2] = {0, 0};
            NvU64 s_rs[2], s_rb[2], s_ti[2]; int c;

            status = uvm_conf_computing_dma_buffer_alloc(&gpu->conf_computing.dma_buffer_pool, &db[0], NULL);
            if (status != NV_OK) goto v2_out;
            status = uvm_conf_computing_dma_buffer_alloc(&gpu->conf_computing.dma_buffer_pool, &db[1], NULL);
            if (status != NV_OK) { uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[0], NULL); goto v2_out; }

            /* Coalesce runs */
            NvU64 *rl_s = NULL, *rl_b = NULL, nr = 0, rc = 0;
            i = 0;
            while (i < gpu_page_cnt) {
                NvU64 rs = i, rb = PAGE_SIZE;
                while (i+1 < gpu_page_cnt && gpu_pages[i+1].phys.address == gpu_pages[i].phys.address + PAGE_SIZE &&
                       gpu_pages[i+1].phys.aperture == gpu_pages[rs].phys.aperture &&
                       gpu_pages[i+1].gpu_buf_offset == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
                       rb + PAGE_SIZE <= max_xfer) { i++; rb += PAGE_SIZE; }
                if (nr >= rc) {
                    NvU64 nc = rc ? rc*2 : 256;
                    NvU64 *a = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
                    NvU64 *b = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
                    if (!a || !b) { kvfree(a); kvfree(b); kvfree(rl_s); kvfree(rl_b);
                        status = NV_ERR_NO_MEMORY; goto v2_dfree; }
                    if (rl_s) { memcpy(a, rl_s, nr*sizeof(NvU64)); kvfree(rl_s); }
                    if (rl_b) { memcpy(b, rl_b, nr*sizeof(NvU64)); kvfree(rl_b); }
                    rl_s = a; rl_b = b; rc = nc;
                }
                rl_s[nr] = rs; rl_b[nr] = rb; nr++; i++;
            }

            /* Ping-pong loop */
            for (i = 0; i < nr; i++) {
                uvm_gpu_address_t sa, da, ta;
                c = i & 1;

                /* Wait + copy previous on this slot */
                if (act[c]) {
                    status = uvm_push_wait(&pu[c]);
                    if (status != NV_OK) goto v2_drain;
                    {
                        void *ci = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                        void *tg = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);
                        NvU64 ti = s_ti[c];
                        crypto_meta_buf[ti].size = s_rb[c];
                        memcpy(crypto_meta_buf[ti].iv, db[c]->decrypt_iv[0].iv, 12);
                        crypto_meta_buf[ti].iv_fresh = db[c]->decrypt_iv[0].fresh;
                        memcpy(crypto_meta_buf[ti].auth_tag, tg, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
                        crypto_meta_buf[ti].key_version = db[c]->key_version[0];
                        if (params->gpu_buf &&
                            copy_to_user((void __user *)(params->gpu_buf + gpu_pages[s_rs[c]].gpu_buf_offset), ci, s_rb[c]))
                        { status = NV_ERR_INVALID_ARGUMENT; goto v2_drain; }
                    }
                    act[c] = 0;
                }

                /* Submit async CE encrypt */
                status = uvm_push_begin(gpu->channel_manager, UVM_CHANNEL_TYPE_GPU_TO_CPU, &pu[c], "LivMig v2[%d]", c);
                if (status != NV_OK) goto v2_drain;

                uvm_conf_computing_log_gpu_encryption(pu[c].channel, rl_b[i], db[c]->decrypt_iv);
                db[c]->key_version[0] = uvm_channel_pool_key_version(pu[c].channel->pool);

                sa = uvm_gpu_address_copy(gpu, gpu_pages[rl_s[i]].phys);
                da = uvm_mem_gpu_address_virtual_kernel(db[c]->alloc, gpu);
                ta = uvm_mem_gpu_address_virtual_kernel(db[c]->auth_tag, gpu);
                gpu->parent->ce_hal->encrypt(&pu[c], da, sa, rl_b[i], ta);

                uvm_push_end(&pu[c]);
                act[c] = 1; s_rs[c] = rl_s[i]; s_rb[c] = rl_b[i]; s_ti[c] = num_transfers;
                num_transfers++;
            }

v2_drain:
            for (c = 0; c < 2; c++) {
                if (!act[c]) continue;
                { NV_STATUS ws = uvm_push_wait(&pu[c]);
                  if (ws != NV_OK && status == NV_OK) status = ws; }
                if (status == NV_OK) {
                    void *ci = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                    void *tg = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);
                    NvU64 ti = s_ti[c];
                    crypto_meta_buf[ti].size = s_rb[c];
                    memcpy(crypto_meta_buf[ti].iv, db[c]->decrypt_iv[0].iv, 12);
                    crypto_meta_buf[ti].iv_fresh = db[c]->decrypt_iv[0].fresh;
                    memcpy(crypto_meta_buf[ti].auth_tag, tg, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
                    crypto_meta_buf[ti].key_version = db[c]->key_version[0];
                    if (params->gpu_buf &&
                        copy_to_user((void __user *)(params->gpu_buf + gpu_pages[s_rs[c]].gpu_buf_offset), ci, s_rb[c]))
                        status = NV_ERR_INVALID_ARGUMENT;
                }
                act[c] = 0;
            }
            kvfree(rl_s); kvfree(rl_b);
v2_dfree:
            uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[0], NULL);
            uvm_conf_computing_dma_buffer_free(&gpu->conf_computing.dma_buffer_pool, db[1], NULL);
            if (status != NV_OK) goto v2_out;
        }

        if (copy_to_user((void __user *)params->residency_map, resmap, n_pages))
        { status = NV_ERR_INVALID_ARGUMENT; goto v2_out; }

        if (num_transfers > 0 && params->crypto_meta) {
            NvU64 mb = num_transfers * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META);
            if (mb > params->crypto_meta_size) { status = NV_ERR_INVALID_ARGUMENT; goto v2_out; }
            if (copy_to_user((void __user *)params->crypto_meta, crypto_meta_buf, mb))
            { status = NV_ERR_INVALID_ARGUMENT; goto v2_out; }
        }

        printk(KERN_DEBUG "UVM LivMig: ENCRYPTED_V2 done: %llu CPU, %llu GPU → %llu xfers (double-buf)\n",
               cpu_off / PAGE_SIZE, gpu_page_cnt, num_transfers);

        params->cpu_bytes_out = cpu_off;
        params->gpu_bytes_out = gpu_off;
        params->num_pages     = n_pages;
        params->num_transfers = num_transfers;
    }

v2_out:
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);
    kvfree(crypto_meta_buf);
    params->rmStatus = status;
    return status;
}

/* ========================================================================= */
/* Multi-threaded variants (ioctl 114 / 115)                                 */
/*                                                                            */
/* Shared worker infrastructure: spawn N kthreads, each owning its own pair   */
/* of DMA buffers and acquiring its own CE channel via uvm_push_begin. Each   */
/* worker processes a contiguous slice of the coalesced run list and does    */
/* its own ping-pong loop, including copy_to_user via kthread_use_mm of the  */
/* ioctl caller's address space.                                              */
/* ========================================================================= */

#define LM_MT_MAX_THREADS 16
#define LM_MT_DEFAULT_THREADS 4

struct lm_mt_worker {
    /* lifecycle */
    struct task_struct *task;
    struct completion   done;

    /* shared inputs */
    struct mm_struct                *caller_mm;
    uvm_gpu_t                       *gpu;
    struct live_migration_gpu_page  *gpu_pages;  /* shared, read-only */
    NvU64                           *rl_s;       /* shared, read-only */
    NvU64                           *rl_b;       /* shared, read-only */
    NvU64                            slice_start; /* index into rl_s/rl_b */
    NvU64                            slice_count;
    NvU64                            user_gpu_buf;  /* params->gpu_buf */

    /* 115-only: shared ciphertext metadata; each worker writes its own slice */
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf;
    NvU64                               crypto_meta_base;  /* slice start index */

    /* selects plaintext (114) vs ciphertext (115) behavior */
    bool is_ciphertext;

    /* outputs */
    NV_STATUS status;
    NvU64     num_transfers;
};

static int lm_mt_worker_fn(void *data)
{
    struct lm_mt_worker              *w  = data;
    uvm_conf_computing_dma_buffer_t  *db[2] = {NULL, NULL};
    uvm_push_t                        pu[2];
    int                               act[2] = {0, 0};
    NvU64                             s_rs[2], s_rb[2], s_ti[2];
    void                             *plain_buf = NULL;
    NV_STATUS                         status = NV_OK;
    NvU64                             i;
    int                               c;
    NvU64                             local_nt = w->crypto_meta_base;

    /* Borrow the caller's mm so copy_to_user can target userspace buffers. */
    kthread_use_mm(w->caller_mm);

    /* Plaintext variant needs a per-worker scratch for CPU decrypt output. */
    if (!w->is_ciphertext) {
        plain_buf = kmalloc(UVM_CONF_COMPUTING_DMA_BUFFER_SIZE, GFP_KERNEL);
        if (!plain_buf) {
            status = NV_ERR_NO_MEMORY;
            goto w_unuse;
        }
    }

    status = uvm_conf_computing_dma_buffer_alloc(&w->gpu->conf_computing.dma_buffer_pool, &db[0], NULL);
    if (status != NV_OK)
        goto w_free_plain;
    status = uvm_conf_computing_dma_buffer_alloc(&w->gpu->conf_computing.dma_buffer_pool, &db[1], NULL);
    if (status != NV_OK) {
        uvm_conf_computing_dma_buffer_free(&w->gpu->conf_computing.dma_buffer_pool, db[0], NULL);
        db[0] = NULL;
        goto w_free_plain;
    }

    /* Ping-pong loop over this worker's slice. */
    for (i = 0; i < w->slice_count; i++) {
        NvU64 rl_i = w->slice_start + i;
        uvm_gpu_address_t sa, da, ta;
        c = i & 1;

        /* Wait + process previous submission on this slot. */
        if (act[c]) {
            status = uvm_push_wait(&pu[c]);
            if (status != NV_OK)
                goto w_drain;

            {
                void *cipher = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
                void *atag   = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);

                if (w->is_ciphertext) {
                    /* 115: hand ciphertext + per-transfer metadata to userspace. */
                    NvU64 ti = s_ti[c];
                    w->crypto_meta_buf[ti].size         = s_rb[c];
                    memcpy(w->crypto_meta_buf[ti].iv, db[c]->decrypt_iv[0].iv, 12);
                    w->crypto_meta_buf[ti].iv_fresh     = db[c]->decrypt_iv[0].fresh;
                    memcpy(w->crypto_meta_buf[ti].auth_tag, atag, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
                    w->crypto_meta_buf[ti].key_version  = db[c]->key_version[0];

                    if (w->user_gpu_buf &&
                        copy_to_user((void __user *)(w->user_gpu_buf +
                                                     w->gpu_pages[s_rs[c]].gpu_buf_offset),
                                     cipher, s_rb[c])) {
                        status = NV_ERR_INVALID_ARGUMENT;
                        goto w_drain;
                    }
                } else {
                    /* 114: CPU decrypt → plaintext → copy_to_user. */
                    status = uvm_conf_computing_cpu_decrypt(pu[c].channel,
                                                            plain_buf, cipher,
                                                            db[c]->decrypt_iv,
                                                            db[c]->key_version[0],
                                                            s_rb[c], atag);
                    if (status != NV_OK)
                        goto w_drain;

                    if (w->user_gpu_buf &&
                        copy_to_user((void __user *)(w->user_gpu_buf +
                                                     w->gpu_pages[s_rs[c]].gpu_buf_offset),
                                     plain_buf, s_rb[c])) {
                        status = NV_ERR_INVALID_ARGUMENT;
                        goto w_drain;
                    }
                }
            }
            act[c] = 0;
        }

        /* Submit async CE encrypt on this slot. */
        status = uvm_push_begin(w->gpu->channel_manager,
                                UVM_CHANNEL_TYPE_GPU_TO_CPU,
                                &pu[c], "LivMig mt[%llu:%d]",
                                (unsigned long long)w->slice_start, c);
        if (status != NV_OK)
            goto w_drain;

        uvm_conf_computing_log_gpu_encryption(pu[c].channel, w->rl_b[rl_i], db[c]->decrypt_iv);
        db[c]->key_version[0] = uvm_channel_pool_key_version(pu[c].channel->pool);

        sa = uvm_gpu_address_copy(w->gpu, w->gpu_pages[w->rl_s[rl_i]].phys);
        da = uvm_mem_gpu_address_virtual_kernel(db[c]->alloc, w->gpu);
        ta = uvm_mem_gpu_address_virtual_kernel(db[c]->auth_tag, w->gpu);
        w->gpu->parent->ce_hal->encrypt(&pu[c], da, sa, w->rl_b[rl_i], ta);

        uvm_push_end(&pu[c]);
        act[c]  = 1;
        s_rs[c] = w->rl_s[rl_i];
        s_rb[c] = w->rl_b[rl_i];
        if (w->is_ciphertext) {
            s_ti[c] = local_nt++;
            w->num_transfers++;
        }
    }

w_drain:
    for (c = 0; c < 2; c++) {
        NV_STATUS ws;
        if (!act[c])
            continue;

        ws = uvm_push_wait(&pu[c]);
        if (ws != NV_OK && status == NV_OK)
            status = ws;

        if (status == NV_OK) {
            void *cipher = uvm_mem_get_cpu_addr_kernel(db[c]->alloc);
            void *atag   = uvm_mem_get_cpu_addr_kernel(db[c]->auth_tag);

            if (w->is_ciphertext) {
                NvU64 ti = s_ti[c];
                w->crypto_meta_buf[ti].size         = s_rb[c];
                memcpy(w->crypto_meta_buf[ti].iv, db[c]->decrypt_iv[0].iv, 12);
                w->crypto_meta_buf[ti].iv_fresh     = db[c]->decrypt_iv[0].fresh;
                memcpy(w->crypto_meta_buf[ti].auth_tag, atag, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
                w->crypto_meta_buf[ti].key_version  = db[c]->key_version[0];

                if (w->user_gpu_buf &&
                    copy_to_user((void __user *)(w->user_gpu_buf +
                                                 w->gpu_pages[s_rs[c]].gpu_buf_offset),
                                 cipher, s_rb[c]))
                    status = NV_ERR_INVALID_ARGUMENT;
            } else {
                NV_STATUS ds = uvm_conf_computing_cpu_decrypt(pu[c].channel,
                                                              plain_buf, cipher,
                                                              db[c]->decrypt_iv,
                                                              db[c]->key_version[0],
                                                              s_rb[c], atag);
                if (ds != NV_OK && status == NV_OK)
                    status = ds;
                if (status == NV_OK && w->user_gpu_buf &&
                    copy_to_user((void __user *)(w->user_gpu_buf +
                                                 w->gpu_pages[s_rs[c]].gpu_buf_offset),
                                 plain_buf, s_rb[c]))
                    status = NV_ERR_INVALID_ARGUMENT;
            }
        }
        act[c] = 0;
    }

    uvm_conf_computing_dma_buffer_free(&w->gpu->conf_computing.dma_buffer_pool, db[1], NULL);
    uvm_conf_computing_dma_buffer_free(&w->gpu->conf_computing.dma_buffer_pool, db[0], NULL);

w_free_plain:
    kfree(plain_buf);

w_unuse:
    kthread_unuse_mm(w->caller_mm);
    w->status = status;
    complete(&w->done);
    /* The main thread does kthread_stop() on us which synchronizes via the
     * completion; return 0 so it exits cleanly. */
    return 0;
}

/*
 * Common PASS 1: classify pages, copy CPU pages to user, build gpu_pages[].
 * Returns NV_OK on success. On success, *va_space_out is held with a read
 * lock that the caller must release, and the global va_spaces list lock is
 * held. On failure, both locks are released.
 *
 * To keep diffs minimal, callers inline this instead — see lm_mt_pass1 below
 * which is a thin wrapper used by both 114 and 115.
 */
static NV_STATUS lm_mt_pass1(NvU64                              base,
                             NvU64                              length,
                             bool                               dirty_only,
                             NvU64                              user_cpu_buf,
                             const NvProcessorUuid             *req_uuid,
                             void                              *page_buf,
                             NvU8                              *resmap,
                             struct live_migration_gpu_page   **gpu_pages_io,
                             NvU64                             *gpu_page_cap_io,
                             NvU64                             *gpu_page_cnt_out,
                             NvU64                             *cpu_off_out,
                             NvU64                             *gpu_off_out,
                             uvm_gpu_t                        **gpu_out)
{
    uvm_va_space_t *va_space = NULL;
    uvm_va_range_t *va_range = NULL;
    uvm_va_space_t *vs_iter;
    uvm_gpu_t      *gpu     = NULL;
    NvU64           end     = base + length - 1;
    NvU64           cpu_off = 0, gpu_off = 0;
    NvU64           gpu_page_cnt = 0;
    NvU64           gpu_page_cap = *gpu_page_cap_io;
    struct live_migration_gpu_page *gpu_pages = *gpu_pages_io;
    NV_STATUS       status = NV_OK;

    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
    list_for_each_entry(vs_iter, &g_uvm_global.va_spaces.list, list_node) {
        uvm_va_space_down_read(vs_iter);
        va_range = uvm_va_range_find(vs_iter, base);
        if (va_range && va_range->type == UVM_VA_RANGE_TYPE_MANAGED) {
            va_space = vs_iter;
            break;
        }
        uvm_va_space_up_read(vs_iter);
    }
    if (!va_space) {
        uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
        return NV_ERR_INVALID_ADDRESS;
    }

    {
        static const NvProcessorUuid z = {{0}};
        gpu = (memcmp(req_uuid, &z, sizeof(z)) == 0)
            ? uvm_processor_mask_find_first_va_space_gpu(&va_space->registered_gpus, va_space)
            : uvm_va_space_get_gpu_by_uuid(va_space, req_uuid);
    }

    uvm_for_each_va_range_in(va_range, va_space, base, end) {
        uvm_va_block_t *vb;
        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
            continue;
        for_each_va_block_in_va_range(va_range, vb) {
            NvU64 bs = max(vb->start, base);
            NvU64 be = min(vb->end,   end);
            NvU64 pv;
            if (bs > be)
                continue;
            /* Tier3-fused: skip non-dirty 2MB blocks entirely. Same
             * predicate + lock as GET_DIRTY_PAGES (ioctl 104); safe
             * because the app is quiesced at AT_BOUNDARY so the dirty
             * set is stable for the duration of this call. */
            if (dirty_only) {
                bool blk_dirty;
                uvm_mutex_lock(&vb->lock);
                blk_dirty = vb->live_migration.migration_tracking_enabled &&
                            vb->live_migration.dirty;
                uvm_mutex_unlock(&vb->lock);
                if (!blk_dirty)
                    continue;
            }
            for (pv = bs; pv <= be; pv += PAGE_SIZE) {
                uvm_page_index_t pi = uvm_va_block_cpu_page_index(vb, pv);
                NvU64            px = (pv - base) / PAGE_SIZE;
                uvm_va_block_gpu_state_t *gs;
                bool ic, ig;

                uvm_mutex_lock(&vb->lock);
                ic = uvm_page_mask_test(&vb->cpu.resident, pi);
                ig = false;
                if (!ic && gpu) {
                    gs = uvm_va_block_gpu_state_get(vb, gpu->id);
                    if (gs)
                        ig = uvm_page_mask_test(&gs->resident, pi);
                }

                if (ic) {
                    struct page *p = uvm_va_block_get_cpu_page(vb, pi);
                    void        *m = kmap(p);
                    memcpy(page_buf, m, PAGE_SIZE);
                    kunmap(m);
                    uvm_mutex_unlock(&vb->lock);
                    resmap[px] = 0;
                    if (user_cpu_buf &&
                        copy_to_user((void __user *)(user_cpu_buf + cpu_off), page_buf, PAGE_SIZE)) {
                        status = NV_ERR_INVALID_ARGUMENT;
                        goto done;
                    }
                    cpu_off += PAGE_SIZE;
                } else if (ig) {
                    uvm_gpu_phys_address_t phys =
                        uvm_va_block_gpu_phys_page_address(vb, pi, gpu);
                    uvm_mutex_unlock(&vb->lock);
                    resmap[px] = 1;
                    if (gpu_page_cnt >= gpu_page_cap) {
                        NvU64 nc = gpu_page_cap * 2;
                        struct live_migration_gpu_page *na =
                            kvmalloc_array(nc, sizeof(*na), GFP_KERNEL);
                        if (!na) {
                            status = NV_ERR_NO_MEMORY;
                            goto done;
                        }
                        memcpy(na, gpu_pages, gpu_page_cnt * sizeof(*na));
                        kvfree(gpu_pages);
                        gpu_pages    = na;
                        gpu_page_cap = nc;
                    }
                    gpu_pages[gpu_page_cnt].phys           = phys;
                    gpu_pages[gpu_page_cnt].gpu_buf_offset = gpu_off;
                    gpu_page_cnt++;
                    gpu_off += PAGE_SIZE;
                } else {
                    uvm_mutex_unlock(&vb->lock);
                }
            }
        }
    }

done:
    uvm_va_space_up_read(va_space);
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);
    *gpu_pages_io     = gpu_pages;
    *gpu_page_cap_io  = gpu_page_cap;
    *gpu_page_cnt_out = gpu_page_cnt;
    *cpu_off_out      = cpu_off;
    *gpu_off_out      = gpu_off;
    *gpu_out          = gpu;
    return status;
}

/*
 * Build the coalesced run list from gpu_pages[]. Each run is a contiguous
 * range of up to UVM_CONF_COMPUTING_DMA_BUFFER_SIZE bytes with sequential
 * physical addresses, same aperture, and sequential gpu_buf_offsets. The
 * caller must kvfree(*rl_s_out) and kvfree(*rl_b_out).
 */
static NV_STATUS lm_mt_build_runs(struct live_migration_gpu_page *gpu_pages,
                                  NvU64   gpu_page_cnt,
                                  NvU64   max_xfer,
                                  NvU64 **rl_s_out,
                                  NvU64 **rl_b_out,
                                  NvU64  *nr_out)
{
    NvU64 *rl_s = NULL, *rl_b = NULL, nr = 0, rc = 0;
    NvU64  i = 0;

    while (i < gpu_page_cnt) {
        NvU64 rs = i, rb = PAGE_SIZE;
        while (i + 1 < gpu_page_cnt &&
               gpu_pages[i+1].phys.address     == gpu_pages[i].phys.address + PAGE_SIZE &&
               gpu_pages[i+1].phys.aperture    == gpu_pages[rs].phys.aperture &&
               gpu_pages[i+1].gpu_buf_offset   == gpu_pages[i].gpu_buf_offset + PAGE_SIZE &&
               rb + PAGE_SIZE <= max_xfer) {
            i++;
            rb += PAGE_SIZE;
        }
        if (nr >= rc) {
            NvU64  nc = rc ? rc * 2 : 256;
            NvU64 *a  = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
            NvU64 *b  = kvmalloc_array(nc, sizeof(NvU64), GFP_KERNEL);
            if (!a || !b) {
                kvfree(a); kvfree(b);
                kvfree(rl_s); kvfree(rl_b);
                return NV_ERR_NO_MEMORY;
            }
            if (rl_s) { memcpy(a, rl_s, nr * sizeof(NvU64)); kvfree(rl_s); }
            if (rl_b) { memcpy(b, rl_b, nr * sizeof(NvU64)); kvfree(rl_b); }
            rl_s = a; rl_b = b; rc = nc;
        }
        rl_s[nr] = rs;
        rl_b[nr] = rb;
        nr++;
        i++;
    }

    *rl_s_out = rl_s;
    *rl_b_out = rl_b;
    *nr_out   = nr;
    return NV_OK;
}

/*
 * Spawn workers, wait for them all, return the first non-OK status.
 * If is_ciphertext, each worker also fills a disjoint slice of crypto_meta_buf.
 */
static NV_STATUS lm_mt_run_workers(uvm_gpu_t                          *gpu,
                                   struct live_migration_gpu_page     *gpu_pages,
                                   NvU64                              *rl_s,
                                   NvU64                              *rl_b,
                                   NvU64                               nr,
                                   NvU64                               user_gpu_buf,
                                   UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf,
                                   bool                                is_ciphertext,
                                   unsigned                            num_threads,
                                   NvU64                              *num_transfers_out)
{
    struct lm_mt_worker *workers;
    struct mm_struct    *mm;
    NvU64                per_thread;
    NvU64                remainder;
    NvU64                cursor = 0;
    NvU64                tnt    = 0;
    NV_STATUS            status = NV_OK;
    unsigned             i;

    mm = get_task_mm(current);
    if (!mm)
        return NV_ERR_INVALID_OPERATION;

    if (num_threads == 0 || num_threads > LM_MT_MAX_THREADS)
        num_threads = LM_MT_DEFAULT_THREADS;
    if (num_threads > nr)
        num_threads = (unsigned)max_t(NvU64, 1, nr);

    workers = kvmalloc_array(num_threads, sizeof(*workers), GFP_KERNEL);
    if (!workers) {
        mmput(mm);
        return NV_ERR_NO_MEMORY;
    }
    memset(workers, 0, num_threads * sizeof(*workers));

    per_thread = nr / num_threads;
    remainder  = nr % num_threads;

    for (i = 0; i < num_threads; i++) {
        NvU64 count = per_thread + (i < remainder ? 1 : 0);

        init_completion(&workers[i].done);
        workers[i].caller_mm         = mm;
        workers[i].gpu               = gpu;
        workers[i].gpu_pages         = gpu_pages;
        workers[i].rl_s              = rl_s;
        workers[i].rl_b              = rl_b;
        workers[i].slice_start       = cursor;
        workers[i].slice_count       = count;
        workers[i].user_gpu_buf      = user_gpu_buf;
        workers[i].crypto_meta_buf   = crypto_meta_buf;
        workers[i].crypto_meta_base  = cursor; /* run-list index == xfer index */
        workers[i].is_ciphertext     = is_ciphertext;
        workers[i].status            = NV_OK;
        workers[i].num_transfers     = 0;

        workers[i].task = kthread_run(lm_mt_worker_fn, &workers[i],
                                      "uvm-livmig-mt-%u", i);
        if (IS_ERR(workers[i].task)) {
            workers[i].task = NULL;
            workers[i].status = NV_ERR_NO_MEMORY;
            complete(&workers[i].done);
        }
        cursor += count;
    }

    /* Wait for all. */
    for (i = 0; i < num_threads; i++) {
        wait_for_completion(&workers[i].done);
        if (workers[i].status != NV_OK && status == NV_OK)
            status = workers[i].status;
        if (is_ciphertext)
            tnt += workers[i].num_transfers;
    }

    kvfree(workers);
    mmput(mm);

    if (is_ciphertext)
        *num_transfers_out = tnt;
    else
        *num_transfers_out = nr;  /* every run is one transfer */
    return status;
}

NV_STATUS uvm_api_live_migration_read_pages_resident_mt(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_MT_PARAMS *params,
    struct file *filp)
{
    uvm_gpu_t                      *gpu          = NULL;
    void                           *page_buf     = NULL;
    NvU8                           *resmap       = NULL;
    struct live_migration_gpu_page *gpu_pages    = NULL;
    NvU64                          *rl_s         = NULL;
    NvU64                          *rl_b         = NULL;
    NvU64                           base, length, n_pages;
    NvU64                           cpu_off = 0, gpu_off = 0;
    NvU64                           gpu_page_cnt = 0;
    NvU64                           gpu_page_cap = 0;
    NvU64                           nr = 0, nt = 0;
    NV_STATUS                       status = NV_OK;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;
    if (!g_uvm_global.conf_computing_enabled) {
        /* Non-CC path isn't wired for MT; fall back to single-threaded 109. */
        UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_PARAMS fb = {0};
        fb.base          = params->base;
        fb.length        = params->length;
        fb.cpu_buf       = params->cpu_buf;
        fb.cpu_buf_size  = params->cpu_buf_size;
        fb.gpu_buf       = params->gpu_buf;
        fb.gpu_buf_size  = params->gpu_buf_size;
        fb.residency_map = params->residency_map;
        fb.gpu_uuid      = params->gpu_uuid;
        status = uvm_api_live_migration_read_pages_resident(&fb, filp);
        params->cpu_bytes_out = fb.cpu_bytes_out;
        params->gpu_bytes_out = fb.gpu_bytes_out;
        params->num_pages     = fb.num_pages;
        params->rmStatus      = status;
        return status;
    }

    n_pages = length / PAGE_SIZE;

    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf) return NV_ERR_NO_MEMORY;

    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) { status = NV_ERR_NO_MEMORY; goto mt_out; }
    memset(resmap, 2, n_pages);

    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages = kvmalloc_array(gpu_page_cap, sizeof(*gpu_pages), GFP_KERNEL);
    if (!gpu_pages) { status = NV_ERR_NO_MEMORY; goto mt_out; }

    status = lm_mt_pass1(base, length, /*dirty_only=*/false,
                         params->cpu_buf, &params->gpu_uuid,
                         page_buf, resmap,
                         &gpu_pages, &gpu_page_cap, &gpu_page_cnt,
                         &cpu_off, &gpu_off, &gpu);
    if (status != NV_OK)
        goto mt_out;
    if (!gpu) { status = NV_ERR_INVALID_DEVICE; goto mt_out; }

    if (gpu_page_cnt > 0) {
        status = lm_mt_build_runs(gpu_pages, gpu_page_cnt,
                                  UVM_CONF_COMPUTING_DMA_BUFFER_SIZE,
                                  &rl_s, &rl_b, &nr);
        if (status != NV_OK)
            goto mt_out;

        status = lm_mt_run_workers(gpu, gpu_pages, rl_s, rl_b, nr,
                                   params->gpu_buf, NULL,
                                   /*is_ciphertext=*/false,
                                   params->num_threads, &nt);
        if (status != NV_OK)
            goto mt_out;
    }

    if (copy_to_user((void __user *)params->residency_map, resmap, n_pages)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto mt_out;
    }

    params->cpu_bytes_out = cpu_off;
    params->gpu_bytes_out = gpu_off;
    params->num_pages     = n_pages;

    printk(KERN_DEBUG "UVM LivMig: MT (114) done: %llu CPU, %llu GPU → %llu xfers (threads req=%u)\n",
           cpu_off / PAGE_SIZE, gpu_page_cnt, nr,
           (unsigned)params->num_threads);

mt_out:
    kvfree(rl_s);
    kvfree(rl_b);
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);
    params->rmStatus = status;
    return status;
}

NV_STATUS uvm_api_live_migration_read_pages_resident_encrypted_mt(
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS *params,
    struct file *filp)
{
    uvm_gpu_t                           *gpu             = NULL;
    void                                *page_buf        = NULL;
    NvU8                                *resmap          = NULL;
    struct live_migration_gpu_page      *gpu_pages       = NULL;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf = NULL;
    NvU64                               *rl_s            = NULL;
    NvU64                               *rl_b            = NULL;
    NvU64                                base, length, n_pages;
    NvU64                                cpu_off = 0, gpu_off = 0;
    NvU64                                gpu_page_cnt = 0;
    NvU64                                gpu_page_cap = 0;
    NvU64                                nr = 0, nt = 0;
    NV_STATUS                            status = NV_OK;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;
    if (!g_uvm_global.conf_computing_enabled)
        return NV_ERR_NOT_SUPPORTED;  /* ciphertext path needs CC */

    n_pages = length / PAGE_SIZE;

    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf) return NV_ERR_NO_MEMORY;

    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) { status = NV_ERR_NO_MEMORY; goto mte_out; }
    memset(resmap, 2, n_pages);

    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages = kvmalloc_array(gpu_page_cap, sizeof(*gpu_pages), GFP_KERNEL);
    if (!gpu_pages) { status = NV_ERR_NO_MEMORY; goto mte_out; }

    /* Worst-case: one transfer per page. We'll only fill the first `nr` entries. */
    crypto_meta_buf = kvmalloc_array(gpu_page_cap,
                                     sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META),
                                     GFP_KERNEL);
    if (!crypto_meta_buf) { status = NV_ERR_NO_MEMORY; goto mte_out; }

    status = lm_mt_pass1(base, length, /*dirty_only=*/false,
                         params->cpu_buf, &params->gpu_uuid,
                         page_buf, resmap,
                         &gpu_pages, &gpu_page_cap, &gpu_page_cnt,
                         &cpu_off, &gpu_off, &gpu);
    if (status != NV_OK)
        goto mte_out;
    if (!gpu) { status = NV_ERR_INVALID_DEVICE; goto mte_out; }

    /* gpu_page_cap may have grown; ensure crypto_meta_buf is at least as big. */
    if (gpu_page_cap > (NvU64)(((size_t)-1) / sizeof(*crypto_meta_buf))) {
        status = NV_ERR_NO_MEMORY;
        goto mte_out;
    }
    {
        UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *nm =
            kvmalloc_array(gpu_page_cap, sizeof(*nm), GFP_KERNEL);
        if (!nm) { status = NV_ERR_NO_MEMORY; goto mte_out; }
        kvfree(crypto_meta_buf);
        crypto_meta_buf = nm;
    }

    if (gpu_page_cnt > 0) {
        status = lm_mt_build_runs(gpu_pages, gpu_page_cnt,
                                  UVM_CONF_COMPUTING_DMA_BUFFER_SIZE,
                                  &rl_s, &rl_b, &nr);
        if (status != NV_OK)
            goto mte_out;

        status = lm_mt_run_workers(gpu, gpu_pages, rl_s, rl_b, nr,
                                   params->gpu_buf, crypto_meta_buf,
                                   /*is_ciphertext=*/true,
                                   params->num_threads, &nt);
        if (status != NV_OK)
            goto mte_out;
    }

    if (copy_to_user((void __user *)params->residency_map, resmap, n_pages)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto mte_out;
    }

    if (nt > 0 && params->crypto_meta) {
        NvU64 mb = nt * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META);
        if (mb > params->crypto_meta_size) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto mte_out;
        }
        if (copy_to_user((void __user *)params->crypto_meta, crypto_meta_buf, mb)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto mte_out;
        }
    }

    params->cpu_bytes_out = cpu_off;
    params->gpu_bytes_out = gpu_off;
    params->num_pages     = n_pages;
    params->num_transfers = nt;

    printk(KERN_DEBUG "UVM LivMig: MT-ENC (115) done: %llu CPU, %llu GPU → %llu xfers (threads req=%u)\n",
           cpu_off / PAGE_SIZE, gpu_page_cnt, nt,
           (unsigned)params->num_threads);

mte_out:
    kvfree(rl_s);
    kvfree(rl_b);
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);
    kvfree(crypto_meta_buf);
    params->rmStatus = status;
    return status;
}

// 116 – READ_DIRTY_DELTA_ENCRYPTED_MT (Tier3-fused dirty-delta, ciphertext)
//
// Byte-for-byte the same MT ciphertext pipeline as ioctl 115, except
// lm_mt_pass1 is invoked with dirty_only=true so the in-kernel VA walk
// includes ONLY pages residing in dirty 2MB va_blocks
// (migration_tracking_enabled && live_migration.dirty — the exact
// predicate GET_DIRTY_PAGES uses). This fuses GET_DIRTY_PAGES + the host's
// per-region 115 loop into ONE ioctl: the kthread / DMA-buffer / CE-channel
// / PASS-1 setup tax is paid once for the entire delta instead of once per
// dirty region. resmap[] is left sparse (entries for skipped non-dirty
// pages stay = 2); the host reconstructs VA → gpu_buf mapping from resmap
// exactly as the coalesced precopy path does.
//
// Requires CAP_SYS_ADMIN + CC. App must be quiesced at AT_BOUNDARY so the
// dirty set is stable for the duration of the call.
NV_STATUS uvm_api_live_migration_read_dirty_delta_encrypted_mt(
    UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT_PARAMS *params,
    struct file *filp)
{
    uvm_gpu_t                           *gpu             = NULL;
    void                                *page_buf        = NULL;
    NvU8                                *resmap          = NULL;
    struct live_migration_gpu_page      *gpu_pages       = NULL;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf = NULL;
    NvU64                               *rl_s            = NULL;
    NvU64                               *rl_b            = NULL;
    NvU64                                base, length, n_pages;
    NvU64                                cpu_off = 0, gpu_off = 0;
    NvU64                                gpu_page_cnt = 0;
    NvU64                                gpu_page_cap = 0;
    NvU64                                nr = 0, nt = 0;
    NV_STATUS                            status = NV_OK;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;
    if (!g_uvm_global.conf_computing_enabled)
        return NV_ERR_NOT_SUPPORTED;  /* ciphertext path needs CC */

    n_pages = length / PAGE_SIZE;

    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf) return NV_ERR_NO_MEMORY;

    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) { status = NV_ERR_NO_MEMORY; goto dte_out; }
    memset(resmap, 2, n_pages);

    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages = kvmalloc_array(gpu_page_cap, sizeof(*gpu_pages), GFP_KERNEL);
    if (!gpu_pages) { status = NV_ERR_NO_MEMORY; goto dte_out; }

    /* Worst-case: one transfer per page. We'll only fill the first `nr` entries. */
    crypto_meta_buf = kvmalloc_array(gpu_page_cap,
                                     sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META),
                                     GFP_KERNEL);
    if (!crypto_meta_buf) { status = NV_ERR_NO_MEMORY; goto dte_out; }

    status = lm_mt_pass1(base, length, /*dirty_only=*/true,
                         params->cpu_buf, &params->gpu_uuid,
                         page_buf, resmap,
                         &gpu_pages, &gpu_page_cap, &gpu_page_cnt,
                         &cpu_off, &gpu_off, &gpu);
    if (status != NV_OK)
        goto dte_out;
    if (!gpu) { status = NV_ERR_INVALID_DEVICE; goto dte_out; }

    /* gpu_page_cap may have grown; ensure crypto_meta_buf is at least as big. */
    if (gpu_page_cap > (NvU64)(((size_t)-1) / sizeof(*crypto_meta_buf))) {
        status = NV_ERR_NO_MEMORY;
        goto dte_out;
    }
    {
        UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *nm =
            kvmalloc_array(gpu_page_cap, sizeof(*nm), GFP_KERNEL);
        if (!nm) { status = NV_ERR_NO_MEMORY; goto dte_out; }
        kvfree(crypto_meta_buf);
        crypto_meta_buf = nm;
    }

    if (gpu_page_cnt > 0) {
        status = lm_mt_build_runs(gpu_pages, gpu_page_cnt,
                                  UVM_CONF_COMPUTING_DMA_BUFFER_SIZE,
                                  &rl_s, &rl_b, &nr);
        if (status != NV_OK)
            goto dte_out;

        status = lm_mt_run_workers(gpu, gpu_pages, rl_s, rl_b, nr,
                                   params->gpu_buf, crypto_meta_buf,
                                   /*is_ciphertext=*/true,
                                   params->num_threads, &nt);
        if (status != NV_OK)
            goto dte_out;
    }

    if (copy_to_user((void __user *)params->residency_map, resmap, n_pages)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto dte_out;
    }

    if (nt > 0 && params->crypto_meta) {
        NvU64 mb = nt * sizeof(UVM_LIVE_MIGRATION_PAGE_CRYPTO_META);
        if (mb > params->crypto_meta_size) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto dte_out;
        }
        if (copy_to_user((void __user *)params->crypto_meta, crypto_meta_buf, mb)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto dte_out;
        }
    }

    params->cpu_bytes_out = cpu_off;
    params->gpu_bytes_out = gpu_off;
    params->num_pages     = n_pages;
    params->num_transfers = nt;

    printk(KERN_DEBUG "UVM LivMig: DIRTY-DELTA (116) done: %llu CPU, %llu GPU → %llu xfers (threads req=%u)\n",
           cpu_off / PAGE_SIZE, gpu_page_cnt, nt,
           (unsigned)params->num_threads);

dte_out:
    kvfree(rl_s);
    kvfree(rl_b);
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);
    kvfree(crypto_meta_buf);
    params->rmStatus = status;
    return status;
}

/* ====================================================================== */
/* 117 - BENCH_DECOMPOSE                                                  */
/*                                                                        */
/* Instrumented FORK of lm_mt_worker_fn / lm_mt_run_workers. Kept separate */
/* on purpose: ioctls 114/115/116 stay free of timers and flag branches so */
/* the numbers we publish for the production path are measured on code     */
/* that has nothing added to it. Before reporting any decomposition, run   */
/* 117 and 115 on the same transfer and confirm aggregate GB/s matches --  */
/* otherwise the breakdown does not describe the shipped path.             */
/* ====================================================================== */

#define LM_BENCH_SLOTS 2

struct lm_bench_worker {
    struct task_struct *task;
    struct completion   done;

    /* shared inputs (read-only) */
    struct mm_struct                *caller_mm;
    uvm_gpu_t                       *gpu;
    struct live_migration_gpu_page  *gpu_pages;
    NvU64                           *rl_s;
    NvU64                           *rl_b;
    NvU64                            slice_start;
    NvU64                            slice_count;
    NvU64                            user_gpu_buf;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf;
    NvU64                            crypto_meta_base;
    NvU32                            flags;

    /* outputs */
    NV_STATUS status;
    NvU64     num_transfers;
    NvU64     t_submit_ns;
    NvU64     t_iv_ns;
    NvU64     t_ce_wait_ns;
    NvU64     t_decrypt_ns;
    NvU64     t_copy_ns;
    NvU64     t_worker_ns;
};

/* Wait for one in-flight CE submission and consume its buffer. */
static NV_STATUS lm_bench_process_slot(struct lm_bench_worker          *w,
                                       uvm_push_t                      *push,
                                       uvm_conf_computing_dma_buffer_t *db,
                                       NvU64                            rs,
                                       NvU64                            rb,
                                       NvU64                            ti,
                                       void                            *plain_buf,
                                       void                            *sink_buf)
{
    NV_STATUS status;
    NvU64     t0, t1;
    void     *cipher, *atag, *src;

    t0 = NV_GETTIME();
    status = uvm_push_wait(push);
    t1 = NV_GETTIME();
    w->t_ce_wait_ns += t1 - t0;
    if (status != NV_OK)
        return status;

    cipher = uvm_mem_get_cpu_addr_kernel(db->alloc);
    atag   = uvm_mem_get_cpu_addr_kernel(db->auth_tag);

    /* Mirror 115's in-loop metadata bookkeeping so the fork performs the
     * same per-transfer work as the production path. */
    if (w->crypto_meta_buf) {
        w->crypto_meta_buf[ti].size        = rb;
        memcpy(w->crypto_meta_buf[ti].iv, db->decrypt_iv[0].iv, 12);
        w->crypto_meta_buf[ti].iv_fresh    = db->decrypt_iv[0].fresh;
        memcpy(w->crypto_meta_buf[ti].auth_tag, atag, UVM_CONF_COMPUTING_AUTH_TAG_SIZE);
        w->crypto_meta_buf[ti].key_version = db->key_version[0];
    }

    src = cipher;

    if (w->flags & UVM_LIVE_MIGRATION_BENCH_DECRYPT) {
        t0 = NV_GETTIME();
        status = uvm_conf_computing_cpu_decrypt(push->channel, plain_buf, cipher,
                                                db->decrypt_iv, db->key_version[0],
                                                rb, atag);
        t1 = NV_GETTIME();
        w->t_decrypt_ns += t1 - t0;
        if (status != NV_OK)
            return status;
        src = plain_buf;
    }

    (void)src;  /* unused when SKIP_COPY is set without DECRYPT */

    if (!(w->flags & UVM_LIVE_MIGRATION_BENCH_SKIP_COPY)) {
        t0 = NV_GETTIME();
        if (w->flags & UVM_LIVE_MIGRATION_BENCH_COPY_TO_SHARED) {
            /* shared -> shared: excludes the TME cost of a private write */
            memcpy(sink_buf, src, rb);
        }
        else if (w->user_gpu_buf) {
            if (copy_to_user((void __user *)(w->user_gpu_buf +
                                             w->gpu_pages[rs].gpu_buf_offset),
                             src, rb))
                status = NV_ERR_INVALID_ARGUMENT;
        }
        t1 = NV_GETTIME();
        w->t_copy_ns += t1 - t0;
    }

    return status;
}

static int lm_bench_worker_fn(void *data)
{
    struct lm_bench_worker          *w = data;
    uvm_conf_computing_dma_buffer_t *db[LM_BENCH_SLOTS] = {NULL, NULL};
    uvm_conf_computing_dma_buffer_t *sink_db = NULL;
    uvm_push_t                       pu[LM_BENCH_SLOTS];
    int                              act[LM_BENCH_SLOTS] = {0, 0};
    NvU64                            s_rs[LM_BENCH_SLOTS];
    NvU64                            s_rb[LM_BENCH_SLOTS];
    NvU64                            s_ti[LM_BENCH_SLOTS];
    void                            *plain_buf = NULL;
    void                            *sink_buf  = NULL;
    NV_STATUS                        status = NV_OK;
    NvU64                            i, w_t0;
    NvU64                            local_nt = w->crypto_meta_base;
    bool                             serial;
    int                              c, nslots;

    serial = (w->flags & UVM_LIVE_MIGRATION_BENCH_SERIAL) != 0;
    nslots = serial ? 1 : LM_BENCH_SLOTS;
    w_t0   = NV_GETTIME();

    kthread_use_mm(w->caller_mm);

    if (w->flags & UVM_LIVE_MIGRATION_BENCH_DECRYPT) {
        plain_buf = kmalloc(UVM_CONF_COMPUTING_DMA_BUFFER_SIZE, GFP_KERNEL);
        if (!plain_buf) { status = NV_ERR_NO_MEMORY; goto b_unuse; }
    }

    for (c = 0; c < nslots; c++) {
        status = uvm_conf_computing_dma_buffer_alloc(&w->gpu->conf_computing.dma_buffer_pool,
                                                     &db[c], NULL);
        if (status != NV_OK)
            goto b_free;
    }

    if (w->flags & UVM_LIVE_MIGRATION_BENCH_COPY_TO_SHARED) {
        status = uvm_conf_computing_dma_buffer_alloc(&w->gpu->conf_computing.dma_buffer_pool,
                                                     &sink_db, NULL);
        if (status != NV_OK)
            goto b_free;
        sink_buf = uvm_mem_get_cpu_addr_kernel(sink_db->alloc);
    }

    for (i = 0; i < w->slice_count; i++) {
        NvU64             rl_i = w->slice_start + i;
        uvm_gpu_address_t sa, da, ta;
        NvU64             t0, t1, iv0;

        c = serial ? 0 : (int)(i & 1);

        /* Pipelined: consume the previous submission on this slot first. */
        if (!serial && act[c]) {
            status = lm_bench_process_slot(w, &pu[c], db[c], s_rs[c], s_rb[c],
                                           s_ti[c], plain_buf, sink_buf);
            act[c] = 0;
            if (status != NV_OK)
                goto b_drain;
        }

        t0 = NV_GETTIME();
        status = uvm_push_begin(w->gpu->channel_manager,
                                UVM_CHANNEL_TYPE_GPU_TO_CPU,
                                &pu[c], "LivMig bench[%llu:%d]",
                                (unsigned long long)w->slice_start, c);
        if (status != NV_OK) {
            w->t_submit_ns += NV_GETTIME() - t0;
            goto b_drain;
        }

        iv0 = NV_GETTIME();
        uvm_conf_computing_log_gpu_encryption(pu[c].channel, w->rl_b[rl_i], db[c]->decrypt_iv);
        w->t_iv_ns += NV_GETTIME() - iv0;

        db[c]->key_version[0] = uvm_channel_pool_key_version(pu[c].channel->pool);

        sa = uvm_gpu_address_copy(w->gpu, w->gpu_pages[w->rl_s[rl_i]].phys);
        da = uvm_mem_gpu_address_virtual_kernel(db[c]->alloc, w->gpu);
        ta = uvm_mem_gpu_address_virtual_kernel(db[c]->auth_tag, w->gpu);
        w->gpu->parent->ce_hal->encrypt(&pu[c], da, sa, w->rl_b[rl_i], ta);
        uvm_push_end(&pu[c]);
        t1 = NV_GETTIME();
        w->t_submit_ns += t1 - t0;

        act[c]  = 1;
        s_rs[c] = w->rl_s[rl_i];
        s_rb[c] = w->rl_b[rl_i];
        s_ti[c] = local_nt++;
        w->num_transfers++;

        /* Serial: no overlap, so per-stage times sum to wall time. */
        if (serial) {
            status = lm_bench_process_slot(w, &pu[c], db[c], s_rs[c], s_rb[c],
                                           s_ti[c], plain_buf, sink_buf);
            act[c] = 0;
            if (status != NV_OK)
                goto b_drain;
        }
    }

b_drain:
    for (c = 0; c < nslots; c++) {
        NV_STATUS ds;
        if (!act[c])
            continue;
        ds = lm_bench_process_slot(w, &pu[c], db[c], s_rs[c], s_rb[c],
                                   s_ti[c], plain_buf, sink_buf);
        if (ds != NV_OK && status == NV_OK)
            status = ds;
        act[c] = 0;
    }

b_free:
    if (sink_db)
        uvm_conf_computing_dma_buffer_free(&w->gpu->conf_computing.dma_buffer_pool, sink_db, NULL);
    for (c = LM_BENCH_SLOTS - 1; c >= 0; c--) {
        if (db[c])
            uvm_conf_computing_dma_buffer_free(&w->gpu->conf_computing.dma_buffer_pool, db[c], NULL);
    }
    kfree(plain_buf);

b_unuse:
    kthread_unuse_mm(w->caller_mm);
    w->t_worker_ns = NV_GETTIME() - w_t0;
    w->status      = status;
    complete(&w->done);
    return 0;
}

static NV_STATUS lm_bench_run_workers(uvm_gpu_t                           *gpu,
                                      struct live_migration_gpu_page      *gpu_pages,
                                      NvU64                               *rl_s,
                                      NvU64                               *rl_b,
                                      NvU64                                nr,
                                      NvU64                                user_gpu_buf,
                                      UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf,
                                      NvU32                                flags,
                                      unsigned                             num_threads,
                                      UVM_LIVE_MIGRATION_BENCH_DECOMPOSE_PARAMS *params)
{
    struct lm_bench_worker *workers;
    struct mm_struct       *mm;
    NvU64                   per_thread, remainder, cursor = 0, tnt = 0;
    NV_STATUS               status = NV_OK;
    unsigned                i;

    mm = get_task_mm(current);
    if (!mm)
        return NV_ERR_INVALID_OPERATION;

    if (num_threads == 0 || num_threads > LM_MT_MAX_THREADS)
        num_threads = LM_MT_DEFAULT_THREADS;
    if (num_threads > nr)
        num_threads = (unsigned)max_t(NvU64, 1, nr);

    workers = kvmalloc_array(num_threads, sizeof(*workers), GFP_KERNEL);
    if (!workers) {
        mmput(mm);
        return NV_ERR_NO_MEMORY;
    }
    memset(workers, 0, num_threads * sizeof(*workers));

    per_thread = nr / num_threads;
    remainder  = nr % num_threads;

    for (i = 0; i < num_threads; i++) {
        NvU64 count = per_thread + (i < remainder ? 1 : 0);

        init_completion(&workers[i].done);
        workers[i].caller_mm        = mm;
        workers[i].gpu              = gpu;
        workers[i].gpu_pages        = gpu_pages;
        workers[i].rl_s             = rl_s;
        workers[i].rl_b             = rl_b;
        workers[i].slice_start      = cursor;
        workers[i].slice_count      = count;
        workers[i].user_gpu_buf     = user_gpu_buf;
        workers[i].crypto_meta_buf  = crypto_meta_buf;
        workers[i].crypto_meta_base = cursor;
        workers[i].flags            = flags;
        workers[i].status           = NV_OK;

        workers[i].task = kthread_run(lm_bench_worker_fn, &workers[i],
                                      "uvm-livmig-bench-%u", i);
        if (IS_ERR(workers[i].task)) {
            workers[i].task   = NULL;
            workers[i].status = NV_ERR_NO_MEMORY;
            complete(&workers[i].done);
        }
        cursor += count;
    }

    for (i = 0; i < num_threads; i++) {
        wait_for_completion(&workers[i].done);
        if (workers[i].status != NV_OK && status == NV_OK)
            status = workers[i].status;

        tnt                   += workers[i].num_transfers;
        params->t_submit_ns   += workers[i].t_submit_ns;
        params->t_iv_ns       += workers[i].t_iv_ns;
        params->t_ce_wait_ns  += workers[i].t_ce_wait_ns;
        params->t_decrypt_ns  += workers[i].t_decrypt_ns;
        params->t_copy_ns     += workers[i].t_copy_ns;
        if (workers[i].t_worker_ns > params->t_worker_max_ns)
            params->t_worker_max_ns = workers[i].t_worker_ns;
    }

    params->num_transfers = tnt;
    params->threads_used  = num_threads;

    kvfree(workers);
    mmput(mm);
    return status;
}

NV_STATUS uvm_api_live_migration_bench_decompose(
    UVM_LIVE_MIGRATION_BENCH_DECOMPOSE_PARAMS *params,
    struct file *filp)
{
    uvm_gpu_t                           *gpu             = NULL;
    void                                *page_buf        = NULL;
    NvU8                                *resmap          = NULL;
    struct live_migration_gpu_page      *gpu_pages       = NULL;
    UVM_LIVE_MIGRATION_PAGE_CRYPTO_META *crypto_meta_buf = NULL;
    NvU64                               *rl_s            = NULL;
    NvU64                               *rl_b            = NULL;
    NvU64                                base, length, n_pages;
    NvU64                                cpu_off = 0, gpu_off = 0;
    NvU64                                gpu_page_cnt = 0;
    NvU64                                gpu_page_cap = 0;
    NvU64                                nr = 0;
    NvU64                                t_start;
    NV_STATUS                            status = NV_OK;

    if (!capable(CAP_SYS_ADMIN))
        return NV_ERR_INSUFFICIENT_PERMISSIONS;

    base   = params->base;
    length = params->length;

    if (length == 0 || (length % PAGE_SIZE) != 0 || (base % PAGE_SIZE) != 0)
        return NV_ERR_INVALID_ARGUMENT;
    if (!params->residency_map)
        return NV_ERR_INVALID_ARGUMENT;
    if (!g_uvm_global.conf_computing_enabled)
        return NV_ERR_NOT_SUPPORTED;

    params->t_submit_ns     = 0;
    params->t_iv_ns         = 0;
    params->t_ce_wait_ns    = 0;
    params->t_decrypt_ns    = 0;
    params->t_copy_ns       = 0;
    params->t_worker_max_ns = 0;
    params->num_transfers   = 0;
    params->threads_used    = 0;

    t_start = NV_GETTIME();
    n_pages = length / PAGE_SIZE;

    page_buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!page_buf)
        return NV_ERR_NO_MEMORY;

    resmap = kvmalloc(n_pages, GFP_KERNEL);
    if (!resmap) { status = NV_ERR_NO_MEMORY; goto bd_out; }
    memset(resmap, 2, n_pages);

    gpu_page_cap = min(n_pages, (NvU64)512);
    gpu_pages = kvmalloc_array(gpu_page_cap, sizeof(*gpu_pages), GFP_KERNEL);
    if (!gpu_pages) { status = NV_ERR_NO_MEMORY; goto bd_out; }

    status = lm_mt_pass1(base, length, /*dirty_only=*/false,
                         params->cpu_buf, &params->gpu_uuid,
                         page_buf, resmap,
                         &gpu_pages, &gpu_page_cap, &gpu_page_cnt,
                         &cpu_off, &gpu_off, &gpu);
    if (status != NV_OK)
        goto bd_out;
    if (!gpu) { status = NV_ERR_INVALID_DEVICE; goto bd_out; }

    crypto_meta_buf = kvmalloc_array(gpu_page_cap, sizeof(*crypto_meta_buf), GFP_KERNEL);
    if (!crypto_meta_buf) { status = NV_ERR_NO_MEMORY; goto bd_out; }

    if (gpu_page_cnt > 0) {
        status = lm_mt_build_runs(gpu_pages, gpu_page_cnt,
                                  UVM_CONF_COMPUTING_DMA_BUFFER_SIZE,
                                  &rl_s, &rl_b, &nr);
        if (status != NV_OK)
            goto bd_out;

        status = lm_bench_run_workers(gpu, gpu_pages, rl_s, rl_b, nr,
                                      params->gpu_buf, crypto_meta_buf,
                                      params->dbg_flags, params->num_threads,
                                      params);
        if (status != NV_OK)
            goto bd_out;
    }

    if (copy_to_user((void __user *)params->residency_map, resmap, n_pages)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto bd_out;
    }

    params->cpu_bytes_out = cpu_off;
    params->gpu_bytes_out = gpu_off;
    params->num_pages     = n_pages;
    params->t_total_ns    = NV_GETTIME() - t_start;

    printk(KERN_DEBUG "UVM LivMig: BENCH (117) flags=0x%x threads=%u xfers=%llu "
                      "gpu=%lluMB submit=%lluus iv=%lluus ce=%lluus dec=%lluus copy=%lluus total=%lluus\n",
           (unsigned)params->dbg_flags, (unsigned)params->threads_used,
           params->num_transfers, gpu_off / (1024 * 1024),
           params->t_submit_ns / 1000, params->t_iv_ns / 1000,
           params->t_ce_wait_ns / 1000, params->t_decrypt_ns / 1000,
           params->t_copy_ns / 1000, params->t_total_ns / 1000);

bd_out:
    kvfree(rl_s);
    kvfree(rl_b);
    kfree(page_buf);
    kvfree(resmap);
    kvfree(gpu_pages);
    kvfree(crypto_meta_buf);
    params->rmStatus = status;
    return status;
}
