/**
 * libckpt_restore.c  –  Checkpoint memory reassembly shared library
 *
 * Implements ckpt_restore_from_files(): reads base.img + delta.img and
 * returns reassembled memory blocks as heap-allocated buffers.
 *
 * Compile as shared library:
 *   gcc -O2 -shared -fPIC -o libckpt_restore.so libckpt_restore.c
 *
 * Compile for static linking:
 *   gcc -O2 -c libckpt_restore.c -o libckpt_restore.o
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <openssl/evp.h>

#include "librst_vllm.h"

/* ------------------------------------------------------------------ */
/* Image format constants — must match test_inc_ckpt_agent.c exactly  */
/* ------------------------------------------------------------------ */
#define CKPT_V2_MAGIC         0xC2C2C2C2u
#define CKPT_V3_VERSION       3u
#define CKPT_INC_DELTA_MAGIC  0xDE17A001u
#define PAGE_SZ               4096ULL
#define MAX_GROUPS            64
#define MAX_RANGES            256

/* UVM ioctl for k1 decryption */
#define UVM_INITIALIZE                           0x30000001
#define UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES 112

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t num_ranges;
    uint32_t page_size;
    uint64_t total_bytes;
} ckpt_v2_file_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va;
    uint64_t length;
    uint64_t num_pages;
    uint64_t data_offset;
    uint64_t resmap_size;
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    uint64_t crypto_meta_bytes;
} ckpt_v2_range_desc_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t num_blocks;
    uint32_t page_size;
    uint64_t alloc_start_va;
    uint64_t alloc_size;
    uint64_t num_dirty_pages_64k;
} ckpt_inc_delta_hdr_t;

typedef struct __attribute__((packed)) {
    uint64_t base_va;
    uint64_t length;
    uint64_t num_pages;
    uint64_t data_offset;
    uint64_t resmap_size;
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    uint64_t crypto_meta_bytes;
} ckpt_inc_delta_block_t;

/* Per-transfer crypto metadata (must match agent's page_crypto_meta_t) */
typedef struct __attribute__((packed)) {
    uint64_t size;
    uint8_t  iv[12];
    uint8_t  iv_fresh;
    uint8_t  auth_tag[16];
    uint32_t key_version;
} crypto_meta_t;  /* 41 bytes */

/* UVM ioctl param structs */
typedef struct {
    uint64_t flags __attribute__((aligned(8)));
    uint32_t rmStatus;
} uvm_init_params_t;

typedef struct { uint8_t uuid[16]; } nv_uuid_t;

typedef struct {
    uint64_t cipher_buf  __attribute__((aligned(8)));
    uint64_t plain_buf   __attribute__((aligned(8)));
    uint64_t total_size  __attribute__((aligned(8)));
    uint64_t crypto_meta __attribute__((aligned(8)));
    uint64_t num_transfers __attribute__((aligned(8)));
    nv_uuid_t gpu_uuid;
    uint32_t rmStatus;
} uvm_decrypt_params_t;

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */
static int cmp_range_va(const void *a, const void *b)
{
    const ckpt_v2_range_desc_t *ra = (const ckpt_v2_range_desc_t *)a;
    const ckpt_v2_range_desc_t *rb = (const ckpt_v2_range_desc_t *)b;
    if (ra->base_va < rb->base_va) return -1;
    if (ra->base_va > rb->base_va) return  1;
    return 0;
}

static int read_bytes(FILE *f, uint64_t offset, void *dst, uint64_t size)
{
    if (size == 0) return 0;
    if (fseek(f, (long)offset, SEEK_SET) != 0) {
        fprintf(stderr, "libckpt_restore: fseek(0x%llx): %s\n",
                (unsigned long long)offset, strerror(errno));
        return -1;
    }
    if (fread(dst, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "libckpt_restore: fread(%llu @ 0x%llx): %s\n",
                (unsigned long long)size,
                (unsigned long long)offset, strerror(errno));
        return -1;
    }
    return 0;
}

/*
 * assemble_data: walk resmap for one range/block, copy CPU/GPU pages
 * into the correct offset within the owning group's buffer.
 * Absent pages stay zero (from calloc).
 */
static int assemble_data(FILE *f,
                         uint64_t base_va,
                         uint64_t num_pages,
                         uint64_t data_offset,
                         uint64_t resmap_size,
                         uint64_t cpu_bytes,
                         uint64_t gpu_bytes,
                         uint8_t *group_buf,
                         uint64_t group_orig_va)
{
    uint8_t *range_buf = group_buf + (base_va - group_orig_va);

    uint8_t *resmap      = malloc(num_pages);
    uint8_t *cpu_section = malloc(cpu_bytes  ? cpu_bytes  : 1);
    uint8_t *gpu_section = malloc(gpu_bytes  ? gpu_bytes  : 1);

    if (!resmap || !cpu_section || !gpu_section) {
        fprintf(stderr, "libckpt_restore: malloc failed\n");
        free(resmap); free(cpu_section); free(gpu_section);
        return -1;
    }

    uint64_t cpu_sec_off = data_offset + resmap_size;
    uint64_t gpu_sec_off = cpu_sec_off + cpu_bytes;

    if (read_bytes(f, data_offset, resmap,      resmap_size) < 0 ||
        read_bytes(f, cpu_sec_off, cpu_section, cpu_bytes)   < 0 ||
        read_bytes(f, gpu_sec_off, gpu_section, gpu_bytes)   < 0) {
        free(resmap); free(cpu_section); free(gpu_section);
        return -1;
    }

    uint64_t cpu_off = 0, gpu_off = 0;

    for (uint64_t p = 0; p < num_pages; p++) {
        uint8_t *dst = range_buf + p * PAGE_SZ;
        if (resmap[p] == 0) {               /* CPU-resident */
            memcpy(dst, cpu_section + cpu_off, PAGE_SZ);
            cpu_off += PAGE_SZ;
        } else if (resmap[p] == 1) {        /* GPU-resident */
            memcpy(dst, gpu_section + gpu_off, PAGE_SZ);
            gpu_off += PAGE_SZ;
        }
        /* absent (2) -> stays zero from calloc */
    }

    free(resmap); free(cpu_section); free(gpu_section);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int ckpt_restore_from_files(const char *base_path,
                            const char *delta_path,
                            ckpt_restore_result_t *result)
{
    memset(result, 0, sizeof(*result));

    /* Temporary working arrays */
    uint64_t  group_va[MAX_GROUPS];
    uint64_t  group_sz[MAX_GROUPS];
    uint8_t  *group_buf[MAX_GROUPS];
    uint32_t  num_groups = 0;

    /* ==============================================================
     * STEP 1: Read baseline image
     * ============================================================== */
    FILE *fb = fopen(base_path, "rb");
    if (!fb) {
        fprintf(stderr, "libckpt_restore: fopen(%s): %s\n",
                base_path, strerror(errno));
        return -1;
    }

    ckpt_v2_file_hdr_t fhdr;
    if (fread(&fhdr, sizeof(fhdr), 1, fb) != 1 ||
        fhdr.magic != CKPT_V2_MAGIC) {
        fprintf(stderr, "libckpt_restore: invalid base image header\n");
        fclose(fb); return -1;
    }

    uint32_t num_ranges = fhdr.num_ranges;
    if (num_ranges > MAX_RANGES) {
        fprintf(stderr, "libckpt_restore: too many ranges (%u)\n", num_ranges);
        fclose(fb); return -1;
    }

    ckpt_v2_range_desc_t *descs =
        malloc(num_ranges * sizeof(ckpt_v2_range_desc_t));
    if (!descs) { fclose(fb); return -1; }

    if (fread(descs, sizeof(ckpt_v2_range_desc_t), num_ranges, fb)
            != (size_t)num_ranges) {
        fprintf(stderr, "libckpt_restore: short read on range descriptors\n");
        free(descs); fclose(fb); return -1;
    }

    /* Sort by VA and group contiguous ranges */
    qsort(descs, num_ranges, sizeof(ckpt_v2_range_desc_t), cmp_range_va);

    int range_group[MAX_RANGES];

    for (uint32_t i = 0; i < num_ranges; i++) {
        uint64_t va_end = descs[i].base_va + descs[i].length;
        if (num_groups == 0 ||
            descs[i].base_va > group_va[num_groups-1] +
                               group_sz[num_groups-1]) {
            if (num_groups >= MAX_GROUPS) {
                fprintf(stderr, "libckpt_restore: too many groups\n");
                free(descs); fclose(fb); return -1;
            }
            group_va[num_groups]  = descs[i].base_va;
            group_sz[num_groups]  = descs[i].length;
            group_buf[num_groups] = NULL;
            num_groups++;
        } else {
            uint64_t gend = group_va[num_groups-1] + group_sz[num_groups-1];
            if (va_end > gend)
                group_sz[num_groups-1] += (va_end - gend);
        }
        range_group[i] = (int)(num_groups - 1);
    }

    /* Allocate zero-filled buffers */
    for (uint32_t g = 0; g < num_groups; g++) {
        group_buf[g] = calloc(1, (size_t)group_sz[g]);
        if (!group_buf[g]) {
            fprintf(stderr, "libckpt_restore: calloc(%.2f MB) failed\n",
                    group_sz[g] / (1024.0*1024.0));
            for (uint32_t j = 0; j < g; j++) free(group_buf[j]);
            free(descs); fclose(fb); return -1;
        }
    }

    /* Assemble each baseline range */
    for (uint32_t ri = 0; ri < num_ranges; ri++) {
        int g = range_group[ri];
        if (assemble_data(fb,
                          descs[ri].base_va,    descs[ri].num_pages,
                          descs[ri].data_offset, descs[ri].resmap_size,
                          descs[ri].cpu_bytes,   descs[ri].gpu_bytes,
                          group_buf[g],          group_va[g]) < 0) {
            for (uint32_t j = 0; j < num_groups; j++) free(group_buf[j]);
            free(descs); fclose(fb); return -1;
        }
    }
    fclose(fb);
    free(descs);

    /* ==============================================================
     * STEP 2: Apply delta
     * ============================================================== */
    FILE *fd = fopen(delta_path, "rb");
    if (!fd) {
        fprintf(stderr, "libckpt_restore: fopen(%s): %s\n",
                delta_path, strerror(errno));
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        return -1;
    }

    ckpt_inc_delta_hdr_t dhdr;
    if (fread(&dhdr, sizeof(dhdr), 1, fd) != 1 ||
        dhdr.magic != CKPT_INC_DELTA_MAGIC) {
        fprintf(stderr, "libckpt_restore: invalid delta image header\n");
        fclose(fd);
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        return -1;
    }

    if (dhdr.num_blocks > 0) {
        ckpt_inc_delta_block_t *blocks =
            malloc(dhdr.num_blocks * sizeof(ckpt_inc_delta_block_t));
        if (!blocks) {
            fclose(fd);
            for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
            return -1;
        }

        if (fread(blocks, sizeof(ckpt_inc_delta_block_t), dhdr.num_blocks, fd)
                != (size_t)dhdr.num_blocks) {
            fprintf(stderr, "libckpt_restore: short read on delta blocks\n");
            free(blocks); fclose(fd);
            for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
            return -1;
        }

        for (uint32_t bi = 0; bi < dhdr.num_blocks; bi++) {
            ckpt_inc_delta_block_t *b = &blocks[bi];

            int block_g = -1;
            for (uint32_t g = 0; g < num_groups; g++) {
                if (b->base_va >= group_va[g] &&
                    b->base_va + b->length <= group_va[g] + group_sz[g]) {
                    block_g = (int)g;
                    break;
                }
            }
            if (block_g < 0) {
                fprintf(stderr,
                        "libckpt_restore: delta block VA 0x%llx not in any group\n",
                        (unsigned long long)b->base_va);
                free(blocks); fclose(fd);
                for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
                return -1;
            }

            if (assemble_data(fd,
                              b->base_va,     b->num_pages,
                              b->data_offset, b->resmap_size,
                              b->cpu_bytes,   b->gpu_bytes,
                              group_buf[block_g], group_va[block_g]) < 0) {
                free(blocks); fclose(fd);
                for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
                return -1;
            }
        }
        free(blocks);
    }
    fclose(fd);

    /* ==============================================================
     * STEP 3: Populate result
     * ============================================================== */
    result->num_blocks = num_groups;
    result->blocks = malloc(num_groups * sizeof(ckpt_mem_block_t));
    if (!result->blocks) {
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        return -1;
    }

    for (uint32_t g = 0; g < num_groups; g++) {
        result->blocks[g].orig_va = group_va[g];
        result->blocks[g].size    = group_sz[g];
        result->blocks[g].data    = group_buf[g];  /* ownership transferred */
    }

    return 0;
}

/*
 * assemble_data_selective: like assemble_data, but only fills pages that
 * overlap with [target_va, target_va + target_size).  Pages outside the
 * target range are skipped.  dst_buf is sized exactly target_size.
 */
static int assemble_data_selective(FILE *f,
                                   uint64_t range_base_va,
                                   uint64_t num_pages,
                                   uint64_t data_offset,
                                   uint64_t resmap_size,
                                   uint64_t cpu_bytes,
                                   uint64_t gpu_bytes,
                                   uint8_t *dst_buf,
                                   uint64_t target_va,
                                   uint64_t target_size)
{
    uint8_t *resmap      = malloc(num_pages);
    uint8_t *cpu_section = malloc(cpu_bytes  ? cpu_bytes  : 1);
    uint8_t *gpu_section = malloc(gpu_bytes  ? gpu_bytes  : 1);

    if (!resmap || !cpu_section || !gpu_section) {
        fprintf(stderr, "libckpt_restore: malloc failed (selective)\n");
        free(resmap); free(cpu_section); free(gpu_section);
        return -1;
    }

    uint64_t cpu_sec_off = data_offset + resmap_size;
    uint64_t gpu_sec_off = cpu_sec_off + cpu_bytes;

    if (read_bytes(f, data_offset, resmap,      resmap_size) < 0 ||
        read_bytes(f, cpu_sec_off, cpu_section, cpu_bytes)   < 0 ||
        read_bytes(f, gpu_sec_off, gpu_section, gpu_bytes)   < 0) {
        free(resmap); free(cpu_section); free(gpu_section);
        return -1;
    }

    uint64_t cpu_off = 0, gpu_off = 0;
    uint64_t target_end = target_va + target_size;

    for (uint64_t p = 0; p < num_pages; p++) {
        uint64_t page_va = range_base_va + p * PAGE_SZ;
        uint64_t page_end = page_va + PAGE_SZ;

        /* Determine source data pointer */
        const uint8_t *src = NULL;
        if (resmap[p] == 0) {
            src = cpu_section + cpu_off;
            cpu_off += PAGE_SZ;
        } else if (resmap[p] == 1) {
            src = gpu_section + gpu_off;
            gpu_off += PAGE_SZ;
        }
        /* absent (2): src stays NULL, target bytes stay zero */

        /* Check if this page overlaps with the target range */
        if (page_end <= target_va || page_va >= target_end)
            continue;  /* no overlap */

        if (!src)
            continue;  /* absent page in target range — already zero from calloc */

        /* Compute overlap region */
        uint64_t copy_start = (page_va >= target_va) ? page_va : target_va;
        uint64_t copy_end   = (page_end <= target_end) ? page_end : target_end;
        uint64_t copy_len   = copy_end - copy_start;

        uint64_t src_offset = copy_start - page_va;
        uint64_t dst_offset = copy_start - target_va;

        memcpy(dst_buf + dst_offset, src + src_offset, copy_len);
    }

    free(resmap); free(cpu_section); free(gpu_section);
    return 0;
}

int ckpt_restore_selective(const char *base_path,
                           const char *delta_path,
                           const uint64_t *requested_vas,
                           const uint64_t *requested_sizes,
                           uint32_t num_requests,
                           ckpt_restore_result_t *result)
{
    memset(result, 0, sizeof(*result));

    if (num_requests == 0) return 0;

    /* Allocate output blocks */
    result->num_blocks = num_requests;
    result->blocks = malloc(num_requests * sizeof(ckpt_mem_block_t));
    if (!result->blocks) return -1;

    for (uint32_t i = 0; i < num_requests; i++) {
        result->blocks[i].orig_va = requested_vas[i];
        result->blocks[i].size    = requested_sizes[i];
        result->blocks[i].data    = calloc(1, (size_t)requested_sizes[i]);
        if (!result->blocks[i].data) {
            fprintf(stderr, "libckpt_restore: calloc(%llu) failed for request %u\n",
                    (unsigned long long)requested_sizes[i], i);
            ckpt_restore_free(result);
            return -1;
        }
    }

    /* ---- Process baseline image ---- */
    FILE *fb = fopen(base_path, "rb");
    if (!fb) {
        fprintf(stderr, "libckpt_restore: fopen(%s): %s\n",
                base_path, strerror(errno));
        ckpt_restore_free(result);
        return -1;
    }

    ckpt_v2_file_hdr_t fhdr;
    if (fread(&fhdr, sizeof(fhdr), 1, fb) != 1 ||
        fhdr.magic != CKPT_V2_MAGIC) {
        fprintf(stderr, "libckpt_restore: invalid base image header\n");
        fclose(fb); ckpt_restore_free(result); return -1;
    }

    uint32_t num_ranges = fhdr.num_ranges;
    if (num_ranges > MAX_RANGES) {
        fclose(fb); ckpt_restore_free(result); return -1;
    }

    ckpt_v2_range_desc_t *descs =
        malloc(num_ranges * sizeof(ckpt_v2_range_desc_t));
    if (!descs) { fclose(fb); ckpt_restore_free(result); return -1; }

    if (fread(descs, sizeof(ckpt_v2_range_desc_t), num_ranges, fb)
            != (size_t)num_ranges) {
        free(descs); fclose(fb); ckpt_restore_free(result); return -1;
    }

    /* For each range in the baseline, check if it overlaps any request */
    for (uint32_t ri = 0; ri < num_ranges; ri++) {
        uint64_t range_start = descs[ri].base_va;
        uint64_t range_end   = range_start + descs[ri].length;

        for (uint32_t req = 0; req < num_requests; req++) {
            uint64_t req_start = requested_vas[req];
            uint64_t req_end   = req_start + requested_sizes[req];

            /* Check overlap */
            if (range_end <= req_start || range_start >= req_end)
                continue;

            if (assemble_data_selective(fb,
                    descs[ri].base_va,    descs[ri].num_pages,
                    descs[ri].data_offset, descs[ri].resmap_size,
                    descs[ri].cpu_bytes,   descs[ri].gpu_bytes,
                    result->blocks[req].data,
                    requested_vas[req], requested_sizes[req]) < 0) {
                free(descs); fclose(fb);
                ckpt_restore_free(result); return -1;
            }
        }
    }
    fclose(fb);
    free(descs);

    /* ---- Process delta image ---- */
    FILE *fd = fopen(delta_path, "rb");
    if (!fd) {
        fprintf(stderr, "libckpt_restore: fopen(%s): %s\n",
                delta_path, strerror(errno));
        ckpt_restore_free(result); return -1;
    }

    ckpt_inc_delta_hdr_t dhdr;
    if (fread(&dhdr, sizeof(dhdr), 1, fd) != 1 ||
        dhdr.magic != CKPT_INC_DELTA_MAGIC) {
        fclose(fd); ckpt_restore_free(result); return -1;
    }

    if (dhdr.num_blocks > 0) {
        ckpt_inc_delta_block_t *dblocks =
            malloc(dhdr.num_blocks * sizeof(ckpt_inc_delta_block_t));
        if (!dblocks) { fclose(fd); ckpt_restore_free(result); return -1; }

        if (fread(dblocks, sizeof(ckpt_inc_delta_block_t), dhdr.num_blocks, fd)
                != (size_t)dhdr.num_blocks) {
            free(dblocks); fclose(fd);
            ckpt_restore_free(result); return -1;
        }

        for (uint32_t bi = 0; bi < dhdr.num_blocks; bi++) {
            ckpt_inc_delta_block_t *b = &dblocks[bi];
            uint64_t blk_start = b->base_va;
            uint64_t blk_end   = blk_start + b->length;

            for (uint32_t req = 0; req < num_requests; req++) {
                uint64_t req_start = requested_vas[req];
                uint64_t req_end   = req_start + requested_sizes[req];

                if (blk_end <= req_start || blk_start >= req_end)
                    continue;

                if (assemble_data_selective(fd,
                        b->base_va,     b->num_pages,
                        b->data_offset, b->resmap_size,
                        b->cpu_bytes,   b->gpu_bytes,
                        result->blocks[req].data,
                        requested_vas[req], requested_sizes[req]) < 0) {
                    free(dblocks); fclose(fd);
                    ckpt_restore_free(result); return -1;
                }
            }
        }
        free(dblocks);
    }
    fclose(fd);

    return 0;
}

/* Forward declarations */
static int decrypt_section(const uint8_t *ciphertext, uint8_t *plaintext,
                            uint64_t section_bytes, const crypto_meta_t *meta,
                            uint64_t num_meta, const uint8_t *k3_key,
                            int uvm_fd, int use_k1);

int ckpt_restore_selective_encrypted(const char *base_path,
                                      const char *delta_path,
                                      const uint64_t *requested_vas,
                                      const uint64_t *requested_sizes,
                                      uint32_t num_requests,
                                      const char *k3_key_path,
                                      const char *uvm_dev,
                                      ckpt_restore_result_t *result)
{
    memset(result, 0, sizeof(*result));
    if (num_requests == 0) return 0;

    /* Load k3 key */
    uint8_t k3_key[32];
    int has_k3 = 0;
    if (k3_key_path) {
        FILE *kf = fopen(k3_key_path, "rb");
        if (kf) { if (fread(k3_key, 1, 32, kf) == 32) has_k3 = 1; fclose(kf); }
    }

    /* Open UVM for k1 decrypt */
    int uvm_fd = -1;
    const char *dev = uvm_dev ? uvm_dev : "/dev/nvidia-uvm";
    uvm_fd = open(dev, O_RDWR);
    if (uvm_fd >= 0) {
        uvm_init_params_t init = {0};
        if (ioctl(uvm_fd, UVM_INITIALIZE, &init) < 0 || init.rmStatus != 0) {
            close(uvm_fd); uvm_fd = -1;
        }
    }

    /* Allocate output blocks */
    result->num_blocks = num_requests;
    result->blocks = malloc(num_requests * sizeof(ckpt_mem_block_t));
    if (!result->blocks) {
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }
    for (uint32_t i = 0; i < num_requests; i++) {
        result->blocks[i].orig_va = requested_vas[i];
        result->blocks[i].size    = requested_sizes[i];
        result->blocks[i].data    = calloc(1, (size_t)requested_sizes[i]);
        if (!result->blocks[i].data) {
            ckpt_restore_free(result);
            if (uvm_fd >= 0) close(uvm_fd);
            return -1;
        }
    }

    /* Process base image: for each range, decrypt gpu_section, extract slices */
    FILE *fb = fopen(base_path, "rb");
    if (!fb) {
        ckpt_restore_free(result);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    ckpt_v2_file_hdr_t fhdr;
    if (fread(&fhdr, sizeof(fhdr), 1, fb) != 1 || fhdr.magic != CKPT_V2_MAGIC) {
        fclose(fb); ckpt_restore_free(result);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    int is_v3 = (fhdr.version == CKPT_V3_VERSION);
    uint32_t num_ranges = fhdr.num_ranges;
    if (num_ranges > MAX_RANGES) {
        fclose(fb); ckpt_restore_free(result);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    ckpt_v2_range_desc_t *descs = malloc(num_ranges * sizeof(ckpt_v2_range_desc_t));
    if (!descs) {
        fclose(fb); ckpt_restore_free(result);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }
    if (fread(descs, sizeof(ckpt_v2_range_desc_t), num_ranges, fb) != (size_t)num_ranges) {
        free(descs); fclose(fb); ckpt_restore_free(result);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    /* For each range that overlaps any request, decrypt and extract */
    for (uint32_t ri = 0; ri < num_ranges; ri++) {
        uint64_t range_start = descs[ri].base_va;
        uint64_t range_end   = range_start + descs[ri].length;
        uint64_t data_off    = descs[ri].data_offset;
        uint64_t resmap_sz   = descs[ri].resmap_size;
        uint64_t cpu_bytes   = descs[ri].cpu_bytes;
        uint64_t gpu_bytes   = descs[ri].gpu_bytes;
        uint64_t meta_bytes  = descs[ri].crypto_meta_bytes;

        /* Check if any request overlaps this range */
        int has_overlap = 0;
        for (uint32_t req = 0; req < num_requests; req++) {
            uint64_t req_end = requested_vas[req] + requested_sizes[req];
            if (req_end > range_start && requested_vas[req] < range_end) {
                has_overlap = 1;
                break;
            }
        }
        if (!has_overlap) continue;

        /* Read resmap */
        uint8_t *resmap = malloc(resmap_sz ? resmap_sz : 1);
        if (!resmap) continue;
        read_bytes(fb, data_off, resmap, resmap_sz);

        /* Read and decrypt GPU section */
        uint8_t *gpu_plain = NULL;
        if (gpu_bytes > 0) {
            uint8_t *gpu_cipher = malloc(gpu_bytes);
            gpu_plain = malloc(gpu_bytes);
            if (gpu_cipher && gpu_plain) {
                read_bytes(fb, data_off + resmap_sz + cpu_bytes, gpu_cipher, gpu_bytes);

                if (meta_bytes > 0) {
                    uint64_t num_meta = meta_bytes / sizeof(crypto_meta_t);
                    crypto_meta_t *meta = malloc(meta_bytes);
                    if (meta) {
                        read_bytes(fb, data_off + resmap_sz + cpu_bytes + gpu_bytes,
                                   meta, meta_bytes);

                        /* Split metadata: GPU entries first */
                        uint64_t gpu_meta_count = 0;
                        if (is_v3) {
                            uint64_t acc = 0;
                            for (uint64_t m = 0; m < num_meta; m++) {
                                if (acc < gpu_bytes) { acc += meta[m].size; gpu_meta_count++; }
                                else break;
                            }
                        } else {
                            gpu_meta_count = num_meta;
                        }

                        int gpu_k1 = is_v3 ? 1 : 0;
                        decrypt_section(gpu_cipher, gpu_plain, gpu_bytes,
                                        meta, gpu_meta_count,
                                        has_k3 ? k3_key : NULL, uvm_fd, gpu_k1);
                        free(meta);
                    } else {
                        memcpy(gpu_plain, gpu_cipher, gpu_bytes);
                    }
                } else {
                    memcpy(gpu_plain, gpu_cipher, gpu_bytes);
                }
            }
            free(gpu_cipher);
        }

        /* Read and decrypt CPU section */
        uint8_t *cpu_plain = NULL;
        if (cpu_bytes > 0) {
            uint8_t *cpu_cipher = malloc(cpu_bytes);
            cpu_plain = malloc(cpu_bytes);
            if (cpu_cipher && cpu_plain) {
                read_bytes(fb, data_off + resmap_sz, cpu_cipher, cpu_bytes);

                if (meta_bytes > 0 && has_k3) {
                    uint64_t num_meta = meta_bytes / sizeof(crypto_meta_t);
                    crypto_meta_t *meta = malloc(meta_bytes);
                    if (meta) {
                        read_bytes(fb, data_off + resmap_sz + cpu_bytes + gpu_bytes,
                                   meta, meta_bytes);

                        uint64_t gpu_meta_count = 0;
                        if (is_v3) {
                            uint64_t acc = 0;
                            for (uint64_t m = 0; m < num_meta; m++) {
                                if (acc < gpu_bytes) { acc += meta[m].size; gpu_meta_count++; }
                                else break;
                            }
                        } else {
                            gpu_meta_count = num_meta;
                        }

                        uint64_t cpu_meta_count = num_meta - gpu_meta_count;
                        if (cpu_meta_count > 0) {
                            decrypt_section(cpu_cipher, cpu_plain, cpu_bytes,
                                            meta + gpu_meta_count, cpu_meta_count,
                                            k3_key, -1, 0);
                        } else {
                            memcpy(cpu_plain, cpu_cipher, cpu_bytes);
                        }
                        free(meta);
                    } else {
                        memcpy(cpu_plain, cpu_cipher, cpu_bytes);
                    }
                } else {
                    memcpy(cpu_plain, cpu_cipher, cpu_bytes);
                }
            }
            free(cpu_cipher);
        }

        /* Extract requested slices by walking the resmap */
        for (uint32_t req = 0; req < num_requests; req++) {
            uint64_t req_start = requested_vas[req];
            uint64_t req_end   = req_start + requested_sizes[req];

            if (req_end <= range_start || req_start >= range_end)
                continue;

            /* Walk resmap pages in the overlap region */
            uint64_t overlap_start = (req_start >= range_start) ? req_start : range_start;
            uint64_t overlap_end   = (req_end <= range_end) ? req_end : range_end;

            /* We need to compute the cpu/gpu offsets for pages before overlap_start
             * to know where in cpu_plain/gpu_plain to read from */
            uint64_t cpu_off = 0, gpu_off = 0;
            uint64_t first_page = (overlap_start - range_start) / PAGE_SZ;

            /* Count cpu/gpu pages before our overlap region */
            for (uint64_t p = 0; p < first_page; p++) {
                if (resmap[p] == 0) cpu_off += PAGE_SZ;
                else if (resmap[p] == 1) gpu_off += PAGE_SZ;
            }

            /* Now extract pages in the overlap region */
            for (uint64_t page_va = overlap_start; page_va < overlap_end; page_va += PAGE_SZ) {
                uint64_t pidx = (page_va - range_start) / PAGE_SZ;
                uint64_t dst_off = page_va - req_start;

                if (resmap[pidx] == 0 && cpu_plain && cpu_off + PAGE_SZ <= cpu_bytes) {
                    memcpy(result->blocks[req].data + dst_off, cpu_plain + cpu_off, PAGE_SZ);
                    cpu_off += PAGE_SZ;
                } else if (resmap[pidx] == 1 && gpu_plain && gpu_off + PAGE_SZ <= gpu_bytes) {
                    memcpy(result->blocks[req].data + dst_off, gpu_plain + gpu_off, PAGE_SZ);
                    gpu_off += PAGE_SZ;
                }
                /* absent stays zero */
            }
        }

        free(resmap);
        free(gpu_plain);
        free(cpu_plain);
    }

    free(descs);
    fclose(fb);

    /* TODO: apply delta similarly if needed */

    if (uvm_fd >= 0) close(uvm_fd);
    return 0;
}

void ckpt_restore_free(ckpt_restore_result_t *result)
{
    if (!result) return;
    for (uint32_t i = 0; i < result->num_blocks; i++)
        free(result->blocks[i].data);
    free(result->blocks);
    memset(result, 0, sizeof(*result));
}

/* ------------------------------------------------------------------ */
/* k3 decryption: AES-256-GCM using OpenSSL                           */
/* Decrypts a single transfer using stored IV + auth_tag.              */
/* ------------------------------------------------------------------ */
static int decrypt_k3_transfer(const uint8_t *key,
                                const uint8_t *ciphertext,
                                uint8_t *plaintext,
                                uint64_t size,
                                const crypto_meta_t *meta)
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;

    int outlen = 0;
    int ret = -1;

    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, meta->iv) != 1)
        goto done;
    if (EVP_DecryptUpdate(ctx, plaintext, &outlen, ciphertext, (int)size) != 1)
        goto done;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16,
                             (void *)meta->auth_tag) != 1)
        goto done;
    if (EVP_DecryptFinal_ex(ctx, plaintext + outlen, &outlen) != 1) {
        fprintf(stderr, "libckpt_restore: k3 decrypt auth failed (size=%llu)\n",
                (unsigned long long)size);
        goto done;
    }
    ret = 0;

done:
    EVP_CIPHER_CTX_free(ctx);
    return ret;
}

/* ------------------------------------------------------------------ */
/* k1 decryption via ioctl 112                                         */
/* ------------------------------------------------------------------ */
static int decrypt_k1_via_ioctl(int uvm_fd,
                                 const uint8_t *ciphertext,
                                 uint8_t *plaintext,
                                 uint64_t total_size,
                                 const crypto_meta_t *meta,
                                 uint64_t num_transfers)
{
    uvm_decrypt_params_t params = {0};
    params.cipher_buf    = (uint64_t)(uintptr_t)ciphertext;
    params.plain_buf     = (uint64_t)(uintptr_t)plaintext;
    params.total_size    = total_size;
    params.crypto_meta   = (uint64_t)(uintptr_t)meta;
    params.num_transfers = num_transfers;

    int ret = ioctl(uvm_fd, UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES, &params);
    if (ret < 0 || params.rmStatus != 0) {
        fprintf(stderr, "libckpt_restore: ioctl 112 decrypt failed: %s status=0x%x\n",
                strerror(errno), params.rmStatus);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* decrypt_section: decrypt a ciphertext section using the specified   */
/* method.  use_k1=1 for GPU pages (ioctl 112), use_k1=0 for k3.      */
/* ------------------------------------------------------------------ */
static int decrypt_section(const uint8_t *ciphertext,
                            uint8_t *plaintext,
                            uint64_t section_bytes,
                            const crypto_meta_t *meta,
                            uint64_t num_meta,
                            const uint8_t *k3_key,
                            int uvm_fd,
                            int use_k1)
{
    if (num_meta == 0 || section_bytes == 0)
        return 0;

    if (use_k1 && uvm_fd >= 0) {
        /* k1 decrypt via ioctl 112 */
        if (decrypt_k1_via_ioctl(uvm_fd, ciphertext, plaintext,
                                  section_bytes, meta, num_meta) < 0)
            return -1;
    } else if (!use_k1 && k3_key) {
        /* k3 decrypt: process each transfer individually */
        uint64_t offset = 0;
        for (uint64_t i = 0; i < num_meta && offset < section_bytes; i++) {
            uint64_t chunk = meta[i].size;
            if (chunk == 0) chunk = PAGE_SZ;
            if (offset + chunk > section_bytes) chunk = section_bytes - offset;

            if (decrypt_k3_transfer(k3_key, ciphertext + offset,
                                     plaintext + offset, chunk, &meta[i]) < 0)
                return -1;
            offset += chunk;
        }
    } else {
        /* No key available — copy as-is */
        memcpy(plaintext, ciphertext, section_bytes);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Main encrypted restore function                                     */
/* ------------------------------------------------------------------ */
int ckpt_restore_from_files_encrypted(const char *base_path,
                                       const char *delta_path,
                                       const char *k3_key_path,
                                       const char *uvm_dev,
                                       ckpt_restore_result_t *result)
{
    memset(result, 0, sizeof(*result));

    /* Load k3 key if provided */
    uint8_t k3_key[32];
    int has_k3 = 0;
    if (k3_key_path) {
        FILE *kf = fopen(k3_key_path, "rb");
        if (kf) {
            if (fread(k3_key, 1, 32, kf) == 32)
                has_k3 = 1;
            fclose(kf);
        }
        if (!has_k3)
            fprintf(stderr, "libckpt_restore: WARNING: could not read k3 key from %s\n",
                    k3_key_path);
    }

    /* Open UVM device for ioctl 112 (v3 images) */
    int uvm_fd = -1;
    const char *dev = uvm_dev ? uvm_dev : "/dev/nvidia-uvm";

    /* Read base header to check version */
    FILE *fb = fopen(base_path, "rb");
    if (!fb) {
        fprintf(stderr, "libckpt_restore: fopen(%s): %s\n", base_path, strerror(errno));
        return -1;
    }

    ckpt_v2_file_hdr_t fhdr;
    if (fread(&fhdr, sizeof(fhdr), 1, fb) != 1 || fhdr.magic != CKPT_V2_MAGIC) {
        fprintf(stderr, "libckpt_restore: invalid base image header\n");
        fclose(fb); return -1;
    }

    int is_encrypted = 0;
    int is_v3 = (fhdr.version == CKPT_V3_VERSION);

    /* Read range descriptors */
    uint32_t num_ranges = fhdr.num_ranges;
    if (num_ranges > MAX_RANGES) {
        fclose(fb); return -1;
    }

    ckpt_v2_range_desc_t *descs = malloc(num_ranges * sizeof(ckpt_v2_range_desc_t));
    if (!descs) { fclose(fb); return -1; }

    if (fread(descs, sizeof(ckpt_v2_range_desc_t), num_ranges, fb)
            != (size_t)num_ranges) {
        free(descs); fclose(fb); return -1;
    }

    /* Check if any range has crypto metadata */
    for (uint32_t i = 0; i < num_ranges; i++) {
        if (descs[i].crypto_meta_bytes > 0) {
            is_encrypted = 1;
            break;
        }
    }

    if (!is_encrypted) {
        /* No encryption — use the standard restore path */
        free(descs);
        fclose(fb);
        return ckpt_restore_from_files(base_path, delta_path, result);
    }

    /* Open UVM device for v3 k1 decryption */
    if (is_v3) {
        uvm_fd = open(dev, O_RDWR);
        if (uvm_fd >= 0) {
            uvm_init_params_t init = {0};
            if (ioctl(uvm_fd, UVM_INITIALIZE, &init) < 0 || init.rmStatus != 0) {
                fprintf(stderr, "libckpt_restore: UVM_INITIALIZE failed\n");
                close(uvm_fd);
                uvm_fd = -1;
            }
        }
    }

    fprintf(stderr, "libckpt_restore: encrypted image v%u, %u ranges, k3=%s, uvm=%s\n",
            fhdr.version, num_ranges,
            has_k3 ? "yes" : "no",
            uvm_fd >= 0 ? "yes" : "no");

    /* Group contiguous ranges (same as ckpt_restore_from_files) */
    uint64_t group_va[MAX_GROUPS];
    uint64_t group_sz[MAX_GROUPS];
    uint8_t *group_buf[MAX_GROUPS];
    uint32_t num_groups = 0;
    int range_group[MAX_RANGES];

    qsort(descs, num_ranges, sizeof(ckpt_v2_range_desc_t), cmp_range_va);

    for (uint32_t i = 0; i < num_ranges; i++) {
        uint64_t va_end = descs[i].base_va + descs[i].length;
        if (num_groups == 0 ||
            descs[i].base_va > group_va[num_groups-1] + group_sz[num_groups-1]) {
            if (num_groups >= MAX_GROUPS) {
                free(descs); fclose(fb);
                if (uvm_fd >= 0) close(uvm_fd);
                return -1;
            }
            group_va[num_groups] = descs[i].base_va;
            group_sz[num_groups] = descs[i].length;
            group_buf[num_groups] = NULL;
            num_groups++;
        } else {
            uint64_t gend = group_va[num_groups-1] + group_sz[num_groups-1];
            if (va_end > gend)
                group_sz[num_groups-1] += (va_end - gend);
        }
        range_group[i] = (int)(num_groups - 1);
    }

    /* Allocate group buffers */
    for (uint32_t g = 0; g < num_groups; g++) {
        group_buf[g] = calloc(1, (size_t)group_sz[g]);
        if (!group_buf[g]) {
            for (uint32_t j = 0; j < g; j++) free(group_buf[j]);
            free(descs); fclose(fb);
            if (uvm_fd >= 0) close(uvm_fd);
            return -1;
        }
    }

    /* Process each range: read ciphertext + metadata, decrypt, assemble */
    int status = 0;
    for (uint32_t ri = 0; ri < num_ranges && status == 0; ri++) {
        int g = range_group[ri];
        uint64_t data_off = descs[ri].data_offset;
        uint64_t resmap_sz = descs[ri].resmap_size;
        uint64_t cpu_bytes = descs[ri].cpu_bytes;
        uint64_t gpu_bytes = descs[ri].gpu_bytes;
        uint64_t meta_bytes = descs[ri].crypto_meta_bytes;

        fprintf(stderr, "libckpt_restore: range[%u] base_va=0x%llx data_off=%llu "
                "cpu=%llu gpu=%llu meta=%llu resmap=%llu num_meta=%llu\n",
                ri, (unsigned long long)descs[ri].base_va,
                (unsigned long long)data_off,
                (unsigned long long)cpu_bytes,
                (unsigned long long)gpu_bytes,
                (unsigned long long)meta_bytes,
                (unsigned long long)resmap_sz,
                (unsigned long long)(meta_bytes / sizeof(crypto_meta_t)));

        /* Read resmap */
        uint8_t *resmap = malloc(resmap_sz ? resmap_sz : 1);
        if (!resmap) { status = -1; break; }
        if (read_bytes(fb, data_off, resmap, resmap_sz) < 0) {
            free(resmap); status = -1; break;
        }

        /* Read CPU section (encrypted with k3 in both v2 and v3) */
        uint8_t *cpu_cipher = malloc(cpu_bytes ? cpu_bytes : 1);
        uint8_t *cpu_plain  = malloc(cpu_bytes ? cpu_bytes : 1);
        if (!cpu_cipher || !cpu_plain) {
            free(resmap); free(cpu_cipher); free(cpu_plain);
            status = -1; break;
        }
        if (cpu_bytes > 0)
            read_bytes(fb, data_off + resmap_sz, cpu_cipher, cpu_bytes);

        /* Read GPU section */
        uint8_t *gpu_cipher = malloc(gpu_bytes ? gpu_bytes : 1);
        uint8_t *gpu_plain  = malloc(gpu_bytes ? gpu_bytes : 1);
        if (!gpu_cipher || !gpu_plain) {
            free(resmap); free(cpu_cipher); free(cpu_plain);
            free(gpu_cipher); free(gpu_plain);
            status = -1; break;
        }
        if (gpu_bytes > 0)
            read_bytes(fb, data_off + resmap_sz + cpu_bytes, gpu_cipher, gpu_bytes);

        /* Read crypto metadata */
        crypto_meta_t *meta = NULL;
        uint64_t num_meta = 0;
        if (meta_bytes > 0) {
            num_meta = meta_bytes / sizeof(crypto_meta_t);
            meta = malloc(meta_bytes);
            if (!meta) {
                free(resmap); free(cpu_cipher); free(cpu_plain);
                free(gpu_cipher); free(gpu_plain);
                status = -1; break;
            }
            read_bytes(fb, data_off + resmap_sz + cpu_bytes + gpu_bytes,
                        meta, meta_bytes);
        }

        /* Decrypt sections */
        if (meta && num_meta > 0) {
            if (is_v3) {
                /* v3: GPU section encrypted with k1, CPU section with k3.
                 * Split metadata by matching cumulative size against gpu_bytes. */
                uint64_t gpu_meta_count = 0;
                uint64_t acc = 0;
                for (uint64_t m = 0; m < num_meta; m++) {
                    if (acc < gpu_bytes) { acc += meta[m].size; gpu_meta_count++; }
                    else break;
                }
                uint64_t cpu_meta_count = num_meta - gpu_meta_count;

                /* Decrypt GPU section with k1 */
                if (gpu_bytes > 0 && gpu_meta_count > 0) {
                    if (decrypt_section(gpu_cipher, gpu_plain, gpu_bytes,
                                         meta, gpu_meta_count,
                                         NULL, uvm_fd, 1) < 0) {
                        fprintf(stderr, "libckpt_restore: GPU decrypt (k1) failed at range %u\n", ri);
                        status = -1;
                    }
                } else if (gpu_bytes > 0) {
                    memcpy(gpu_plain, gpu_cipher, gpu_bytes);
                }

                /* Decrypt CPU section with k3 */
                if (cpu_bytes > 0 && cpu_meta_count > 0 && has_k3) {
                    if (decrypt_section(cpu_cipher, cpu_plain, cpu_bytes,
                                         meta + gpu_meta_count, cpu_meta_count,
                                         k3_key, -1, 0) < 0) {
                        fprintf(stderr, "libckpt_restore: CPU decrypt (k3) failed at range %u\n", ri);
                        status = -1;
                    }
                } else if (cpu_bytes > 0) {
                    memcpy(cpu_plain, cpu_cipher, cpu_bytes);
                }
            } else {
                /* v2: cpu+gpu were encrypted together as one combined buffer.
                 * Must combine, decrypt as one unit, then split back. */
                uint64_t combined_size = cpu_bytes + gpu_bytes;
                fprintf(stderr, "libckpt_restore: v2 range[%u] combined decrypt: "
                        "cpu=%llu gpu=%llu combined=%llu num_meta=%llu "
                        "meta[0].size=%llu meta[0].iv=[%02x%02x%02x%02x]\n",
                        ri, (unsigned long long)cpu_bytes,
                        (unsigned long long)gpu_bytes,
                        (unsigned long long)combined_size,
                        (unsigned long long)num_meta,
                        num_meta > 0 ? (unsigned long long)meta[0].size : 0ULL,
                        num_meta > 0 ? meta[0].iv[0] : 0,
                        num_meta > 0 ? meta[0].iv[1] : 0,
                        num_meta > 0 ? meta[0].iv[2] : 0,
                        num_meta > 0 ? meta[0].iv[3] : 0);
                uint8_t *combined_cipher = malloc(combined_size ? combined_size : 1);
                uint8_t *combined_plain  = malloc(combined_size ? combined_size : 1);
                if (combined_cipher && combined_plain && combined_size > 0) {
                    /* Agent encrypted in order: cpu_buf first, gpu_buf second */
                    if (cpu_bytes > 0) memcpy(combined_cipher, cpu_cipher, cpu_bytes);
                    if (gpu_bytes > 0) memcpy(combined_cipher + cpu_bytes, gpu_cipher, gpu_bytes);

                    if (decrypt_section(combined_cipher, combined_plain, combined_size,
                                         meta, num_meta, k3_key, -1, 0) < 0) {
                        fprintf(stderr, "libckpt_restore: v2 combined decrypt failed at range %u\n", ri);
                        status = -1;
                    } else {
                        /* Split back into cpu_plain and gpu_plain */
                        if (cpu_bytes > 0) memcpy(cpu_plain, combined_plain, cpu_bytes);
                        if (gpu_bytes > 0) memcpy(gpu_plain, combined_plain + cpu_bytes, gpu_bytes);
                    }
                }
                free(combined_cipher);
                free(combined_plain);
            }
        } else {
            /* No metadata — data is plaintext */
            if (gpu_bytes > 0) memcpy(gpu_plain, gpu_cipher, gpu_bytes);
            if (cpu_bytes > 0) memcpy(cpu_plain, cpu_cipher, cpu_bytes);
        }

        /* Assemble into group buffer (same logic as assemble_data) */
        if (status == 0) {
            uint8_t *range_buf = group_buf[g] + (descs[ri].base_va - group_va[g]);
            uint64_t cpu_off = 0, gpu_off = 0;

            for (uint64_t p = 0; p < descs[ri].num_pages; p++) {
                uint8_t *dst = range_buf + p * PAGE_SZ;
                if (resmap[p] == 0 && cpu_off + PAGE_SZ <= cpu_bytes) {
                    memcpy(dst, cpu_plain + cpu_off, PAGE_SZ);
                    cpu_off += PAGE_SZ;
                } else if (resmap[p] == 1 && gpu_off + PAGE_SZ <= gpu_bytes) {
                    memcpy(dst, gpu_plain + gpu_off, PAGE_SZ);
                    gpu_off += PAGE_SZ;
                }
                /* absent (2): stays zero from calloc */
            }
        }

        free(resmap);
        free(cpu_cipher); free(cpu_plain);
        free(gpu_cipher); free(gpu_plain);
        free(meta);
    }

    free(descs);
    fclose(fb);

    if (status != 0) {
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    /* Apply delta (same logic — read delta, decrypt if needed, patch groups) */
    FILE *fd = fopen(delta_path, "rb");
    if (!fd) {
        fprintf(stderr, "libckpt_restore: fopen(%s): %s\n", delta_path, strerror(errno));
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    ckpt_inc_delta_hdr_t dhdr;
    if (fread(&dhdr, sizeof(dhdr), 1, fd) != 1 || dhdr.magic != CKPT_INC_DELTA_MAGIC) {
        fclose(fd);
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        if (uvm_fd >= 0) close(uvm_fd);
        return -1;
    }

    if (dhdr.num_blocks > 0) {
        ckpt_inc_delta_block_t *dblocks =
            malloc(dhdr.num_blocks * sizeof(ckpt_inc_delta_block_t));
        if (!dblocks) {
            fclose(fd);
            for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
            if (uvm_fd >= 0) close(uvm_fd);
            return -1;
        }

        if (fread(dblocks, sizeof(ckpt_inc_delta_block_t), dhdr.num_blocks, fd)
                != (size_t)dhdr.num_blocks) {
            free(dblocks); fclose(fd);
            for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
            if (uvm_fd >= 0) close(uvm_fd);
            return -1;
        }

        for (uint32_t bi = 0; bi < dhdr.num_blocks && status == 0; bi++) {
            ckpt_inc_delta_block_t *b = &dblocks[bi];

            int block_g = -1;
            for (uint32_t g = 0; g < num_groups; g++) {
                if (b->base_va >= group_va[g] &&
                    b->base_va + b->length <= group_va[g] + group_sz[g]) {
                    block_g = (int)g;
                    break;
                }
            }

            if (block_g < 0) {
                /* Delta block not in any group — skip (may be from different VA range) */
                continue;
            }

            uint64_t d_off = b->data_offset;
            uint64_t d_resmap_sz = b->resmap_size;
            uint64_t d_cpu = b->cpu_bytes;
            uint64_t d_gpu = b->gpu_bytes;
            uint64_t d_meta = b->crypto_meta_bytes;

            /* Read and decrypt delta data (same approach as baseline) */
            uint8_t *resmap = malloc(d_resmap_sz ? d_resmap_sz : 1);
            uint8_t *cpu_data = malloc(d_cpu ? d_cpu : 1);
            uint8_t *gpu_data = malloc(d_gpu ? d_gpu : 1);
            if (!resmap || !cpu_data || !gpu_data) {
                free(resmap); free(cpu_data); free(gpu_data);
                status = -1; break;
            }

            read_bytes(fd, d_off, resmap, d_resmap_sz);
            if (d_cpu > 0) read_bytes(fd, d_off + d_resmap_sz, cpu_data, d_cpu);
            if (d_gpu > 0) read_bytes(fd, d_off + d_resmap_sz + d_cpu, gpu_data, d_gpu);

            /* Read + apply crypto metadata for delta if present */
            if (d_meta > 0) {
                uint64_t d_num_meta = d_meta / sizeof(crypto_meta_t);
                crypto_meta_t *dmeta = malloc(d_meta);
                if (dmeta) {
                    read_bytes(fd, d_off + d_resmap_sz + d_cpu + d_gpu, dmeta, d_meta);

                    uint8_t *gpu_dec = malloc(d_gpu ? d_gpu : 1);
                    uint8_t *cpu_dec = malloc(d_cpu ? d_cpu : 1);
                    if (gpu_dec && cpu_dec) {
                        /* Split delta metadata same way as baseline */
                        uint64_t d_gpu_meta = 0;
                        if (is_v3) {
                            uint64_t acc = 0;
                            for (uint64_t m = 0; m < d_num_meta; m++) {
                                if (acc < d_gpu) { acc += dmeta[m].size; d_gpu_meta++; }
                                else break;
                            }
                        } else {
                            d_gpu_meta = d_num_meta;
                        }

                        if (d_gpu > 0 && d_gpu_meta > 0) {
                            int gpu_k1 = is_v3 ? 1 : 0;
                            decrypt_section(gpu_data, gpu_dec, d_gpu,
                                            dmeta, d_gpu_meta,
                                            has_k3 ? k3_key : NULL, uvm_fd,
                                            gpu_k1);
                            memcpy(gpu_data, gpu_dec, d_gpu);
                        }
                        if (d_cpu > 0 && has_k3 && (d_num_meta - d_gpu_meta) > 0) {
                            decrypt_section(cpu_data, cpu_dec, d_cpu,
                                            dmeta + d_gpu_meta, d_num_meta - d_gpu_meta,
                                            k3_key, -1, 0);
                            memcpy(cpu_data, cpu_dec, d_cpu);
                        }
                    }
                    free(gpu_dec); free(cpu_dec);
                    free(dmeta);
                }
            }

            /* Patch group buffer */
            uint8_t *range_buf = group_buf[block_g] + (b->base_va - group_va[block_g]);
            uint64_t cpu_off = 0, gpu_off = 0;
            for (uint64_t p = 0; p < b->num_pages; p++) {
                uint8_t *dst = range_buf + p * PAGE_SZ;
                if (resmap[p] == 0 && cpu_off + PAGE_SZ <= d_cpu) {
                    memcpy(dst, cpu_data + cpu_off, PAGE_SZ);
                    cpu_off += PAGE_SZ;
                } else if (resmap[p] == 1 && gpu_off + PAGE_SZ <= d_gpu) {
                    memcpy(dst, gpu_data + gpu_off, PAGE_SZ);
                    gpu_off += PAGE_SZ;
                }
            }

            free(resmap); free(cpu_data); free(gpu_data);
        }
        free(dblocks);
    }
    fclose(fd);
    if (uvm_fd >= 0) close(uvm_fd);

    if (status != 0) {
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        return -1;
    }

    /* Populate result */
    result->num_blocks = num_groups;
    result->blocks = malloc(num_groups * sizeof(ckpt_mem_block_t));
    if (!result->blocks) {
        for (uint32_t g = 0; g < num_groups; g++) free(group_buf[g]);
        return -1;
    }

    for (uint32_t g = 0; g < num_groups; g++) {
        result->blocks[g].orig_va = group_va[g];
        result->blocks[g].size    = group_sz[g];
        result->blocks[g].data    = group_buf[g];
    }

    return 0;
}
