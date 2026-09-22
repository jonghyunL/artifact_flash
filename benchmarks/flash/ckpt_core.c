/**
 * test_inc_ckpt_agent.c  –  Incremental pre-copy checkpoint agent
 *
 * Checkpoint strategy: pre-copy baseline + dirty-page delta
 *
 *   Phase 1  (concurrent with app running K1)
 *     PREPARE_ALL (ioctl 102): write-protect all pages, zero dirty bitmaps
 *     gate->ckpt_req = 1:      tell app to stop at next cudaStreamSynchronize
 *     READ_PAGES_RESIDENT:     baseline capture of all user ranges
 *     write /tmp/ckpt_inc_base.img   (ckpt_v2 format, magic 0xC2C2C2C2)
 *
 *   Phase 2  (app quiesced at AT_BOUNDARY)
 *     GET_DIRTY_PAGES (ioctl 104): 64KB-aligned VA list of all written pages
 *     sort + merge into contiguous regions
 *     READ_PAGES_RESIDENT on each region: final at-stop values
 *     write /tmp/ckpt_inc_delta.img  (ckpt_inc_delta format, magic 0xDE17A001)
 *     gate->phase = RESUME
 *
 * --pause-first mode (for fast workloads like PyTorch training):
 *   Sets ckpt_req=1 IMMEDIATELY after PREPARE_ALL — before the baseline read.
 *   The app pauses at its very next sync. Baseline is then read while the
 *   app is frozen (stop-and-copy). Delta will have 0 dirty pages.
 *   Use this when training steps are faster than the baseline read time.
 *
 * Usage:
 *   sudo ./test_inc_ckpt_agent [--pause-first] [gate_file] [base_img] [delta_img]
 *   e.g.: sudo ./test_inc_ckpt_agent /tmp/ckpt_gate \
 *                  /tmp/ckpt_inc_base.img /tmp/ckpt_inc_delta.img
 *   e.g.: sudo ./test_inc_ckpt_agent --pause-first /tmp/ckpt_gate \
 *                  /tmp/ckpt_inc_base.img /tmp/ckpt_inc_delta.img
 *
 * State file snapshot:
 *   The agent copies ckpt_train_state.json (and ckpt_tensor_map.json) into
 *   the output directory while the app is frozen at AT_BOUNDARY.  This
 *   ensures the saved step number matches the GPU memory snapshot — without
 *   this, the app continues training after RESUME and overwrites the state
 *   file to the final step, making restore think there's nothing to resume.
 *
 * Compile:
 *   gcc -O2 -o test_inc_ckpt_agent test_inc_ckpt_agent.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>

#include "ckpt_gate.h"
#include "ckpt_crypto.h"

/* ------------------------------------------------------------------ */
/* --no-image-write: skip the actual /dev/shm/ckpt_*.img file writes,
 * substitute MAP_PRIVATE|MAP_ANON for MAP_SHARED+fd at every output
 * mmap site. Keeps transfer + encrypt + per-page metadata accumulation
 * identical (so the GPU-side mechanism cost is unchanged) but removes
 * the image-persistence cost that phos_emul/gcr_emul don't pay. Used
 * for apples-to-apples mechanism-throughput comparisons; restore is
 * non-functional in this mode. Default OFF (image written as before).
 *
 * Owned by the ckpt_core binary. ckpt_baseline.c (linked only into
 * libckpt_vllm.so) has its own independent definition; the two binaries
 * don't share a link, so there's no multiple-definition concern. */
int g_no_image_write = 0;

/* --track-only: SKIP the concurrent-precopy D2H (per-range ioctl 115 at
 * the per-range capture site, and ioctl 115 at the coalesced site IF used
 * for precopy). The post-quiesce stop-phase dirty delta (ioctl 116, fired
 * after AT_BOUNDARY when the app is frozen) STILL runs for real, so the
 * delta image contains all pages dirtied during the tracking window.
 *
 * Tests whether vLLM TBT regression during the concurrent-precopy window
 * is from the COPY (CE/bounce-buffer pressure, HBM/PCIe traffic) or from
 * the TRACKING (PREPARE write-protect → write fault → fault servicing).
 * The kernel dirty-tracking machinery (PREPARE_ALL, write-fault count,
 * FINALIZE bulk-restore) runs unchanged either way.
 *
 * Base image will be empty under --track-only (precopy was skipped); the
 * delta image is a real capture of the dirtied pages. Combined image is
 * NOT a valid checkpoint for restore — runs are for TBT measurement +
 * dirty-rate observation only. Combine with --track-only-budget-ms N to
 * sleep N ms per skipped ioctl so the concurrent window duration matches
 * production. */
static int g_track_only = 0;
static int g_track_only_budget_ms = 0;

/* CKPT_VERBOSE: when 1, dump every alloc/range entry. Default 0 keeps
 * the per-round terminal output to summary lines + timing. Set via env
 * (CKPT_VERBOSE=1 sudo ./ckpt_agent ...) — read once in main(). */
static int g_verbose = 0;

/* ------------------------------------------------------------------ */
/* UVM ioctl numbers                                                   */
/* ------------------------------------------------------------------ */
#define UVM_INITIALIZE                           0x30000001
#define UVM_LIVE_MIGRATION_PREPARE_ALL           102
#define UVM_LIVE_MIGRATION_GET_DIRTY_PAGES       104
#define UVM_LIVE_MIGRATION_GET_VA_RANGES         106
#define UVM_LIVE_MIGRATION_FINALIZE              101
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT   109
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED 111
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT 115
/* Tier3-fused: same MT ciphertext pipeline as 115 but the kernel internally
 * filters to pages in dirty 2MB va_blocks (fuses GET_DIRTY_PAGES + the
 * per-region 115 loop into ONE ioctl — setup tax paid once per delta). */
#define UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT 116

/* Default worker count for ioctl 115. Tunable via CKPT_MT_THREADS env var.
 * Bench at 4 threads matches the cudaMemcpy ceiling x2; 2 threads also works
 * but 4 is the sweet spot at 16+ GB sizes. */
#define CKPT_MT_THREADS_DEFAULT 4

#define NV_OK 0x00000000u
typedef uint32_t NV_STATUS;
typedef uint32_t NvU32;
typedef uint64_t NvU64;

/* ------------------------------------------------------------------ */
/* UVM parameter structs (must match uvm_ioctl.h exactly)             */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t  flags __attribute__((aligned(8)));
    uint32_t  rmStatus;
} UVM_INITIALIZE_PARAMS;

typedef struct {
    NvU64     base     __attribute__((aligned(8)));
    NvU64     length   __attribute__((aligned(8)));
    NV_STATUS rmStatus;
} UVM_LIVE_MIGRATION_FINALIZE_PARAMS;

typedef struct {
    NvU32     flags;        /* IN  – migration flags (0 = default) */
    NV_STATUS rmStatus;     /* OUT */
} UVM_LIVE_MIGRATION_PREPARE_ALL_PARAMS;

typedef struct {
    NvU64     max_pages;           /* IN  – capacity of dirty_addresses[] */
    NvU64     num_pages;           /* OUT – number of dirty 64KB page VAs */
    NvU64    *dirty_addresses;     /* IN  – caller NvU64[max_pages]       */
    NV_STATUS rmStatus;            /* OUT */
} UVM_LIVE_MIGRATION_GET_DIRTY_PAGES_PARAMS;

typedef struct {
    NvU64     max_ranges  __attribute__((aligned(8)));
    NvU64     num_ranges  __attribute__((aligned(8)));
    NvU64    *base_addrs;
    NvU64    *range_sizes;
    NV_STATUS rmStatus;
} UVM_LIVE_MIGRATION_GET_VA_RANGES_PARAMS;

typedef struct { uint8_t uuid[16]; } NvProcessorUuid;

typedef struct {
    NvU64           base          __attribute__((aligned(8)));
    NvU64           length        __attribute__((aligned(8)));
    NvU64           cpu_buf       __attribute__((aligned(8)));
    NvU64           cpu_buf_size  __attribute__((aligned(8)));
    NvU64           gpu_buf       __attribute__((aligned(8)));
    NvU64           gpu_buf_size  __attribute__((aligned(8)));
    NvU64           residency_map __attribute__((aligned(8)));
    NvProcessorUuid gpu_uuid;
    NvU64           cpu_bytes_out __attribute__((aligned(8)));
    NvU64           gpu_bytes_out __attribute__((aligned(8)));
    NvU64           num_pages     __attribute__((aligned(8)));
    NV_STATUS       rmStatus;
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_PARAMS;

/* Per-transfer crypto metadata (must match kernel UVM_LIVE_MIGRATION_PAGE_CRYPTO_META) */
typedef struct __attribute__((packed)) {
    uint64_t size;         /* ciphertext bytes this entry covers */
    uint8_t  iv[12];
    uint8_t  iv_fresh;
    uint8_t  auth_tag[16];
    uint32_t key_version;
} page_crypto_meta_t;   /* 41 bytes */

typedef struct {
    NvU64           base          __attribute__((aligned(8)));
    NvU64           length        __attribute__((aligned(8)));
    NvU64           cpu_buf       __attribute__((aligned(8)));
    NvU64           cpu_buf_size  __attribute__((aligned(8)));
    NvU64           gpu_buf       __attribute__((aligned(8)));
    NvU64           gpu_buf_size  __attribute__((aligned(8)));
    NvU64           residency_map __attribute__((aligned(8)));
    NvU64           crypto_meta   __attribute__((aligned(8)));
    NvU64           crypto_meta_size __attribute__((aligned(8)));
    NvProcessorUuid gpu_uuid;
    NvU64           cpu_bytes_out __attribute__((aligned(8)));
    NvU64           gpu_bytes_out __attribute__((aligned(8)));
    NvU64           num_pages     __attribute__((aligned(8)));
    NvU64           num_transfers __attribute__((aligned(8)));
    NV_STATUS       rmStatus;
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS;

/* Multi-threaded variant of the encrypted ciphertext read (ioctl 115).
 * Same on-disk format as the single-threaded encrypted path; just adds a
 * num_threads field that the kernel uses to spawn N parallel CE workers. */
typedef struct {
    NvU64           base             __attribute__((aligned(8)));
    NvU64           length           __attribute__((aligned(8)));
    NvU64           cpu_buf          __attribute__((aligned(8)));
    NvU64           cpu_buf_size     __attribute__((aligned(8)));
    NvU64           gpu_buf          __attribute__((aligned(8)));
    NvU64           gpu_buf_size     __attribute__((aligned(8)));
    NvU64           residency_map    __attribute__((aligned(8)));
    NvU64           crypto_meta      __attribute__((aligned(8)));
    NvU64           crypto_meta_size __attribute__((aligned(8)));
    uint32_t        num_threads;
    uint32_t        _pad;
    NvProcessorUuid gpu_uuid;
    NvU64           cpu_bytes_out    __attribute__((aligned(8)));
    NvU64           gpu_bytes_out    __attribute__((aligned(8)));
    NvU64           num_pages        __attribute__((aligned(8)));
    NvU64           num_transfers    __attribute__((aligned(8)));
    NV_STATUS       rmStatus;
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS;

/* ------------------------------------------------------------------ */
/* Baseline image format  (ckpt_v2 — identical to ckpt_resident_agent)*/
/* ------------------------------------------------------------------ */
#define CKPT_V2_MAGIC    0xC2C2C2C2u
#define CKPT_V2_VERSION  2u

/*
 * File layout (v2/v3):
 *   [ ckpt_v2_file_hdr_t       ]  24 bytes
 *   [ ckpt_v2_range_desc_t × N ]  64 bytes each
 *   for each range i:
 *     [ residency_map ]  range[i].num_pages bytes   (0=CPU 1=GPU 2=absent)
 *     [ cpu_section   ]  range[i].cpu_bytes bytes
 *     [ gpu_section   ]  range[i].gpu_bytes bytes   (plaintext or ciphertext)
 *     [ crypto_meta   ]  range[i].crypto_meta_bytes bytes (encrypted mode only)
 *                         = encrypted_page_count × sizeof(page_crypto_meta_t)
 *
 * Version 2: no crypto metadata (plaintext gpu_section, crypto_meta_bytes=0)
 * Version 3: encrypted gpu_section + per-GPU-page crypto metadata
 */
#define CKPT_V3_VERSION  3u

typedef struct __attribute__((packed)) {
    uint32_t magic;          /* CKPT_V2_MAGIC   */
    uint32_t version;        /* CKPT_V2_VERSION or CKPT_V3_VERSION */
    uint32_t num_ranges;
    uint32_t page_size;      /* 4096            */
    uint64_t total_bytes;    /* sum of all range lengths */
} ckpt_v2_file_hdr_t;        /* 24 bytes        */

typedef struct __attribute__((packed)) {
    uint64_t base_va;
    uint64_t length;
    uint64_t num_pages;
    uint64_t data_offset;    /* file offset of resmap+cpu+gpu+crypto for this range */
    uint64_t resmap_size;    /* == num_pages                                 */
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    uint64_t crypto_meta_bytes; /* 0 for v2/plaintext; encrypted_page_count*33 for v3 */
} ckpt_v2_range_desc_t;      /* 64 bytes        */

/* ------------------------------------------------------------------ */
/* Delta image format  (ckpt_inc_delta)                               */
/* ------------------------------------------------------------------ */
#define CKPT_INC_DELTA_MAGIC  0xDE17A001u
#define CKPT_INC_DELTA_VER    1u

/*
 * File layout:
 *   [ ckpt_inc_delta_hdr_t        ]  32 bytes
 *   [ ckpt_inc_delta_block_t × N  ]  56 bytes each
 *   for each block i:
 *     [ residency_map ]  block[i].resmap_size bytes
 *     [ cpu_section   ]  block[i].cpu_bytes bytes
 *     [ gpu_section   ]  block[i].gpu_bytes bytes
 *
 * alloc_start_va: the original VA of gate->user_va.
 * Restore uses: buf_offset = block.base_va - alloc_start_va
 */
typedef struct __attribute__((packed)) {
    uint32_t magic;                /* CKPT_INC_DELTA_MAGIC              */
    uint32_t version;              /* CKPT_INC_DELTA_VER                */
    uint32_t num_blocks;           /* number of merged dirty regions    */
    uint32_t page_size;            /* 4096                              */
    uint64_t alloc_start_va;       /* gate->user_va                     */
    uint64_t alloc_size;           /* gate->user_size                   */
    uint64_t num_dirty_pages_64k;  /* raw count from GET_DIRTY_PAGES    */
} ckpt_inc_delta_hdr_t;            /* 32 bytes                          */

typedef struct __attribute__((packed)) {
    uint64_t base_va;
    uint64_t length;
    uint64_t num_pages;
    uint64_t data_offset;          /* file offset of resmap+cpu+gpu+crypto */
    uint64_t resmap_size;          /* == num_pages                      */
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    uint64_t crypto_meta_bytes;    /* 0 or encrypted_page_count*33      */
} ckpt_inc_delta_block_t;          /* 64 bytes                          */

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
#define PAGE_64K    (64ULL * 1024ULL)
#define MAX_RANGES  256
#define PATH_MAX_LEN 1024

/* ------------------------------------------------------------------ */
/* Timing helper                                                       */
/* ------------------------------------------------------------------ */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* Captured data for one range / dirty region                         */
/* ------------------------------------------------------------------ */
typedef struct {
    NvU64    base;
    NvU64    length;
    NvU64    n_pages;
    NvU64    cpu_bytes;
    NvU64    gpu_bytes;
    NvU64    encrypted_page_count; /* number of crypto_meta entries */
    uint8_t *resmap;
    uint8_t *cpu_buf;
    uint8_t *gpu_buf;
    page_crypto_meta_t *crypto_meta;  /* per-GPU-page encryption metadata (or NULL) */
} range_capture_t;

static void range_capture_free(range_capture_t *rc)
{
    free(rc->resmap);
    free(rc->cpu_buf);
    free(rc->gpu_buf);
    free(rc->crypto_meta);
    memset(rc, 0, sizeof(*rc));
}

/* ------------------------------------------------------------------ */
/* write_delta_parallel: ckpt_inc_delta format via mmap + N writer       */
/* threads — the same fast path the P3-B baseline uses, but emitting the */
/* delta header/descriptors and copying each region's REAL sparse        */
/* resmap (+ cpu section) instead of an all-1s scratch. Used by the      */
/* Tier3-fused dirty-delta branch in capture_and_write_coalesced so the  */
/* delta write is no longer a single-threaded stdio fwrite (~2.6 GB/s).  */
/* Inputs are slices into the big ioctl buffers — caller owns them.      */
/* ------------------------------------------------------------------ */
typedef struct {
    uint8_t                      *img;        /* mmap'd delta image      */
    const ckpt_inc_delta_block_t *blocks;     /* descriptor array        */
    const range_capture_t        *caps;       /* per-region slice srcs   */
    uint32_t                      reg_start;
    uint32_t                      reg_end;    /* exclusive               */
} delta_writer_ctx_t;

static void *delta_writer_thread(void *arg)
{
    delta_writer_ctx_t *c = (delta_writer_ctx_t *)arg;
    for (uint32_t i = c->reg_start; i < c->reg_end; i++) {
        const ckpt_inc_delta_block_t *b = &c->blocks[i];
        const range_capture_t        *r = &c->caps[i];
        uint8_t *p = c->img + b->data_offset;
        memcpy(p, r->resmap,      (size_t)b->resmap_size);  p += b->resmap_size;
        memcpy(p, r->cpu_buf,     (size_t)b->cpu_bytes);    p += b->cpu_bytes;
        memcpy(p, r->gpu_buf,     (size_t)b->gpu_bytes);    p += b->gpu_bytes;
        if (b->crypto_meta_bytes)
            memcpy(p, r->crypto_meta, (size_t)b->crypto_meta_bytes);
    }
    return NULL;
}

static int write_delta_parallel(const char *path,
                                range_capture_t *caps, uint32_t num_blocks,
                                NvU64 alloc_start_va, NvU64 alloc_size,
                                NvU64 num_dirty_pages_64k, double *write_ms_out)
{
    ckpt_inc_delta_block_t *blocks =
        malloc((num_blocks ? num_blocks : 1) * sizeof(ckpt_inc_delta_block_t));
    if (!blocks) { fprintf(stderr, "  ✗ delta-par: malloc blocks\n"); return -1; }

    uint64_t cur_off = sizeof(ckpt_inc_delta_hdr_t)
                     + (uint64_t)num_blocks * sizeof(ckpt_inc_delta_block_t);
    uint64_t total_delta = 0;
    for (uint32_t i = 0; i < num_blocks; i++) {
        uint64_t meta_bytes =
            caps[i].encrypted_page_count * sizeof(page_crypto_meta_t);
        blocks[i].base_va           = caps[i].base;
        blocks[i].length            = caps[i].length;
        blocks[i].num_pages         = caps[i].n_pages;
        blocks[i].data_offset       = cur_off;
        blocks[i].resmap_size       = caps[i].n_pages;
        blocks[i].cpu_bytes         = caps[i].cpu_bytes;
        blocks[i].gpu_bytes         = caps[i].gpu_bytes;
        blocks[i].crypto_meta_bytes = meta_bytes;
        cur_off     += caps[i].n_pages + caps[i].cpu_bytes
                     + caps[i].gpu_bytes + meta_bytes;
        total_delta += caps[i].length;
    }
    uint64_t image_size = cur_off;

    ckpt_inc_delta_hdr_t dhdr = {
        .magic               = CKPT_INC_DELTA_MAGIC,
        .version             = CKPT_INC_DELTA_VER,
        .num_blocks          = num_blocks,
        .page_size           = 4096,
        .alloc_start_va      = alloc_start_va,
        .alloc_size          = alloc_size,
        .num_dirty_pages_64k = num_dirty_pages_64k,
    };

    printf("  [delta-par] writing %s  (%u block(s), %.2f MB, image=%.2f GB)\n",
           path, num_blocks, total_delta / (1024.0*1024.0),
           image_size / (1024.0*1024.0*1024.0));

    double tw0 = now_ms();
    int    fd  = -1;
    void  *mp;

    if (g_no_image_write) {
        mp = mmap(NULL, (size_t)image_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        /* Unlike the baseline (must-prealloc) path, the delta image is
         * written fresh each round — O_CREAT + grow as needed. */
        fd = open(path, O_RDWR | O_CREAT, 0644);
        if (fd < 0) {
            fprintf(stderr, "  ✗ delta-par: open(%s): %s\n",
                    path, strerror(errno));
            free(blocks); return -1;
        }
        struct stat st;
        if (fstat(fd, &st) < 0) {
            fprintf(stderr, "  ✗ delta-par: fstat: %s\n", strerror(errno));
            close(fd); free(blocks); return -1;
        }
        if ((uint64_t)st.st_size < image_size &&
            ftruncate(fd, (off_t)image_size) < 0) {
            fprintf(stderr, "  ✗ delta-par: ftruncate(%llu): %s\n",
                    (unsigned long long)image_size, strerror(errno));
            close(fd); free(blocks); return -1;
        }
        mp = mmap(NULL, (size_t)image_size, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);
    }
    if (mp == MAP_FAILED) {
        fprintf(stderr, "  ✗ delta-par: mmap(%llu): %s\n",
                (unsigned long long)image_size, strerror(errno));
        if (fd >= 0) close(fd);
        free(blocks); return -1;
    }
    (void)madvise(mp, (size_t)image_size, MADV_HUGEPAGE);
    uint8_t *img = (uint8_t *)mp;

    memcpy(img, &dhdr, sizeof(dhdr));
    memcpy(img + sizeof(dhdr), blocks,
           (size_t)num_blocks * sizeof(ckpt_inc_delta_block_t));

    int n_writers = 8;
    {
        const char *env = getenv("CKPT_WRITER_THREADS");
        if (env) n_writers = atoi(env);
        if (n_writers < 1)  n_writers = 1;
        if (n_writers > 16) n_writers = 16;
        if (num_blocks && (uint32_t)n_writers > num_blocks)
            n_writers = (int)num_blocks;
        if (n_writers < 1) n_writers = 1;
    }

    pthread_t          tids[16];
    delta_writer_ctx_t ctxs[16];
    uint32_t per = num_blocks / (uint32_t)n_writers;
    uint32_t extra = num_blocks % (uint32_t)n_writers;
    uint32_t nxt = 0;
    for (int w = 0; w < n_writers; w++) {
        uint32_t chunk = per + (((uint32_t)w < extra) ? 1 : 0);
        ctxs[w].img       = img;
        ctxs[w].blocks    = blocks;
        ctxs[w].caps      = caps;
        ctxs[w].reg_start = nxt;
        ctxs[w].reg_end   = nxt + chunk;
        nxt += chunk;
        pthread_create(&tids[w], NULL, delta_writer_thread, &ctxs[w]);
    }
    for (int w = 0; w < n_writers; w++)
        pthread_join(tids[w], NULL);
    printf("  [delta-par] parallel write with %d writer thread(s)\n", n_writers);

    if (!g_no_image_write)
        (void)msync(mp, (size_t)image_size, MS_ASYNC);
    if (munmap(mp, (size_t)image_size) < 0)
        fprintf(stderr, "  ✗ delta-par: munmap: %s\n", strerror(errno));
    if (fd >= 0) close(fd);

    double wms = now_ms() - tw0;
    if (write_ms_out) *write_ms_out = wms;
    printf("  ✓ delta parallel-written  %.1f ms  (%.2f GB/s)\n\n",
           wms, wms > 0
                ? total_delta / (1024.0*1024.0*1024.0) / (wms/1000.0) : 0.0);

    free(blocks);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Capture buffer pool — pre-allocated, reused across all ranges      */
/*                                                                    */
/* Large mallocs (4 GB+) on every range are expensive in TDX/CC mode: */
/* glibc routes them through mmap, and first-touch to each new page   */
/* pays confidential-memory allocation overhead. We observed ~12 s    */
/* of excess capture-wall time over the pure ioctl sum on 81 ranges.  */
/*                                                                    */
/* Fix: pre-allocate N slots at startup, each sized to the largest    */
/* range. Main thread acquires a free slot, passes it to the capture  */
/* function (which uses the slot's buffers instead of mallocing), and */
/* enqueues the slot index along with the cap. The writer releases    */
/* the slot back to the pool after memcpy — the buffers are not      */
/* freed until the pool is destroyed at end of checkpoint.             */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t            *resmap;        /* max_n_pages bytes                */
    uint8_t            *cpu_buf;       /* max_length bytes                 */
    uint8_t            *gpu_buf;       /* max_length bytes                 */
    page_crypto_meta_t *crypto_meta;   /* max_n_pages entries              */
    uint64_t            max_length;
    uint64_t            max_n_pages;
    int                 verified;      /* sanity: 1 once kernel write seen */
} capture_slot_t;

typedef struct {
    capture_slot_t *slots;
    int             n_slots;
    int            *free_stack;   /* LIFO of free slot indices            */
    int             n_free;
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;

    /* Shm-backed gpu_buf + cpu_buf storage. resmap + crypto_meta stay
     * malloc'd (small, MB-scale). Layout in shm:
     *   [ slot0.gpu_buf ][ slot1.gpu_buf ]...[ slotN-1.gpu_buf ]
     *   [ slot0.cpu_buf ][ slot1.cpu_buf ]...[ slotN-1.cpu_buf ]
     * Stride is rounded up to a generous round number (POOL_SHM_STRIDE_MIN)
     * to leave headroom for workloads with larger ranges, since the shm
     * is already pre-faulted by ckpt_prealloc — extra stride is free. */
    int       shm_fd;
    uint8_t  *shm_base;
    size_t    shm_mapped_size;
    uint64_t  stride;             /* per-slot byte size in shm            */
} capture_pool_t;

#define POOL_SHM_NAME         "/ckpt_core_gpu_staging"
#define POOL_SHM_STRIDE_MIN   (4ULL * 1024 * 1024 * 1024)  /* 4 GB        */

/* Sanity-check toggle: when 1, the pool poisons each slot's gpu_buf with
 * a known 0xA5 byte at allocation, and the first time the writer thread
 * consumes a slot it spot-checks that the kernel actually overwrote it
 * (i.e. the bytes really landed at the expected shm offset). Once we've
 * confirmed shm-backed slots work end-to-end on real workloads, set to
 * 0 to skip the poison + verify cost. */
#define POOL_SHM_SANITY       1

static int capture_pool_init(capture_pool_t *p, int n_slots,
                              uint64_t max_length, uint64_t max_n_pages)
{
    memset(p, 0, sizeof(*p));
    p->shm_fd   = -1;
    p->shm_base = NULL;
    p->slots      = calloc(n_slots, sizeof(*p->slots));
    p->free_stack = calloc(n_slots, sizeof(int));
    if (!p->slots || !p->free_stack) {
        free(p->slots); free(p->free_stack);
        return -1;
    }
    p->n_slots = n_slots;
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->not_empty, NULL);

    /* Round stride up so the same prealloc'd shm survives bigger workloads
     * without resizing — shm RAM is already paid for, headroom is free. */
    uint64_t stride = max_length;
    if (stride < POOL_SHM_STRIDE_MIN) stride = POOL_SHM_STRIDE_MIN;
    p->stride = stride;

    size_t meta_bytes  = (size_t)max_n_pages * sizeof(page_crypto_meta_t);
    size_t need_shm    = (size_t)2 * (size_t)n_slots * (size_t)stride;

    /* Open the prealloc'd shm staging file. Required dependency. */
    p->shm_fd = shm_open(POOL_SHM_NAME, O_RDWR, 0);
    if (p->shm_fd < 0) {
        fprintf(stderr,
                "  ✗ pool: shm_open(/dev/shm%s): %s\n"
                "  ✗ pool: per-range capture pool needs the prealloc'd staging\n"
                "  ✗ pool: file. Run this once per boot:\n"
                "\n"
                "      sudo ./ckpt_prealloc --gpu-size-gb %.0f --out <base.img> --out-size-gb 90\n"
                "\n",
                POOL_SHM_NAME, strerror(errno),
                (need_shm / (1024.0*1024.0*1024.0)) + 1.0);
        free(p->slots); free(p->free_stack);
        pthread_mutex_destroy(&p->lock);
        pthread_cond_destroy(&p->not_empty);
        return -1;
    }
    struct stat st;
    if (fstat(p->shm_fd, &st) < 0) {
        fprintf(stderr, "  ✗ pool: fstat shm: %s\n", strerror(errno));
        close(p->shm_fd);
        free(p->slots); free(p->free_stack);
        pthread_mutex_destroy(&p->lock);
        pthread_cond_destroy(&p->not_empty);
        return -1;
    }
    if ((size_t)st.st_size < need_shm) {
        fprintf(stderr,
                "  ✗ pool: shm staging is %.2f GB but the pool needs %.2f GB\n"
                "  ✗ pool: (%d slots × 2 × %.2f GB stride). Rerun ckpt_prealloc\n"
                "  ✗ pool: with --gpu-size-gb %.0f or larger.\n",
                st.st_size / (1024.0*1024.0*1024.0),
                need_shm   / (1024.0*1024.0*1024.0),
                n_slots, stride / (1024.0*1024.0*1024.0),
                (need_shm / (1024.0*1024.0*1024.0)) + 1.0);
        close(p->shm_fd);
        free(p->slots); free(p->free_stack);
        pthread_mutex_destroy(&p->lock);
        pthread_cond_destroy(&p->not_empty);
        return -1;
    }
    p->shm_base = mmap(NULL, need_shm, PROT_READ | PROT_WRITE,
                       MAP_SHARED, p->shm_fd, 0);
    if (p->shm_base == MAP_FAILED) {
        fprintf(stderr, "  ✗ pool: mmap shm (%.2f GB): %s\n",
                need_shm / (1024.0*1024.0*1024.0), strerror(errno));
        p->shm_base = NULL;
        close(p->shm_fd);
        free(p->slots); free(p->free_stack);
        pthread_mutex_destroy(&p->lock);
        pthread_cond_destroy(&p->not_empty);
        return -1;
    }
    p->shm_mapped_size = need_shm;
    (void)madvise(p->shm_base, need_shm, MADV_HUGEPAGE);

    printf("  [pool] %d slots × stride %.2f GB → %.2f GB shm partition "
           "(reuses prealloc'd %s, no malloc/first-touch in timed region)\n",
           n_slots,
           stride    / (1024.0*1024.0*1024.0),
           need_shm  / (1024.0*1024.0*1024.0),
           POOL_SHM_NAME);

    /* Per-slot small mallocs (resmap + crypto_meta) — KB to ~MB each,
     * cheap enough that we don't bother shm-backing them. */
    for (int i = 0; i < n_slots; i++) {
        p->slots[i].max_length  = stride;
        p->slots[i].max_n_pages = max_n_pages;
        p->slots[i].resmap      = malloc(max_n_pages);
        p->slots[i].crypto_meta = malloc(meta_bytes);
        /* Partition: gpu_bufs first half, cpu_bufs second half. */
        p->slots[i].gpu_buf = p->shm_base + (size_t)i * stride;
        p->slots[i].cpu_buf = p->shm_base + (size_t)(n_slots + i) * stride;
        if (!p->slots[i].resmap || !p->slots[i].crypto_meta) {
            fprintf(stderr, "  ✗ capture_pool slot[%d] small alloc failed\n", i);
            for (int j = 0; j <= i; j++) {
                free(p->slots[j].resmap);
                free(p->slots[j].crypto_meta);
            }
            munmap(p->shm_base, p->shm_mapped_size);
            close(p->shm_fd);
            free(p->slots); free(p->free_stack);
            pthread_mutex_destroy(&p->lock);
            pthread_cond_destroy(&p->not_empty);
            return -1;
        }
        (void)madvise(p->slots[i].crypto_meta, meta_bytes, MADV_HUGEPAGE);
#if POOL_SHM_SANITY
        /* Poison the first few pages of each gpu_buf slot. After the
         * kernel writes encrypted bytes via ioctl 115, capture_range_*
         * verifies the poison is gone — confirming bytes landed at the
         * expected shm offset and we partitioned correctly. */
        memset(p->slots[i].gpu_buf, 0xA5, 16384);
#endif
        p->free_stack[i] = i;
    }
    p->n_free = n_slots;
    return 0;
}

static void capture_pool_destroy(capture_pool_t *p)
{
    for (int i = 0; i < p->n_slots; i++) {
        free(p->slots[i].resmap);
        free(p->slots[i].crypto_meta);
        /* gpu_buf + cpu_buf are partitions of shm — not freed. */
    }
    if (p->shm_base) munmap(p->shm_base, p->shm_mapped_size);
    if (p->shm_fd >= 0) close(p->shm_fd);
    free(p->slots);
    free(p->free_stack);
    pthread_mutex_destroy(&p->lock);
    pthread_cond_destroy(&p->not_empty);
    memset(p, 0, sizeof(*p));
}

static int capture_pool_acquire(capture_pool_t *p)
{
    pthread_mutex_lock(&p->lock);
    while (p->n_free == 0)
        pthread_cond_wait(&p->not_empty, &p->lock);
    int idx = p->free_stack[--p->n_free];
    pthread_mutex_unlock(&p->lock);
    return idx;
}

static void capture_pool_release(capture_pool_t *p, int idx)
{
    pthread_mutex_lock(&p->lock);
    p->free_stack[p->n_free++] = idx;
    pthread_cond_signal(&p->not_empty);
    pthread_mutex_unlock(&p->lock);
}

/* ------------------------------------------------------------------ */
/* capture_range: issue ioctl 109 for one range, fill cap.            */
/* Returns 0 on success, -1 on error.                                 */
/* ------------------------------------------------------------------ */
static int capture_range(int uvm_fd, NvU64 base, NvU64 length,
                         range_capture_t *cap, double *read_ms_acc)
{
    cap->base    = base;
    cap->length  = length;
    cap->n_pages = length / 4096;

    cap->resmap  = malloc(cap->n_pages);
    cap->cpu_buf = malloc(length);
    cap->gpu_buf = malloc(length);
    if (!cap->resmap || !cap->cpu_buf || !cap->gpu_buf) {
        fprintf(stderr, "  ✗ malloc failed for range base=0x%llx\n",
                (unsigned long long)base);
        range_capture_free(cap);
        return -1;
    }
    memset(cap->resmap, 2, cap->n_pages);   /* default: absent */

    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_PARAMS rp = {0};
    rp.base          = base;
    rp.length        = length;
    rp.cpu_buf       = (NvU64)(uintptr_t)cap->cpu_buf;
    rp.cpu_buf_size  = length;
    rp.gpu_buf       = (NvU64)(uintptr_t)cap->gpu_buf;
    rp.gpu_buf_size  = length;
    rp.residency_map = (NvU64)(uintptr_t)cap->resmap;
    /* gpu_uuid left zero → kernel picks first registered GPU */

    double t0  = now_ms();
    int    ret = ioctl(uvm_fd, UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT, &rp);
    double ms  = now_ms() - t0;
    if (read_ms_acc) *read_ms_acc += ms;

    if (ret < 0 || rp.rmStatus != NV_OK) {
        fprintf(stderr, "  ✗ READ_PAGES_RESIDENT base=0x%llx: %s  status=0x%08x\n",
                (unsigned long long)base, strerror(errno), rp.rmStatus);
        range_capture_free(cap);
        return -1;
    }

    cap->cpu_bytes = rp.cpu_bytes_out;
    cap->gpu_bytes = rp.gpu_bytes_out;

    NvU64 absent = 0;
    for (NvU64 i = 0; i < cap->n_pages; i++)
        if (cap->resmap[i] == 2) absent++;

    if (g_verbose)
        printf("    ✓ base=0x%llx  %.1f ms  CPU=%llu  GPU=%llu  absent=%llu pages\n",
               (unsigned long long)base, ms,
               (unsigned long long)(cap->cpu_bytes / 4096),
               (unsigned long long)(cap->gpu_bytes / 4096),
               (unsigned long long)absent);
    return 0;
}

/* ------------------------------------------------------------------ */
/* capture_range_encrypted: read GPU pages as CE-encrypted ciphertext +    */
/* per-transfer crypto metadata via ioctl 115 (multi-threaded variant of   */
/* the v3 path). Each ioctl invocation spawns N kthread workers in the     */
/* kernel; thread count is read once from CKPT_MT_THREADS env var or       */
/* defaults to CKPT_MT_THREADS_DEFAULT.                                     */
/* ------------------------------------------------------------------ */
static int capture_range_encrypted(int uvm_fd, NvU64 base, NvU64 length,
                                    range_capture_t *cap,
                                    capture_slot_t *slot,
                                    double *read_ms_acc)
{
    static int  s_mt_threads      = -1;
    if (s_mt_threads < 0) {
        const char *env = getenv("CKPT_MT_THREADS");
        s_mt_threads = env ? atoi(env) : CKPT_MT_THREADS_DEFAULT;
        if (s_mt_threads < 1)  s_mt_threads = 1;
        if (s_mt_threads > 16) s_mt_threads = 16;
    }

    cap->base    = base;
    cap->length  = length;
    cap->n_pages = length / 4096;

    if (slot) {
        /* Borrow pre-allocated buffers from the pool slot. Ownership stays
         * with the pool — range_capture_free MUST NOT be called on this cap. */
        if (length > slot->max_length || cap->n_pages > slot->max_n_pages) {
            fprintf(stderr, "  ✗ range too large for pool slot "
                    "(length=%llu > max=%llu)\n",
                    (unsigned long long)length,
                    (unsigned long long)slot->max_length);
            return -1;
        }
        cap->resmap      = slot->resmap;
        cap->cpu_buf     = slot->cpu_buf;
        cap->gpu_buf     = slot->gpu_buf;
        cap->crypto_meta = slot->crypto_meta;
    } else {
        /* Legacy per-range malloc path (non-encrypted flow). */
        cap->resmap  = malloc(cap->n_pages);
        cap->cpu_buf = malloc(length);
        cap->gpu_buf = malloc(length);
        cap->crypto_meta = malloc(cap->n_pages * sizeof(page_crypto_meta_t));
        if (!cap->resmap || !cap->cpu_buf || !cap->gpu_buf || !cap->crypto_meta) {
            fprintf(stderr, "  ✗ malloc failed for range base=0x%llx\n",
                    (unsigned long long)base);
            range_capture_free(cap);
            return -1;
        }
    }
    memset(cap->resmap, 2, cap->n_pages);

    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS rp = {0};
    rp.base             = base;
    rp.length           = length;
    rp.cpu_buf          = (NvU64)(uintptr_t)cap->cpu_buf;
    rp.cpu_buf_size     = length;
    rp.gpu_buf          = (NvU64)(uintptr_t)cap->gpu_buf;
    rp.gpu_buf_size     = length;
    rp.residency_map    = (NvU64)(uintptr_t)cap->resmap;
    rp.crypto_meta      = (NvU64)(uintptr_t)cap->crypto_meta;
    rp.crypto_meta_size = cap->n_pages * sizeof(page_crypto_meta_t);
    rp.num_threads      = (uint32_t)s_mt_threads;

    double t0  = now_ms();
    int    ret;
    if (g_track_only) {
        /* Skip the D2H. resmap is already memset(2)=absent above, so callers
         * see "no resident pages, no transfers." POOL_SHM_SANITY won't fire
         * because gpu_bytes stays 0. Optional per-call sleep keeps the round
         * duration comparable to the production path. */
        if (g_track_only_budget_ms > 0)
            usleep((useconds_t)g_track_only_budget_ms * 1000);
        rp.cpu_bytes_out = 0;
        rp.gpu_bytes_out = 0;
        rp.num_transfers = 0;
        rp.rmStatus      = NV_OK;
        ret = 0;
    } else {
        ret = ioctl(uvm_fd, UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT, &rp);
    }
    double ms  = now_ms() - t0;
    if (read_ms_acc) *read_ms_acc += ms;

    if (ret < 0 || rp.rmStatus != NV_OK) {
        fprintf(stderr, "  ✗ READ_PAGES_RESIDENT_ENCRYPTED_MT base=0x%llx: %s  status=0x%08x\n",
                (unsigned long long)base, strerror(errno), rp.rmStatus);
        /* Only free buffers we own. Pool-backed buffers stay with the pool. */
        if (!slot)
            range_capture_free(cap);
        return -1;
    }

    cap->cpu_bytes      = rp.cpu_bytes_out;
    cap->gpu_bytes      = rp.gpu_bytes_out;
    cap->encrypted_page_count = rp.num_transfers;

#if POOL_SHM_SANITY
    /* Shm-partition sanity: confirm the kernel actually wrote into the
     * portion of /dev/shm/ckpt_core_gpu_staging we partitioned for this
     * slot. Only check on a slot's first use (subsequent uses may legitly
     * begin with 0xA5 ciphertext bytes). Disable POOL_SHM_SANITY once
     * we've confirmed the partitioning works on real workloads. */
    if (slot && !slot->verified && cap->gpu_bytes > 0) {
        int still_poison = 1;
        for (int j = 0; j < 16; j++) {
            if (slot->gpu_buf[j] != 0xA5) { still_poison = 0; break; }
        }
        if (still_poison) {
            fprintf(stderr,
                    "  ✗ pool-sanity: gpu_buf still 0xA5 after ioctl 115 "
                    "(base=0x%llx gpu_bytes=%llu) — kernel did NOT write to "
                    "the shm-partition slot\n",
                    (unsigned long long)base,
                    (unsigned long long)cap->gpu_bytes);
            return -1;
        }
        slot->verified = 1;
        if (g_verbose)
            fprintf(stderr, "  [pool-sanity] slot @gpu_buf=%p verified "
                            "(base=0x%llx, kernel wrote into shm partition)\n",
                    slot->gpu_buf, (unsigned long long)base);
    }
#endif

    NvU64 absent = 0;
    for (NvU64 i = 0; i < cap->n_pages; i++)
        if (cap->resmap[i] == 2) absent++;

    if (g_verbose)
        printf("    ✓ base=0x%llx  %.1f ms  CPU=%llu  GPU=%llu(encrypted)  absent=%llu pages  meta=%llu\n",
           (unsigned long long)base, ms,
           (unsigned long long)(cap->cpu_bytes / 4096),
           (unsigned long long)(cap->gpu_bytes / 4096),
           (unsigned long long)absent,
           (unsigned long long)cap->encrypted_page_count);
    return 0;
}

/* ====================================================================== */
/* P3-B: capture_and_write_coalesced                                        */
/*                                                                          */
/* Replaces the per-range capture_range_encrypted loop with a SINGLE        */
/* ioctl 115 call covering all filtered ranges as one big VA span.          */
/*                                                                          */
/* Rationale: each ioctl 115 call pays ~150-300 ms of per-call setup        */
/* (kthread spawn, DMA buffer alloc, channel acquire, PASS 1 walk).         */
/* With ~80 ranges for vLLM that's 12+ seconds of amortizable overhead.     */
/* Collapsing into one call approaches the ~22 GB/s kernel ceiling we       */
/* measured in bench_transfer_with_prepareall vs the ~6 GB/s we currently   */
/* see with per-range calls.                                                */
/*                                                                          */
/* The kernel's PASS 1 uses uvm_for_each_va_range_in() which walks va_range */
/* entries via rbtree — gap addresses (between allocs) are NATURALLY        */
/* skipped with no per-page cost. Only constraint: `base` must fall inside  */
/* a managed va_range, which is guaranteed when we pick base = min(alloc).  */
/*                                                                          */
/* Memory cost: gpu_buf = sum(filt_size) ≈ 72 GB for vLLM. resmap =         */
/* span/4096 ≈ 18 MB. meta = worst-case (total_payload/4096)×41 ≈ 754 MB    */
/* but kernel fills only ~total_payload/2MB entries so actual is ~1.5 MB.   */
/*                                                                          */
/* Assumption: all pages are GPU-resident. Fails loudly (returns error) if  */
/* the kernel reports any CPU-resident pages in cpu_bytes_out.              */
/* ====================================================================== */
typedef struct {
    uint64_t va;
    uint64_t size;
} p3b_alloc_t;

static int p3b_cmp_alloc_va(const void *a, const void *b)
{
    const p3b_alloc_t *aa = (const p3b_alloc_t *)a;
    const p3b_alloc_t *bb = (const p3b_alloc_t *)b;
    if (aa->va < bb->va) return -1;
    if (aa->va > bb->va) return  1;
    return 0;
}

static ssize_t p3b_write_full(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(fd, p, remaining);
        if (n <= 0) return -1;
        p += n;
        remaining -= (size_t)n;
    }
    return (ssize_t)len;
}

/* Parallel writer worker for the P3-B mmap write phase. Each worker is
 * assigned a contiguous slice of the range index space; it memcpys each
 * range's (resmap + gpu_section + crypto_meta) into the mmap'd output
 * file at the per-range desc.data_offset. Writers are fully disjoint —
 * no synchronization needed. */
typedef struct {
    uint8_t                     *img;           /* mmap'd output file */
    const ckpt_v2_range_desc_t  *descs;
    const uint8_t               *big_gpu_buf;
    const page_crypto_meta_t    *big_meta;
    const uint64_t              *alloc_gpu_off;
    const uint64_t              *alloc_meta_idx;
    const uint8_t               *ones_scratch;
    uint32_t                     range_start;
    uint32_t                     range_end;      /* exclusive */
} p3b_writer_ctx_t;

static void *p3b_writer_thread(void *arg)
{
    p3b_writer_ctx_t *c = (p3b_writer_ctx_t *)arg;
    for (uint32_t i = c->range_start; i < c->range_end; i++) {
        uint8_t *p = c->img + c->descs[i].data_offset;
        /* resmap: all 1s (verified GPU-resident for this filt alloc) */
        memcpy(p, c->ones_scratch, (size_t)c->descs[i].resmap_size);
        p += c->descs[i].resmap_size;
        /* cpu_section: empty (cpu_bytes=0) */
        /* gpu_section: slice of big_gpu_buf at alloc_gpu_off[i] */
        memcpy(p, c->big_gpu_buf + c->alloc_gpu_off[i],
               (size_t)c->descs[i].gpu_bytes);
        p += c->descs[i].gpu_bytes;
        /* crypto_meta: slice of big_meta at alloc_meta_idx[i] */
        memcpy(p, c->big_meta + c->alloc_meta_idx[i],
               (size_t)c->descs[i].crypto_meta_bytes);
    }
    return NULL;
}

static int capture_and_write_coalesced(int uvm_fd,
                                         const char *base_path,
                                         const uint64_t *filt_base,
                                         const uint64_t *filt_size,
                                         uint32_t n_filt,
                                         double *read_ms_out,
                                         double *write_ms_out,
                                         unsigned long ioctl_cmd,
                                         uint64_t *gpu_bytes_out)
{
    int ret_code = -1;

    /* Read mt thread count from env (same as capture_range_encrypted) */
    static int s_mt_threads = -1;
    if (s_mt_threads < 0) {
        const char *env = getenv("CKPT_MT_THREADS");
        s_mt_threads = env ? atoi(env) : CKPT_MT_THREADS_DEFAULT;
        if (s_mt_threads < 1)  s_mt_threads = 1;
        if (s_mt_threads > 16) s_mt_threads = 16;
    }

    /* 1. Sort allocations by VA ---------------------------------------- */
    p3b_alloc_t *allocs = calloc(n_filt, sizeof(*allocs));
    if (!allocs) { fprintf(stderr, "  ✗ coalesce: calloc allocs\n"); return -1; }
    for (uint32_t i = 0; i < n_filt; i++) {
        allocs[i].va   = filt_base[i];
        allocs[i].size = filt_size[i];
    }
    qsort(allocs, n_filt, sizeof(*allocs), p3b_cmp_alloc_va);

    /* 2. Compute base, span, total_payload ----------------------------- */
    uint64_t base          = allocs[0].va;
    uint64_t end           = allocs[n_filt - 1].va + allocs[n_filt - 1].size;
    uint64_t span          = end - base;
    uint64_t total_payload = 0;
    for (uint32_t i = 0; i < n_filt; i++) total_payload += allocs[i].size;

    const int is_dirty_delta =
        (ioctl_cmd == UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT);

    if (is_dirty_delta)
        /* span/payload here is only the SCAN WINDOW handed to the kernel —
         * the kernel internally filters to dirty 2MB blocks and copies just
         * that subset. Actual copied bytes are reported as gpu_out below. */
        printf("  [P3-B coalesce] %u ranges: base=0x%016llx scan-window span=%.2f GB "
               "resident=%.2f GB (kernel copies only the dirty subset)\n",
               n_filt, (unsigned long long)base,
               span          / (1024.0*1024.0*1024.0),
               total_payload / (1024.0*1024.0*1024.0));
    else
        printf("  [P3-B coalesce] %u ranges: base=0x%016llx span=%.2f GB payload=%.2f GB\n",
               n_filt, (unsigned long long)base,
               span          / (1024.0*1024.0*1024.0),
               total_payload / (1024.0*1024.0*1024.0));

    /* 3. Acquire shared buffers ----------------------------------------
     *
     * big_gpu_buf: persistent tmpfs file created by ckpt_prealloc.
     *   - Name: /dev/shm/ckpt_core_gpu_staging (via shm_open)
     *   - Must be pre-warmed (pages faulted in) to avoid per-invocation
     *     first-touch cost under TDX encryption.
     *   - We just open + fstat (verify size ≥ span) + mmap + madvise.
     *   - No pre-touch here — ckpt_prealloc already did it.
     *   - No unlink — lifetime is user-managed via `rm` or reboot.
     *
     * big_cpu_buf: 2 GB malloc (per-invocation). CPU-resident pages are
     *   rare (~8 MB observed for vLLM); 2 GB is a safe upper bound. If
     *   the kernel writes more than 2 GB we fail loudly.
     *
     * big_resmap, big_meta: small malloc (KB to ~750 MB) per-invocation. */

    #define P3B_GPU_STAGING_NAME  "/ckpt_core_gpu_staging"
    #define P3B_CPU_BUF_CAP       (2ULL * 1024 * 1024 * 1024)  /* 2 GB safety */

    int      gpu_fd          = -1;
    uint8_t *big_gpu_buf     = NULL;
    size_t   gpu_mapped_size = 0;
    size_t   resmap_bytes    = (size_t)(span / 4096);
    uint8_t *big_resmap      = calloc(resmap_bytes, 1);
    uint8_t *big_cpu_buf     = malloc(P3B_CPU_BUF_CAP);
    size_t   max_transfers   = (size_t)(span / 4096);
    size_t   meta_cap_bytes  = max_transfers * sizeof(page_crypto_meta_t);
    page_crypto_meta_t *big_meta = malloc(meta_cap_bytes);

    if (!big_resmap || !big_cpu_buf || !big_meta) {
        fprintf(stderr, "  ✗ coalesce: small malloc failed\n");
        goto out_free;
    }

    /* Open persistent GPU staging file */
    gpu_fd = shm_open(P3B_GPU_STAGING_NAME, O_RDWR, 0644);
    if (gpu_fd < 0) {
        fprintf(stderr,
                "  ✗ P3-B: shm_open(/dev/shm%s): %s\n"
                "  ✗ P3-B: the persistent GPU staging file does not exist.\n"
                "  ✗ P3-B: run this BEFORE starting vLLM (or once per boot):\n"
                "\n"
                "      sudo ./ckpt_prealloc --gpu-size-gb 90 "
                "--out %s --out-size-gb 90\n"
                "\n"
                "  ✗ P3-B: (--gpu-size-gb must be ≥ %.2f for this workload)\n",
                P3B_GPU_STAGING_NAME, strerror(errno), base_path,
                span / (1024.0*1024.0*1024.0));
        goto out_free;
    }
    struct stat gpu_st;
    if (fstat(gpu_fd, &gpu_st) < 0) {
        fprintf(stderr, "  ✗ P3-B: fstat gpu staging: %s\n", strerror(errno));
        close(gpu_fd); gpu_fd = -1;
        goto out_free;
    }
    if ((uint64_t)gpu_st.st_size < span) {
        fprintf(stderr,
                "  ✗ P3-B: gpu staging is %.2f GB but this workload needs %.2f GB.\n"
                "  ✗ P3-B: rerun ckpt_prealloc with --gpu-size-gb %.0f or larger.\n",
                gpu_st.st_size / (1024.0*1024.0*1024.0),
                span           / (1024.0*1024.0*1024.0),
                span           / (1024.0*1024.0*1024.0) + 1.0);
        close(gpu_fd); gpu_fd = -1;
        goto out_free;
    }
    /* Map exactly the span we need (not the full file). */
    gpu_mapped_size = (size_t)span;
    big_gpu_buf = mmap(NULL, gpu_mapped_size, PROT_READ|PROT_WRITE,
                        MAP_SHARED, gpu_fd, 0);
    if (big_gpu_buf == MAP_FAILED) {
        fprintf(stderr, "  ✗ P3-B: mmap gpu staging (%.2f GB): %s\n",
                gpu_mapped_size / (1024.0*1024.0*1024.0), strerror(errno));
        big_gpu_buf = NULL;
        close(gpu_fd); gpu_fd = -1;
        goto out_free;
    }
    (void)madvise(big_gpu_buf, gpu_mapped_size, MADV_HUGEPAGE);

    /* MADV hints for the small buffers. */
    (void)madvise(big_meta, meta_cap_bytes, MADV_HUGEPAGE);

    printf("  [P3-B coalesce] using persistent gpu staging (%.2f GB warm-mapped)\n",
           gpu_mapped_size / (1024.0*1024.0*1024.0));

    /* 4. Single ioctl 115 call spanning everything ---------------------
     * gpu_buf_size is span (the mmap'd extent of our persistent staging);
     * cpu_buf_size is the 2 GB safety cap. If the kernel tries to write
     * more than 2 GB of CPU-resident pages into cpu_buf, it'll error out
     * rather than corrupting heap. */
    UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS rp = {0};
    rp.base             = base;
    rp.length           = span;
    rp.cpu_buf          = (NvU64)(uintptr_t)big_cpu_buf;
    rp.cpu_buf_size     = P3B_CPU_BUF_CAP;
    rp.gpu_buf          = (NvU64)(uintptr_t)big_gpu_buf;
    rp.gpu_buf_size     = gpu_mapped_size;
    rp.residency_map    = (NvU64)(uintptr_t)big_resmap;
    rp.crypto_meta      = (NvU64)(uintptr_t)big_meta;
    rp.crypto_meta_size = meta_cap_bytes;
    rp.num_threads      = (uint32_t)s_mt_threads;

    /* track-only: SKIP only the precopy ioctl 115 path here. The dirty-delta
     * ioctl 116 path runs after AT_BOUNDARY (app quiesced) — there's nothing
     * to contend with, AND the delta is the only useful payload of the
     * round, so we always do that for real. */
    int track_only_skip_here = g_track_only && !is_dirty_delta;
    printf("  [P3-B coalesce] calling coalesced MT ioctl %lu (mt_threads=%d)%s ...\n",
           ioctl_cmd, s_mt_threads, track_only_skip_here ? " [TRACK-ONLY: skipped]" : "");
    double t0 = now_ms();
    int ret;
    if (track_only_skip_here) {
        /* Make the sparse consumer below see "all pages absent" so it emits
         * an empty delta instead of treating uninitialized calloc'd resmap
         * (zeros) as "all CPU-resident" — which would loop over the full
         * span as if every page had real data behind it. */
        memset(big_resmap, 2, resmap_bytes);
        if (g_track_only_budget_ms > 0)
            usleep((useconds_t)g_track_only_budget_ms * 1000);
        rp.cpu_bytes_out = 0;
        rp.gpu_bytes_out = 0;
        rp.num_transfers = 0;
        rp.rmStatus      = NV_OK;
        ret = 0;
    } else {
        ret = ioctl(uvm_fd, ioctl_cmd, &rp);
    }
    double read_ms = now_ms() - t0;
    if (ret < 0 || rp.rmStatus != NV_OK) {
        fprintf(stderr, "  ✗ coalesced MT ioctl %lu failed: %s  rmStatus=0x%08x\n",
                ioctl_cmd, strerror(errno), rp.rmStatus);
        goto out_free;
    }
    if (read_ms_out)   *read_ms_out   = read_ms;
    if (gpu_bytes_out) *gpu_bytes_out = (uint64_t)rp.gpu_bytes_out;

    /* Effective rate must be over bytes the kernel actually moved
     * (cpu_out + gpu_out), NOT the scan-window payload — otherwise the
     * dirty-delta path reports a fictitious rate (full footprint / time). */
    uint64_t moved_bytes = rp.gpu_bytes_out + rp.cpu_bytes_out;
    double   eff_gbps    = moved_bytes / (1024.0*1024.0*1024.0)
                           / (read_ms / 1000.0);
    printf("  [P3-B coalesce] ioctl done: %.1f ms  (%.2f GB/s over %.2f GB moved)  "
           "cpu_out=%llu gpu_out=%llu transfers=%llu\n",
           read_ms, eff_gbps,
           moved_bytes / (1024.0*1024.0*1024.0),
           (unsigned long long)rp.cpu_bytes_out,
           (unsigned long long)rp.gpu_bytes_out,
           (unsigned long long)rp.num_transfers);

    /* 5. Soft sanity info ---------------------------------------------
     * With span-sized buffers and non-filt allocs inside the span, we
     * can safely tolerate:
     *   - cpu_bytes_out > 0   (CPU-resident pages exist — either in
     *                          filt allocs or non-filt extras)
     *   - gpu_bytes_out > total_payload  (non-filt allocs' GPU pages
     *                                     got written to gpu_buf too)
     * Correctness enforcement is done later per filt alloc against the
     * resmap: each filt alloc must have ALL its pages resmap==1
     * (GPU-resident) for the all-GPU-resident slicing path to work. */
    if (rp.cpu_bytes_out > 0) {
        printf("  [P3-B coalesce] note: %llu bytes CPU-resident in span "
               "(will verify filt allocs are fully GPU-resident below)\n",
               (unsigned long long)rp.cpu_bytes_out);
    }
    if (rp.gpu_bytes_out > total_payload) {
        printf("  [P3-B coalesce] note: gpu_bytes_out=%llu > filt payload=%llu "
               "(%llu bytes from non-filt allocs in span)\n",
               (unsigned long long)rp.gpu_bytes_out,
               (unsigned long long)total_payload,
               (unsigned long long)(rp.gpu_bytes_out - total_payload));
    }

    /* ---- Tier3-fused dirty-delta: SPARSE consumer (Option A) -------- *
     * The kernel filtered to dirty 2MB blocks, so big_resmap is sparse:
     * present pages are resmap 0/1, every page in a non-dirty block is
     * resmap 2 (absent). The dense per-alloc slicer below assumes every
     * page of every alloc is resmap==1 and aborts on the first absent
     * page ("Mixed residency not yet supported"). Instead: coalesce
     * maximal runs of present (resmap!=2) pages into (va,len) regions
     * and emit the existing ckpt_inc_delta format via write_delta(), so
     * restore is byte-format-identical with the legacy [7] path. The
     * kernel packs gpu_buf / cpu_buf / big_meta in VA order, so each
     * region's sections are contiguous slices — no copies. Page-granular
     * coalescing may yield more, smaller regions than the legacy 2MB
     * oracle if a dirty block has internal evicted pages, but the union
     * of present pages (hence decrypted plaintext per VA) is identical. */
    if (is_dirty_delta) {
        const uint64_t span_pages = resmap_bytes;   /* == span / 4096 */

        /* Pass 1: count present-page runs (regions) + total present. */
        uint32_t n_regions     = 0;
        uint64_t total_present = 0;
        for (uint64_t p = 0; p < span_pages; ) {
            if (big_resmap[p] == 2) { p++; continue; }
            n_regions++;
            while (p < span_pages && big_resmap[p] != 2) { total_present++; p++; }
        }

        printf("  [P3-B coalesce] dirty-delta: %u present region(s), "
               "%llu present 4K page(s) (%.2f MB)\n",
               n_regions, (unsigned long long)total_present,
               total_present * 4096.0 / (1024.0*1024.0));

        range_capture_t *fcaps = calloc(n_regions ? n_regions : 1,
                                        sizeof(range_capture_t));
        if (!fcaps) {
            fprintf(stderr, "  ✗ dirty-delta: calloc fcaps\n");
            goto out_free;
        }

        /* Pass 2: one range_capture_t per region, all members slices
         * into the big buffers. cpu_off / gpu_off / meta_idx advance
         * monotonically since regions are visited in VA order. */
        uint64_t cpu_off  = 0;       /* cumulative resmap==0 bytes  */
        uint64_t gpu_off  = 0;       /* cumulative resmap==1 bytes  */
        uint64_t meta_idx = 0;       /* next big_meta entry         */
        uint64_t meta_acc = 0;       /* cumulative meta size bytes  */
        uint32_t ri       = 0;
        int      walk_err = 0;

        for (uint64_t p = 0; p < span_pages && !walk_err; ) {
            if (big_resmap[p] == 2) { p++; continue; }

            uint64_t s = p, cpu_in = 0, gpu_in = 0;
            while (p < span_pages && big_resmap[p] != 2) {
                if (big_resmap[p] == 0) cpu_in += 4096;
                else                    gpu_in += 4096;   /* resmap==1 */
                p++;
            }
            uint64_t reg_pages = p - s;

            /* meta_acc must already sit on this region's gpu_off: the
             * kernel breaks transfer runs at absent pages, so no
             * transfer can straddle a region (resmap!=2 run) boundary. */
            if (meta_acc != gpu_off) {
                fprintf(stderr,
                        "  ✗ dirty-delta: region %u meta misaligned "
                        "(meta_acc=%llu gpu_off=%llu)\n",
                        ri, (unsigned long long)meta_acc,
                        (unsigned long long)gpu_off);
                walk_err = 1; break;
            }
            uint64_t meta_start = meta_idx;
            uint64_t target     = gpu_off + gpu_in;
            while (meta_idx < rp.num_transfers && meta_acc < target) {
                meta_acc += big_meta[meta_idx].size;
                meta_idx++;
            }
            if (meta_acc != target) {
                fprintf(stderr,
                        "  ✗ dirty-delta: region %u meta coverage %llu "
                        "!= gpu_bytes %llu\n",
                        ri, (unsigned long long)(meta_acc - gpu_off),
                        (unsigned long long)gpu_in);
                walk_err = 1; break;
            }

            fcaps[ri].base                 = base + s * 4096ULL;
            fcaps[ri].length               = reg_pages * 4096ULL;
            fcaps[ri].n_pages              = reg_pages;
            fcaps[ri].cpu_bytes            = cpu_in;
            fcaps[ri].gpu_bytes            = gpu_in;
            fcaps[ri].encrypted_page_count = meta_idx - meta_start;
            fcaps[ri].resmap               = big_resmap  + s;
            fcaps[ri].cpu_buf              = big_cpu_buf + cpu_off;
            fcaps[ri].gpu_buf              = big_gpu_buf + gpu_off;
            fcaps[ri].crypto_meta          = big_meta    + meta_start;

            cpu_off += cpu_in;
            gpu_off += gpu_in;
            ri++;
        }

        if (walk_err) { free(fcaps); goto out_free; }

        /* restore keys blocks by absolute base_va; alloc_start_va/size
         * are reference-only for the multi-range path and
         * num_dirty_pages_64k is informational (restore ignores it).
         * Parallel mmap writer (not single-threaded fwrite) so the
         * delta write keeps pace with the ~17 GB/s fused read. */
        int werr = write_delta_parallel(base_path, fcaps, n_regions,
                                        /*alloc_start_va=*/base,
                                        /*alloc_size=*/span,
                                        /*num_dirty_pages_64k=*/total_present,
                                        write_ms_out);
        free(fcaps);             /* slices — must NOT range_capture_free */
        if (werr == 0) ret_code = 0;
        goto out_free;
    }

    /* 6. Slice big_gpu_buf and big_meta back into N per-range descriptors.
     *    The kernel fills outputs in VA order (via uvm_for_each_va_range_in)
     *    with cumulative gpu_buf_offset, so gpu slices are packed contiguously
     *    in VA order within gpu_buf. Since non-filt allocs may also have
     *    pages in the span, we cannot assume cumulative byte count over
     *    filt allocs matches the gpu_buf layout. Instead we walk the
     *    resmap to compute each filt alloc's actual byte offset in gpu_buf:
     *
     *      for each filt alloc in VA order:
     *        start_page = (alloc.va - base) / 4096
     *        end_page   = (alloc.va + alloc.size - base) / 4096
     *        verify all resmap[start_page..end_page) == 1 (GPU-resident)
     *        gpu_offset[i] = (# of 1s in resmap[0..start_page)) * 4096
     *
     *    We walk resmap incrementally since filt allocs are in VA order,
     *    giving O(span_pages) total time and O(1) extra space.
     *
     *    For meta slicing, we walk big_meta in parallel with filt allocs,
     *    skipping meta entries that fall before the current alloc's gpu
     *    range (those belong to non-filt allocs at lower VAs). Each run
     *    doesn't cross va_range boundaries (different phys backings), so
     *    meta entries align cleanly with alloc boundaries. */
    ckpt_v2_range_desc_t *descs = calloc(n_filt, sizeof(*descs));
    uint64_t *alloc_gpu_off     = calloc(n_filt, sizeof(uint64_t));
    uint64_t *alloc_meta_idx    = calloc(n_filt, sizeof(uint64_t));
    uint64_t *alloc_meta_count  = calloc(n_filt, sizeof(uint64_t));
    if (!descs || !alloc_gpu_off || !alloc_meta_idx || !alloc_meta_count) {
        fprintf(stderr, "  ✗ coalesce: calloc descs/aux arrays\n");
        free(descs); free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
        goto out_free;
    }

    uint64_t hdr_off = sizeof(ckpt_v2_file_hdr_t)
                     + (uint64_t)n_filt * sizeof(ckpt_v2_range_desc_t);
    uint64_t cur_off = hdr_off;
    const uint64_t num_meta = rp.num_transfers;

    /* Walk state:
     *   resmap_cursor     — byte-wise index into resmap[], advances monotonically
     *   gpu_bytes_so_far  — cumulative GPU bytes visited in gpu_buf so far
     *   meta_idx          — next meta[] entry to consider
     *   meta_bytes_so_far — cumulative byte coverage of meta[0..meta_idx) */
    uint64_t resmap_cursor     = 0;
    uint64_t gpu_bytes_so_far  = 0;
    uint64_t meta_idx          = 0;
    uint64_t meta_bytes_so_far = 0;
    int walk_err = 0;

    for (uint32_t i = 0; i < n_filt && !walk_err; i++) {
        uint64_t alloc_size = allocs[i].size;
        uint64_t n_pages    = alloc_size / 4096;
        uint64_t start_page = (allocs[i].va - base) / 4096;
        uint64_t end_page   = start_page + n_pages;

        /* Advance resmap_cursor to start_page, summing GPU bytes (these
         * come from non-filt allocs and prior filt allocs we've moved past). */
        while (resmap_cursor < start_page) {
            if (big_resmap[resmap_cursor] == 1)
                gpu_bytes_so_far += 4096;
            resmap_cursor++;
        }
        alloc_gpu_off[i] = gpu_bytes_so_far;

        /* Verify every page in this filt alloc is GPU-resident. */
        for (uint64_t p = start_page; p < end_page; p++) {
            if (big_resmap[p] != 1) {
                fprintf(stderr, "  ✗ P3-B: filt alloc[%u] va=0x%llx page[%llu] "
                                "has resmap=%u (expected 1=GPU-resident). "
                                "Mixed residency not yet supported.\n",
                        i, (unsigned long long)allocs[i].va,
                        (unsigned long long)(p - start_page),
                        big_resmap[p]);
                walk_err = 1;
                break;
            }
        }
        if (walk_err) break;

        /* All pages GPU-resident: advance cursor and running counters. */
        resmap_cursor    = end_page;
        gpu_bytes_so_far += alloc_size;

        /* Now locate meta entries covering [alloc_gpu_off[i], alloc_gpu_off[i]+alloc_size). */
        while (meta_idx < num_meta &&
               meta_bytes_so_far + big_meta[meta_idx].size <= alloc_gpu_off[i]) {
            /* This meta entry is entirely before our alloc's gpu range —
             * belongs to a non-filt alloc. Skip. */
            meta_bytes_so_far += big_meta[meta_idx].size;
            meta_idx++;
        }
        if (meta_bytes_so_far != alloc_gpu_off[i]) {
            fprintf(stderr, "  ✗ P3-B: meta[%llu] boundary=%llu doesn't align with "
                            "filt alloc[%u] gpu_off=%llu (meta entry spans alloc boundary?)\n",
                    (unsigned long long)meta_idx,
                    (unsigned long long)meta_bytes_so_far,
                    i, (unsigned long long)alloc_gpu_off[i]);
            walk_err = 1;
            break;
        }

        alloc_meta_idx[i] = meta_idx;
        uint64_t target = alloc_gpu_off[i] + alloc_size;
        while (meta_idx < num_meta && meta_bytes_so_far < target) {
            meta_bytes_so_far += big_meta[meta_idx].size;
            meta_idx++;
        }
        if (meta_bytes_so_far != target) {
            fprintf(stderr, "  ✗ P3-B: filt alloc[%u] meta coverage = %llu, expected %llu\n",
                    i, (unsigned long long)(meta_bytes_so_far - alloc_gpu_off[i]),
                    (unsigned long long)alloc_size);
            walk_err = 1;
            break;
        }
        alloc_meta_count[i] = meta_idx - alloc_meta_idx[i];

        /* Fill per-range descriptor */
        uint64_t meta_bytes = alloc_meta_count[i] * sizeof(page_crypto_meta_t);
        descs[i].base_va           = allocs[i].va;
        descs[i].length            = alloc_size;
        descs[i].num_pages         = n_pages;
        descs[i].data_offset       = cur_off;
        descs[i].resmap_size       = n_pages;
        descs[i].cpu_bytes         = 0;         /* verified all GPU-resident */
        descs[i].gpu_bytes         = alloc_size;
        descs[i].crypto_meta_bytes = meta_bytes;

        cur_off += n_pages + 0 + alloc_size + meta_bytes;
    }
    if (walk_err) {
        free(descs); free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
        goto out_free;
    }
    uint64_t image_size = cur_off;

    /* 7. Write the output file — ftruncate + mmap + memcpy approach,
     *    matching the existing baseline_stream_start pattern:
     *      - open + ftruncate output to image_size
     *      - mmap(PROT_READ|PROT_WRITE, MAP_SHARED)
     *      - madvise(MADV_HUGEPAGE) so tmpfs uses 2 MB pages (fewer faults)
     *      - memcpy header + descs + per-range data into the mmap
     *      - msync + munmap + close
     *
     *    This avoids the 243-small-write() syscall pattern that was hitting
     *    ~2.4 GB/s. With huge pages backing the tmpfs file, first-touch
     *    faults drop from 18M to 35K → first write is ~100 ms, the rest is
     *    memcpy bandwidth (~15 GB/s). */
    uint64_t max_n_pages = 0;
    for (uint32_t i = 0; i < n_filt; i++) {
        if (descs[i].num_pages > max_n_pages) max_n_pages = descs[i].num_pages;
    }
    uint8_t *ones_scratch = malloc((size_t)max_n_pages);
    if (!ones_scratch) {
        fprintf(stderr, "  ✗ coalesce: malloc ones_scratch\n");
        free(descs); free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
        goto out_free;
    }
    memset(ones_scratch, 1, (size_t)max_n_pages);  /* 1 = GPU-resident */

    printf("  [P3-B coalesce] writing %s  (v3, %u ranges, image=%.2f GB)\n",
           base_path, n_filt, image_size / (1024.0*1024.0*1024.0));

    /* Open the persistent output file. It should already exist and be
     * sized >= image_size from a prior ckpt_prealloc run. We do NOT
     * O_TRUNC (that would release all the warm tmpfs pages we paid to
     * fault in). We conditionally ftruncate only if the file size needs
     * to grow. If the file is larger than image_size, we just write our
     * image at the start and leave the tail untouched — readers use the
     * header.num_ranges + per-range data_offset to know where valid data
     * ends. */
    double tw0 = now_ms();
    int fd = -1;
    void *mp;

    if (g_no_image_write) {
        /* No-disk mode: anonymous mapping, discarded on munmap. Skip
         * open/ftruncate entirely. Same offset math + memcpy from here on. */
        mp = mmap(NULL, (size_t)image_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        fd = open(base_path, O_RDWR, 0644);
        if (fd < 0) {
            fprintf(stderr,
                    "  ✗ P3-B: open(%s): %s\n"
                    "  ✗ P3-B: the persistent output file does not exist.\n"
                    "  ✗ P3-B: run this BEFORE starting vLLM (or once per boot):\n"
                    "\n"
                    "      sudo ./ckpt_prealloc --gpu-size-gb 90 "
                    "--out %s --out-size-gb 90\n"
                    "\n",
                    base_path, strerror(errno), base_path);
            free(ones_scratch); free(descs);
            free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
            goto out_free;
        }
        struct stat out_st;
        if (fstat(fd, &out_st) < 0) {
            fprintf(stderr, "  ✗ P3-B: fstat output: %s\n", strerror(errno));
            close(fd);
            free(ones_scratch); free(descs);
            free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
            goto out_free;
        }
        if ((uint64_t)out_st.st_size < image_size) {
            /* Need to grow: the new tail pages will be fresh-fault (slower).
             * Keep it simple — grow in place. Warns the user. */
            printf("  [P3-B coalesce] output file %.2f GB < image %.2f GB, growing (fault cost)\n",
                   out_st.st_size / (1024.0*1024.0*1024.0),
                   image_size      / (1024.0*1024.0*1024.0));
            if (ftruncate(fd, (off_t)image_size) < 0) {
                fprintf(stderr, "  ✗ ftruncate(%llu): %s\n",
                        (unsigned long long)image_size, strerror(errno));
                close(fd);
                free(ones_scratch); free(descs);
                free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
                goto out_free;
            }
        }
        /* Map image_size bytes from offset 0. If the file is larger, the tail
         * is not part of our mapping — still physically allocated on tmpfs but
         * not touched by our writes. */
        mp = mmap(NULL, (size_t)image_size, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);
    }
    if (mp == MAP_FAILED) {
        fprintf(stderr, "  ✗ mmap(%llu): %s\n",
                (unsigned long long)image_size, strerror(errno));
        if (fd >= 0) close(fd);
        free(ones_scratch); free(descs);
        free(alloc_gpu_off); free(alloc_meta_idx); free(alloc_meta_count);
        goto out_free;
    }
    (void)madvise(mp, (size_t)image_size, MADV_HUGEPAGE);
    uint8_t *img = (uint8_t *)mp;

    /* File header */
    ckpt_v2_file_hdr_t fhdr = {
        .magic       = CKPT_V2_MAGIC,
        .version     = CKPT_V3_VERSION,
        .num_ranges  = n_filt,
        .page_size   = 4096,
        .total_bytes = total_payload,
    };
    memcpy(img, &fhdr, sizeof(fhdr));
    memcpy(img + sizeof(fhdr), descs, (size_t)n_filt * sizeof(*descs));

    /* Per-range data sections — parallel memcpys into disjoint regions of
     * the mmap'd output file. With warm tmpfs pages (from ckpt_prealloc),
     * we expect ~40-50 GB/s at 4-8 threads (A-1 warm memcpy rate from
     * bench_mem_write), vs ~11 GB/s single-threaded. Thread count is
     * controlled by CKPT_WRITER_THREADS env var (default 8). */
    int n_writers = 8;
    {
        const char *env = getenv("CKPT_WRITER_THREADS");
        if (env) n_writers = atoi(env);
        if (n_writers < 1)  n_writers = 1;
        if (n_writers > 16) n_writers = 16;
        if ((uint32_t)n_writers > n_filt) n_writers = (int)n_filt;
    }

    pthread_t          writer_tids[16];
    p3b_writer_ctx_t   writer_ctxs[16];
    /* Split n_filt ranges across n_writers as evenly as possible. */
    uint32_t base_per_writer = n_filt / (uint32_t)n_writers;
    uint32_t extra            = n_filt % (uint32_t)n_writers;
    uint32_t next_start       = 0;
    for (int w = 0; w < n_writers; w++) {
        uint32_t chunk = base_per_writer + (((uint32_t)w < extra) ? 1 : 0);
        writer_ctxs[w].img            = img;
        writer_ctxs[w].descs          = descs;
        writer_ctxs[w].big_gpu_buf    = big_gpu_buf;
        writer_ctxs[w].big_meta       = big_meta;
        writer_ctxs[w].alloc_gpu_off  = alloc_gpu_off;
        writer_ctxs[w].alloc_meta_idx = alloc_meta_idx;
        writer_ctxs[w].ones_scratch   = ones_scratch;
        writer_ctxs[w].range_start    = next_start;
        writer_ctxs[w].range_end      = next_start + chunk;
        next_start += chunk;
        pthread_create(&writer_tids[w], NULL, p3b_writer_thread, &writer_ctxs[w]);
    }
    for (int w = 0; w < n_writers; w++)
        pthread_join(writer_tids[w], NULL);
    printf("  [P3-B coalesce] parallel write with %d writer threads\n", n_writers);

    /* Flush to tmpfs (skip in --no-image-write — anon mapping has nothing
     * to sync) and release the mapping. */
    if (!g_no_image_write) {
        (void)msync(mp, (size_t)image_size, MS_ASYNC);
    }
    if (munmap(mp, (size_t)image_size) < 0) {
        fprintf(stderr, "  ✗ munmap: %s\n", strerror(errno));
    }
    if (fd >= 0) close(fd);

    double write_ms = now_ms() - tw0;
    if (write_ms_out) *write_ms_out = write_ms;
    double write_gbps = total_payload / (1024.0*1024.0*1024.0) / (write_ms / 1000.0);
    printf("  ✓ P3-B coalesced baseline written  %.1f ms  (%.2f GB/s)\n\n",
           write_ms, write_gbps);

    ret_code = 0;
    free(ones_scratch);
    free(descs);
    free(alloc_gpu_off);
    free(alloc_meta_idx);
    free(alloc_meta_count);

out_free:
    free(big_resmap);
    if (big_gpu_buf && big_gpu_buf != MAP_FAILED) {
        (void)munmap(big_gpu_buf, gpu_mapped_size);
    }
    if (gpu_fd >= 0) close(gpu_fd);
    free(big_cpu_buf);
    free(big_meta);
    free(allocs);
    return ret_code;
}

/* ====================================================================== */
/* Streaming baseline writer (encrypted v3 path) — multi-threaded writer    */
/*                                                                          */
/* Main capture thread reads each range via ioctl 115, atomically assigns   */
/* a file offset for that range's data sections, fills the corresponding    */
/* descriptor entry, and enqueues (cap, offset, desc_index) onto a bounded  */
/* work queue. A pool of N writer threads pulls items off the queue and     */
/* pwrites the 4 sections (resmap, cpu, gpu, crypto_meta) at the assigned   */
/* offsets. tmpfs and real disks handle concurrent pwrite() to disjoint     */
/* offsets natively — no synchronization needed between writers.             */
/*                                                                          */
/* Peak memory: ~(queue_cap + N) ranges in flight, typically ≤32 GB.        */
/* On-disk layout is unchanged: header + descs + range data.                */
/* ====================================================================== */

#define BASELINE_WRITER_DEFAULT 4
#define BASELINE_WRITER_MAX     16

typedef struct {
    range_capture_t *cap;
    uint64_t         file_offset;   /* where this cap's data sections start */
    NvU32            desc_index;
    int              slot_idx;      /* -1 = cap is self-owned; ≥0 = pool-backed */
} baseline_work_item_t;

typedef struct {
    baseline_work_item_t *slots;
    int                   capacity;
    int                   head;     /* next dequeue */
    int                   tail;     /* next enqueue */
    int                   count;
    int                   closed;   /* producer signals end-of-stream */
    pthread_mutex_t       lock;
    pthread_cond_t        not_empty;
    pthread_cond_t        not_full;
} baseline_wq_t;

static int baseline_wq_init(baseline_wq_t *q, int capacity)
{
    memset(q, 0, sizeof(*q));
    q->slots = calloc(capacity, sizeof(baseline_work_item_t));
    if (!q->slots) return -1;
    q->capacity = capacity;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
    return 0;
}

static void baseline_wq_destroy(baseline_wq_t *q)
{
    free(q->slots);
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

static void baseline_wq_push(baseline_wq_t *q, baseline_work_item_t item)
{
    pthread_mutex_lock(&q->lock);
    while (q->count == q->capacity)
        pthread_cond_wait(&q->not_full, &q->lock);
    q->slots[q->tail] = item;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

/* Returns 1 on dequeue, 0 on closed+empty. */
static int baseline_wq_pop(baseline_wq_t *q, baseline_work_item_t *out)
{
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->closed)
        pthread_cond_wait(&q->not_empty, &q->lock);
    if (q->count == 0 && q->closed) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    *out = q->slots[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

static void baseline_wq_close(baseline_wq_t *q)
{
    pthread_mutex_lock(&q->lock);
    q->closed = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

typedef struct {
    uint8_t              *map;          /* mmap'd file, shared among workers */
    baseline_wq_t        *wq;
    capture_pool_t       *pool;         /* NULL if legacy path; set for pooled */
    int                   worker_id;
    volatile int         *shared_err;
    /* Output — accumulated by this worker. */
    uint64_t              bytes_written;
} baseline_writer_ctx_t;

/*
 * mmap-based writer. Workers memcpy into disjoint regions of the shared
 * mapping — no inode lock, no pwrite syscalls, no page-cache path. On tmpfs
 * this is ~10× faster than pwrite for concurrent writes.
 */
static void *baseline_writer_thread(void *arg)
{
    baseline_writer_ctx_t *ctx = arg;
    baseline_work_item_t   item;

    while (baseline_wq_pop(ctx->wq, &item)) {
        if (*ctx->shared_err) {
            /* Someone else errored — drain + release without writing. */
            if (item.slot_idx >= 0 && ctx->pool)
                capture_pool_release(ctx->pool, item.slot_idx);
            else
                range_capture_free(item.cap);
            free(item.cap);
            continue;
        }

        range_capture_t *cap = item.cap;
        uint8_t *dst = ctx->map + item.file_offset;
        uint64_t meta_bytes = 0;
        if (cap->crypto_meta && cap->encrypted_page_count > 0)
            meta_bytes = cap->encrypted_page_count * sizeof(page_crypto_meta_t);

        /* resmap + cpu + gpu + crypto_meta — disjoint memcpys. No syscalls. */
        memcpy(dst, cap->resmap, cap->n_pages);
        dst += cap->n_pages;

        if (cap->cpu_bytes) {
            memcpy(dst, cap->cpu_buf, cap->cpu_bytes);
            dst += cap->cpu_bytes;
        }
        if (cap->gpu_bytes) {
            memcpy(dst, cap->gpu_buf, cap->gpu_bytes);
            dst += cap->gpu_bytes;
        }
        if (meta_bytes) {
            memcpy(dst, cap->crypto_meta, meta_bytes);
        }

        ctx->bytes_written += cap->n_pages + cap->cpu_bytes + cap->gpu_bytes + meta_bytes;

        /* Release the slot (or free self-owned buffers), then free the
         * small cap struct. */
        if (item.slot_idx >= 0 && ctx->pool)
            capture_pool_release(ctx->pool, item.slot_idx);
        else
            range_capture_free(cap);
        free(cap);
    }
    return NULL;
}

/* State handed back to the caller for the capture loop. */
typedef struct {
    int                    fd;
    uint8_t               *map;               /* mmap'd output file */
    uint64_t               map_size;          /* worst-case file size (used for munmap) */
    NvU32                  num_ranges;
    uint32_t               version;
    ckpt_v2_range_desc_t  *descs;             /* filled inline as caps are assigned offsets */
    uint64_t               cur_off;            /* running offset of next range's data */
    uint64_t               total_data_bytes;

    /* Format selector: 0 = ckpt_v2 baseline, 1 = ckpt_inc_delta. The descs
     * layout is byte-identical between v2_range_desc_t and inc_delta_block_t,
     * so the writer pool is format-agnostic; only the header bytes at
     * offset 0 differ (24 vs 32 bytes) and so does the reservation amount. */
    int                    is_delta;
    NvU64                  delta_alloc_start_va;   /* gate->user_va  (delta) */
    NvU64                  delta_alloc_size;       /* gate->user_size (delta) */
    NvU64                  delta_num_dirty_64k;    /* raw count from GET_DIRTY_PAGES */

    /* Writer pool. */
    unsigned               n_writers;
    pthread_t              writer_tids[BASELINE_WRITER_MAX];
    baseline_writer_ctx_t  writer_ctx[BASELINE_WRITER_MAX];
    baseline_wq_t          wq;
    volatile int           shared_err;

    /* Timing points. */
    double                 t_start;       /* stream_start entry                    */
    double                 t_capture_end; /* wq_close (producer done)               */
    double                 t_drain_end;   /* after all writer threads joined        */
    double                 t_finalize_end;/* after msync + ftruncate trim + close   */
} baseline_stream_t;

static int baseline_stream_resolve_workers(void)
{
    const char *env = getenv("CKPT_WRITER_THREADS");
    int n = env ? atoi(env) : BASELINE_WRITER_DEFAULT;
    if (n < 1) n = 1;
    if (n > BASELINE_WRITER_MAX) n = BASELINE_WRITER_MAX;
    return n;
}

/*
 * Compute worst-case file size:
 *   header
 * + descriptor table
 * + per range: n_pages (resmap) + length (cpu+gpu combined max) + n_pages * 41 (meta max)
 *
 * n_pages[i] is derived from range_sizes[i] / PAGE_SIZE.
 *
 * Headers differ: ckpt_v2_file_hdr_t is 24 B, ckpt_inc_delta_hdr_t is 32 B.
 * Descriptors are byte-identical (64 B).
 */
static uint64_t baseline_worst_case_size(NvU32 num_ranges, const NvU64 *range_sizes,
                                         int is_delta)
{
    uint64_t hdr_bytes  = is_delta ? sizeof(ckpt_inc_delta_hdr_t)
                                   : sizeof(ckpt_v2_file_hdr_t);
    uint64_t desc_bytes = is_delta ? sizeof(ckpt_inc_delta_block_t)
                                   : sizeof(ckpt_v2_range_desc_t);
    uint64_t total = hdr_bytes + (uint64_t)num_ranges * desc_bytes;
    for (NvU32 i = 0; i < num_ranges; i++) {
        uint64_t n_pages = range_sizes[i] / 4096;
        total += n_pages                                      /* resmap */
               + range_sizes[i]                                /* cpu+gpu combined */
               + n_pages * sizeof(page_crypto_meta_t);         /* worst-case meta */
    }
    return total;
}

static int baseline_stream_start(const char *path, NvU32 num_ranges,
                                 const NvU64 *range_sizes,
                                 uint32_t version,
                                 capture_pool_t *pool,
                                 int      is_delta,
                                 NvU64    delta_alloc_start_va,
                                 NvU64    delta_alloc_size,
                                 NvU64    delta_num_dirty_64k,
                                 baseline_stream_t *s)
{
    memset(s, 0, sizeof(*s));
    s->t_start = now_ms();
    s->is_delta             = is_delta;
    s->delta_alloc_start_va = delta_alloc_start_va;
    s->delta_alloc_size     = delta_alloc_size;
    s->delta_num_dirty_64k  = delta_num_dirty_64k;

    /* Open without O_TRUNC so we keep any pre-warmed tmpfs pages from a
     * prior ckpt_prealloc run. Only ftruncate UP if the existing file is
     * smaller than what we need; never truncate down (releases warm pages).
     * Mode 0666 so that ckpt_core (usually root) and vLLM's libckpt_vllm.c
     * (regular user) can both write to the same output file. */
    s->fd = open(path, O_RDWR | O_CREAT, 0666);
    if (s->fd < 0) {
        fprintf(stderr, "  ✗ open(%s): %s\n", path, strerror(errno));
        return -1;
    }
    (void)fchmod(s->fd, 0666);

    uint64_t hdr_bytes  = is_delta ? sizeof(ckpt_inc_delta_hdr_t)
                                   : sizeof(ckpt_v2_file_hdr_t);
    uint64_t desc_bytes = is_delta ? sizeof(ckpt_inc_delta_block_t)
                                   : sizeof(ckpt_v2_range_desc_t);
    uint64_t reserve = hdr_bytes + (uint64_t)num_ranges * desc_bytes;

    /* Worst-case file size — cpu/gpu split isn't known until after capture,
     * so we reserve the max (length) for each range's data section. tmpfs
     * doesn't consume RAM for untouched pages, so this is free. We ftruncate
     * down to the actual size in baseline_stream_join. */
    uint64_t worst_size = baseline_worst_case_size(num_ranges, range_sizes, is_delta);

    struct stat stream_st;
    if (fstat(s->fd, &stream_st) < 0) {
        fprintf(stderr, "  ✗ fstat(%s): %s\n", path, strerror(errno));
        close(s->fd); return -1;
    }
    if ((uint64_t)stream_st.st_size < worst_size) {
        if (ftruncate(s->fd, (off_t)worst_size) < 0) {
            fprintf(stderr, "  ✗ ftruncate(%llu): %s\n",
                    (unsigned long long)worst_size, strerror(errno));
            close(s->fd); return -1;
        }
    }
    /* If the existing file is already >= worst_size, reuse — its pre-warmed
     * tmpfs pages save the first-touch cost we'd otherwise pay. */

    /* Map the whole file — workers memcpy into disjoint regions. */
    void *mp = mmap(NULL, worst_size, PROT_READ | PROT_WRITE, MAP_SHARED, s->fd, 0);
    if (mp == MAP_FAILED) {
        fprintf(stderr, "  ✗ mmap(%llu): %s\n",
                (unsigned long long)worst_size, strerror(errno));
        close(s->fd); return -1;
    }
    s->map      = (uint8_t *)mp;
    s->map_size = worst_size;

    /* Ask the kernel to use 2 MB THP pages for this mapping. With tmpfs
     * mounted huge=advise and global shmem_enabled=advise, this hints the
     * allocator to satisfy first-touch faults with 2 MB pages instead of
     * 4 KB — 512× fewer faults, dramatically reducing per-page alloc cost
     * under CC memory. Non-fatal if it fails. */
    if (madvise(s->map, worst_size, MADV_HUGEPAGE) < 0) {
        fprintf(stderr, "  [stream] MADV_HUGEPAGE failed: %s (continuing)\n",
                strerror(errno));
    }

    s->descs = calloc(num_ranges, sizeof(ckpt_v2_range_desc_t));
    if (!s->descs) {
        fprintf(stderr, "  ✗ calloc descs\n");
        munmap(s->map, s->map_size); close(s->fd); return -1;
    }

    s->num_ranges = num_ranges;
    s->version    = version;
    s->cur_off    = reserve;

    s->n_writers = (unsigned)baseline_stream_resolve_workers();
    if (baseline_wq_init(&s->wq, (int)(s->n_writers * 2)) < 0) {
        fprintf(stderr, "  ✗ wq_init\n");
        free(s->descs); munmap(s->map, s->map_size); close(s->fd); return -1;
    }

    printf("  [stream] writers=%u  queue=%d  mmap=%.2f GB  path=%s\n",
           s->n_writers, s->wq.capacity,
           worst_size / (1024.0*1024.0*1024.0), path);

    for (unsigned i = 0; i < s->n_writers; i++) {
        s->writer_ctx[i].map        = s->map;
        s->writer_ctx[i].wq         = &s->wq;
        s->writer_ctx[i].pool       = pool;    /* may be NULL (legacy path) */
        s->writer_ctx[i].worker_id  = (int)i;
        s->writer_ctx[i].shared_err = &s->shared_err;
        s->writer_ctx[i].bytes_written = 0;
        if (pthread_create(&s->writer_tids[i], NULL,
                           baseline_writer_thread, &s->writer_ctx[i]) != 0) {
            fprintf(stderr, "  ✗ pthread_create writer[%u]: %s\n", i, strerror(errno));
            /* Close queue so already-running workers exit. */
            baseline_wq_close(&s->wq);
            for (unsigned j = 0; j < i; j++)
                pthread_join(s->writer_tids[j], NULL);
            baseline_wq_destroy(&s->wq);
            free(s->descs); munmap(s->map, s->map_size); close(s->fd);
            return -1;
        }
    }
    return 0;
}

/*
 * Called by the capture thread after each successful capture_range_encrypted.
 * Assigns the file offset, fills descs[i], and hands off to a writer thread.
 * Does NOT free the cap — the writer thread is now the owner and will
 * either free the buffers (slot_idx < 0) or release the pool slot.
 */
static void baseline_stream_enqueue(baseline_stream_t *s, NvU32 i,
                                    range_capture_t *cap, int slot_idx)
{
    uint64_t meta_bytes = 0;
    if (cap->crypto_meta && cap->encrypted_page_count > 0)
        meta_bytes = cap->encrypted_page_count * sizeof(page_crypto_meta_t);

    s->descs[i].base_va           = cap->base;
    s->descs[i].length            = cap->length;
    s->descs[i].num_pages         = cap->n_pages;
    s->descs[i].data_offset       = s->cur_off;
    s->descs[i].resmap_size       = cap->n_pages;
    s->descs[i].cpu_bytes         = cap->cpu_bytes;
    s->descs[i].gpu_bytes         = cap->gpu_bytes;
    s->descs[i].crypto_meta_bytes = meta_bytes;

    baseline_work_item_t item = {
        .cap         = cap,
        .file_offset = s->cur_off,
        .desc_index  = i,
        .slot_idx    = slot_idx,
    };

    s->cur_off          += cap->n_pages + cap->cpu_bytes + cap->gpu_bytes + meta_bytes;
    s->total_data_bytes += cap->length;

    baseline_wq_push(&s->wq, item);
}

/*
 * Close the queue, wait for all writers to drain, fill descriptor table and
 * header, trim the file to actual size, close fd. Returns 0 on success.
 *
 * Emits three timing values to write_ms_out[0..2]:
 *   [0] capture wall     (stream_start → wq_close, main thread)
 *   [1] drain wall        (wq_close → all writers exit)
 *   [2] total pipeline wall (stream_start → finalize complete)
 */
static int baseline_stream_join(baseline_stream_t *s, double *timings_out /*[3]*/)
{
    /* Capture phase ends when the main thread closes the queue. */
    baseline_wq_close(&s->wq);
    s->t_capture_end = now_ms();

    uint64_t aggregate_bytes = 0;
    for (unsigned i = 0; i < s->n_writers; i++) {
        pthread_join(s->writer_tids[i], NULL);
        aggregate_bytes += s->writer_ctx[i].bytes_written;
    }
    s->t_drain_end = now_ms();

    int err = s->shared_err;

    /* Write descriptor table + header directly into the mmap — workers are
     * done so there's no concurrent access. v2_range_desc_t and
     * inc_delta_block_t are byte-identical, so the same descs array works
     * for both formats; only the header bytes (and offset) differ. */
    if (!err) {
        if (s->is_delta) {
            memcpy(s->map + sizeof(ckpt_inc_delta_hdr_t), s->descs,
                   (size_t)s->num_ranges * sizeof(ckpt_inc_delta_block_t));
            ckpt_inc_delta_hdr_t dhdr = {
                .magic               = CKPT_INC_DELTA_MAGIC,
                .version             = CKPT_INC_DELTA_VER,
                .num_blocks          = s->num_ranges,
                .page_size           = 4096,
                .alloc_start_va      = s->delta_alloc_start_va,
                .alloc_size          = s->delta_alloc_size,
                .num_dirty_pages_64k = s->delta_num_dirty_64k,
            };
            memcpy(s->map, &dhdr, sizeof(dhdr));
        } else {
            memcpy(s->map + sizeof(ckpt_v2_file_hdr_t), s->descs,
                   (size_t)s->num_ranges * sizeof(ckpt_v2_range_desc_t));
            ckpt_v2_file_hdr_t fhdr = {
                .magic       = CKPT_V2_MAGIC,
                .version     = s->version,
                .num_ranges  = s->num_ranges,
                .page_size   = 4096,
                .total_bytes = s->total_data_bytes,
            };
            memcpy(s->map, &fhdr, sizeof(fhdr));
        }
    }

    /* Flush dirty pages to the tmpfs backing store (cheap on tmpfs — just
     * inode metadata update). We only sync the range [0..cur_off), not the
     * full worst-case mapping, to avoid touching untouched pages. */
    if (!err) {
        if (msync(s->map, s->cur_off, MS_SYNC) < 0) {
            fprintf(stderr, "  ✗ msync: %s\n", strerror(errno));
            err = 1;
        }
    }

    /* Unmap the worst-case region. */
    if (munmap(s->map, s->map_size) < 0) {
        fprintf(stderr, "  ✗ munmap: %s\n", strerror(errno));
        if (!err) err = 1;
    }
    s->map = NULL;

    /* Do NOT ftruncate the file down to s->cur_off here.
     *
     * Trimming would release the warm tmpfs pages we want to reuse on the
     * next ckpt_core invocation. The file may end up slightly larger than
     * the actual used data (worst-case estimate vs actual cpu/gpu split),
     * but the on-disk format is self-describing — the header has
     * num_ranges and each desc has data_offset, so restore_agent reads
     * exactly the right bytes regardless of trailing slack. */

    close(s->fd);
    s->t_finalize_end = now_ms();

    double capture_wall = s->t_capture_end   - s->t_start;
    double drain_wall   = s->t_drain_end     - s->t_capture_end;
    double pipeline_wall= s->t_finalize_end  - s->t_start;

    if (timings_out) {
        timings_out[0] = capture_wall;
        timings_out[1] = drain_wall;
        timings_out[2] = pipeline_wall;
    }

    if (!err) {
        const char *label = s->is_delta ? "Delta" : "Baseline";
        printf("  ✓ %s streamed\n"
               "      capture wall : %8.1f ms  (main thread producing into queue)\n"
               "      drain   wall : %8.1f ms  (writers finishing after capture done)\n"
               "      pipeline tot : %8.1f ms  (%.2f GB / %.1f GB/s)\n"
               "      %u range(s), %u writer(s), used %.2f GB of %.2f GB mapping\n\n",
               label,
               capture_wall, drain_wall, pipeline_wall,
               aggregate_bytes / (1024.0*1024.0*1024.0),
               pipeline_wall > 0
                   ? aggregate_bytes / (1024.0*1024.0*1024.0) / (pipeline_wall/1000.0) : 0.0,
               s->num_ranges, s->n_writers,
               s->cur_off / (1024.0*1024.0*1024.0),
               s->map_size / (1024.0*1024.0*1024.0));
    }

    free(s->descs);
    baseline_wq_destroy(&s->wq);
    return err ? -1 : 0;
}

/* ------------------------------------------------------------------ */
/* write_baseline: ckpt_v2 format (same as ckpt_resident_agent)       */
/* ------------------------------------------------------------------ */
/* version: CKPT_V2_VERSION (plaintext or k3), CKPT_V3_VERSION (encrypted/k1) */
static int write_baseline(const char *path, range_capture_t *caps,
                          NvU32 num_ranges, uint32_t version, double *write_ms_out)
{
    uint64_t hdr_off = sizeof(ckpt_v2_file_hdr_t)
                     + (uint64_t)num_ranges * sizeof(ckpt_v2_range_desc_t);

    ckpt_v2_range_desc_t *descs =
        malloc(num_ranges * sizeof(ckpt_v2_range_desc_t));
    if (!descs) { fprintf(stderr, "  ✗ malloc descs\n"); return -1; }

    /* Compute total image size and fill descriptors */
    uint64_t cur_off = hdr_off, total_bytes = 0, image_size;
    for (NvU32 i = 0; i < num_ranges; i++) {
        uint64_t meta_bytes = 0;
        if (caps[i].crypto_meta && caps[i].encrypted_page_count > 0)
            meta_bytes = caps[i].encrypted_page_count * sizeof(page_crypto_meta_t);

        descs[i].base_va          = caps[i].base;
        descs[i].length           = caps[i].length;
        descs[i].num_pages        = caps[i].n_pages;
        descs[i].data_offset      = cur_off;
        descs[i].resmap_size      = caps[i].n_pages;
        descs[i].cpu_bytes        = caps[i].cpu_bytes;
        descs[i].gpu_bytes        = caps[i].gpu_bytes;
        descs[i].crypto_meta_bytes = meta_bytes;
        cur_off     += caps[i].n_pages + caps[i].cpu_bytes + caps[i].gpu_bytes + meta_bytes;
        total_bytes += caps[i].length;
    }
    image_size = cur_off;

    ckpt_v2_file_hdr_t fhdr = {
        .magic       = CKPT_V2_MAGIC,
        .version     = version,
        .num_ranges  = num_ranges,
        .page_size   = 4096,
        .total_bytes = total_bytes,
    };

    printf("  Writing baseline → %s  (version=%u, %u range(s), %.2f MB, image=%.2f MB) ...\n",
           path, fhdr.version, num_ranges, total_bytes / (1024.0 * 1024.0),
           image_size / (1024.0 * 1024.0));

    /* Assemble full image into one aligned buffer for O_DIRECT */
    #define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))
    uint64_t aligned_size = ALIGN_UP(image_size, 4096);
    uint8_t *img_buf = NULL;
    if (posix_memalign((void **)&img_buf, 4096, aligned_size) != 0) {
        fprintf(stderr, "  ✗ posix_memalign(%llu): %s\n",
                (unsigned long long)aligned_size, strerror(errno));
        free(descs); return -1;
    }
    memset(img_buf + image_size, 0, aligned_size - image_size); /* zero padding */

    /* Pack header + descriptors */
    uint64_t pos = 0;
    memcpy(img_buf + pos, &fhdr, sizeof(fhdr));
    pos += sizeof(fhdr);
    memcpy(img_buf + pos, descs, num_ranges * sizeof(ckpt_v2_range_desc_t));
    pos += num_ranges * sizeof(ckpt_v2_range_desc_t);

    /* Pack range data */
    for (NvU32 i = 0; i < num_ranges; i++) {
        memcpy(img_buf + pos, caps[i].resmap, caps[i].n_pages);
        pos += caps[i].n_pages;
        memcpy(img_buf + pos, caps[i].cpu_buf, caps[i].cpu_bytes);
        pos += caps[i].cpu_bytes;
        memcpy(img_buf + pos, caps[i].gpu_buf, caps[i].gpu_bytes);
        pos += caps[i].gpu_bytes;
        if (caps[i].crypto_meta && caps[i].encrypted_page_count > 0) {
            size_t meta_bytes = caps[i].encrypted_page_count * sizeof(page_crypto_meta_t);
            memcpy(img_buf + pos, caps[i].crypto_meta, meta_bytes);
            pos += meta_bytes;
        }
    }

    /* Open WITHOUT O_TRUNC so warm tmpfs pages from a prior ckpt_prealloc
     * run survive (matches the v3 baseline path's pattern at line 1015).
     * If the file is larger than aligned_size, the trailing bytes are
     * leftover from a prior run — harmless, since restore reads only
     * what the header descriptors point to (descs[i].data_offset + length).
     * Use ftruncate to grow if needed. */
    double tw0 = now_ms();
    int err = 0;
    int fd = open(path, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) {
        /* Fallback: O_DIRECT not supported (e.g. tmpfs) — use normal write */
        fd = open(path, O_WRONLY | O_CREAT, 0644);
        if (fd < 0) {
            fprintf(stderr, "  ✗ open(%s): %s\n", path, strerror(errno));
            free(img_buf); free(descs); return -1;
        }
    }
    {
        struct stat out_st;
        if (fstat(fd, &out_st) == 0 && (uint64_t)out_st.st_size < aligned_size) {
            if (ftruncate(fd, (off_t)aligned_size) < 0) {
                fprintf(stderr, "  ✗ ftruncate(%s, %llu): %s\n", path,
                        (unsigned long long)aligned_size, strerror(errno));
                close(fd);
                free(img_buf); free(descs); return -1;
            }
        }
    }

    /* Write in large chunks */
    uint64_t written = 0;
    while (written < aligned_size) {
        uint64_t chunk = aligned_size - written;
        if (chunk > (256ULL * 1024 * 1024)) chunk = 256ULL * 1024 * 1024; /* 256MB max per write */
        ssize_t ret = write(fd, img_buf + written, chunk);
        if (ret <= 0) {
            fprintf(stderr, "  ✗ write: %s (written=%llu/%llu)\n",
                    strerror(errno), (unsigned long long)written,
                    (unsigned long long)aligned_size);
            err = 1; break;
        }
        written += ret;
    }

    /* Truncate to exact size (O_DIRECT may have written padding) */
    if (!err) ftruncate(fd, image_size);
    close(fd);

    if (!err) {
        double wms = now_ms() - tw0;
        if (write_ms_out) *write_ms_out = wms;
        printf("  ✓ Baseline written  %.1f ms  (%.1f MB/s)\n\n", wms,
               wms > 0 ? total_bytes / (1024.0*1024.0) / (wms/1000.0) : 0.0);
    }
    free(img_buf);
    free(descs);
    return err ? -1 : 0;
    #undef ALIGN_UP
}

/* ------------------------------------------------------------------ */
/* write_delta: ckpt_inc_delta format                                 */
/* ------------------------------------------------------------------ */
static int write_delta(const char *path,
                       range_capture_t *caps, NvU32 num_blocks,
                       NvU64 alloc_start_va, NvU64 alloc_size,
                       NvU64 num_dirty_pages_64k, double *write_ms_out)
{
    uint64_t base_off = sizeof(ckpt_inc_delta_hdr_t)
                      + (uint64_t)num_blocks * sizeof(ckpt_inc_delta_block_t);

    ckpt_inc_delta_block_t *blocks =
        malloc((num_blocks ? num_blocks : 1) * sizeof(ckpt_inc_delta_block_t));
    if (!blocks) { fprintf(stderr, "  ✗ malloc blocks\n"); return -1; }

    uint64_t cur_off = base_off, total_delta = 0;
    for (NvU32 i = 0; i < num_blocks; i++) {
        uint64_t meta_bytes = 0;
        if (caps[i].crypto_meta && caps[i].encrypted_page_count > 0)
            meta_bytes = caps[i].encrypted_page_count * sizeof(page_crypto_meta_t);

        blocks[i].base_va          = caps[i].base;
        blocks[i].length           = caps[i].length;
        blocks[i].num_pages        = caps[i].n_pages;
        blocks[i].data_offset      = cur_off;
        blocks[i].resmap_size      = caps[i].n_pages;
        blocks[i].cpu_bytes        = caps[i].cpu_bytes;
        blocks[i].gpu_bytes        = caps[i].gpu_bytes;
        blocks[i].crypto_meta_bytes = meta_bytes;
        cur_off     += caps[i].n_pages + caps[i].cpu_bytes + caps[i].gpu_bytes + meta_bytes;
        total_delta += caps[i].length;
    }

    ckpt_inc_delta_hdr_t dhdr = {
        .magic               = CKPT_INC_DELTA_MAGIC,
        .version             = CKPT_INC_DELTA_VER,
        .num_blocks          = num_blocks,
        .page_size           = 4096,
        .alloc_start_va      = alloc_start_va,
        .alloc_size          = alloc_size,
        .num_dirty_pages_64k = num_dirty_pages_64k,
    };

    printf("  Writing delta → %s  (%u block(s), %.2f MB) ...\n",
           path, num_blocks, total_delta / (1024.0 * 1024.0));

    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "  ✗ fopen(%s): %s\n", path, strerror(errno));
        free(blocks); return -1;
    }

    double tw0 = now_ms();
    int err = 0;

    if (fwrite(&dhdr,   sizeof(dhdr),  1, out) != 1 ||
        fwrite(blocks, sizeof(ckpt_inc_delta_block_t), num_blocks, out) != num_blocks) {
        err = 1; goto done;
    }
    for (NvU32 i = 0; i < num_blocks; i++) {
        if (fwrite(caps[i].resmap,  1, caps[i].n_pages,   out) != caps[i].n_pages   ||
            fwrite(caps[i].cpu_buf, 1, caps[i].cpu_bytes, out) != caps[i].cpu_bytes ||
            fwrite(caps[i].gpu_buf, 1, caps[i].gpu_bytes, out) != caps[i].gpu_bytes) {
            err = 1; goto done;
        }
        if (caps[i].crypto_meta && caps[i].encrypted_page_count > 0) {
            size_t meta_bytes = caps[i].encrypted_page_count * sizeof(page_crypto_meta_t);
            if (fwrite(caps[i].crypto_meta, 1, meta_bytes, out) != meta_bytes) {
                err = 1; goto done;
            }
        }
    }

done:
    if (err) fprintf(stderr, "  ✗ fwrite: %s\n", strerror(errno));
    else {
        double wms = now_ms() - tw0;
        if (write_ms_out) *write_ms_out = wms;
        printf("  ✓ Delta written  %.1f ms  (%.1f MB/s)\n\n", wms,
               wms > 0 ? total_delta / (1024.0*1024.0) / (wms/1000.0) : 0.0);
    }
    fclose(out);
    free(blocks);
    return err ? -1 : 0;
}

/* ------------------------------------------------------------------ */
/* Gate helpers                                                        */
/* ------------------------------------------------------------------ */
static ckpt_gate_t *gate_open(const char *path)
{
    int fd = -1;
    for (int i = 0; i < 600; i++) {
        fd = open(path, O_RDWR);
        if (fd >= 0) break;
        if (errno != ENOENT) {
            fprintf(stderr, "gate_open: %s: %s\n", path, strerror(errno));
            exit(1);
        }
        usleep(100000);
    }
    if (fd < 0) {
        fprintf(stderr, "ERROR: gate file %s not found after 60s\n", path);
        exit(1);
    }

    ckpt_gate_t *g = (ckpt_gate_t *)mmap(NULL, GATE_FILE_SIZE,
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd, 0);
    close(fd);
    if (g == MAP_FAILED) { perror("gate mmap"); exit(1); }

    /* Wait for either INIT (first ckpt_core run against a fresh app) or
     * RESUME (a prior ckpt_core against this same app already finished —
     * the app resumed and is running, ready for another checkpoint). */
    for (int i = 0; i < 600; i++) {
        if (g->phase == GATE_PHASE_INIT || g->phase == GATE_PHASE_RESUME)
            break;
        usleep(100000);
    }
    if (g->phase != GATE_PHASE_INIT && g->phase != GATE_PHASE_RESUME) {
        fprintf(stderr, "ERROR: gate never ready (phase=%u) — stale file?\n",
                g->phase);
        exit(1);
    }
    if (g->phase == GATE_PHASE_RESUME) {
        printf("  (gate in RESUME state — this is a follow-up checkpoint run)\n");
        /* Reset the request fields so we start clean. Don't touch
         * app_pid/user_va/user_size which the app set once at INIT. */
        g->ckpt_req      = 0;
        g->baseline_req  = 0;
        g->baseline_done = 0;
        g->baseline_err  = 0;
        g->delta_req     = 0;
        g->delta_done    = 0;
        g->delta_err     = 0;
        __sync_synchronize();
    }
    return g;
}

/* ------------------------------------------------------------------ */
/* qsort comparator for NvU64                                         */
/* ------------------------------------------------------------------ */
static int cmp_u64(const void *a, const void *b)
{
    NvU64 va = *(const NvU64 *)a;
    NvU64 vb = *(const NvU64 *)b;
    return (va > vb) - (va < vb);
}

/* ------------------------------------------------------------------ */
/* snapshot_file: copy src to <dst_dir>/<basename(src)> while app is   */
/* frozen.  Returns 0 on success, -1 if src doesn't exist (warning),  */
/* or -2 on write error.                                               */
/* ------------------------------------------------------------------ */
static int snapshot_file(const char *src, const char *dst_dir)
{
    FILE *in = fopen(src, "rb");
    if (!in) {
        printf("    WARN: %s not found — skipping snapshot\n", src);
        return -1;
    }

    /*
     * Write to a snapshot-prefixed name so we don't collide with the
     * app's live file (which may be owned by a different uid on /tmp).
     */
    const char *base = strrchr(src, '/');
    base = base ? base + 1 : src;

    char dst[PATH_MAX_LEN];
    snprintf(dst, sizeof(dst), "%s/snap_%s", dst_dir, base);

    FILE *out = fopen(dst, "wb");
    if (!out) {
        fprintf(stderr, "    ERROR: cannot create %s: %s\n", dst, strerror(errno));
        fclose(in);
        return -2;
    }

    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, n, out) != n) {
            fprintf(stderr, "    ERROR: write to %s: %s\n", dst, strerror(errno));
            fclose(in); fclose(out);
            return -2;
        }
    }

    fclose(in);
    fclose(out);
    printf("    ✓ %s → %s\n", src, dst);
    return 0;
}

/* ------------------------------------------------------------------ */
/* encrypt_captures_batch: encrypt all range_capture_t buffers in one  */
/* pass.  Gathers all CPU and GPU pages into a single contiguous       */
/* buffer, encrypts once (multi-threaded), then scatters back.         */
/* This avoids per-capture thread spawn overhead and allows the        */
/* thread pool to operate on the full dataset.                         */
/* Returns 0 on success, updates *encrypt_ms_out with elapsed time.    */
/* ------------------------------------------------------------------ */
static int encrypt_captures_batch(range_capture_t *caps, NvU32 n_caps,
                                  const uint8_t k3_key[CKPT_CRYPTO_KEY_SIZE],
                                  uint64_t *iv_counter,
                                  int num_threads,
                                  size_t chunk_size,
                                  double *encrypt_ms_out)
{
    if (chunk_size == 0) chunk_size = CKPT_CRYPTO_PAGE_SIZE;

    /* Count total pages across all captures */
    uint64_t total_bytes = 0;
    for (NvU32 i = 0; i < n_caps; i++)
        total_bytes += caps[i].cpu_bytes + caps[i].gpu_bytes;

    if (total_bytes == 0) {
        if (encrypt_ms_out) *encrypt_ms_out = 0.0;
        return 0;
    }

    uint32_t total_pages = (uint32_t)(total_bytes / chunk_size);
    if (total_pages == 0) {
        if (encrypt_ms_out) *encrypt_ms_out = 0.0;
        return 0;
    }

    /* Gather all page data into one contiguous buffer */
    uint8_t *gathered = malloc(total_bytes);
    ckpt_page_meta_t *meta = calloc(total_pages, sizeof(ckpt_page_meta_t));
    if (!gathered || !meta) {
        free(gathered); free(meta);
        return -1;
    }

    uint64_t off = 0;
    for (NvU32 i = 0; i < n_caps; i++) {
        if (caps[i].cpu_bytes > 0) {
            memcpy(gathered + off, caps[i].cpu_buf, caps[i].cpu_bytes);
            off += caps[i].cpu_bytes;
        }
        if (caps[i].gpu_bytes > 0) {
            memcpy(gathered + off, caps[i].gpu_buf, caps[i].gpu_bytes);
            off += caps[i].gpu_bytes;
        }
    }

    /* Encrypt all pages in one multi-threaded call */
    double t0 = now_ms();
    int ret = ckpt_crypto_encrypt_pages(k3_key, gathered, gathered,
                                         meta, total_pages,
                                         chunk_size,
                                         *iv_counter, num_threads);
    double elapsed = now_ms() - t0;

    if (ret != 0) {
        free(gathered); free(meta);
        return -1;
    }

    *iv_counter += total_pages;

    /* Scatter encrypted data + metadata back into capture buffers */
    off = 0;
    uint32_t meta_off = 0;
    for (NvU32 i = 0; i < n_caps; i++) {
        uint32_t cap_pages = (uint32_t)((caps[i].cpu_bytes + caps[i].gpu_bytes)
                                        / chunk_size);

        if (caps[i].cpu_bytes > 0) {
            memcpy(caps[i].cpu_buf, gathered + off, caps[i].cpu_bytes);
            off += caps[i].cpu_bytes;
        }
        if (caps[i].gpu_bytes > 0) {
            memcpy(caps[i].gpu_buf, gathered + off, caps[i].gpu_bytes);
            off += caps[i].gpu_bytes;
        }

        /* Store per-page k3 crypto metadata into the capture */
        if (cap_pages > 0) {
            caps[i].crypto_meta = malloc(cap_pages * sizeof(page_crypto_meta_t));
            if (caps[i].crypto_meta) {
                for (uint32_t p = 0; p < cap_pages; p++) {
                    memcpy(caps[i].crypto_meta[p].iv,
                           meta[meta_off + p].iv, CKPT_CRYPTO_IV_SIZE);
                    caps[i].crypto_meta[p].iv_fresh = 0;
                    memcpy(caps[i].crypto_meta[p].auth_tag,
                           meta[meta_off + p].auth_tag, CKPT_CRYPTO_AUTH_TAG_SIZE);
                    caps[i].crypto_meta[p].size = (uint32_t)chunk_size;
                    caps[i].crypto_meta[p].key_version = 0; /* k3 — key stored externally */
                }
                caps[i].encrypted_page_count = cap_pages;
            }
        }
        meta_off += cap_pages;
    }

    free(gathered);
    free(meta);

    if (encrypt_ms_out)
        *encrypt_ms_out = elapsed;

    return 0;
}

/* ------------------------------------------------------------------ */
/* encrypt_cpu_pages_batch: encrypt ONLY the CPU-resident pages in     */
/* the captures with k3.  Used by --encrypted mode where GPU pages     */
/* are already CE-encrypted with k1, but CPU pages need k3 encryption  */
/* for data-at-rest security.                                          */
/* ------------------------------------------------------------------ */
static int encrypt_cpu_pages_batch(range_capture_t *caps, NvU32 n_caps,
                                   const uint8_t k3_key[CKPT_CRYPTO_KEY_SIZE],
                                   uint64_t *iv_counter,
                                   int num_threads,
                                   double *encrypt_ms_out)
{
    uint64_t total_cpu_bytes = 0;
    for (NvU32 i = 0; i < n_caps; i++)
        total_cpu_bytes += caps[i].cpu_bytes;

    if (total_cpu_bytes == 0) {
        if (encrypt_ms_out) *encrypt_ms_out = 0.0;
        return 0;
    }

    uint32_t total_pages = (uint32_t)(total_cpu_bytes / CKPT_CRYPTO_PAGE_SIZE);
    if (total_pages == 0) {
        if (encrypt_ms_out) *encrypt_ms_out = 0.0;
        return 0;
    }

    uint8_t *gathered = malloc(total_cpu_bytes);
    ckpt_page_meta_t *meta = calloc(total_pages, sizeof(ckpt_page_meta_t));
    if (!gathered || !meta) {
        free(gathered); free(meta);
        return -1;
    }

    /* Gather CPU pages only */
    uint64_t off = 0;
    for (NvU32 i = 0; i < n_caps; i++) {
        if (caps[i].cpu_bytes > 0) {
            memcpy(gathered + off, caps[i].cpu_buf, caps[i].cpu_bytes);
            off += caps[i].cpu_bytes;
        }
    }

    double t0 = now_ms();
    int ret = ckpt_crypto_encrypt_pages(k3_key, gathered, gathered,
                                         meta, total_pages,
                                         CKPT_CRYPTO_PAGE_SIZE,
                                         *iv_counter, num_threads);
    double elapsed = now_ms() - t0;

    if (ret != 0) {
        free(gathered); free(meta);
        return -1;
    }

    *iv_counter += total_pages;

    /* Scatter encrypted CPU data back + store metadata */
    off = 0;
    uint32_t meta_off = 0;
    for (NvU32 i = 0; i < n_caps; i++) {
        if (caps[i].cpu_bytes > 0) {
            uint32_t cpu_pages = (uint32_t)(caps[i].cpu_bytes / CKPT_CRYPTO_PAGE_SIZE);

            memcpy(caps[i].cpu_buf, gathered + off, caps[i].cpu_bytes);
            off += caps[i].cpu_bytes;

            /* Append CPU page metadata to existing crypto_meta array.
             * For --encrypted mode, crypto_meta already has GPU page entries
             * from the kernel. We need to extend it with CPU page entries.
             * For simplicity, store CPU metadata in a separate field or
             * append after the GPU entries. */
            if (cpu_pages > 0) {
                uint64_t existing = caps[i].encrypted_page_count;
                uint64_t new_count = existing + cpu_pages;
                page_crypto_meta_t *new_meta = realloc(caps[i].crypto_meta,
                    new_count * sizeof(page_crypto_meta_t));
                if (new_meta) {
                    for (uint32_t p = 0; p < cpu_pages; p++) {
                        memcpy(new_meta[existing + p].iv,
                               meta[meta_off + p].iv, CKPT_CRYPTO_IV_SIZE);
                        new_meta[existing + p].iv_fresh = 0;
                        memcpy(new_meta[existing + p].auth_tag,
                               meta[meta_off + p].auth_tag, CKPT_CRYPTO_AUTH_TAG_SIZE);
                        new_meta[existing + p].size = CKPT_CRYPTO_PAGE_SIZE;
                        new_meta[existing + p].key_version = 0; /* k3 */
                    }
                    caps[i].crypto_meta = new_meta;
                    caps[i].encrypted_page_count = new_count;
                }
            }
            meta_off += cpu_pages;
        }
    }

    free(gathered);
    free(meta);

    if (encrypt_ms_out)
        *encrypt_ms_out = elapsed;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    int pause_first = 0;
    int encrypt_k3 = 0;
    int use_encrypted_ioctl = 0;  /* --encrypted: use ioctl 115 (MT v3) */
    int encrypt_threads = 0;  /* 0 = auto */
    int stun_copy = 0;            /* --stun-copy: skip PREPARE_ALL, just stun + copy ranges */
    int precopy_baseline = 0;     /* --precopy-baseline: P1e in-app precopy + in-app delta */
    int baseline_mstreams = 1;    /* --mstreams N: P1f multi-stream count (1 = single-stream) */
    int coalesce = 0;             /* --coalesce: P3-B — collapse N per-range ioctl 115 calls
                                   *             into ONE call spanning [min_va, max_end).
                                   *             Requires --encrypted, forbids --precopy-baseline. */
    const char *ranges_file = NULL; /* --ranges-file: binary file with dynamic VA ranges */
    const char *dump_dirty_path = NULL; /* --dump-dirty: write dirty addrs to file */
    const char *static_img_path = NULL; /* P6 --static-img: emit-once / skip H2D allocs */
    int argoff = 1;

    /* Parse flags */
    while (argoff < argc && argv[argoff][0] == '-') {
        if (strcmp(argv[argoff], "--pause-first") == 0) {
            pause_first = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--encrypted") == 0) {
            use_encrypted_ioctl = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--encrypt-k3") == 0) {
            encrypt_k3 = 1;
            argoff++;
            /* Optional: --encrypt-threads N */
            if (argoff < argc && strcmp(argv[argoff], "--encrypt-threads") == 0) {
                argoff++;
                if (argoff < argc) {
                    encrypt_threads = atoi(argv[argoff]);
                    argoff++;
                }
            }
        } else if (strcmp(argv[argoff], "--stun-copy") == 0) {
            stun_copy = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--precopy-baseline") == 0) {
            precopy_baseline = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--mstreams") == 0) {
            argoff++;
            if (argoff < argc) {
                baseline_mstreams = atoi(argv[argoff]);
                argoff++;
            }
        } else if (strcmp(argv[argoff], "--coalesce") == 0) {
            coalesce = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--ranges-file") == 0) {
            argoff++;
            if (argoff < argc) {
                ranges_file = argv[argoff];
                argoff++;
            }
        } else if (strcmp(argv[argoff], "--dump-dirty") == 0) {
            argoff++;
            if (argoff < argc) {
                dump_dirty_path = argv[argoff];
                argoff++;
            }
        } else if (strcmp(argv[argoff], "--static-img") == 0) {
            argoff++;
            if (argoff < argc) {
                static_img_path = argv[argoff];
                argoff++;
            }
        } else if (strcmp(argv[argoff], "--no-image-write") == 0) {
            g_no_image_write = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--track-only") == 0) {
            g_track_only = 1;
            argoff++;
        } else if (strcmp(argv[argoff], "--track-only-budget-ms") == 0) {
            argoff++;
            if (argoff < argc) {
                g_track_only_budget_ms = atoi(argv[argoff]);
                argoff++;
            }
        } else {
            break;
        }
    }

    /* --coalesce (P3-B) validation: only compatible with --encrypted,
     * incompatible with --precopy-baseline. */
    if (coalesce) {
        if (!use_encrypted_ioctl) {
            fprintf(stderr, "Error: --coalesce requires --encrypted\n");
            return 1;
        }
        if (precopy_baseline) {
            fprintf(stderr, "Error: --coalesce is not compatible with --precopy-baseline "
                            "(precopy uses in-app cudaMemcpyAsync, not ioctl 115)\n");
            return 1;
        }
    }

    const char *gate_path  = (argc > argoff+0) ? argv[argoff+0] : GATE_FILE_DEFAULT;
    const char *base_path  = (argc > argoff+1) ? argv[argoff+1] : "/tmp/ckpt_inc_base.img";
    const char *delta_path = (argc > argoff+2) ? argv[argoff+2] : "/tmp/ckpt_inc_delta.img";

    {
        const char *vv = getenv("CKPT_VERBOSE");
        g_verbose = (vv && atoi(vv) > 0) ? 1 : 0;
    }

    printf("=== Incremental Pre-Copy Checkpoint Agent ===\n");
    printf("Gate  : %s\n", gate_path);
    printf("Base  : %s\n", base_path);
    printf("Delta : %s\n", delta_path);
    printf("Mode  : %s\n", stun_copy ? "stun-copy (adaptive, no PREPARE_ALL)" :
                          pause_first ? "pause-first (stop-and-copy)" : "pre-copy (concurrent baseline)");
    if (ranges_file)
        printf("Ranges: %s\n", ranges_file);
    if (dump_dirty_path)
        printf("Dump  : %s\n", dump_dirty_path);
    if (g_track_only) {
        printf("Image : --track-only — ioctl 115/116 SKIPPED, image INVALID. "
               "Per-skipped-call sleep budget=%d ms.\n", g_track_only_budget_ms);
    }
    if (g_no_image_write) {
        printf("Image : --no-image-write — output goes to anon mmap, "
               "discarded on munmap (restore disabled)\n");
    } else {
        printf("Image : persistent (writes to %s / %s)\n", base_path, delta_path);
    }
    if (encrypt_k3) {
        printf("Crypto: naive baseline (ioctl 109 + k3 re-encryption)\n\n");
    } else if (use_encrypted_ioctl) {
        const char *env = getenv("CKPT_MT_THREADS");
        int t = env ? atoi(env) : CKPT_MT_THREADS_DEFAULT;
        if (t < 1)  t = 1;
        if (t > 16) t = 16;
        printf("Crypto: optimized v3 (ioctl 115 — multi-threaded encrypted, %d worker(s), no CPU crypto)\n\n", t);
    } else {
        printf("Crypto: none (plaintext checkpoint)\n\n");
    }

    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: requires root (CAP_SYS_ADMIN)\n");
        return 1;
    }

    /* ---- Open UVM ------------------------------------------------ */
    int uvm_fd = open("/dev/nvidia-uvm", O_RDWR);
    if (uvm_fd < 0) { perror("open /dev/nvidia-uvm"); return 1; }

    UVM_INITIALIZE_PARAMS init = {0};
    if (ioctl(uvm_fd, UVM_INITIALIZE, &init) < 0 || init.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_INITIALIZE: %s  status=0x%08x\n",
                strerror(errno), init.rmStatus);
        close(uvm_fd); return 1;
    }
    printf("✓ UVM initialized\n\n");

    /* ---- Generate k3 key if any encryption mode is enabled -------- */
    /* encrypt-k3: k3 encrypts everything (CPU + GPU pages)           */
    /* encrypted:  k3 encrypts CPU pages only (GPU pages use k1)      */
    uint8_t k3_key[CKPT_CRYPTO_KEY_SIZE];
    uint64_t k3_iv_counter = 0;
    double base_encrypt_ms = 0.0;
    double delta_encrypt_ms = 0.0;
    /* P6: --static-img always emits k3-encrypted static.img, so we need
     * the key in ckpt_core regardless of which baseline mode is active.
     * The key file is shared with libckpt_vllm.c (sidecar at <base>.k3.key
     * derived from delta_path below); whichever process touches it first
     * generates+saves, others reuse — same chicken/egg pattern as the
     * existing --encrypt-k3 path. */
    int need_k3 = encrypt_k3 || use_encrypted_ioctl || (static_img_path != NULL);

    if (need_k3) {
        /* Save k3 key to file for restore — derive dir from delta_path.
         * If the key file already exists (from an earlier ckpt_core run
         * against the same app), REUSE it. Regenerating would break any
         * previously-written checkpoint image encrypted with the old key,
         * and is unnecessary: one k3 per app lifetime is the right scope. */
        char k3_path[PATH_MAX_LEN];
        {
            char tmp[PATH_MAX_LEN - 16];
            strncpy(tmp, delta_path, sizeof(tmp) - 1);
            tmp[sizeof(tmp) - 1] = '\0';
            char *sl = strrchr(tmp, '/');
            if (sl) *sl = '\0';
            else    strcpy(tmp, ".");
            snprintf(k3_path, sizeof(k3_path), "%s/ckpt_k3.key", tmp);
        }

        /* Try to load existing key first */
        int loaded = 0;
        FILE *kr = fopen(k3_path, "rb");
        if (kr) {
            size_t n = fread(k3_key, 1, CKPT_CRYPTO_KEY_SIZE, kr);
            fclose(kr);
            if (n == CKPT_CRYPTO_KEY_SIZE) {
                loaded = 1;
                printf("✓ K3 key reused from existing %s\n\n", k3_path);
            } else {
                fprintf(stderr,
                        "WARNING: %s exists but is %zu bytes (expected %d) — regenerating\n",
                        k3_path, n, CKPT_CRYPTO_KEY_SIZE);
            }
        }

        if (!loaded) {
            if (ckpt_crypto_gen_key(k3_key) != 0) {
                fprintf(stderr, "ERROR: k3 key generation failed\n");
                return 1;
            }
            FILE *kf = fopen(k3_path, "wb");
            if (kf) {
                fwrite(k3_key, 1, CKPT_CRYPTO_KEY_SIZE, kf);
                fclose(kf);
                printf("✓ K3 key generated and saved to %s\n\n", k3_path);
            } else {
                fprintf(stderr, "WARNING: could not save k3 key to %s: %s\n",
                        k3_path, strerror(errno));
                printf("✓ K3 key generated (not saved to file)\n\n");
            }
        }
    }

    /* ---- Wait for app gate --------------------------------------- */
    printf("Waiting for gate ...\n");
    ckpt_gate_t *gate = gate_open(gate_path);
    printf("✓ Gate mapped  pid=%d\n\n", gate->app_pid);

    NvU64 user_va   = gate->user_va;
    NvU64 user_size = gate->user_size;

    /* Read allocation table (multi-buffer support) */
    ckpt_alloc_hdr_t   *alloc_hdr   = GATE_ALLOC_HDR(gate);
    ckpt_alloc_entry_t *alloc_table = GATE_ALLOC_TABLE(gate);
    uint32_t n_allocs = alloc_hdr->count;

    if (n_allocs > 0) {
        uint32_t n_h2d = 0;
        uint64_t h2d_bytes = 0, total_alloc_bytes = 0;
        printf("App allocs (%u buffer(s)):\n", n_allocs);
        for (uint32_t i = 0; i < n_allocs; i++) {
            int is_h2d = (alloc_table[i].flags & CKPT_ALLOC_FLAG_H2D) != 0;
            if (g_verbose)
                printf("  [%u] VA=0x%016llx  size=%.2f MB  %s\n", i,
                       (unsigned long long)alloc_table[i].va,
                       alloc_table[i].size / (1024.0 * 1024.0),
                       is_h2d ? "(H2D)" : "(GPU-only)");
            total_alloc_bytes += alloc_table[i].size;
            if (is_h2d) { n_h2d++; h2d_bytes += alloc_table[i].size; }
        }
        printf("  H2D allocs: %u (%.2f MB) / total: %u (%.2f MB)\n\n",
               n_h2d, h2d_bytes / (1024.0 * 1024.0),
               n_allocs, total_alloc_bytes / (1024.0 * 1024.0));
        printf("\n");
    } else if (user_va > 0) {
        printf("App alloc (legacy): VA=0x%016llx  size=%.2f MB\n\n",
               (unsigned long long)user_va,
               user_size / (1024.0 * 1024.0));
    } else {
        printf("App alloc VA unknown — all ranges will be captured\n\n");
    }

    /* ================================================================ */
    /* P1 STRAWMAN BASELINE (in-app cudaMemcpyAsync stop-and-copy)      */
    /*                                                                   */
    /* Default plaintext path: no PREPARE_ALL, no GET_VA_RANGES, no      */
    /* dirty tracking, no delta. Agent signals baseline_req on the gate, */
    /* the LD_PRELOAD'd app catches it at its next sync point and runs   */
    /* ckpt_baseline_run() — single stop-and-copy via cudaMemcpyAsync.   */
    /* This is the kernel-mod-free baseline for paper comparison.        */
    /* ================================================================ */
    if (!use_encrypted_ioctl && !encrypt_k3 && !stun_copy && !pause_first && !precopy_baseline) {
        printf("=== Baseline (in-app cudaMemcpyAsync) ===\n");

        if (n_allocs == 0) {
            fprintf(stderr, "ERROR: no allocations registered in gate "
                            "(LD_PRELOAD active and app started?)\n");
            close(uvm_fd); return 1;
        }

        /* Tell the app where to write the image. */
        snprintf(gate->baseline_path, sizeof(gate->baseline_path), "%s", base_path);
        gate->baseline_done      = 0;
        gate->baseline_err       = 0;
        gate->baseline_ms        = 0;
        gate->baseline_bytes     = 0;
        gate->baseline_mstreams  = (uint32_t)baseline_mstreams;
        __sync_synchronize();

        printf("[1] Signal baseline_req=1 → %s  (mstreams=%d)\n",
               base_path, baseline_mstreams);
        double t_signal = now_ms();
        gate->baseline_req = 1;
        __sync_synchronize();

        printf("[2] Waiting for app to reach next sync point and finish ...\n");
        while (gate->baseline_done == 0) {
            usleep(1000);
        }
        __sync_synchronize();
        double t_done = now_ms();

        if (gate->baseline_err) {
            fprintf(stderr, "ERROR: in-app baseline reported failure\n");
            close(uvm_fd); return 1;
        }

        printf("  ✓ Baseline complete in %.1f ms (app-reported %.1f ms, %.2f GB)\n\n",
               t_done - t_signal, gate->baseline_ms,
               gate->baseline_bytes / (1024.0*1024.0*1024.0));

        /* Write an empty delta image so restore_agent has something to mmap. */
        printf("[3] Writing empty delta image → %s\n", delta_path);
        {
            range_capture_t empty_cap = {0};
            double dummy_ms = 0.0;
            if (write_delta(delta_path, &empty_cap, 0,
                            user_va, user_size, 0, &dummy_ms) < 0) {
                fprintf(stderr, "ERROR: write empty delta failed\n");
                close(uvm_fd); return 1;
            }
        }

        /* Snapshot the gate file so restore can read the alloc table. */
        char out_dir_sm[PATH_MAX_LEN];
        strncpy(out_dir_sm, delta_path, sizeof(out_dir_sm) - 1);
        out_dir_sm[sizeof(out_dir_sm) - 1] = '\0';
        {
            char *sl = strrchr(out_dir_sm, '/');
            if (sl) *sl = '\0'; else strcpy(out_dir_sm, ".");
        }
        printf("[4] Snapshot gate → %s\n", out_dir_sm);
        snapshot_file(gate_path, out_dir_sm);

        /* Tell app it can continue (clear the request bits). */
        gate->baseline_req = 0;
        __sync_synchronize();

        double base_gbps = gate->baseline_ms > 0
            ? gate->baseline_bytes / (1024.0*1024.0*1024.0) /
              (gate->baseline_ms / 1000.0)
            : 0.0;
        printf("\n=== Strawman Baseline Timing ===\n");
        printf("  in-app cudaMemcpyAsync : %8.1f ms  (%.2f GB, %.2f GB/s)\n",
               gate->baseline_ms,
               gate->baseline_bytes / (1024.0*1024.0*1024.0),
               base_gbps);
        printf("  e2e (signal → done)    : %8.1f ms\n", t_done - t_signal);

        munmap(gate, GATE_FILE_SIZE);
        close(uvm_fd);
        printf("\n=== Agent done (strawman baseline) ===\n");
        return 0;
    }

    /* ================================================================ */
    /* P1e PRECOPY BASELINE: concurrent baseline + in-app dirty delta    */
    /*                                                                    */
    /* PREPARE_ALL arms kernel dirty tracking at 2 MB granularity.       */
    /* Agent signals concurrent baseline (app keeps running; kernel      */
    /* tracks dirtied pages). After baseline, stun app, GET_DIRTY_PAGES, */
    /* round up to 2 MB ranges, populate gate dirty table, signal        */
    /* delta_req. App's in-app cudaMemcpyAsync does the delta into a     */
    /* v4 delta image sharing the same k3 key + continuing IV counter.   */
    /* ================================================================ */
    if (precopy_baseline) {
        printf("=== P1e Precopy Baseline (in-app cudaMemcpyAsync + in-app delta) ===\n");

        if (n_allocs == 0) {
            fprintf(stderr, "ERROR: no allocations registered in gate\n");
            close(uvm_fd); return 1;
        }

        /* [1] PREPARE_ALL (arm dirty tracking) */
        printf("[1] PREPARE_ALL ...\n");
        UVM_LIVE_MIGRATION_PREPARE_ALL_PARAMS pa = {0};
        double t_prep0 = now_ms();
        int prep_rc = ioctl(uvm_fd, UVM_LIVE_MIGRATION_PREPARE_ALL, &pa);
        double t_prep1 = now_ms();
        if (prep_rc < 0 || pa.rmStatus != NV_OK) {
            fprintf(stderr, "PREPARE_ALL: %s  status=0x%08x\n",
                    strerror(errno), pa.rmStatus);
            close(uvm_fd); return 1;
        }
        fprintf(stderr, "[ckpt-tim] PREPARE_ALL wall=%.2f ms\n",
                t_prep1 - t_prep0);
        printf("  ✓ dirty tracking armed\n\n");

        /* [2] Signal concurrent baseline */
        snprintf(gate->baseline_path, sizeof(gate->baseline_path), "%s", base_path);
        gate->baseline_done        = 0;
        gate->baseline_err         = 0;
        gate->baseline_ms          = 0;
        gate->baseline_bytes       = 0;
        gate->baseline_concurrent  = 1;   /* skip cudaDeviceSynchronize */
        gate->baseline_mstreams    = (uint32_t)baseline_mstreams;
        __sync_synchronize();

        printf("[2] Signal concurrent baseline → %s  (mstreams=%d)\n",
               base_path, baseline_mstreams);
        double t_base_start = now_ms();
        gate->baseline_req = 1;
        __sync_synchronize();

        printf("[3] Waiting for baseline (app running concurrently) ...\n");
        while (gate->baseline_done == 0) usleep(1000);
        __sync_synchronize();
        double t_base_done = now_ms();

        if (gate->baseline_err) {
            fprintf(stderr, "ERROR: concurrent baseline reported failure\n");
            close(uvm_fd); return 1;
        }
        printf("  ✓ Baseline: %.1f ms (app-reported %.1f ms, %.2f GB)\n\n",
               t_base_done - t_base_start, gate->baseline_ms,
               gate->baseline_bytes / (1024.0*1024.0*1024.0));

        /* [4] Stun app */
        printf("[4] ckpt_req=1 (stun app at next sync) ...\n");
        double t_stun_start = now_ms();
        __sync_synchronize();
        gate->ckpt_req = 1;
        __sync_synchronize();
        while (gate->phase != GATE_PHASE_AT_BOUNDARY) usleep(100);
        __sync_synchronize();
        double t_stun_end = now_ms();
        printf("  ✓ AT_BOUNDARY in %.1f ms\n\n", t_stun_end - t_stun_start);

        /* [5] GET_DIRTY_PAGES → 2 MB-aligned ranges */
        printf("[5] GET_DIRTY_PAGES ...\n");
        NvU64  max_dirty   = (gate->baseline_bytes > 0)
                              ? (gate->baseline_bytes / PAGE_64K + 1)
                              : 65536;
        NvU64 *dirty_addrs = malloc(max_dirty * sizeof(NvU64));
        if (!dirty_addrs) { fprintf(stderr, "malloc dirty\n"); return 1; }

        UVM_LIVE_MIGRATION_GET_DIRTY_PAGES_PARAMS dp = {0};
        dp.max_pages       = max_dirty;
        dp.dirty_addresses = dirty_addrs;
        if (ioctl(uvm_fd, UVM_LIVE_MIGRATION_GET_DIRTY_PAGES, &dp) < 0 ||
            dp.rmStatus != NV_OK) {
            fprintf(stderr, "GET_DIRTY_PAGES: %s  status=0x%08x\n",
                    strerror(errno), dp.rmStatus);
            free(dirty_addrs); close(uvm_fd); return 1;
        }
        NvU64 num_dirty_64k = dp.num_pages;
        printf("  ✓ %llu dirty 64KB page(s) (%.2f MB)\n",
               (unsigned long long)num_dirty_64k,
               num_dirty_64k * PAGE_64K / (1024.0*1024.0));

        /* Coalesce to 2 MB-aligned unique regions, then merge contiguous. */
        #define CHUNK_2MB  (2ULL * 1024 * 1024)
        /* Collapse each dirty 64KB page to its containing 2MB boundary. */
        for (NvU64 i = 0; i < num_dirty_64k; i++)
            dirty_addrs[i] &= ~(CHUNK_2MB - 1);
        qsort(dirty_addrs, (size_t)num_dirty_64k, sizeof(NvU64), cmp_u64);
        /* Dedup in place */
        NvU64 n_uniq = 0;
        for (NvU64 i = 0; i < num_dirty_64k; i++) {
            if (i == 0 || dirty_addrs[i] != dirty_addrs[n_uniq - 1])
                dirty_addrs[n_uniq++] = dirty_addrs[i];
        }
        /* Filter to user alloc ranges and merge contiguous 2 MB blocks. */
        ckpt_dirty_hdr_t   *dhdr = GATE_DIRTY_HDR(gate);
        ckpt_dirty_range_t *dtbl = GATE_DIRTY_TABLE(gate);
        uint32_t n_ranges_out = 0;
        uint64_t total_delta_bytes = 0;
        NvU64 cur_start = 0, cur_end = 0;
        int cur_valid = 0;

        for (NvU64 i = 0; i < n_uniq; i++) {
            NvU64 va   = dirty_addrs[i];
            NvU64 vend = va + CHUNK_2MB;
            /* in any live alloc? */
            int in_range = 0;
            for (uint32_t a = 0; a < n_allocs && !in_range; a++) {
                NvU64 a_end = alloc_table[a].va + alloc_table[a].size;
                if (va >= alloc_table[a].va && vend <= a_end) in_range = 1;
            }
            if (!in_range) continue;

            if (!cur_valid) {
                cur_start = va; cur_end = vend; cur_valid = 1;
            } else if (va == cur_end) {
                cur_end = vend;  /* extend run */
            } else {
                if (n_ranges_out < GATE_DIRTY_MAX) {
                    dtbl[n_ranges_out].va   = cur_start;
                    dtbl[n_ranges_out].size = cur_end - cur_start;
                    total_delta_bytes      += cur_end - cur_start;
                    n_ranges_out++;
                }
                cur_start = va; cur_end = vend;
            }
        }
        if (cur_valid && n_ranges_out < GATE_DIRTY_MAX) {
            dtbl[n_ranges_out].va   = cur_start;
            dtbl[n_ranges_out].size = cur_end - cur_start;
            total_delta_bytes      += cur_end - cur_start;
            n_ranges_out++;
        }
        dhdr->count = n_ranges_out;
        __sync_synchronize();
        free(dirty_addrs);
        printf("  coalesced → %u 2 MB-aligned range(s), %.2f MB delta\n\n",
               n_ranges_out, total_delta_bytes / (1024.0*1024.0));

        /* [6] Signal delta_req while app is still at AT_BOUNDARY.
         * maybe_checkpoint_post_sync's wait loop services delta_req in
         * place without letting the app run, so the snapshot stays
         * consistent between GET_DIRTY_PAGES and the delta copy. */
        snprintf(gate->delta_path, sizeof(gate->delta_path), "%s", delta_path);
        gate->delta_done  = 0;
        gate->delta_err   = 0;
        gate->delta_ms    = 0;
        gate->delta_bytes = 0;
        __sync_synchronize();

        printf("[6] Signal delta_req=1 → %s (app still at AT_BOUNDARY)\n", delta_path);
        double t_delta_start = now_ms();
        gate->delta_req = 1;
        __sync_synchronize();

        printf("[7] Waiting for delta ...\n");
        while (gate->delta_done == 0) usleep(1000);
        __sync_synchronize();
        double t_delta_done = now_ms();

        /* NOW release the app. */
        gate->phase    = GATE_PHASE_RESUME;
        gate->ckpt_req = 0;
        __sync_synchronize();

        if (gate->delta_err) {
            fprintf(stderr, "ERROR: in-app delta reported failure\n");
            close(uvm_fd); return 1;
        }
        printf("  ✓ Delta: %.1f ms (app-reported %.1f ms, %.2f MB)\n\n",
               t_delta_done - t_delta_start, gate->delta_ms,
               gate->delta_bytes / (1024.0*1024.0));

        /* Snapshot the gate file for restore. */
        char out_dir_pc[PATH_MAX_LEN];
        strncpy(out_dir_pc, delta_path, sizeof(out_dir_pc) - 1);
        out_dir_pc[sizeof(out_dir_pc) - 1] = '\0';
        { char *sl = strrchr(out_dir_pc, '/');
          if (sl) *sl = '\0'; else strcpy(out_dir_pc, "."); }
        printf("[8] Snapshot gate → %s\n", out_dir_pc);
        snapshot_file(gate_path, out_dir_pc);

        /* Clear request bits */
        gate->baseline_req = 0;
        gate->delta_req    = 0;
        __sync_synchronize();

        /* FINALIZE */
        printf("[9] FINALIZE ...\n");
        UVM_LIVE_MIGRATION_FINALIZE_PARAMS fp = {0};
        double t_fin0 = now_ms();
        int fin_rc = ioctl(uvm_fd, UVM_LIVE_MIGRATION_FINALIZE, &fp);
        double t_fin1 = now_ms();
        fprintf(stderr, "[ckpt-tim] FINALIZE wall=%.2f ms (rc=%d rm=0x%x)\n",
                t_fin1 - t_fin0, fin_rc, fp.rmStatus);
        if (fin_rc == 0 && fp.rmStatus == NV_OK)
            printf("  ✓ tracking disabled\n");

        double base_gbps = gate->baseline_ms > 0
            ? gate->baseline_bytes / (1024.0*1024.0*1024.0) /
              (gate->baseline_ms / 1000.0) : 0.0;
        double delta_gbps = gate->delta_ms > 0
            ? gate->delta_bytes / (1024.0*1024.0*1024.0) /
              (gate->delta_ms / 1000.0) : 0.0;

        printf("\n=== P1e Precopy Baseline Timing ===\n");
        printf("  baseline (concurrent): %8.1f ms  (%.2f GB, %.2f GB/s)\n",
               gate->baseline_ms,
               gate->baseline_bytes / (1024.0*1024.0*1024.0), base_gbps);
        printf("  stun latency         : %8.1f ms\n", t_stun_end - t_stun_start);
        printf("  delta (stop&copy)    : %8.1f ms  (%.2f MB, %.2f GB/s)\n",
               gate->delta_ms,
               gate->delta_bytes / (1024.0*1024.0), delta_gbps);
        printf("  e2e (PREPARE → done) : %8.1f ms\n", t_delta_done - t_base_start);

        munmap(gate, GATE_FILE_SIZE);
        close(uvm_fd);
        printf("\n=== Agent done (P1e precopy) ===\n");
        return 0;
        #undef CHUNK_2MB
    }

    /* ================================================================ */
    /* STUN-COPY MODE: skip PREPARE_ALL, just stun + copy given ranges */
    /* ================================================================ */
    if (stun_copy) {
        if (!ranges_file) {
            fprintf(stderr, "ERROR: --stun-copy requires --ranges-file\n");
            close(uvm_fd); return 1;
        }

        /* Read dynamic ranges from binary file: [uint32_t count][(va,size) x count] */
        FILE *rf = fopen(ranges_file, "rb");
        if (!rf) {
            fprintf(stderr, "ERROR: open ranges file %s: %s\n", ranges_file, strerror(errno));
            close(uvm_fd); return 1;
        }
        uint32_t n_dyn_ranges = 0;
        fread(&n_dyn_ranges, sizeof(uint32_t), 1, rf);

        typedef struct { uint64_t va; uint64_t size; } dyn_range_t;
        dyn_range_t *dyn_ranges = malloc(n_dyn_ranges * sizeof(dyn_range_t));
        if (!dyn_ranges) { fclose(rf); close(uvm_fd); return 1; }
        fread(dyn_ranges, sizeof(dyn_range_t), n_dyn_ranges, rf);
        fclose(rf);

        uint64_t dyn_total = 0;
        printf("[stun-copy] %u dynamic range(s):\n", n_dyn_ranges);
        for (uint32_t i = 0; i < n_dyn_ranges; i++) {
            if (g_verbose)
                printf("  [%u] VA=0x%016llx  size=%.2f MB\n", i,
                       (unsigned long long)dyn_ranges[i].va,
                       dyn_ranges[i].size / (1024.0 * 1024.0));
            dyn_total += dyn_ranges[i].size;
        }
        printf("  Total dynamic: %.2f MB\n\n", dyn_total / (1024.0 * 1024.0));

        /* Stun: set ckpt_req=1, wait for AT_BOUNDARY */
        printf("[stun-copy] Stunning app (ckpt_req=1) ...\n");
        double t_stun_start = now_ms();
        __sync_synchronize();
        gate->ckpt_req = 1;
        __sync_synchronize();

        while (gate->phase != GATE_PHASE_AT_BOUNDARY)
            usleep(100);
        __sync_synchronize();
        double t_at_boundary_sc = now_ms();
        printf("  App stunned in %.1f ms\n\n", t_at_boundary_sc - t_stun_start);

        /* Read dynamic ranges */
        printf("[stun-copy] READ_PAGES_RESIDENT (%u range(s)) ...\n", n_dyn_ranges);
        range_capture_t *dyn_caps = calloc(n_dyn_ranges, sizeof(range_capture_t));
        if (!dyn_caps) { free(dyn_ranges); close(uvm_fd); return 1; }

        double dyn_read_ms = 0.0;
        int sc_err = 0;
        for (uint32_t i = 0; i < n_dyn_ranges && !sc_err; i++) {
            if (use_encrypted_ioctl) {
                if (capture_range_encrypted(uvm_fd, dyn_ranges[i].va, dyn_ranges[i].size,
                                             &dyn_caps[i], NULL, &dyn_read_ms) < 0)
                    sc_err = 1;
            } else {
                if (capture_range(uvm_fd, dyn_ranges[i].va, dyn_ranges[i].size,
                                  &dyn_caps[i], &dyn_read_ms) < 0)
                    sc_err = 1;
            }
        }

        /* Encrypt if needed */
        double sc_encrypt_ms = 0.0;
        if (!sc_err && encrypt_k3) {
            printf("  [k3] Encrypting dynamic captures ...\n");
            if (encrypt_captures_batch(dyn_caps, n_dyn_ranges, k3_key, &k3_iv_counter,
                                       encrypt_threads, CKPT_CRYPTO_PAGE_SIZE,
                                       &sc_encrypt_ms) < 0)
                sc_err = 1;
        } else if (!sc_err && use_encrypted_ioctl) {
            printf("  [k3] Encrypting dynamic CPU pages ...\n");
            if (encrypt_cpu_pages_batch(dyn_caps, n_dyn_ranges, k3_key, &k3_iv_counter,
                                         encrypt_threads, &sc_encrypt_ms) < 0)
                sc_err = 1;
        }

        /* Write as baseline image (contains all data needed for this checkpoint) */
        double sc_write_ms = 0.0;
        if (!sc_err) {
            uint32_t img_version = use_encrypted_ioctl ? CKPT_V3_VERSION : CKPT_V2_VERSION;
            sc_err = write_baseline(base_path, dyn_caps, n_dyn_ranges, img_version, &sc_write_ms);
        }

        /* Write empty delta */
        if (!sc_err) {
            double dummy_ms = 0.0;
            range_capture_t empty_cap = {0};
            sc_err = write_delta(delta_path, &empty_cap, 0,
                                 user_va, user_size, 0, &dummy_ms);
        }

        for (uint32_t i = 0; i < n_dyn_ranges; i++)
            range_capture_free(&dyn_caps[i]);
        free(dyn_caps);
        free(dyn_ranges);

        /* Snapshot state files */
        {
            char out_dir_sc[PATH_MAX_LEN];
            strncpy(out_dir_sc, delta_path, sizeof(out_dir_sc) - 1);
            out_dir_sc[sizeof(out_dir_sc) - 1] = '\0';
            char *sl = strrchr(out_dir_sc, '/');
            if (sl) *sl = '\0'; else strcpy(out_dir_sc, ".");

            const char *ts = getenv("CKPT_TRAIN_STATE");
            if (!ts) ts = "/tmp/ckpt_train_state.json";
            snapshot_file(ts, out_dir_sc);
            const char *tm = getenv("CKPT_TENSOR_MAP");
            if (!tm) tm = "/tmp/ckpt_tensor_map.json";
            snapshot_file(tm, out_dir_sc);
            snapshot_file(gate_path, out_dir_sc);
        }

        /* Resume — no FINALIZE needed since we never did PREPARE_ALL */
        __sync_synchronize();
        gate->ckpt_req = 0;
        gate->phase    = GATE_PHASE_RESUME;
        __sync_synchronize();
        double t_sc_end = now_ms();

        printf("\n=== Stun-Copy Timing ===\n");
        printf("  stun latency        : %8.1f ms\n", t_at_boundary_sc - t_stun_start);
        printf("  read (dynamic only) : %8.1f ms  (%.1f MB/s  %.2f MB)\n",
               dyn_read_ms,
               dyn_read_ms > 0 ? dyn_total / (1024.0*1024.0) / (dyn_read_ms/1000.0) : 0.0,
               dyn_total / (1024.0*1024.0));
        printf("  write               : %8.1f ms\n", sc_write_ms);
        if (sc_encrypt_ms > 0)
            printf("  encryption          : %8.1f ms\n", sc_encrypt_ms);
        printf("  total (stun→resume) : %8.1f ms\n", t_sc_end - t_stun_start);

        munmap(gate, GATE_FILE_SIZE);
        close(uvm_fd);
        printf("\n=== Agent done (stun-copy) ===\n");
        return sc_err ? 1 : 0;
    }

    /* ---- [1] GET_VA_RANGES + filter ------------------------------ */
    printf("[1] GET_VA_RANGES ...\n");
    NvU64 base_addrs[MAX_RANGES], range_sizes[MAX_RANGES];
    UVM_LIVE_MIGRATION_GET_VA_RANGES_PARAMS gr = {0};
    gr.max_ranges  = MAX_RANGES;
    gr.base_addrs  = base_addrs;
    gr.range_sizes = range_sizes;

    if (ioctl(uvm_fd, UVM_LIVE_MIGRATION_GET_VA_RANGES, &gr) < 0 ||
        gr.rmStatus != NV_OK) {
        fprintf(stderr, "GET_VA_RANGES: %s  status=0x%08x\n",
                strerror(errno), gr.rmStatus);
        close(uvm_fd); return 1;
    }

    NvU64 filt_base[MAX_RANGES], filt_size[MAX_RANGES];
    NvU32 n_filt = 0;
    NvU64 base_total_bytes = 0;
    /* P6 — static (H2D-flagged) ranges separated out when --static-img is set.
     * If static.img already exists on disk, these ranges are excluded from
     * base/delta entirely (saved earlier, decryptable on restore). If it does
     * not exist yet, they are captured into n_static_filt for a one-time
     * emit_static_img() call after the regular baseline write completes. */
    NvU64 static_filt_base[MAX_RANGES], static_filt_size[MAX_RANGES];
    NvU32 n_static_filt = 0;
    NvU64 static_total_bytes = 0;
    /* P6: "static.img exists" means it has a captured CKPT_V2 image — not
     * just a raw prealloc'd buffer. ckpt_prealloc creates a zero-filled
     * file of the requested size; the magic check distinguishes warm
     * scratch from a real prior capture. */
    int static_img_exists = 0;
    if (static_img_path) {
        int sf = open(static_img_path, O_RDONLY);
        if (sf >= 0) {
            uint32_t magic = 0;
            if (read(sf, &magic, sizeof(magic)) == (ssize_t)sizeof(magic) &&
                magic == CKPT_V2_MAGIC) {
                static_img_exists = 1;
            }
            close(sf);
        }
    }
    {
        for (NvU64 i = 0; i < gr.num_ranges; i++) {
            NvU64 r_start = base_addrs[i];
            NvU64 r_end   = base_addrs[i] + range_sizes[i];
            int keep = 0;
            int is_static = 0;

            if (n_allocs > 0) {
                /* Multi-buffer path: keep every user alloc (H2D weights + GPU-only
                 * KV cache / workspace) so baseline captures the full GPU-resident
                 * state. Needed for live-migration-style restore. */
                for (uint32_t a = 0; a < n_allocs && !keep; a++) {
                    NvU64 a_end = alloc_table[a].va + alloc_table[a].size;
                    if (r_start >= alloc_table[a].va && r_end <= a_end) {
                        keep = 1;
                        is_static = (alloc_table[a].flags & CKPT_ALLOC_FLAG_H2D) != 0;
                    }
                }
            } else {
                /* Legacy single-alloc path */
                NvU64 user_end = user_va + user_size;
                keep = (user_va == 0) ||
                       (r_start >= user_va && r_end <= user_end);
            }

            if (!keep) {
                printf("  skip 0x%016llx (CUDA internal)\n",
                       (unsigned long long)base_addrs[i]);
                continue;
            }

            if (static_img_path && is_static) {
                /* Excluded from base/delta this run. Captured into static.img
                 * later if file does not exist yet. */
                if (!static_img_exists) {
                    static_filt_base[n_static_filt] = r_start;
                    static_filt_size[n_static_filt] = range_sizes[i];
                    static_total_bytes += range_sizes[i];
                    n_static_filt++;
                }
                continue;
            }

            filt_base[n_filt] = r_start;
            filt_size[n_filt] = range_sizes[i];
            base_total_bytes += range_sizes[i];
            n_filt++;
        }
    }
    printf("  %u user range(s)  (%.2f MB)\n",
           n_filt, base_total_bytes / (1024.0 * 1024.0));
    if (static_img_path) {
        printf("  [P6] static.img: %s%s — %u H2D range(s) %s (%.2f MB)\n",
               static_img_path,
               static_img_exists ? " (exists, skipping H2D from base/delta)"
                                 : " (will emit this run)",
               n_static_filt,
               static_img_exists ? "excluded" : "to capture",
               static_total_bytes / (1024.0 * 1024.0));
    }
    printf("\n");

    if (n_filt == 0) {
        fprintf(stderr, "ERROR: no user ranges found — is LD_PRELOAD active?\n");
        close(uvm_fd); return 1;
    }

    /* ---- [P6] Emit static.img once ---------------------------------
     * Mode-agnostic: runs for --precopy-baseline, --encrypted, and
     * --encrypt-k3 alike. Placed before PREPARE_ALL because static memory
     * (H2D-loaded weights, embeddings, tokenizer buffers) is bit-stable
     * after the H2D copy — no need for write-protection semantics.
     * Skipped when file already exists or --static-img not passed.
     *
     * Two paths:
     *  1. --encrypted (use_encrypted_ioctl): use capture_and_write_coalesced
     *     so static.img is v3 format, kernel-encrypted with k1 (single ioctl
     *     115 call across all H2D ranges). Restore decrypts via ioctl 112,
     *     same path as base.img. ~10× faster gen vs the legacy path.
     *  2. Otherwise (legacy): per-range capture_range + CPU k3 encrypt +
     *     write_baseline (v2 format, 2 MB chunks for v4-delta restore reuse).
     *     Kept for --encrypt-k3 / no-encryption paths.
     */
    if (n_static_filt > 0) {
        double static_read_ms = 0.0, static_write_ms = 0.0;
        int s_err = 0;
        printf("[P6] Capturing %u static H2D range(s) (%.2f MB) → %s\n",
               n_static_filt, static_total_bytes / (1024.0 * 1024.0),
               static_img_path);

        if (use_encrypted_ioctl) {
            /* v3 path: single coalesced ioctl 115 with kernel-side encrypt. */
            if (capture_and_write_coalesced(uvm_fd, static_img_path,
                                             static_filt_base, static_filt_size,
                                             n_static_filt,
                                             &static_read_ms, &static_write_ms,
                                             UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT,
                                             NULL) < 0)
                s_err = 1;
            if (!s_err)
                printf("[P6] static.img v3 written: read=%.1f ms write=%.1f ms\n\n",
                       static_read_ms, static_write_ms);
        } else {
            /* Legacy path: per-range capture + CPU k3 encrypt + write_baseline. */
            range_capture_t *static_caps = calloc(n_static_filt,
                                                   sizeof(range_capture_t));
            if (!static_caps) {
                fprintf(stderr, "calloc static_caps\n");
                close(uvm_fd); return 1;
            }
            double static_encrypt_ms = 0.0;
            for (NvU32 i = 0; i < n_static_filt && !s_err; i++) {
                if (capture_range(uvm_fd, static_filt_base[i], static_filt_size[i],
                                  &static_caps[i], &static_read_ms) < 0)
                    s_err = 1;
            }
            if (!s_err) {
                if (encrypt_captures_batch(static_caps, n_static_filt, k3_key,
                                           &k3_iv_counter, encrypt_threads,
                                           2u * 1024u * 1024u,
                                           &static_encrypt_ms) < 0)
                    s_err = 1;
            }
            if (!s_err)
                s_err = write_baseline(static_img_path, static_caps,
                                       n_static_filt, CKPT_V2_VERSION,
                                       &static_write_ms);
            for (NvU32 i = 0; i < n_static_filt; i++)
                range_capture_free(&static_caps[i]);
            free(static_caps);
            if (!s_err)
                printf("[P6] static.img v2 written: read=%.1f ms encrypt=%.1f ms write=%.1f ms\n\n",
                       static_read_ms, static_encrypt_ms, static_write_ms);
        }

        if (s_err) {
            fprintf(stderr, "✗ static.img emit failed\n");
            close(uvm_fd); return 1;
        }
    }

    /* ---- [2] PREPARE_ALL ---------------------------------------- */
    printf("[2] PREPARE_ALL ...\n");
    UVM_LIVE_MIGRATION_PREPARE_ALL_PARAMS pa = {0};
    double t_prep0 = now_ms();
    int prep_rc = ioctl(uvm_fd, UVM_LIVE_MIGRATION_PREPARE_ALL, &pa);
    double t_prep1 = now_ms();
    if (prep_rc < 0 || pa.rmStatus != NV_OK) {
        fprintf(stderr, "PREPARE_ALL: %s  status=0x%08x\n",
                strerror(errno), pa.rmStatus);
        close(uvm_fd); return 1;
    }
    fprintf(stderr, "[ckpt-tim] PREPARE_ALL wall=%.2f ms\n",
            t_prep1 - t_prep0);
    printf("  ✓ Write protection active, dirty bitmaps zeroed\n\n");

    double base_read_ms   = 0.0;
    double base_write_ms  = 0.0;
    double t_at_boundary  = 0.0;
    int    err            = 0;
    double t_start        = now_ms();

    /* Derive output directory from delta_path for state file snapshots */
    char out_dir[PATH_MAX_LEN];
    {
        strncpy(out_dir, delta_path, sizeof(out_dir) - 1);
        out_dir[sizeof(out_dir) - 1] = '\0';
        char *last_slash = strrchr(out_dir, '/');
        if (last_slash) *last_slash = '\0';
        else            strcpy(out_dir, ".");
    }

    /* ---- [3/4] Baseline read + ckpt_req handshake ---------------- */
    if (pause_first) {
        /*
         * --pause-first mode: stop the app BEFORE reading baseline.
         * Use this when training steps are faster than the baseline read
         * (e.g. PyTorch MLP on H100 ~10ms/step vs ~1s baseline read).
         * The app pauses at the next cudaDeviceSynchronize; baseline is
         * then read as a clean stop-and-copy snapshot.
         */
        printf("[3] pause-first: setting ckpt_req=1 before baseline read ...\n");
        __sync_synchronize();
        gate->ckpt_req = 1;
        __sync_synchronize();
        printf("  ✓ ckpt_req=1 set\n\n");

        printf("[4] Waiting for AT_BOUNDARY (app pausing at next sync) ...\n");
        while (gate->phase != GATE_PHASE_AT_BOUNDARY)
            usleep(100);
        __sync_synchronize();
        t_at_boundary = now_ms();
        printf("  ✓ AT_BOUNDARY (%.1f ms) — app frozen, reading baseline\n\n",
               t_at_boundary - t_start);

        printf("[3b] Baseline READ_PAGES_RESIDENT (%u range(s), app frozen) ...\n", n_filt);
    } else {
        /*
         * Pre-copy mode (default): read baseline concurrently while app runs.
         * ckpt_req is still 0 here — the app has not been asked to stop.
         * Pages written after PREPARE_ALL generate dirty bits caught in delta.
         */
        printf("[3] Baseline READ_PAGES_RESIDENT (%u range(s), concurrent with app) ...\n", n_filt);
    }

    if (use_encrypted_ioctl && coalesce) {
        /* ---- P3-B coalesced path (single ioctl 115 spanning all ranges) ----
         * Replaces ~N per-range ioctl calls with one big call, amortizing
         * per-call setup (kthread spawn, DMA pool, channel acquire, PASS 1).
         * Expected win: ~3× throughput improvement based on POC bench. */
        printf("[3c] P3-B coalesced baseline (%u range(s), single ioctl 115) ...\n",
               n_filt);
        if (capture_and_write_coalesced(uvm_fd, base_path,
                                          filt_base, filt_size, n_filt,
                                          &base_read_ms, &base_write_ms,
                                          UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT,
                                          NULL) < 0) {
            fprintf(stderr, "✗ P3-B coalesced baseline failed\n");
            return 1;
        }
    } else if (use_encrypted_ioctl) {
        /* ---- Streaming pipeline path (encrypted v3) ----
         * Capture thread (main) reads each range via ioctl 115 into a
         * range_capture_t whose buffers are borrowed from a pre-allocated
         * pool slot. It hands the cap + slot_idx to a pool of N writer
         * threads, which memcpy their sections into a pre-allocated mmap'd
         * file region at pre-assigned offsets and release the pool slot
         * back to the pool — NO per-range malloc/free.
         */
        capture_pool_t    pool;
        baseline_stream_t stream;
        uint32_t          img_version = CKPT_V3_VERSION;
        double            stream_timings[3] = {0};

        /* Size the pool to the largest range, with as many slots as the
         * writer pool's work-queue capacity (2 × n_writers). */
        NvU64 max_length = 0;
        for (NvU32 i = 0; i < n_filt; i++)
            if (filt_size[i] > max_length) max_length = filt_size[i];
        NvU64 max_n_pages = max_length / 4096;

        int n_writers = baseline_stream_resolve_workers();
        int n_slots   = n_writers * 2;

        if (capture_pool_init(&pool, n_slots, max_length, max_n_pages) < 0) {
            fprintf(stderr, "✗ capture_pool_init failed\n");
            return 1;
        }

        if (baseline_stream_start(base_path, n_filt, filt_size, img_version,
                                  &pool,
                                  /*is_delta=*/0,
                                  /*delta_alloc_start_va=*/0,
                                  /*delta_alloc_size=*/0,
                                  /*delta_num_dirty_64k=*/0,
                                  &stream) < 0) {
            capture_pool_destroy(&pool);
            return 1;
        }

        for (NvU32 i = 0; i < n_filt && !err && !stream.shared_err; i++) {
            if (g_verbose)
                printf("  range[%u] 0x%016llx  %.2f MB\n", i,
                       (unsigned long long)filt_base[i],
                       filt_size[i] / (1024.0 * 1024.0));

            int slot_idx = capture_pool_acquire(&pool);

            range_capture_t *cap = calloc(1, sizeof(*cap));
            if (!cap) {
                fprintf(stderr, "calloc cap\n");
                capture_pool_release(&pool, slot_idx);
                err = 1;
                break;
            }

            if (capture_range_encrypted(uvm_fd, filt_base[i], filt_size[i],
                                         cap, &pool.slots[slot_idx],
                                         &base_read_ms) < 0) {
                free(cap);
                capture_pool_release(&pool, slot_idx);
                err = 1;
                break;
            }
            baseline_stream_enqueue(&stream, i, cap, slot_idx);
        }

        if (baseline_stream_join(&stream, stream_timings) < 0)
            err = 1;

        /* Feed the final timing block from the pipeline-total number. */
        base_write_ms = stream_timings[2];

        capture_pool_destroy(&pool);

        if (err) { fprintf(stderr, "✗ Baseline failed\n"); return 1; }
    } else {
        /* ---- Legacy buffered path (plaintext / encrypt-k3) ---- */
        range_capture_t *base_caps = calloc(n_filt, sizeof(range_capture_t));
        if (!base_caps) { fprintf(stderr, "calloc base_caps\n"); return 1; }

        for (NvU32 i = 0; i < n_filt && !err; i++) {
            if (g_verbose)
                printf("  range[%u] 0x%016llx  %.2f MB\n", i,
                       (unsigned long long)filt_base[i],
                       filt_size[i] / (1024.0 * 1024.0));
            if (capture_range(uvm_fd, filt_base[i], filt_size[i],
                              &base_caps[i], &base_read_ms) < 0)
                err = 1;
        }

        if (!err && encrypt_k3) {
            printf("  [k3] Encrypting baseline captures (%u range(s), %.2f MB) ...\n",
                   n_filt, base_total_bytes / (1024.0 * 1024.0));
            if (encrypt_captures_batch(base_caps, n_filt, k3_key, &k3_iv_counter,
                                       encrypt_threads, CKPT_CRYPTO_PAGE_SIZE,
                                       &base_encrypt_ms) < 0)
                err = 1;
            if (!err)
                printf("  [k3] Baseline encrypted (%.1f ms, %.1f GB/s)\n\n",
                       base_encrypt_ms,
                       base_encrypt_ms > 0
                           ? base_total_bytes / (1024.0*1024.0*1024.0) / (base_encrypt_ms/1000.0) : 0.0);
        }

        if (!err)
            err = write_baseline(base_path, base_caps, n_filt, CKPT_V2_VERSION, &base_write_ms);

        for (NvU32 i = 0; i < n_filt; i++)
            range_capture_free(&base_caps[i]);
        free(base_caps);

        if (err) { fprintf(stderr, "✗ Baseline failed\n"); return 1; }
    }

    /* ---- [4] Set ckpt_req=1 (pre-copy mode only) ----------------- */
    if (!pause_first) {
        /*
         * Pre-copy mode: baseline safely on disk, now stop the app.
         * Works well when the app runs long enough for the baseline read
         * to complete before all sync points are exhausted.
         */
        printf("[4] Setting ckpt_req=1 (app stops at next cudaStreamSynchronize) ...\n");
        __sync_synchronize();
        gate->ckpt_req = 1;
        __sync_synchronize();
        printf("  ✓ ckpt_req=1 set\n\n");
    }

    /* ---- [5] Wait for AT_BOUNDARY -------------------------------- */
    /* ---- [5] Wait for AT_BOUNDARY (pre-copy mode only) ----------- */
    if (!pause_first) {
        printf("[5] Waiting for AT_BOUNDARY ...\n");
        while (gate->phase != GATE_PHASE_AT_BOUNDARY)
            usleep(100);
        __sync_synchronize();
        t_at_boundary = now_ms();
        printf("  ✓ AT_BOUNDARY (%.1f ms from ckpt_req) — app quiesced\n\n",
               t_at_boundary - t_start);
    }

    /* delta-phase shared outputs (set by either the fused or legacy path,
     * consumed by the [stop] delta read/write prints below). */
    double   delta_read_ms     = 0.0;
    double   delta_write_ms    = 0.0;
    uint64_t delta_total_bytes = 0;

    /* ---- Tier3-fused delta path (default) ------------------------ *
     * ONE ioctl 116 over the full footprint span: the kernel internally
     * snapshots the dirty bitmap and copies+encrypts ONLY pages in dirty
     * 2MB va_blocks, paying the kthread/DMA/CE-channel/PASS-1 setup tax
     * ONCE for the whole delta instead of once per merged dirty region
     * (the per-region ioctl-115 loop is what pinned delta read at ~2 GB/s).
     * Set CKPT_DELTA_LEGACY=1 to fall back to the GET_DIRTY + merge +
     * per-region-115 loop, kept as a byte-identical correctness oracle. */
    if (use_encrypted_ioctl && n_filt > 0 &&
        getenv("CKPT_DELTA_LEGACY") == NULL) {
        printf("[6] Tier3-fused dirty-delta: ONE ioctl 116 over %u range(s) "
               "(no GET_DIRTY/merge/per-region loop) ...\n", n_filt);
        uint64_t fused_gpu_bytes = 0;
        if (capture_and_write_coalesced(uvm_fd, delta_path,
                                        filt_base, filt_size, n_filt,
                                        &delta_read_ms, &delta_write_ms,
                                        UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT,
                                        &fused_gpu_bytes) < 0) {
            fprintf(stderr, "✗ Tier3-fused dirty-delta failed\n");
            return 1;
        }
        delta_total_bytes = fused_gpu_bytes;
        printf("  ✓ fused delta captured %.2f MB in one ioctl 116 "
               "(read %.1f ms, write %.1f ms)\n\n",
               delta_total_bytes / (1024.0 * 1024.0),
               delta_read_ms, delta_write_ms);
        goto delta_captured;
    }

    /* ---- [6] GET_DIRTY_PAGES (legacy oracle path) ---------------- */
    printf("[6] GET_DIRTY_PAGES (legacy delta path) ...\n");

    /*
     * Allocate enough slots for the worst case: every 64KB page in every
     * user range is dirty.  base_total_bytes covers all filtered ranges.
     * Fall back to 65536 if nothing was captured.
     */
    NvU64  max_dirty   = (base_total_bytes > 0)
                         ? (base_total_bytes / PAGE_64K + 1)
                         : 65536;
    NvU64 *dirty_addrs = malloc(max_dirty * sizeof(NvU64));
    if (!dirty_addrs) { fprintf(stderr, "malloc dirty_addrs\n"); return 1; }

    UVM_LIVE_MIGRATION_GET_DIRTY_PAGES_PARAMS dp = {0};
    dp.max_pages       = max_dirty;
    dp.dirty_addresses = dirty_addrs;

    if (ioctl(uvm_fd, UVM_LIVE_MIGRATION_GET_DIRTY_PAGES, &dp) < 0 ||
        dp.rmStatus != NV_OK) {
        fprintf(stderr, "GET_DIRTY_PAGES: %s  status=0x%08x\n",
                strerror(errno), dp.rmStatus);
        free(dirty_addrs); return 1;
    }

    NvU64 num_dirty = dp.num_pages;
    /* Each entry is one dirty 2 MB va_block (uvm_live_migration.c:514). */
    printf("  ✓ %llu dirty 2MB block(s)  (%.2f MB total dirty)\n\n",
           (unsigned long long)num_dirty,
           num_dirty * (2ULL * 1024 * 1024) / (1024.0 * 1024.0));

    /* Dump raw dirty addresses to file if requested */
    if (dump_dirty_path && num_dirty > 0) {
        FILE *df = fopen(dump_dirty_path, "wb");
        if (df) {
            fwrite(&num_dirty, sizeof(NvU64), 1, df);
            fwrite(dirty_addrs, sizeof(NvU64), (size_t)num_dirty, df);
            fclose(df);
            printf("  Dirty addresses dumped to %s (%llu entries)\n\n",
                   dump_dirty_path, (unsigned long long)num_dirty);
        } else {
            fprintf(stderr, "  WARNING: could not write %s: %s\n",
                    dump_dirty_path, strerror(errno));
        }
    }

    /* Filter dirty pages to the user allocation range(s) */
    {
        NvU64 n_kept = 0;
        for (NvU64 i = 0; i < num_dirty; i++) {
            NvU64 page_start = dirty_addrs[i];
            NvU64 page_end   = page_start + PAGE_64K;
            int   in_range   = 0;

            if (n_allocs > 0) {
                /* Multi-buffer: keep if the page falls within ANY recorded alloc.
                 * P6: when --static-img is set, dirty pages inside H2D-flagged
                 * (static) allocs are excluded from the delta — those allocs are
                 * captured once into static.img and treated as immutable. If a
                 * dirty page does land in an H2D alloc, that's a heuristic
                 * violation Option 1's validation experiment will surface. */
                for (uint32_t a = 0; a < n_allocs && !in_range; a++) {
                    NvU64 a_end = alloc_table[a].va + alloc_table[a].size;
                    if (page_start >= alloc_table[a].va && page_end <= a_end) {
                        if (static_img_path &&
                            (alloc_table[a].flags & CKPT_ALLOC_FLAG_H2D))
                            break;  /* skip H2D alloc page from delta */
                        in_range = 1;
                    }
                }
            } else {
                /* Legacy single-alloc path */
                NvU64 user_end = user_va + user_size;
                in_range = (user_va == 0) ||
                           (page_start >= user_va && page_end <= user_end);
            }

            if (in_range)
                dirty_addrs[n_kept++] = dirty_addrs[i];
        }
        if (n_kept < num_dirty)
            printf("  (filtered %llu → %llu: only user alloc range(s))\n\n",
                   (unsigned long long)num_dirty,
                   (unsigned long long)n_kept);
        num_dirty = n_kept;
    }

    /* Each entry from GET_DIRTY_PAGES is a 2 MB-aligned va_block start
     * (uvm_live_migration.c:514 — kernel writes va_block->start, not a 64K
     * page address). Sort and merge adjacent 2 MB blocks into runs.        */
    #define DELTA_CHUNK_2MB  (2ULL * 1024 * 1024)
    qsort(dirty_addrs, (size_t)num_dirty, sizeof(NvU64), cmp_u64);

    typedef struct { NvU64 start; NvU64 len; } dirty_region_t;
    dirty_region_t *regions = malloc((num_dirty + 1) * sizeof(dirty_region_t));
    if (!regions) { fprintf(stderr, "malloc regions\n"); free(dirty_addrs); return 1; }

    NvU32 n_regions = 0;
    if (num_dirty > 0) {
        NvU64 reg_start = dirty_addrs[0];
        NvU64 reg_end   = dirty_addrs[0] + DELTA_CHUNK_2MB;
        for (NvU64 i = 1; i < num_dirty; i++) {
            if (dirty_addrs[i] == reg_end) {
                reg_end += DELTA_CHUNK_2MB;
            } else {
                regions[n_regions++] = (dirty_region_t){reg_start, reg_end - reg_start};
                reg_start = dirty_addrs[i];
                reg_end   = dirty_addrs[i] + DELTA_CHUNK_2MB;
            }
        }
        regions[n_regions++] = (dirty_region_t){reg_start, reg_end - reg_start};
    }
    free(dirty_addrs);
    #undef DELTA_CHUNK_2MB

    delta_total_bytes = 0;
    for (NvU32 i = 0; i < n_regions; i++)
        delta_total_bytes += regions[i].len;
    printf("  %u dirty region(s) after merge  (%.2f MB)\n\n",
           n_regions, delta_total_bytes / (1024.0 * 1024.0));

    /* ---- [7] Delta read (app frozen — no concurrent writes) ------ */
    printf("[7] Delta READ_PAGES_RESIDENT (%u region(s)) ...\n", n_regions);

    /* delta_read_ms / delta_write_ms hoisted above the fused branch */
    err = 0;

    if (use_encrypted_ioctl && n_regions > 0) {
        /* ---- v3 streaming pipeline ----
         * Same pool + writer-pool architecture the baseline uses, but with
         * is_delta=1 so the file gets a ckpt_inc_delta header. Capture
         * thread reads each dirty region via ioctl 115 into a borrowed
         * pool slot, then hands the cap off to the writer pool which
         * memcpys into the pre-allocated /dev/shm/ckpt_delta.img mmap.
         * Read and write are concurrent, so wall time = max(read, write)
         * instead of read + write. */
        capture_pool_t    delta_pool;
        baseline_stream_t delta_stream;
        double            dstream_timings[3] = {0};

        NvU64 max_region_len = 0;
        for (NvU32 i = 0; i < n_regions; i++)
            if (regions[i].len > max_region_len) max_region_len = regions[i].len;
        NvU64 max_region_pages = max_region_len / 4096;

        int n_delta_writers = baseline_stream_resolve_workers();
        int n_delta_slots   = n_delta_writers * 2;

        if (capture_pool_init(&delta_pool, n_delta_slots,
                              max_region_len, max_region_pages) < 0) {
            fprintf(stderr, "✗ delta capture_pool_init failed\n");
            free(regions); return 1;
        }

        /* Pack region sizes for worst-case file sizing. */
        NvU64 *region_sizes = malloc(n_regions * sizeof(NvU64));
        if (!region_sizes) {
            fprintf(stderr, "malloc region_sizes\n");
            capture_pool_destroy(&delta_pool); free(regions); return 1;
        }
        for (NvU32 i = 0; i < n_regions; i++) region_sizes[i] = regions[i].len;

        if (baseline_stream_start(delta_path, n_regions, region_sizes,
                                  CKPT_V3_VERSION, &delta_pool,
                                  /*is_delta=*/1,
                                  /*delta_alloc_start_va=*/user_va,
                                  /*delta_alloc_size=*/user_size,
                                  /*delta_num_dirty_64k=*/num_dirty,
                                  &delta_stream) < 0) {
            free(region_sizes);
            capture_pool_destroy(&delta_pool); free(regions); return 1;
        }

        for (NvU32 i = 0; i < n_regions && !err && !delta_stream.shared_err; i++) {
            int slot_idx = capture_pool_acquire(&delta_pool);

            range_capture_t *cap = calloc(1, sizeof(*cap));
            if (!cap) {
                fprintf(stderr, "calloc delta cap\n");
                capture_pool_release(&delta_pool, slot_idx);
                err = 1; break;
            }

            if (capture_range_encrypted(uvm_fd, regions[i].start, regions[i].len,
                                         cap, &delta_pool.slots[slot_idx],
                                         &delta_read_ms) < 0) {
                free(cap);
                capture_pool_release(&delta_pool, slot_idx);
                err = 1; break;
            }

            /* CPU-page encryption: vLLM's dirty pages are GPU-resident in
             * steady state, so cap->cpu_bytes is 0 in practice and the
             * loop below is skipped. If a workload ever has cpu_bytes > 0
             * we'd need to encrypt before enqueue (writer thread copies
             * cap->cpu_buf into the image), but we can't realloc the pool
             * slot's crypto_meta safely — punt with a warning. */
            if (cap->cpu_bytes > 0) {
                static int warned_cpu_pages = 0;
                if (!warned_cpu_pages) {
                    warned_cpu_pages = 1;
                    fprintf(stderr,
                            "  [delta] WARNING: region %u has %llu CPU bytes "
                            "but streaming path skips CPU k3 encryption — "
                            "restore will see plaintext.\n",
                            i, (unsigned long long)cap->cpu_bytes);
                }
            }

            baseline_stream_enqueue(&delta_stream, i, cap, slot_idx);
        }

        if (baseline_stream_join(&delta_stream, dstream_timings) < 0)
            err = 1;

        /* dstream_timings: [0]=capture wall, [1]=drain wall, [2]=pipeline tot */
        delta_write_ms = dstream_timings[2];

        capture_pool_destroy(&delta_pool);
        free(region_sizes);
        free(regions);

        if (err) { fprintf(stderr, "✗ Delta failed\n"); return 1; }
    } else {
        /* ---- Legacy buffered path (plaintext / encrypt-k3 / empty) ---- */
        range_capture_t *delta_caps = calloc(n_regions ? n_regions : 1,
                                             sizeof(range_capture_t));
        if (!delta_caps) {
            fprintf(stderr, "calloc delta_caps\n"); free(regions); return 1;
        }

        for (NvU32 i = 0; i < n_regions && !err; i++) {
            if (use_encrypted_ioctl) {
                if (capture_range_encrypted(uvm_fd, regions[i].start, regions[i].len,
                                             &delta_caps[i], NULL, &delta_read_ms) < 0)
                    err = 1;
            } else {
                if (capture_range(uvm_fd, regions[i].start, regions[i].len,
                                  &delta_caps[i], &delta_read_ms) < 0)
                    err = 1;
            }
        }
        free(regions);

        if (!err && encrypt_k3) {
            printf("  [k3] Encrypting delta captures (%u region(s), %.2f MB) ...\n",
                   n_regions, delta_total_bytes / (1024.0 * 1024.0));
            if (encrypt_captures_batch(delta_caps, n_regions, k3_key, &k3_iv_counter,
                                       encrypt_threads, CKPT_CRYPTO_PAGE_SIZE,
                                       &delta_encrypt_ms) < 0)
                err = 1;
            if (!err)
                printf("  [k3] Delta encrypted (%.1f ms, %.1f GB/s)\n\n",
                       delta_encrypt_ms,
                       delta_encrypt_ms > 0
                           ? delta_total_bytes / (1024.0*1024.0*1024.0) / (delta_encrypt_ms/1000.0) : 0.0);
        } else if (!err && use_encrypted_ioctl) {
            printf("  [k3] Encrypting delta CPU pages ...\n");
            if (encrypt_cpu_pages_batch(delta_caps, n_regions, k3_key, &k3_iv_counter,
                                        encrypt_threads, &delta_encrypt_ms) < 0)
                err = 1;
            if (!err)
                printf("  [k3] Delta CPU pages encrypted (%.1f ms)\n\n", delta_encrypt_ms);
        }

        if (!err) {
            err = write_delta(delta_path, delta_caps, n_regions,
                              user_va, user_size, num_dirty, &delta_write_ms);
        }

        for (NvU32 i = 0; i < n_regions; i++)
            range_capture_free(&delta_caps[i]);
        free(delta_caps);

        if (err) { fprintf(stderr, "✗ Delta failed\n"); return 1; }
    }

delta_captured: ;
    double t_delta_done = now_ms();

    /* ---- [7b] Snapshot state files while app is frozen ----------- */
    /*
     * The app writes ckpt_train_state.json every step, so we must copy it
     * NOW (app is at AT_BOUNDARY) before RESUME lets training continue.
     * After RESUME the app overwrites the file to later steps, making the
     * state inconsistent with the GPU memory snapshot.
     */
    {
        const char *train_state_src = getenv("CKPT_TRAIN_STATE");
        if (!train_state_src) train_state_src = "/tmp/ckpt_train_state.json";

        const char *tensor_map_src = getenv("CKPT_TENSOR_MAP");
        if (!tensor_map_src) tensor_map_src = "/tmp/ckpt_tensor_map.json";

        printf("[7b] Snapshot state files → %s ...\n", out_dir);
        snapshot_file(train_state_src, out_dir);
        snapshot_file(tensor_map_src, out_dir);
        snapshot_file(gate_path, out_dir);

        const char *engine_state_src = getenv("CKPT_ENGINE_STATE");
        if (!engine_state_src) engine_state_src = "/tmp/ckpt_engine_state.json";
        snapshot_file(engine_state_src, out_dir);
        printf("\n");
    }

    /* ---- [8] Finalize + Resume ---------------------------------- */
    /* Finalize BEFORE resume: clear tracking flags so the app doesn't
     * continue faulting on every GPU write after checkpoint completes. */
    printf("[8] FINALIZE (clear tracking) ...\n");
    {
        UVM_LIVE_MIGRATION_FINALIZE_PARAMS fp = {0};
        double t_fin0 = now_ms();
        int fin_rc = ioctl(uvm_fd, UVM_LIVE_MIGRATION_FINALIZE, &fp);
        double t_fin1 = now_ms();
        fprintf(stderr, "[ckpt-tim] FINALIZE wall=%.2f ms (rc=%d rm=0x%x)\n",
                t_fin1 - t_fin0, fin_rc, fp.rmStatus);
        if (fin_rc < 0 || fp.rmStatus != NV_OK) {
            fprintf(stderr, "FINALIZE: %s  status=0x%08x\n",
                    strerror(errno), fp.rmStatus);
        } else {
            printf("  ✓ Tracking disabled, 2M PTEs re-enabled\n");
        }
    }

    printf("[9] RESUME ...\n");
    __sync_synchronize();
    gate->ckpt_req = 0;
    gate->phase    = GATE_PHASE_RESUME;
    __sync_synchronize();
    double t_end = now_ms();
    printf("  ✓ App released\n");

    /* ---- Timing report ------------------------------------------ */
    printf("\n=== Checkpoint Timing ===\n");
    if (coalesce) {
        /* P3-B: sequential ioctl → mmap write, NOT pipelined.
         * The two times add up to the total baseline phase wall time. */
        printf("  [pre-copy] ioctl read (kernel) : %8.1f ms  (%.1f MB/s  %.2f MB)\n",
               base_read_ms,
               base_read_ms > 0
                   ? base_total_bytes / (1024.0*1024.0) / (base_read_ms/1000.0) : 0.0,
               base_total_bytes / (1024.0*1024.0));
        printf("  [pre-copy] mmap write (coalesce): %8.1f ms  (%.2f GB / %.2f GB/s)\n",
               base_write_ms,
               base_total_bytes / (1024.0*1024.0*1024.0),
               base_write_ms > 0
                   ? base_total_bytes / (1024.0*1024.0*1024.0) / (base_write_ms/1000.0) : 0.0);
        printf("  [pre-copy] baseline total      : %8.1f ms  (ioctl + write, sequential)\n",
               base_read_ms + base_write_ms);
    } else {
        /* Per-range pipelined path: the writer pool runs CONCURRENTLY with
         * the ioctl capture loop. base_read_ms is the accumulated per-call
         * ioctl time (informational sub-metric) and base_write_ms is the
         * total wall time of the capture+write pipeline — it ALREADY
         * INCLUDES the ioctl time, they overlap in real wall clock.
         * Do not add them to get total baseline time. */
        printf("  [pre-copy] ioctl time (in-pipe): %8.1f ms  (%.1f MB/s  %.2f MB) [sub-metric]\n",
               base_read_ms,
               base_read_ms > 0
                   ? base_total_bytes / (1024.0*1024.0) / (base_read_ms/1000.0) : 0.0,
               base_total_bytes / (1024.0*1024.0));
        printf("  [pre-copy] baseline wall       : %8.1f ms  (pipelined capture+write, %.2f GB / %.2f GB/s)\n",
               base_write_ms,
               base_total_bytes / (1024.0*1024.0*1024.0),
               base_write_ms > 0
                   ? base_total_bytes / (1024.0*1024.0*1024.0) / (base_write_ms/1000.0) : 0.0);
    }
    if (encrypt_k3)
        printf("  [pre-copy] baseline k3 enc     : %8.1f ms\n", base_encrypt_ms);
    printf("  [stop]     delta read       : %8.1f ms  (%.1f MB/s  %.2f MB)\n",
           delta_read_ms,
           delta_read_ms > 0
               ? delta_total_bytes / (1024.0*1024.0) / (delta_read_ms/1000.0) : 0.0,
           delta_total_bytes / (1024.0*1024.0));
    printf("  [stop]     delta write      : %8.1f ms\n", delta_write_ms);
    if (encrypt_k3)
        printf("  [stop]     delta k3 enc     : %8.1f ms\n", delta_encrypt_ms);
    printf("  stop time (AT_BOUNDARY → RESUME): %8.1f ms\n",
           t_delta_done - t_at_boundary);
    printf("  e2e (ckpt_req → RESUME)         : %8.1f ms\n",
           t_end - t_start);

    munmap(gate, GATE_FILE_SIZE);
    close(uvm_fd);
    printf("\n=== Agent done ===\n");
    return 0;
}
