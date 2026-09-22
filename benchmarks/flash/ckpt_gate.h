/**
 * ckpt_gate.h  –  Inter-kernel checkpoint synchronization gate
 *
 * Shared between ckpt_two_kernel_app.cu (target app) and
 * ckpt_two_kernel_agent.c (checkpoint agent).
 *
 * Protocol (agent-driven — agent requests, app responds):
 *
 *   App:    signal(SIGUSR1, handler)         ← install handler at startup
 *           gate_create(path, pid=getpid())  ← register; agent reads our PID
 *           run K1
 *           cudaStreamSynchronize(0)         ← GPU fully idle after K1
 *           if (g_ckpt_requested):
 *               gate->phase = AT_BOUNDARY    ← tell agent: safe to ckpt now
 *               spin until phase == RESUME   ← block until agent is done
 *           run K2
 *           gate->phase = DONE
 *
 *   Agent:  wait for gate file to appear     ← created by app on startup
 *           kill(app_pid, SIGUSR1)           ← request checkpoint (app responds
 *                                               at the next kernel boundary)
 *           spin until phase == AT_BOUNDARY  ← wait for GPU-idle confirmation
 *           GET_VA_RANGES → MIGRATE_TO_CPU → READ_CPU_PAGES  (ioctls 106/107/108)
 *           write image to disk
 *           gate->phase = RESUME             ← release app to continue
 *           spin until phase == DONE         ← e2e timing only
 *
 * Key property: the app never decides WHEN to checkpoint.  It only
 * responds to the agent's SIGUSR1 at the next safe kernel boundary.
 * This mirrors how real CR systems (CRIU, DMTCP) work.
 *
 * Memory barrier discipline:
 *   Writers use __sync_synchronize() before and after writing phase.
 *   Readers spin on the volatile field; compiler cannot cache it.
 */

#pragma once

#include <stdint.h>
#include <sys/types.h>

/* Phase values -------------------------------------------------------- */
#define GATE_PHASE_INVALID     0u   /* uninitialized / stale gate file     */
#define GATE_PHASE_INIT        1u   /* gate created; app registered        */
#define GATE_PHASE_AT_BOUNDARY 2u   /* GPU idle; app quiesced; ckpt now    */
#define GATE_PHASE_RESUME      3u   /* checkpoint saved; app may continue  */
#define GATE_PHASE_DONE        4u   /* app finished all kernels            */

/* File path and size -------------------------------------------------- */
#define GATE_FILE_DEFAULT  "/tmp/ckpt_gate"
#define GATE_FILE_SIZE     262144 /* 256 KB — gate + alloc table + dirty range table */

/* Agent config flags (set by agent before app starts) */
#define CKPT_CFG_SAVE_WEIGHTS   0x01u  /* save model weights at H2D time with k3 */
#define CKPT_CFG_ENCRYPTED      0x02u  /* use v3 encrypted mode (ioctl 111) */
#define CKPT_CFG_RESTORE        0x04u  /* restore mode: load checkpoint into new allocs */
#define CKPT_CFG_BASELINE       0x08u  /* in-app cudaMemcpyAsync baseline (no kernel mods) */
#define CKPT_CFG_PRECOPY_BASELINE 0x10u /* in-app precopy: concurrent baseline + in-app delta */

/* Shared gate struct (lives in the mmap'd file) ----------------------- */
typedef struct {
    volatile uint32_t phase;           /* GATE_PHASE_*                  offset  0 */
    uint32_t          config_flags;    /* CKPT_CFG_* set by agent       offset  4 */
    int32_t           app_pid;         /* set by app; read by agent     offset  8 */
    volatile uint32_t ckpt_req;        /* agent sets 1 to request ckpt  offset 12 */
    uint64_t          buf_bytes;       /* allocation size in bytes      offset 16 */
    uint64_t          ckpt_seq;        /* incremented per checkpoint    offset 24 */
    uint64_t          user_va;         /* VA of main managed allocation offset 32 */
    uint64_t          user_size;       /* bytes of main managed alloc   offset 40 */
    volatile uint32_t hash_req;        /* agent sets 1 to request hash  offset 48 */
    volatile uint32_t hash_done;       /* intercept sets 1 when done    offset 52 */
    /* In-app cudaMemcpyAsync baseline (P1) ----------------------------- */
    volatile uint32_t baseline_req;    /* agent → app: start baseline   offset 56 */
    volatile uint32_t baseline_done;   /* app → agent: 1 when finished  offset 60 */
    volatile uint32_t baseline_err;    /* app → agent: !=0 on failure   offset 64 */
    uint32_t          baseline_concurrent; /* agent → app: 1=precopy, 0=stop&copy  offset 68 */
    double            baseline_ms;     /* app → agent: elapsed ms       offset 72 */
    uint64_t          baseline_bytes;  /* app → agent: bytes copied     offset 80 */
    char              baseline_path[256]; /* agent → app: out file path  offset 88 */
    /* In-app delta (P1e) ----------------------------------------------- */
    volatile uint32_t delta_req;       /* agent → app: start delta     offset 344 */
    volatile uint32_t delta_done;      /* app → agent: 1 when finished offset 348 */
    volatile uint32_t delta_err;       /* app → agent: !=0 on failure  offset 352 */
    uint32_t          baseline_mstreams; /* agent → app: P1f stream count  offset 356
                                            * (0 or 1 = single-stream default;
                                            *  2..8 = multi-stream slot-bound) */
    double            delta_ms;        /* app → agent: elapsed ms      offset 360 */
    uint64_t          delta_bytes;     /* app → agent: bytes encrypted offset 368 */
    char              delta_path[256]; /* agent → app: out delta path  offset 376 */
} ckpt_gate_t;                    /* 632 bytes total                         */

/*
 * Allocation table — follows ckpt_gate_t at offset 56 inside the gate file.
 * libckpt_intercept.so appends one entry per cudaMallocManaged call so the
 * agent can keep ALL user buffers regardless of size order.
 *
 * Layout inside the 64KB gate file:
 *   [0 .. 55]   ckpt_gate_t
 *   [56 .. 63]  ckpt_alloc_hdr_t   (count + padding)
 *   [64 .. 63 + count*24]  ckpt_alloc_entry_t[]
 *
 * Maximum entries: (65536 - 64) / 24 = 2728
 */
#define GATE_ALLOC_MAX   2715u
#define GATE_ALLOC_HDR_OFF  512u  /* byte offset of ckpt_alloc_hdr_t in gate file
                                   * (bumped from 56 to leave room for ckpt_gate_t
                                   * growth — current ckpt_gate_t is 344 bytes) */

typedef struct {
    uint32_t count;    /* number of valid entries written so far */
    uint32_t _pad;     /* reserved                               */
} ckpt_alloc_hdr_t;   /* 8 bytes at offset 56                   */

/* Per-allocation flags */
#define CKPT_ALLOC_FLAG_H2D    0x01u  /* received cudaMemcpy H2D (has host data) */
#define CKPT_ALLOC_FLAG_FREED  0x02u  /* cudaFree called — entry is stale, skip */

typedef struct {
    uint64_t va;    /* base VA returned by cudaMallocManaged */
    uint64_t size;  /* size in bytes                        */
    uint32_t flags; /* CKPT_ALLOC_FLAG_* */
    uint32_t _pad;
} ckpt_alloc_entry_t; /* 24 bytes each, starting at offset 56 */

/* Convenience accessors given a ckpt_gate_t pointer                  */
#define GATE_ALLOC_HDR(g)    ((ckpt_alloc_hdr_t   *)((char *)(g) + GATE_ALLOC_HDR_OFF))
#define GATE_ALLOC_TABLE(g)  ((ckpt_alloc_entry_t *)((char *)(g) + GATE_ALLOC_HDR_OFF + sizeof(ckpt_alloc_hdr_t)))

/*
 * Dirty range table — agent → app transport for P1e precopy delta phase.
 *
 * Layout inside the 256 KB gate file:
 *   [0 ..  631]  ckpt_gate_t
 *   [ 512 ..]    alloc header + alloc_entry[]  (app → agent)
 *   [131072..]   dirty header + dirty_range[]  (agent → app)
 *
 * Reserved region [131072, 262144) = 128 KB = up to 8187 dirty 2 MB ranges,
 * which covers ~16 TB of dirty memory — far beyond what the tracker will
 * ever report in practice.
 */
#define GATE_DIRTY_HDR_OFF      131072u
#define GATE_DIRTY_MAX          8187u

typedef struct {
    uint32_t count;
    uint32_t _pad;
} ckpt_dirty_hdr_t;   /* 8 bytes */

typedef struct {
    uint64_t va;       /* 2 MB-aligned GPU VA */
    uint64_t size;     /* multiple of 2 MB    */
} ckpt_dirty_range_t; /* 16 bytes */

#define GATE_DIRTY_HDR(g)   ((ckpt_dirty_hdr_t   *)((char *)(g) + GATE_DIRTY_HDR_OFF))
#define GATE_DIRTY_TABLE(g) ((ckpt_dirty_range_t *)((char *)(g) + GATE_DIRTY_HDR_OFF + sizeof(ckpt_dirty_hdr_t)))

