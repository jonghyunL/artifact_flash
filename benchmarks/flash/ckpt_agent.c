/**
 * ckpt_agent.c — Adaptive checkpoint agent with page classification
 *
 * The app starts first with LD_PRELOAD=./libckpt_vllm.so, which creates
 * the gate file. This agent opens the existing gate, waits for the app
 * to initialize, then triggers checkpoints.
 *
 * Modes:
 *   --interval 0       : checkpoint once, then exit (default)
 *   --interval N        : checkpoint every N seconds
 *   --learn N           : use first N rounds to classify pages as STATIC/DYNAMIC
 *                         at 2MB granularity (default: 3). After learning,
 *                         subsequent checkpoints skip PREPARE_ALL and only
 *                         copy DYNAMIC pages (stun-copy).
 *   --learn 0           : disable adaptive mode, always use precopy
 *
 * During the learning phase (rounds 1..N), ckpt_core runs with --dump-dirty
 * to record which 64KB pages were dirtied. The agent aggregates these into
 * a 2MB bitmap and classifies:
 *   STATIC  = never dirty across all learning rounds (e.g. model weights)
 *   DYNAMIC = dirty in at least one round (e.g. KV cache, activations)
 *
 * After classification, subsequent rounds call ckpt_core with --stun-copy
 * --ranges-file, which skips PREPARE_ALL entirely — no write-protection,
 * no page fault storms, zero overhead on the running application.
 *
 * Usage:
 *   # Terminal 1: start app
 *   LD_PRELOAD=./libckpt_vllm.so python run_vllm.py --model ... --text-only ...
 *
 *   # Terminal 2: adaptive checkpoint
 *   sudo ./ckpt_agent --gate /tmp/ckpt_gate --interval 10 --learn 3
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>

#include "ckpt_gate.h"

/* ------------------------------------------------------------------ */
/* Checkpoint mode selector                                            */
/*                                                                     */
/* Each mode maps to a specific ckpt_core invocation (see build_cmd):  */
/*   p1d        — plain stop-and-copy via cudaMemcpyAsync              */
/*                (default ckpt_core, no flag)                         */
/*   p1e        — concurrent precopy + delta via cudaMemcpyAsync       */
/*                (--precopy-baseline)                                  */
/*   v3         — encrypted precopy + delta via ioctl 115              */
/*                (--encrypted)                                         */
/*   v3-static  — v3 with --static-img split (one-time static.img +    */
/*                per-round delta of mutable state)                     */
/* ------------------------------------------------------------------ */
typedef enum {
    MODE_P1D       = 0,
    MODE_P1E       = 1,
    MODE_V3        = 2,
    MODE_V3_STATIC = 3,
} ckpt_mode_t;

static const char *mode_name(ckpt_mode_t m)
{
    switch (m) {
        case MODE_P1D:       return "p1d";
        case MODE_P1E:       return "p1e";
        case MODE_V3:        return "v3";
        case MODE_V3_STATIC: return "v3-static";
    }
    return "?";
}

static int parse_mode(const char *s, ckpt_mode_t *out)
{
    if (!strcmp(s, "p1d"))       { *out = MODE_P1D;       return 0; }
    if (!strcmp(s, "p1e"))       { *out = MODE_P1E;       return 0; }
    if (!strcmp(s, "v3"))        { *out = MODE_V3;        return 0; }
    if (!strcmp(s, "v3-static")) { *out = MODE_V3_STATIC; return 0; }
    return -1;
}

#define PAGE_2M   (2ULL * 1024 * 1024)
#define PAGE_64K  (64ULL * 1024)
#define PAGES_PER_2M  (PAGE_2M / PAGE_64K)  /* 32 */

/* Maximum 2MB pages we track (80GB / 2MB = 40960) */
#define MAX_2M_PAGES  65536

static volatile int g_running = 1;

static void sig_handler(int sig) { (void)sig; g_running = 0; }

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* 2MB page classification bitmap                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t va;         /* 2MB-aligned VA */
    uint32_t dirty_count; /* number of rounds this page was dirty */
    uint32_t total_rounds;
} page_class_entry_t;

typedef struct {
    page_class_entry_t pages[MAX_2M_PAGES];
    uint32_t count;
    uint32_t rounds_completed;
} page_classifier_t;

static void classifier_init(page_classifier_t *pc)
{
    memset(pc, 0, sizeof(*pc));
}

/* Find or insert a 2MB page entry */
static page_class_entry_t *classifier_get(page_classifier_t *pc, uint64_t va_2m)
{
    for (uint32_t i = 0; i < pc->count; i++) {
        if (pc->pages[i].va == va_2m)
            return &pc->pages[i];
    }
    if (pc->count >= MAX_2M_PAGES) return NULL;
    page_class_entry_t *e = &pc->pages[pc->count++];
    e->va = va_2m;
    e->dirty_count = 0;
    e->total_rounds = 0;
    return e;
}

/**
 * Read dirty addresses dumped by ckpt_core --dump-dirty and update
 * the 2MB classification bitmap.
 */
static int classifier_update(page_classifier_t *pc, const char *dump_path)
{
    FILE *f = fopen(dump_path, "rb");
    if (!f) {
        fprintf(stderr, "  WARN: cannot read dirty dump %s: %s\n",
                dump_path, strerror(errno));
        return -1;
    }

    uint64_t num_dirty = 0;
    if (fread(&num_dirty, sizeof(uint64_t), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    /* Track which 2MB pages were dirty THIS round (avoid double-counting) */
    uint64_t seen_2m[MAX_2M_PAGES];
    uint32_t n_seen = 0;

    for (uint64_t i = 0; i < num_dirty; i++) {
        uint64_t addr;
        if (fread(&addr, sizeof(uint64_t), 1, f) != 1) break;

        uint64_t va_2m = addr & ~(PAGE_2M - 1);

        /* Check if we already counted this 2MB page this round */
        int already = 0;
        for (uint32_t j = 0; j < n_seen; j++) {
            if (seen_2m[j] == va_2m) { already = 1; break; }
        }
        if (already) continue;
        if (n_seen < MAX_2M_PAGES) seen_2m[n_seen++] = va_2m;

        page_class_entry_t *e = classifier_get(pc, va_2m);
        if (e) e->dirty_count++;
    }

    fclose(f);
    pc->rounds_completed++;

    /* Update total_rounds for all known pages */
    for (uint32_t i = 0; i < pc->count; i++)
        pc->pages[i].total_rounds = pc->rounds_completed;

    return 0;
}

/**
 * Write dynamic (always-dirty) ranges to a binary file for ckpt_core --ranges-file.
 * Format: [uint32_t count][(uint64_t va, uint64_t size) x count]
 *
 * Also includes any allocation ranges that were never seen in dirty tracking
 * but are NOT marked H2D (i.e. GPU-only allocations like KV cache that may
 * not have been dirtied during the specific learning windows but are dynamic
 * by nature).
 *
 * Returns number of dynamic ranges written.
 */
static uint32_t classifier_write_ranges(page_classifier_t *pc,
                                         const char *out_path,
                                         ckpt_alloc_entry_t *alloc_table,
                                         uint32_t n_allocs)
{
    /* Collect all DYNAMIC 2MB page VAs (dirty at least once) */
    /* Sort them to merge contiguous ranges */
    uint64_t *dyn_vas = malloc(pc->count * sizeof(uint64_t));
    uint32_t n_dyn = 0;

    for (uint32_t i = 0; i < pc->count; i++) {
        if (pc->pages[i].dirty_count > 0)
            dyn_vas[n_dyn++] = pc->pages[i].va;
    }

    /* Also add GPU-only alloc ranges that might not have appeared in dirty tracking
     * (e.g. allocations made after PREPARE_ALL, or pages that were only read) */
    for (uint32_t a = 0; a < n_allocs; a++) {
        if (alloc_table[a].flags & CKPT_ALLOC_FLAG_H2D)
            continue; /* Skip weights — STATIC */

        /* Add all 2MB pages in this GPU-only allocation */
        uint64_t start = alloc_table[a].va & ~(PAGE_2M - 1);
        uint64_t end = alloc_table[a].va + alloc_table[a].size;
        for (uint64_t va = start; va < end; va += PAGE_2M) {
            /* Check if already in dyn_vas */
            int found = 0;
            for (uint32_t j = 0; j < n_dyn; j++) {
                if (dyn_vas[j] == va) { found = 1; break; }
            }
            if (!found && n_dyn < pc->count + MAX_2M_PAGES) {
                dyn_vas = realloc(dyn_vas, (n_dyn + 1) * sizeof(uint64_t));
                dyn_vas[n_dyn++] = va;
            }
        }
    }

    /* Sort */
    for (uint32_t i = 1; i < n_dyn; i++) {
        uint64_t key = dyn_vas[i];
        int j = i - 1;
        while (j >= 0 && dyn_vas[j] > key) {
            dyn_vas[j + 1] = dyn_vas[j];
            j--;
        }
        dyn_vas[j + 1] = key;
    }

    /* Merge contiguous 2MB pages into ranges */
    typedef struct { uint64_t va; uint64_t size; } range_t;
    range_t *ranges = malloc((n_dyn + 1) * sizeof(range_t));
    uint32_t n_ranges = 0;

    if (n_dyn > 0) {
        uint64_t r_start = dyn_vas[0];
        uint64_t r_end = dyn_vas[0] + PAGE_2M;
        for (uint32_t i = 1; i < n_dyn; i++) {
            if (dyn_vas[i] == r_end) {
                r_end += PAGE_2M;
            } else {
                ranges[n_ranges++] = (range_t){r_start, r_end - r_start};
                r_start = dyn_vas[i];
                r_end = dyn_vas[i] + PAGE_2M;
            }
        }
        ranges[n_ranges++] = (range_t){r_start, r_end - r_start};
    }
    free(dyn_vas);

    /* Write to file */
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot write ranges file %s: %s\n",
                out_path, strerror(errno));
        free(ranges);
        return 0;
    }
    fwrite(&n_ranges, sizeof(uint32_t), 1, f);
    for (uint32_t i = 0; i < n_ranges; i++) {
        fwrite(&ranges[i].va, sizeof(uint64_t), 1, f);
        fwrite(&ranges[i].size, sizeof(uint64_t), 1, f);
    }
    fclose(f);
    free(ranges);

    return n_ranges;
}

static void classifier_print(page_classifier_t *pc,
                               ckpt_alloc_entry_t *alloc_table,
                               uint32_t n_allocs)
{
    uint32_t n_static = 0, n_dynamic = 0;
    uint64_t static_bytes = 0, dynamic_bytes = 0;

    for (uint32_t i = 0; i < pc->count; i++) {
        if (pc->pages[i].dirty_count == 0) {
            n_static++;
            static_bytes += PAGE_2M;
        } else {
            n_dynamic++;
            dynamic_bytes += PAGE_2M;
        }
    }

    /* Count H2D (weight) bytes and GPU-only (KV cache) bytes from alloc table */
    uint64_t h2d_bytes = 0, gpu_only_bytes = 0;
    for (uint32_t a = 0; a < n_allocs; a++) {
        if (alloc_table[a].flags & CKPT_ALLOC_FLAG_H2D)
            h2d_bytes += alloc_table[a].size;
        else
            gpu_only_bytes += alloc_table[a].size;
    }

    printf("\n=== Page Classification (2MB granularity) ===\n");
    printf("  Learning rounds : %u\n", pc->rounds_completed);
    printf("  Total 2MB pages : %u (%.2f GB)\n",
           pc->count, pc->count * PAGE_2M / (1024.0 * 1024.0 * 1024.0));
    printf("  STATIC pages    : %u (%.2f GB) — never dirty (model weights)\n",
           n_static, static_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  DYNAMIC pages   : %u (%.2f GB) — dirty at least once (KV cache/activations)\n",
           n_dynamic, dynamic_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  H2D allocs      : %.2f GB (weights loaded from host)\n",
           h2d_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  GPU-only allocs : %.2f GB (KV cache, activations)\n",
           gpu_only_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Savings         : skip %.2f GB STATIC data per checkpoint\n",
           static_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Overhead        : ZERO after learning (no PREPARE_ALL, no page faults)\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* /dev/shm size + prealloc helpers                                    */
/* ------------------------------------------------------------------ */

/* Returns size of /dev/shm in GiB, or 0 on failure. */
static uint64_t shm_size_gb(void)
{
    struct statvfs st;
    if (statvfs("/dev/shm", &st) != 0) return 0;
    uint64_t bytes = (uint64_t)st.f_blocks * (uint64_t)st.f_frsize;
    return bytes / (1024ULL * 1024ULL * 1024ULL);
}

/* If /dev/shm < want_gb, attempt `mount -o remount,size=<remount_gb>G`.
 * Caller must be root. Non-fatal on failure (caller may still succeed). */
static void ensure_shm_size(uint64_t want_gb, uint64_t remount_gb)
{
    uint64_t cur = shm_size_gb();
    if (cur >= want_gb) {
        printf("[shm] /dev/shm = %lu GB (>= %lu GB needed)\n",
               (unsigned long)cur, (unsigned long)want_gb);
        return;
    }
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mount -o remount,size=%luG /dev/shm",
             (unsigned long)remount_gb);
    printf("[shm] /dev/shm = %lu GB < %lu GB needed; remounting to %lu GB\n",
           (unsigned long)cur, (unsigned long)want_gb,
           (unsigned long)remount_gb);
    printf("[shm] running: %s\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[shm] WARNING: remount failed (exit=%d). "
                        "If you're not root, run this manually first.\n", ret);
    } else {
        printf("[shm] /dev/shm now %lu GB\n", (unsigned long)shm_size_gb());
    }
}

/* Run ckpt_prealloc if any of the required images are missing. The shm
 * staging file (/dev/shm/ckpt_core_gpu_staging) is auto-created by
 * ckpt_prealloc whenever it's invoked, so we only need to ask for it
 * via --gpu-size-gb. The delta image is preallocated to base_size_gb
 * (worst case = full GPU footprint dirty between checkpoints). */
static int ensure_prealloc(const char *base_path,
                           const char *delta_path,
                           const char *static_path,  /* NULL if unused */
                           int gpu_size_gb,
                           int base_size_gb,
                           int static_size_gb)
{
    struct stat st;
    int need_staging = (stat("/dev/shm/ckpt_core_gpu_staging", &st) != 0);
    int need_base    = (base_path  && stat(base_path,  &st) != 0);
    int need_delta   = (delta_path && stat(delta_path, &st) != 0);
    int need_static  = (static_path && stat(static_path, &st) != 0);

    if (!need_staging && !need_base && !need_delta && !need_static) {
        printf("[prealloc] all artifacts present; skipping ckpt_prealloc\n");
        return 0;
    }

    /* Use --file PATH:GB (modern flag, supports multiple) instead of
     * --out/--out-size-gb which only takes a single path. */
    char cmd[1536];
    int  pos = 0;
    pos += snprintf(cmd + pos, sizeof(cmd) - pos,
                    "./ckpt_prealloc --gpu-size-gb %d", gpu_size_gb);
    if (need_base)
        pos += snprintf(cmd + pos, sizeof(cmd) - pos,
                        " --file %s:%d", base_path, base_size_gb);
    if (need_delta)
        pos += snprintf(cmd + pos, sizeof(cmd) - pos,
                        " --file %s:%d", delta_path, base_size_gb);
    if (need_static)
        pos += snprintf(cmd + pos, sizeof(cmd) - pos,
                        " --file %s:%d", static_path, static_size_gb);

    printf("[prealloc] missing: staging=%d base=%d delta=%d static=%d\n",
           need_staging, need_base, need_delta, need_static);
    printf("[prealloc] running: %s\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[prealloc] FAILED (exit=%d)\n", ret);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-round metrics collection                                        */
/*                                                                     */
/* Parse ckpt_core stdout for timing/size lines and stash one row per  */
/* round. Final CSV is printed at end-of-run for spreadsheet copy.    */
/* ------------------------------------------------------------------ */
typedef struct {
    int    round;
    double baseline_wall_ms;
    double baseline_gb;
    double baseline_gbps;
    double delta_mb;
    double stop_time_ms;
    double e2e_ms;
    int    success;
} round_metrics_t;

#define MAX_METRIC_ROUNDS 1024
static round_metrics_t g_metrics[MAX_METRIC_ROUNDS];
static int             g_n_metrics = 0;

/* Run ckpt_core via popen, echo every line, and scrape known timing
 * lines into `m`. Returns the system()-style exit status. */
static int run_and_capture(const char *cmd, round_metrics_t *m)
{
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[capture] popen failed: %s\n", strerror(errno));
        return -1;
    }
    char line[2048];
    while (fgets(line, sizeof(line), fp)) {
        fputs(line, stdout);
        fflush(stdout);
        /* Match the printf formats produced by ckpt_core / ckpt_baseline.
         * Lines we care about (samples):
         *   "  [pre-copy] baseline wall       :   8128.7 ms  (pipelined capture+write, 69.60 GB / 8.56 GB/s)"
         *   "  [stop]     delta read       :    131.7 ms  (86.8 MB/s  11.44 MB)"
         *   "  stop time (AT_BOUNDARY → RESUME):    153.6 ms"
         *   "  e2e (ckpt_req → RESUME)         :   8488.4 ms"
         *   "  ✓ Baseline: 16445.2 ms (app-reported 16444.4 ms, 69.60 GB)"   (p1e/p1d in-app banner)
         *   "      pipeline tot :   5477.3 ms  (53.56 GB / 9.8 GB/s)"        (v3-static streaming)
         */
        const char *p;
        double ms, gb, gbps, mb;

        if ((p = strstr(line, "baseline wall"))) {
            if (sscanf(p, "baseline wall %*[^:]: %lf ms", &ms) >= 1)
                m->baseline_wall_ms = ms;
            const char *q = strstr(line, "GB / ");
            if (q && sscanf(q, "GB / %lf GB/s", &gbps) == 1)
                m->baseline_gbps = gbps;
            q = strchr(line, ',');
            if (q && sscanf(q, ", %lf GB", &gb) == 1)
                m->baseline_gb = gb;
        }
        else if ((p = strstr(line, "Baseline:"))) {
            /* In-app p1d/p1e banner. */
            if (sscanf(p, "Baseline: %lf ms", &ms) == 1)
                m->baseline_wall_ms = ms;
            const char *q = strchr(line, ',');
            if (q && sscanf(q, ", %lf GB", &gb) == 1) {
                m->baseline_gb = gb;
                if (ms > 0.0) m->baseline_gbps = gb / (ms / 1000.0);
            }
        }
        else if ((p = strstr(line, "pipeline tot"))) {
            /* Format: "      pipeline tot : 8128.7 ms  (66.06 GB / 12.63 GB/s)"
             * Only one space between "tot" and ":", so %*[^:] can't be used —
             * skip everything non-digit between "tot" and the number. */
            if (sscanf(p, "pipeline tot%*[^0-9]%lf ms", &ms) == 1) {
                if (m->baseline_wall_ms == 0.0) m->baseline_wall_ms = ms;
            }
            const char *q = strstr(line, "GB / ");
            if (q && sscanf(q, "GB / %lf GB/s", &gbps) == 1)
                if (m->baseline_gbps == 0.0) m->baseline_gbps = gbps;
            q = strchr(line, '(');
            if (q && sscanf(q, "(%lf GB", &gb) == 1)
                if (m->baseline_gb == 0.0) m->baseline_gb = gb;
        }
        else if ((p = strstr(line, "delta read"))) {
            const char *q = strrchr(line, '(');
            /* "(86.8 MB/s  11.44 MB)" — last MB number. */
            if (q && sscanf(q, "(%*f MB/s %lf MB", &mb) == 1)
                m->delta_mb = mb;
        }
        else if ((p = strstr(line, "stop time"))) {
            if (sscanf(p, "stop time%*[^:]: %lf ms", &ms) == 1)
                m->stop_time_ms = ms;
        }
        else if ((p = strstr(line, "e2e"))) {
            if (sscanf(p, "e2e%*[^:]: %lf ms", &ms) == 1)
                m->e2e_ms = ms;
        }
        else if (strstr(line, "delta]") && strstr(line, " MB total")) {
            /* p1e in-app delta banner: "[delta] 64 ranges, 142.00 MB total, ..." */
            const char *q = strchr(line, ',');
            if (q) q = strchr(q + 1, ' ');
            if (q && sscanf(q, " %lf MB", &mb) == 1)
                m->delta_mb = mb;
        }
    }
    int status = pclose(fp);
    return status;
}

static void emit_csv_summary(const char *mode_name_str)
{
    if (g_n_metrics == 0) return;
    printf("\n=== Per-round metrics (CSV) ===\n");
    printf("mode,round,baseline_wall_ms,baseline_gb,baseline_gbps,delta_mb,stop_time_ms,e2e_ms,ok\n");
    for (int i = 0; i < g_n_metrics; i++) {
        round_metrics_t *m = &g_metrics[i];
        printf("%s,%d,%.1f,%.2f,%.2f,%.2f,%.1f,%.1f,%d\n",
               mode_name_str, m->round,
               m->baseline_wall_ms, m->baseline_gb, m->baseline_gbps,
               m->delta_mb, m->stop_time_ms, m->e2e_ms, m->success);
    }
    /* Aggregate (success rounds only). */
    int    n   = 0;
    double sum_wall = 0, sum_gbps = 0, sum_e2e = 0, sum_delta = 0, sum_stop = 0;
    for (int i = 0; i < g_n_metrics; i++) {
        if (!g_metrics[i].success) continue;
        n++;
        sum_wall  += g_metrics[i].baseline_wall_ms;
        sum_gbps  += g_metrics[i].baseline_gbps;
        sum_e2e   += g_metrics[i].e2e_ms;
        sum_delta += g_metrics[i].delta_mb;
        sum_stop  += g_metrics[i].stop_time_ms;
    }
    if (n > 0) {
        printf("\n=== Aggregate (n=%d successful round(s)) ===\n", n);
        printf("  avg baseline_wall : %.1f ms\n", sum_wall  / n);
        printf("  avg baseline_gbps : %.2f GB/s\n", sum_gbps / n);
        printf("  avg delta size    : %.2f MB\n", sum_delta / n);
        printf("  avg stop_time     : %.1f ms\n", sum_stop  / n);
        printf("  avg e2e           : %.1f ms\n", sum_e2e   / n);
    }
}

/* ------------------------------------------------------------------ */
/* Make image paths writable by both root (agent path: ckpt_core) and  */
/* the non-root vLLM process (in-app path: ckpt_baseline_run/_delta).  */
/* Cross-mode runs hit EACCES because round-N (root) creates with 0644 */
/* and round-N+1 (vLLM as user) can't write to it.                     */
/* ------------------------------------------------------------------ */
static void make_writable(const char *path)
{
    if (!path) return;
    if (chmod(path, 0666) == 0) {
        printf("[perms] chmod 0666 %s\n", path);
    } else if (errno != ENOENT) {
        fprintf(stderr, "[perms] chmod %s: %s\n", path, strerror(errno));
    }
}

/* ------------------------------------------------------------------ */
/* Build the ckpt_core command line for a given mode                  */
/* ------------------------------------------------------------------ */
static void build_cmd(char *out, size_t cap, ckpt_mode_t mode,
                      const char *gate_path, const char *base_path,
                      const char *delta_path, const char *static_path,
                      int is_learning, int is_adaptive,
                      const char *dirty_dump_path,
                      const char *ranges_path,
                      int track_only, int track_only_budget_ms)
{
    int pos = 0;
    pos += snprintf(out + pos, cap - pos, "./ckpt_core");

    if (track_only) {
        pos += snprintf(out + pos, cap - pos,
                        " --track-only --track-only-budget-ms %d",
                        track_only_budget_ms);
    }

    /* Adaptive mode (post-learning): override mode-specific flags with
     * --stun-copy + --ranges-file so we only re-copy DYNAMIC ranges. */
    if (is_adaptive) {
        pos += snprintf(out + pos, cap - pos,
                        " --stun-copy --ranges-file %s", ranges_path);
        if (mode == MODE_V3 || mode == MODE_V3_STATIC)
            pos += snprintf(out + pos, cap - pos, " --encrypted");
    } else {
        if (is_learning)
            pos += snprintf(out + pos, cap - pos,
                            " --dump-dirty %s", dirty_dump_path);
        switch (mode) {
            case MODE_P1D:
                /* default ckpt_core, no flag — stop-and-copy in-app baseline */
                break;
            case MODE_P1E:
                pos += snprintf(out + pos, cap - pos, " --precopy-baseline");
                break;
            case MODE_V3:
                pos += snprintf(out + pos, cap - pos, " --encrypted");
                break;
            case MODE_V3_STATIC:
                pos += snprintf(out + pos, cap - pos,
                                " --encrypted --static-img %s",
                                static_path);
                break;
        }
    }

    snprintf(out + pos, cap - pos, " %s %s %s",
             gate_path, base_path, delta_path);
}

int main(int argc, char **argv)
{
    const char *gate_path   = GATE_FILE_DEFAULT;
    const char *base_path   = "/dev/shm/ckpt_base.img";
    const char *delta_path  = "/dev/shm/ckpt_delta.img";
    const char *static_path = "/dev/shm/ckpt_static.img";
    ckpt_mode_t mode        = MODE_V3;       /* default = v3 (most-tested) */
    int interval_s          = 0;
    int delay_s             = 0;             /* initial delay before round 1 */
    int max_rounds          = 0;             /* 0 = unlimited (loop until app exits) */
    int learn_rounds        = 0;             /* default OFF (was 3 — opt-in only) */
    int do_prealloc         = 1;             /* run ckpt_prealloc if needed */
    int do_remount          = 0;             /* remount /dev/shm if too small */
    int shm_need_gb         = 180;           /* threshold below which we remount */
    int shm_remount_gb      = 200;           /* target size when we do remount */
    int prealloc_gpu_gb     = 90;            /* --gpu-size-gb for ckpt_prealloc */
    int prealloc_base_gb    = 90;            /* base.img size */
    int prealloc_static_gb  = 17;            /* static.img size */
    /* Legacy backwards-compat flags (kept so old invocations still work).
     * They're translated to --mode after argv parsing. */
    int legacy_encrypted    = 0;
    int legacy_encrypt_k3   = 0;
    /* --track-only pass-through: forwards to ckpt_core to skip ioctl 115/116
     * D2H copy while keeping PREPARE_ALL/FINALIZE armed. Used to isolate
     * tracking cost (write-protect → fault servicing) from copy cost (CE/
     * bounce-buffer pressure). Image produced is INVALID. */
    int track_only          = 0;
    int track_only_budget_ms = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            if (parse_mode(argv[++i], &mode) < 0) {
                fprintf(stderr, "ERROR: unknown --mode '%s' "
                                "(want p1d|p1e|v3|v3-static)\n", argv[i]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--gate")       && i + 1 < argc) gate_path   = argv[++i];
        else if (!strcmp(argv[i], "--base")       && i + 1 < argc) base_path   = argv[++i];
        else if (!strcmp(argv[i], "--delta")      && i + 1 < argc) delta_path  = argv[++i];
        else if (!strcmp(argv[i], "--static-img") && i + 1 < argc) static_path = argv[++i];
        else if (!strcmp(argv[i], "--interval")   && i + 1 < argc) interval_s  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--delay")      && i + 1 < argc) delay_s     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds")     && i + 1 < argc) max_rounds  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--learn")      && i + 1 < argc) learn_rounds= atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-prealloc"))                do_prealloc = 0;
        else if (!strcmp(argv[i], "--auto-remount"))               do_remount  = 1;
        else if (!strcmp(argv[i], "--prealloc-gpu-gb")    && i + 1 < argc) prealloc_gpu_gb    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prealloc-base-gb")   && i + 1 < argc) prealloc_base_gb   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prealloc-static-gb") && i + 1 < argc) prealloc_static_gb = atoi(argv[++i]);
        /* Track-only experiment knobs (forwarded to ckpt_core). */
        else if (!strcmp(argv[i], "--track-only")) track_only = 1;
        else if (!strcmp(argv[i], "--track-only-budget-ms") && i + 1 < argc)
            track_only_budget_ms = atoi(argv[++i]);
        /* Legacy: translated to --mode after the loop. */
        else if (!strcmp(argv[i], "--encrypted"))    legacy_encrypted = 1;
        else if (!strcmp(argv[i], "--encrypt-k3"))   legacy_encrypt_k3 = 1;
        else {
            fprintf(stderr,
                "Usage: %s --mode {p1d|p1e|v3|v3-static} [opts]\n"
                "  --gate PATH         (default: %s)\n"
                "  --base PATH         (default: /dev/shm/ckpt_base.img)\n"
                "  --delta PATH        (default: /dev/shm/ckpt_delta.img)\n"
                "  --static-img PATH   (v3-static only; default: /dev/shm/ckpt_static.img)\n"
                "  --delay  SEC        initial sleep before first checkpoint (default: 0)\n"
                "  --interval SEC      seconds between checkpoints  (default: 0 = once)\n"
                "  --rounds N          stop after N checkpoints     (default: 0 = until app exits)\n"
                "  --learn N           use first N rounds to classify pages, then adaptive (default: 0)\n"
                "  --no-prealloc       skip ckpt_prealloc auto-run\n"
                "  --auto-remount      mount -o remount,size=200G /dev/shm if too small (needs root)\n"
                "  --prealloc-gpu-gb N      ckpt_prealloc --gpu-size-gb (default: 90)\n"
                "  --prealloc-base-gb N     ckpt_prealloc base.img size (default: 90)\n"
                "  --prealloc-static-gb N   ckpt_prealloc static.img size (default: 17)\n"
                "  --track-only             skip ioctl 115/116 D2H (image INVALID; TBT test only)\n"
                "  --track-only-budget-ms N usleep per skipped ioctl call (default: 0)\n"
                "Env passed to ckpt_core: CKPT_MT_THREADS, CKPT_WRITER_THREADS\n",
                argv[0], GATE_FILE_DEFAULT);
            return 1;
        }
    }

    /* Legacy --encrypted / --encrypt-k3 → mode mapping (only if --mode
     * wasn't explicitly given; default is MODE_V3 anyway, so this is a
     * no-op unless someone passes --encrypt-k3 alone for the old k3
     * baseline behavior). */
    if (legacy_encrypt_k3 && !legacy_encrypted) {
        /* k3 is the in-app CPU baseline (P1e variant). Map to p1e — both
         * are in-app cudaMemcpyAsync; ckpt_core will pick up k3 from
         * --encrypt-k3 once we re-add it via the legacy path. For now
         * map to p1e and warn so the caller knows. */
        fprintf(stderr,
                "[ckpt_agent] note: --encrypt-k3 is legacy; mapping to --mode p1e\n");
        mode = MODE_P1E;
    }
    (void)legacy_encrypted;  /* default mode is v3, so already covered */

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    printf("=== ckpt_agent (mode=%s) ===\n", mode_name(mode));
    printf("Gate          : %s\n", gate_path);
    printf("Base image    : %s\n", base_path);
    printf("Delta image   : %s\n", delta_path);
    if (mode == MODE_V3_STATIC)
        printf("Static image  : %s\n", static_path);
    printf("Interval      : %s\n", interval_s > 0 ? "" : "once");
    if (interval_s > 0)
        printf("                every %d s (Ctrl-C to stop)\n", interval_s);
    if (delay_s > 0)
        printf("Initial delay : %d s\n", delay_s);
    if (max_rounds > 0)
        printf("Max rounds    : %d\n", max_rounds);
    printf("Learn rounds  : %d%s\n",
           learn_rounds,
           learn_rounds == 0 ? " (off — every round runs the chosen mode)" : "");
    printf("Prealloc      : %s\n",
           do_prealloc ? "auto (run ckpt_prealloc if missing)" : "skipped");
    if (track_only)
        printf("Track-only    : ON — ioctl 115/116 SKIPPED, image INVALID. "
               "Budget=%d ms/skipped-call.\n", track_only_budget_ms);
    printf("\n");

    /* ---- Recommended thread defaults for v3 modes ----
     * ckpt_core's built-in defaults (CKPT_MT_THREADS=4, CKPT_WRITER_THREADS=4)
     * leave throughput on the table for ioctl-115-based v3. We bump MT to 8
     * here so the kernel-side encrypt pool saturates ~11 GB/s. setenv with
     * overwrite=0 means the caller can still override (e.g. CKPT_MT_THREADS=16
     * sudo ./ckpt_agent --mode v3). */
    if (mode == MODE_V3 || mode == MODE_V3_STATIC) {
        setenv("CKPT_MT_THREADS",     "8", 0);
        setenv("CKPT_WRITER_THREADS", "4", 0);
        printf("v3 env        : CKPT_MT_THREADS=%s CKPT_WRITER_THREADS=%s\n",
               getenv("CKPT_MT_THREADS"), getenv("CKPT_WRITER_THREADS"));
    }

    /* ---- /dev/shm size + ckpt_prealloc (run during initial delay) ---- */
    if (do_remount)
        ensure_shm_size((uint64_t)shm_need_gb, (uint64_t)shm_remount_gb);
    if (do_prealloc) {
        const char *static_for_prealloc =
            (mode == MODE_V3_STATIC) ? static_path : NULL;
        if (ensure_prealloc(base_path, delta_path, static_for_prealloc,
                            prealloc_gpu_gb, prealloc_base_gb,
                            prealloc_static_gb) < 0) {
            fprintf(stderr, "ERROR: ckpt_prealloc failed; aborting\n");
            return 1;
        }
    }
    /* Make image paths writable by both root and the non-root vLLM
     * process. Critical for p1d/p1e (in-app paths) when the same files
     * were previously created by root (e.g. by a v3 run). */
    make_writable(base_path);
    make_writable(delta_path);
    if (mode == MODE_V3_STATIC) make_writable(static_path);
    printf("\n");

    /* Open existing gate file (created by the app's LD_PRELOAD library) */
    printf("Waiting for gate file %s ...\n", gate_path);
    int fd = -1;
    while (fd < 0 && g_running) {
        fd = open(gate_path, O_RDWR);
        if (fd < 0) usleep(200000);
    }
    if (fd < 0) return 1;

    ckpt_gate_t *gate = mmap(NULL, GATE_FILE_SIZE, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, 0);
    close(fd);
    if (gate == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap: %s\n", strerror(errno));
        return 1;
    }

    printf("Gate file opened.\n");

    /* Wait for app to set app_pid */
    while (gate->app_pid == 0 && g_running) {
        __sync_synchronize();
        usleep(100000);
    }
    if (!g_running) { munmap(gate, GATE_FILE_SIZE); return 0; }

    printf("App connected: pid=%d\n", gate->app_pid);

    /* Wait for INIT phase */
    while (gate->phase < GATE_PHASE_INIT && g_running) {
        __sync_synchronize();
        usleep(100000);
    }
    if (!g_running) { munmap(gate, GATE_FILE_SIZE); return 0; }

    printf("App initialized (GATE_PHASE_INIT)\n\n");

    /* Print allocation summary */
    ckpt_alloc_hdr_t *hdr = GATE_ALLOC_HDR(gate);
    ckpt_alloc_entry_t *tbl = GATE_ALLOC_TABLE(gate);
    uint32_t n = hdr->count;
    uint32_t n_h2d = 0;
    uint64_t h2d_bytes = 0, total_bytes = 0;

    for (uint32_t i = 0; i < n; i++) {
        total_bytes += tbl[i].size;
        if (tbl[i].flags & CKPT_ALLOC_FLAG_H2D) {
            n_h2d++;
            h2d_bytes += tbl[i].size;
        }
    }

    printf("Allocations: %u total (%.2f GB)\n", n,
           total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  H2D (weights): %u allocs (%.2f GB)\n",
           n_h2d, h2d_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  GPU-only (KV): %u allocs (%.2f GB)\n\n",
           n - n_h2d, (total_bytes - h2d_bytes) / (1024.0 * 1024.0 * 1024.0));

    /* ---- Initialize classifier ---- */
    page_classifier_t *classifier = malloc(sizeof(page_classifier_t));
    classifier_init(classifier);

    const char *dirty_dump_path = "/tmp/ckpt_dirty_dump.bin";
    const char *ranges_path = "/tmp/ckpt_dynamic_ranges.bin";
    int classified = 0;  /* set to 1 after learning phase completes */

    /* ---- Initial delay before first checkpoint ---- */
    if (delay_s > 0 && g_running) {
        printf("Sleeping %d s before first checkpoint...\n", delay_s);
        for (int s = 0; s < delay_s && g_running; s++)
            sleep(1);
        printf("\n");
    }

    /* ---- Checkpoint loop ---- */
    int round = 0;
    int last_ret = 0;

    do {
        /* Check if app is still alive */
        if (gate->app_pid > 0 && kill(gate->app_pid, 0) != 0) {
            printf("App exited (pid=%d), stopping.\n", gate->app_pid);
            break;
        }
        if (max_rounds > 0 && round >= max_rounds) {
            printf("Reached --rounds %d limit, stopping.\n", max_rounds);
            break;
        }

        round++;
        int is_learning = (learn_rounds > 0 && round <= learn_rounds && !classified);
        int is_adaptive = (classified && learn_rounds > 0);

        if (is_learning)
            printf("--- Checkpoint #%d (LEARNING round %d/%d, base mode=%s) ---\n",
                   round, round, learn_rounds, mode_name(mode));
        else if (is_adaptive)
            printf("--- Checkpoint #%d (ADAPTIVE stun-copy on top of %s) ---\n",
                   round, mode_name(mode));
        else
            printf("--- Checkpoint #%d (mode=%s) ---\n", round, mode_name(mode));

        char cmd[2048];
        build_cmd(cmd, sizeof(cmd), mode,
                  gate_path, base_path, delta_path, static_path,
                  is_learning, is_adaptive,
                  dirty_dump_path, ranges_path,
                  track_only, track_only_budget_ms);
        printf("  $ %s\n", cmd);

        round_metrics_t *m = NULL;
        if (g_n_metrics < MAX_METRIC_ROUNDS) {
            m = &g_metrics[g_n_metrics++];
            memset(m, 0, sizeof(*m));
            m->round = round;
        }

        double t0 = now_ms();
        last_ret = m ? run_and_capture(cmd, m) : system(cmd);
        double ms = now_ms() - t0;
        if (m) m->success = (last_ret == 0);

        if (last_ret == 0)
            printf("  Checkpoint #%d complete: %.1f ms\n", round, ms);
        else
            printf("  Checkpoint #%d FAILED (exit=%d, %.1f ms)\n", round, last_ret, ms);

        /* Update classifier during learning phase */
        if (is_learning && last_ret == 0) {
            classifier_update(classifier, dirty_dump_path);
            printf("  Classifier updated (round %d/%d, %u 2MB pages tracked)\n",
                   round, learn_rounds, classifier->count);

            /* Check if learning is complete */
            if (round >= learn_rounds) {
                classifier_print(classifier, tbl, n);

                uint32_t n_ranges = classifier_write_ranges(
                    classifier, ranges_path, tbl, n);
                printf("  Dynamic ranges written to %s (%u ranges)\n",
                       ranges_path, n_ranges);
                printf("  >> Switching to ADAPTIVE stun-copy mode <<\n\n");
                classified = 1;
            }
        }

        /* Save gate snapshot (overwrite each round) */
        char snap_path[512];
        snprintf(snap_path, sizeof(snap_path), "/tmp/snap_ckpt_gate");
        int snap_fd = open(snap_path, O_CREAT | O_WRONLY | O_TRUNC, 0666);
        if (snap_fd >= 0) {
            write(snap_fd, gate, GATE_FILE_SIZE);
            close(snap_fd);
        }

        printf("\n");

        /* Wait for next interval */
        if (interval_s > 0 && g_running) {
            for (int s = 0; s < interval_s && g_running; s++)
                sleep(1);
        }

    } while (interval_s > 0 && g_running);

    /* ---- Restore timing emulation ---- */
    if (last_ret == 0) {
        printf("--- Restore Emulation ---\n");

        struct stat st_base, st_delta;
        if (stat(base_path, &st_base) == 0 && stat(delta_path, &st_delta) == 0) {
            uint64_t img_size = st_base.st_size + st_delta.st_size;

            double t_load = now_ms();
            int bfd = open(base_path, O_RDONLY);
            if (bfd >= 0) {
                char *buf = malloc(st_base.st_size);
                if (buf) { read(bfd, buf, st_base.st_size); free(buf); }
                close(bfd);
            }
            int dfd = open(delta_path, O_RDONLY);
            if (dfd >= 0) {
                char *buf = malloc(st_delta.st_size);
                if (buf) { read(dfd, buf, st_delta.st_size); free(buf); }
                close(dfd);
            }
            double load_ms = now_ms() - t_load;

            double gb = img_size / (1024.0 * 1024.0 * 1024.0);
            printf("  Image load (disk->host): %.2f GB in %.1f ms (%.1f GB/s)\n",
                   gb, load_ms, load_ms > 0 ? gb / (load_ms / 1000.0) : 0);

            double ce_bw_gbs = 7.8;
            double h2d_ms = (gb / ce_bw_gbs) * 1000.0;
            printf("  Estimated H2D restore : %.2f GB at %.1f GB/s = %.1f ms\n",
                   gb, ce_bw_gbs, h2d_ms);
            printf("  Estimated total restore: %.1f ms\n\n",
                   load_ms + h2d_ms);
        }
    }

    printf("Agent done (%d checkpoint(s), %d learning, %d adaptive).\n",
           round, learn_rounds > round ? round : learn_rounds,
           classified ? round - learn_rounds : 0);

    emit_csv_summary(mode_name(mode));

    free(classifier);
    munmap(gate, GATE_FILE_SIZE);
    return last_ret ? 1 : 0;
}
