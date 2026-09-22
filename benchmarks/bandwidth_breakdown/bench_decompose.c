/*
 * bench_decompose.c — cudaMemcpy vs kernel CE path, with CC stage breakdown.
 *
 * Sweeps 20/30/40/50 GB and reports, per size:
 *   1. cudaMemcpy D2H   (userspace baseline)
 *   2. ioctl 115        (production kernel path)
 *   3. ioctl 117        (instrumented fork — per-stage split)
 *
 * Methodology matches the earlier bench_transfer work: one discarded warmup
 * call per configuration, then N timed repeats, reported as median with
 * min/max. The prior 18 GB/s ioctl-115 ceiling was measured as "steady state,
 * excluding cold first run, 5 repeats" — single-shot numbers are not
 * comparable to it and swing up to 3x.
 *
 * ioctl 117 is a FORK of the 114/115 worker; the production paths carry no
 * timers. If 117's aggregate GB/s does not match 115's on the same transfer,
 * the breakdown does not describe the shipped path and must not be reported.
 * The tool prints "117 vs 115: +-x%" on the medians and flags >5%.
 *
 * PREPARE_ALL is ON by default. --no-prepare disables it, but those numbers
 * are not production-comparable.
 *
 * Build: make bench_decompose
 * Run:   sudo ./bench_decompose --sweep --repeat 5
 *        sudo ./bench_decompose --sweep --breakdown --repeat 3
 *        sudo ./bench_decompose --lo 50 --hi 50 --thread-sweep --repeat 3
 *
 * --breakdown honors --threads (default 8). Use --threads 1 for the
 * single-stream cost model, higher for the scaling picture.
 *
 * --thread-sweep runs 1/2/4/8/16 threads twice, once pipelined with
 * ciphertext out and once serial with host decrypt, and prints the per-stage
 * times for each. Read the stage lines, not just the GB/s: the copy stage
 * needs no CE channel and should keep scaling, while ce_wait acquires one per
 * transfer and flattens once threads exceed channel_pool_type_ce_num_channels
 * in uvm_channel.c (2 in this tree). Whichever stage stops improving is the
 * limit.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <cuda_runtime.h>

#define UVM_INITIALIZE                                      0x30000001
#define UVM_LIVE_MIGRATION_FINALIZE                         101
#define UVM_LIVE_MIGRATION_PREPARE_ALL                      102
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT 115
#define UVM_LIVE_MIGRATION_BENCH_DECOMPOSE                  117

/* dbg_flags — must match uvm_ioctl.h */
#define BENCH_SERIAL         0x1u
#define BENCH_DECRYPT        0x2u
#define BENCH_COPY_TO_SHARED 0x4u
#define BENCH_SKIP_COPY      0x8u

#define NV_OK       0x00000000u
#define MAX_REPEAT  16

typedef uint32_t NV_STATUS;
typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef struct { uint8_t uuid[16]; } NvProcessorUuid;

typedef struct {
    uint64_t  flags __attribute__((aligned(8)));
    uint32_t  rmStatus;
} UVM_INITIALIZE_PARAMS;

typedef struct {
    NvU32     flags;
    NV_STATUS rmStatus;
} PREPARE_ALL_PARAMS;

typedef struct {
    NvU64     base   __attribute__((aligned(8)));
    NvU64     length __attribute__((aligned(8)));
    NV_STATUS rmStatus;
} FINALIZE_PARAMS;

typedef struct __attribute__((packed)) {
    uint64_t size;
    uint8_t  iv[12];
    uint8_t  iv_fresh;
    uint8_t  auth_tag[16];
    uint32_t key_version;
} page_crypto_meta_t;

/* ioctl 115 */
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
} MT_PARAMS;

/* ioctl 117 — must match UVM_LIVE_MIGRATION_BENCH_DECOMPOSE_PARAMS */
typedef struct {
    NvU64           base            __attribute__((aligned(8)));
    NvU64           length          __attribute__((aligned(8)));
    NvU64           cpu_buf         __attribute__((aligned(8)));
    NvU64           cpu_buf_size    __attribute__((aligned(8)));
    NvU64           gpu_buf         __attribute__((aligned(8)));
    NvU64           gpu_buf_size    __attribute__((aligned(8)));
    NvU64           residency_map   __attribute__((aligned(8)));
    uint32_t        num_threads;
    uint32_t        dbg_flags;
    NvProcessorUuid gpu_uuid;
    NvU64           cpu_bytes_out   __attribute__((aligned(8)));
    NvU64           gpu_bytes_out   __attribute__((aligned(8)));
    NvU64           num_pages       __attribute__((aligned(8)));
    NvU64           num_transfers   __attribute__((aligned(8)));
    uint32_t        threads_used;
    uint32_t        _pad2;
    NvU64           t_submit_ns     __attribute__((aligned(8)));
    NvU64           t_iv_ns         __attribute__((aligned(8)));
    NvU64           t_ce_wait_ns    __attribute__((aligned(8)));
    NvU64           t_decrypt_ns    __attribute__((aligned(8)));
    NvU64           t_copy_ns       __attribute__((aligned(8)));
    NvU64           t_worker_max_ns __attribute__((aligned(8)));
    NvU64           t_total_ns      __attribute__((aligned(8)));
    NV_STATUS       rmStatus;
} BENCH_PARAMS;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double gbps(size_t bytes, double ms)
{
    if (ms <= 0.0)
        return 0.0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double absdiff(double a, double b)
{
    return (a > b) ? (a - b) : (b - a);
}

/* Median of v[0..n-1]. Sorts v in place. */
static double median_of(double *v, int n)
{
    if (n <= 0)
        return -1.0;
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    if (n % 2)
        return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int                 g_fd = -1;
static int                 g_prepare = 1;
static int                 g_threads = 8;
static void               *g_gpu = NULL;
static uint8_t            *g_resmap = NULL;
static uint8_t            *g_cpu_buf = NULL;
static uint8_t            *g_out = NULL;
static page_crypto_meta_t *g_meta = NULL;

static int prep(void)
{
    PREPARE_ALL_PARAMS p;

    if (!g_prepare)
        return 0;

    memset(&p, 0, sizeof(p));
    if (ioctl(g_fd, UVM_LIVE_MIGRATION_PREPARE_ALL, &p) != 0 || p.rmStatus != NV_OK) {
        fprintf(stderr, "PREPARE_ALL failed: errno=%d status=0x%08x\n",
                errno, p.rmStatus);
        return -1;
    }
    return 0;
}

static void fin(size_t len)
{
    FINALIZE_PARAMS f;

    if (!g_prepare)
        return;

    memset(&f, 0, sizeof(f));
    f.base   = (NvU64)(uintptr_t)g_gpu;
    f.length = (NvU64)len;
    ioctl(g_fd, UVM_LIVE_MIGRATION_FINALIZE, &f);
}

/* ---------------- cudaMemcpy ---------------- */

static int memcpy_once(size_t len, double *ms_out)
{
    cudaError_t e;
    double t0;

    t0 = now_ms();
    e  = cudaMemcpy(g_out, g_gpu, len, cudaMemcpyDeviceToHost);
    *ms_out = now_ms() - t0;
    return (e == cudaSuccess) ? 0 : -1;
}

static double run_memcpy(size_t len, int repeat)
{
    double v[MAX_REPEAT], ms;
    int r, n = 0;

    if (memcpy_once(len, &ms) != 0) {                 /* warmup, discarded */
        printf("    %-28s FAILED (warmup)\n", "cudaMemcpy D2H");
        return -1.0;
    }
    for (r = 0; r < repeat; r++) {
        if (memcpy_once(len, &ms) != 0) {
            printf("    %-28s FAILED (run %d)\n", "cudaMemcpy D2H", r);
            return -1.0;
        }
        v[n++] = gbps(len, ms);
    }
    ms = median_of(v, n);
    printf("    %-28s %6.2f GB/s   (median of %d: %.2f-%.2f)\n",
           "cudaMemcpy D2H", ms, n, v[0], v[n - 1]);
    return ms;
}

/* ---------------- ioctl 115 ---------------- */

static int ioctl115_once(size_t len, size_t npages, double *ms_out, MT_PARAMS *out)
{
    MT_PARAMS m;
    double t0;
    int rc;

    memset(&m, 0, sizeof(m));
    m.base             = (NvU64)(uintptr_t)g_gpu;
    m.length           = (NvU64)len;
    m.cpu_buf          = (NvU64)(uintptr_t)g_cpu_buf;
    m.cpu_buf_size     = (NvU64)len;
    m.gpu_buf          = (NvU64)(uintptr_t)g_out;
    m.gpu_buf_size     = (NvU64)len;
    m.residency_map    = (NvU64)(uintptr_t)g_resmap;
    m.crypto_meta      = (NvU64)(uintptr_t)g_meta;
    m.crypto_meta_size = (NvU64)(npages * sizeof(page_crypto_meta_t));
    m.num_threads      = (uint32_t)g_threads;

    if (prep() < 0)
        return -1;

    t0 = now_ms();
    rc = ioctl(g_fd, UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT, &m);
    *ms_out = now_ms() - t0;
    fin(len);

    if (rc != 0 || m.rmStatus != NV_OK) {
        fprintf(stderr, "ioctl 115: errno=%d status=0x%08x\n", errno, m.rmStatus);
        return -1;
    }
    *out = m;
    return 0;
}

static double run_115(size_t len, size_t npages, int repeat)
{
    double v[MAX_REPEAT], ms, med;
    MT_PARAMS m;
    int r, n = 0;

    if (ioctl115_once(len, npages, &ms, &m) != 0) {    /* warmup, discarded */
        printf("    %-28s FAILED (warmup)\n", "ioctl 115 (production)");
        return -1.0;
    }
    for (r = 0; r < repeat; r++) {
        if (ioctl115_once(len, npages, &ms, &m) != 0) {
            printf("    %-28s FAILED (run %d)\n", "ioctl 115 (production)", r);
            return -1.0;
        }
        v[n++] = gbps((size_t)m.gpu_bytes_out, ms);
    }
    med = median_of(v, n);
    printf("    %-28s %6.2f GB/s   (median of %d: %.2f-%.2f, %llu xfers)\n",
           "ioctl 115 (production)", med, n, v[0], v[n - 1],
           (unsigned long long)m.num_transfers);
    return med;
}

/* ---------------- ioctl 117 ---------------- */

static int ioctl117_once(size_t len, uint32_t flags, int threads,
                         double *ms_out, BENCH_PARAMS *out)
{
    BENCH_PARAMS b;
    double t0;
    int rc;

    memset(&b, 0, sizeof(b));
    b.base          = (NvU64)(uintptr_t)g_gpu;
    b.length        = (NvU64)len;
    b.cpu_buf       = (NvU64)(uintptr_t)g_cpu_buf;
    b.cpu_buf_size  = (NvU64)len;
    b.gpu_buf       = (NvU64)(uintptr_t)g_out;
    b.gpu_buf_size  = (NvU64)len;
    b.residency_map = (NvU64)(uintptr_t)g_resmap;
    b.num_threads   = (uint32_t)threads;
    b.dbg_flags     = flags;

    if (prep() < 0)
        return -1;

    t0 = now_ms();
    rc = ioctl(g_fd, UVM_LIVE_MIGRATION_BENCH_DECOMPOSE, &b);
    *ms_out = now_ms() - t0;
    fin(len);

    if (rc != 0 || b.rmStatus != NV_OK) {
        fprintf(stderr, "ioctl 117: errno=%d status=0x%08x\n", errno, b.rmStatus);
        return -1;
    }
    *out = b;
    return 0;
}

static double run_117(const char *label, size_t len, uint32_t flags,
                      int threads, int repeat)
{
    double        v[MAX_REPEAT], walls[MAX_REPEAT], ms, med, sum_ms;
    double        worker_ms, pass1_ms, lo, hi;
    BENCH_PARAMS  runs[MAX_REPEAT], b;
    int           r, n = 0, mid;

    if (ioctl117_once(len, flags, threads, &ms, &b) != 0) {  /* warmup */
        printf("    %-28s FAILED (warmup)\n", label);
        return -1.0;
    }
    for (r = 0; r < repeat; r++) {
        if (ioctl117_once(len, flags, threads, &ms, &b) != 0) {
            printf("    %-28s FAILED (run %d)\n", label, r);
            return -1.0;
        }
        walls[n] = ms;
        runs[n]  = b;
        v[n]     = gbps((size_t)b.gpu_bytes_out, ms);
        n++;
    }

    /* Median by throughput; report that run's stage breakdown so the split
     * and the headline number describe the same call. */
    {
        double tmp[MAX_REPEAT];
        memcpy(tmp, v, sizeof(double) * (size_t)n);
        med = median_of(tmp, n);   /* sorts tmp, leaves v in run order */
        lo  = tmp[0];
        hi  = tmp[n - 1];
    }
    mid = 0;
    for (r = 1; r < n; r++) {
        if (absdiff(v[r], med) < absdiff(v[mid], med))
            mid = r;
    }

    b         = runs[mid];
    ms        = walls[mid];
    sum_ms    = (double)(b.t_submit_ns + b.t_ce_wait_ns +
                         b.t_decrypt_ns + b.t_copy_ns) / 1e6;
    worker_ms = (double)b.t_worker_max_ns / 1e6;
    pass1_ms  = ms - worker_ms;
    if (pass1_ms < 0.0)
        pass1_ms = 0.0;

    printf("    %-28s %6.2f GB/s   (median of %d: %.2f-%.2f, %llu xfers, %u thr)\n",
           label, med, n, lo, hi,
           (unsigned long long)b.num_transfers, (unsigned)b.threads_used);
    printf("        submit %7.0f  iv %7.0f  ce_wait %7.0f  decrypt %7.0f  copy %7.0f  [ms]\n",
           b.t_submit_ns / 1e6, b.t_iv_ns / 1e6, b.t_ce_wait_ns / 1e6,
           b.t_decrypt_ns / 1e6, b.t_copy_ns / 1e6);
    printf("        wall %.0f = worker %.0f + serial %.0f ms  (serial = PASS 1 + setup, %.0f%%)\n",
           ms, worker_ms, pass1_ms, ms > 0.0 ? 100.0 * pass1_ms / ms : 0.0);
    printf("        stage-sum %.0f ms = %.0f%% of worker phase%s\n",
           sum_ms, worker_ms > 0.0 ? 100.0 * sum_ms / worker_ms : 0.0,
           (flags & BENCH_SERIAL) ? "  (serial: expect ~100%)"
                                  : "  (pipelined: overlap expected)");
    return med;
}

int main(int argc, char **argv)
{
    size_t lo_gb = 20, hi_gb = 50, step_gb = 10;
    size_t max_bytes, max_pages, gb;
    int sweep = 0, breakdown = 0, thread_sweep = 0, repeat = 3, i;
    UVM_INITIALIZE_PARAMS ip;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sweep"))
            sweep = 1;
        else if (!strcmp(argv[i], "--breakdown"))
            breakdown = 1;
        else if (!strcmp(argv[i], "--thread-sweep"))
            thread_sweep = 1;
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
            repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lo") && i + 1 < argc)
            lo_gb = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hi") && i + 1 < argc)
            hi_gb = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--step") && i + 1 < argc)
            step_gb = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            g_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-prepare"))
            g_prepare = 0;
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (!sweep && !breakdown && !thread_sweep)
        sweep = 1;
    if (step_gb == 0)
        step_gb = 10;
    if (repeat < 1)
        repeat = 1;
    if (repeat > MAX_REPEAT)
        repeat = MAX_REPEAT;

    max_bytes = hi_gb * 1024ULL * 1024ULL * 1024ULL;
    max_pages = max_bytes / 4096ULL;

    printf("=== cudaMemcpy vs kernel CE path (CC mode) ===\n");
    printf("Sweep %zu..%zu GB step %zu | threads %d | repeat %d (+1 warmup) | PREPARE_ALL %s\n",
           lo_gb, hi_gb, step_gb, g_threads, repeat,
           g_prepare ? "on" : "OFF (not production-comparable)");
    printf("Host RAM needed: ~%zu GB (out + cpu buffers). Ctrl-C now if too much.\n\n",
           2 * hi_gb);

    if (cudaMallocManaged(&g_gpu, max_bytes, cudaMemAttachGlobal) != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged %zu GB failed\n", hi_gb);
        return 1;
    }
    cudaMemAdvise(g_gpu, max_bytes, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemPrefetchAsync(g_gpu, max_bytes, 0, 0);
    cudaMemset(g_gpu, 0xAB, max_bytes);
    cudaDeviceSynchronize();

    g_fd = open("/dev/nvidia-uvm", O_RDWR);
    if (g_fd < 0) {
        fprintf(stderr, "open /dev/nvidia-uvm: %s (need root)\n", strerror(errno));
        return 1;
    }
    memset(&ip, 0, sizeof(ip));
    ioctl(g_fd, UVM_INITIALIZE, &ip);

    g_resmap  = (uint8_t *)calloc(max_pages, 1);
    g_cpu_buf = (uint8_t *)malloc(max_bytes);
    g_meta    = (page_crypto_meta_t *)malloc(max_pages * sizeof(page_crypto_meta_t));
    if (posix_memalign((void **)&g_out, 4096, max_bytes) != 0)
        g_out = NULL;

    if (!g_resmap || !g_cpu_buf || !g_out || !g_meta) {
        fprintf(stderr, "host allocation failed (need ~%zu GB)\n", 2 * hi_gb);
        return 1;
    }
    memset(g_out, 0, max_bytes);   /* pre-fault: don't time page faults */

    for (gb = lo_gb; gb <= hi_gb; gb += step_gb) {
        size_t len = gb * 1024ULL * 1024ULL * 1024ULL;
        size_t np  = len / 4096ULL;
        double gm, g5, g7;

        printf("--- %zu GB ---\n", gb);
        gm = run_memcpy(len, repeat);
        g5 = run_115(len, np, repeat);
        g7 = run_117("ioctl 117 (same config)", len, 0u, g_threads, repeat);

        if (gm > 0.0 && g5 > 0.0)
            printf("    speedup 115 / cudaMemcpy: %.2fx\n", g5 / gm);

        if (g5 > 0.0 && g7 > 0.0) {
            double d = 100.0 * (g7 - g5) / g5;
            printf("    117 vs 115 (medians): %+.1f%%%s\n", d,
                   (d < -5.0 || d > 5.0)
                       ? "   <-- DIVERGENT: do not report the stage split"
                       : "   (ok)");
        }

        if (breakdown) {
            /* Stage attribution needs BENCH_SERIAL so the per-stage times sum
             * to the worker phase. Thread count is independent of that: each
             * worker still runs its own stages back to back. At 1 thread the
             * shares are the single-stream cost model; at N they show which
             * stage stops scaling. */
            printf("    additive breakdown (serial, %d thread%s):\n",
                   g_threads, g_threads == 1 ? "" : "s");
            run_117("serial, cipher out",     len, BENCH_SERIAL, g_threads, repeat);
            run_117("serial, + host decrypt", len, BENCH_SERIAL | BENCH_DECRYPT, g_threads, repeat);
            run_117("serial, copy->shared",   len, BENCH_SERIAL | BENCH_COPY_TO_SHARED, g_threads, repeat);
            run_117("serial, no output copy", len, BENCH_SERIAL | BENCH_SKIP_COPY, g_threads, repeat);
        }

        if (thread_sweep) {
            /* Which stage stops scaling? copy needs no CE channel, so it
             * should keep improving; ce_wait acquires a channel per transfer,
             * so it flattens once threads exceed the CE channel pool (see
             * channel_pool_type_ce_num_channels in uvm_channel.c). Compare
             * the per-stage lines across rows, not just the GB/s. */
            int t;
            printf("    thread sweep (pipelined, cipher out):\n");
            for (t = 1; t <= 16; t *= 2) {
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "%2d thread%s", t, t == 1 ? " " : "s");
                run_117(lbl, len, 0u, t, repeat);
            }
            printf("    thread sweep (serial, + host decrypt):\n");
            for (t = 1; t <= 16; t *= 2) {
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "%2d thread%s, decrypt", t, t == 1 ? " " : "s");
                run_117(lbl, len, BENCH_SERIAL | BENCH_DECRYPT, t, repeat);
            }
        }
        printf("\n");
    }

    close(g_fd);
    cudaFree(g_gpu);
    return 0;
}
