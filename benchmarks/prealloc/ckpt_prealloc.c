/**
 * ckpt_prealloc.c — pre-allocate and warm tmpfs-backed checkpoint buffers
 *
 * Build:
 *   gcc -O2 -Wall -o ckpt_prealloc ckpt_prealloc.c -lpthread
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
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define MAX_BUFS 16

typedef enum { BUF_SHM, BUF_FILE } buf_kind_t;

typedef struct {
    buf_kind_t  kind;
    const char *name_or_path;   /* "/ckpt_core_gpu_staging" or "/dev/shm/foo.img" */
    size_t      size_bytes;
} buf_spec_t;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double gbps(size_t bytes, double ms)
{
    if (ms <= 0) return 0.0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

/* ------------------------------------------------------------------ */
/* Parallel memset worker harness                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    uint8_t *dst;
    size_t   bytes;
} memset_arg_t;

static void *memset_thread(void *arg)
{
    memset_arg_t *a = (memset_arg_t *)arg;
    memset(a->dst, 0, a->bytes);
    return NULL;
}

static double parallel_memset(void *buf, size_t total, int n_threads)
{
    if (n_threads <= 1) {
        double t0 = now_ms();
        memset(buf, 0, total);
        return now_ms() - t0;
    }

    pthread_t   tids[32];
    memset_arg_t args[32];
    if (n_threads > 32) n_threads = 32;

    size_t per = total / (size_t)n_threads;
    per &= ~((size_t)4095);  /* round to 4 KB for clean splits */
    size_t tail = total - per * (size_t)(n_threads - 1);

    double t0 = now_ms();
    for (int t = 0; t < n_threads; t++) {
        args[t].dst   = (uint8_t *)buf + (size_t)t * per;
        args[t].bytes = (t == n_threads - 1) ? tail : per;
        pthread_create(&tids[t], NULL, memset_thread, &args[t]);
    }
    for (int t = 0; t < n_threads; t++)
        pthread_join(tids[t], NULL);
    return now_ms() - t0;
}

/* ------------------------------------------------------------------ */
/* Prepare a single tmpfs-backed buffer at a given fd and size.        */
/* ------------------------------------------------------------------ */
static int prepare_fd(int fd, const char *label, size_t size, int threads)
{
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "%s: fstat: %s\n", label, strerror(errno));
        return -1;
    }
    if ((size_t)st.st_size != size) {
        printf("  %s: ftruncate %zu → %zu bytes (%.2f GB)\n",
               label, (size_t)st.st_size, size, size/(1024.0*1024.0*1024.0));
        if (ftruncate(fd, (off_t)size) < 0) {
            fprintf(stderr, "%s: ftruncate: %s\n", label, strerror(errno));
            return -1;
        }
    }

    void *map = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "%s: mmap(%zu): %s\n", label, size, strerror(errno));
        return -1;
    }
    if (madvise(map, size, MADV_HUGEPAGE) < 0) {
        fprintf(stderr, "%s: MADV_HUGEPAGE advisory failed: %s (continuing)\n",
                label, strerror(errno));
    }

    printf("  %s: parallel memset %d threads ...\n", label, threads);
    double ms = parallel_memset(map, size, threads);
    printf("  %s: warmed %.2f GB in %.1f ms  (%.2f GB/s)\n",
           label, size/(1024.0*1024.0*1024.0), ms, gbps(size, ms));

    (void)msync(map, size, MS_ASYNC);

    if (munmap(map, size) < 0) {
        fprintf(stderr, "%s: munmap: %s\n", label, strerror(errno));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Open + warm one buf_spec_t. Handles both shm_open and open paths.   */
/* mode 0666 so non-root processes can write (vLLM runs as user).      */
/* ------------------------------------------------------------------ */
static int prepare_buf(const buf_spec_t *b, int threads)
{
    int fd;
    char label[64];

    if (b->kind == BUF_SHM) {
        snprintf(label, sizeof(label), "  shm  %s", b->name_or_path);
        printf("[shm ] %s  (%.1f GB)\n",
               b->name_or_path, b->size_bytes/(1024.0*1024.0*1024.0));
        fd = shm_open(b->name_or_path, O_RDWR|O_CREAT, 0666);
        if (fd < 0) {
            fprintf(stderr, "shm_open(%s): %s\n",
                    b->name_or_path, strerror(errno));
            return -1;
        }
    } else {
        snprintf(label, sizeof(label), "  file %s", b->name_or_path);
        printf("[file] %s  (%.1f GB)\n",
               b->name_or_path, b->size_bytes/(1024.0*1024.0*1024.0));
        fd = open(b->name_or_path, O_RDWR|O_CREAT, 0666);
        if (fd < 0) {
            fprintf(stderr, "open(%s): %s\n",
                    b->name_or_path, strerror(errno));
            return -1;
        }
        (void)fchmod(fd, 0666);
    }

    int rc = prepare_fd(fd, label, b->size_bytes, threads);
    close(fd);
    printf("\n");
    return rc;
}

/* ------------------------------------------------------------------ */
/* Parse "name:GB" or "path:GB" into a buf_spec_t.                     */
/* Returns 0 on success, -1 on parse error.                            */
/* ------------------------------------------------------------------ */
static int parse_spec(const char *arg, buf_kind_t kind, buf_spec_t *out)
{
    const char *colon = strrchr(arg, ':');
    if (!colon || colon == arg) {
        fprintf(stderr, "bad spec '%s' — expected NAME:GB\n", arg);
        return -1;
    }
    int gb = atoi(colon + 1);
    if (gb <= 0) {
        fprintf(stderr, "bad size in '%s' — expected positive GB\n", arg);
        return -1;
    }
    /* dup the name portion so we own it */
    size_t name_len = (size_t)(colon - arg);
    char *name = malloc(name_len + 1);
    if (!name) return -1;
    memcpy(name, arg, name_len);
    name[name_len] = '\0';
    out->kind = kind;
    out->name_or_path = name;
    out->size_bytes = (size_t)gb * 1024ULL * 1024ULL * 1024ULL;
    return 0;
}

static void usage(const char *argv0)
{
    printf("Usage: %s [options]\n"
           "\n"
           "  --shm  NAME:GB       add a POSIX shm region (e.g. --shm /ckpt_core_gpu_staging:90)\n"
           "  --file PATH:GB       add a regular file (e.g. --file /dev/shm/ckpt_base.img:90)\n"
           "  --threads N          parallel memset threads (default 8)\n"
           "\n"
           "Legacy/compat flags (still supported):\n"
           "  --gpu-size-gb N      shorthand for: --shm /ckpt_core_gpu_staging:N\n"
           "  --out PATH           output file path (used with --out-size-gb)\n"
           "  --out-size-gb N      shorthand for: --file <out-path>:N\n"
           "\n"
           "Nothing is created by default. Specify what you need per cell:\n"
           "\n"
           "  v3 / v3-static cell:\n"
           "    sudo %s --shm /ckpt_core_gpu_staging:90 \\\n"
           "             --file /dev/shm/ckpt_base.img:90 \\\n"
           "             --file /dev/shm/ckpt_static.img:17\n"
           "\n"
           "  phos / gcr / p1d / p1e cell (no GPU staging buffer needed):\n"
           "    sudo %s --file /dev/shm/ckpt_base.img:90\n",
           argv0, argv0, argv0);
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    buf_spec_t bufs[MAX_BUFS];
    int        n_bufs  = 0;
    int        threads = 8;

    /* Legacy compat staging area */
    int         legacy_out_size_gb = 0;
    const char *legacy_out_path    = NULL;
    int         legacy_gpu_size_gb = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--shm") && i + 1 < argc) {
            if (n_bufs >= MAX_BUFS) {
                fprintf(stderr, "too many bufs (max %d)\n", MAX_BUFS);
                return 1;
            }
            if (parse_spec(argv[++i], BUF_SHM, &bufs[n_bufs]) < 0) return 1;
            n_bufs++;
        }
        else if (!strcmp(argv[i], "--file") && i + 1 < argc) {
            if (n_bufs >= MAX_BUFS) {
                fprintf(stderr, "too many bufs (max %d)\n", MAX_BUFS);
                return 1;
            }
            if (parse_spec(argv[++i], BUF_FILE, &bufs[n_bufs]) < 0) return 1;
            n_bufs++;
        }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        }
        /* Legacy compat */
        else if (!strcmp(argv[i], "--gpu-size-gb") && i + 1 < argc) {
            legacy_gpu_size_gb = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            legacy_out_path = argv[++i];
        }
        else if (!strcmp(argv[i], "--out-size-gb") && i + 1 < argc) {
            legacy_out_size_gb = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Translate legacy flags into buf_spec_t entries. */
    if (legacy_gpu_size_gb > 0) {
        if (n_bufs >= MAX_BUFS) { fprintf(stderr, "too many bufs\n"); return 1; }
        bufs[n_bufs].kind         = BUF_SHM;
        bufs[n_bufs].name_or_path = "/ckpt_core_gpu_staging";
        bufs[n_bufs].size_bytes   = (size_t)legacy_gpu_size_gb * 1024ULL*1024ULL*1024ULL;
        n_bufs++;
    }
    if (legacy_out_path && legacy_out_size_gb > 0) {
        if (n_bufs >= MAX_BUFS) { fprintf(stderr, "too many bufs\n"); return 1; }
        bufs[n_bufs].kind         = BUF_FILE;
        bufs[n_bufs].name_or_path = legacy_out_path;
        bufs[n_bufs].size_bytes   = (size_t)legacy_out_size_gb * 1024ULL*1024ULL*1024ULL;
        n_bufs++;
    }

    if (n_bufs == 0) {
        fprintf(stderr, "no buffers specified — nothing to do\n\n");
        usage(argv[0]);
        return 1;
    }

    printf("=== ckpt_prealloc ===\n");
    printf("Threads     : %d\n", threads);
    printf("Buffers     : %d\n", n_bufs);
    size_t total_bytes = 0;
    for (int i = 0; i < n_bufs; i++) {
        printf("  [%d] %-4s  %s  (%.1f GB)\n", i,
               bufs[i].kind == BUF_SHM ? "shm" : "file",
               bufs[i].name_or_path,
               bufs[i].size_bytes / (1024.0*1024.0*1024.0));
        total_bytes += bufs[i].size_bytes;
    }
    printf("Total       : %.2f GB\n\n", total_bytes / (1024.0*1024.0*1024.0));

    double t_total = now_ms();
    for (int i = 0; i < n_bufs; i++) {
        if (prepare_buf(&bufs[i], threads) < 0) return 1;
    }
    double total_ms = now_ms() - t_total;

    printf("=== done ===\n");
    printf("Total: %.1f ms  (%.2f GB/s aggregate)\n",
           total_ms, gbps(total_bytes, total_ms));
    printf("\nBuffers persist until you remove them or /dev/shm is cleared.\n");
    return 0;
}
