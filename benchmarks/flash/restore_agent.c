/**
 * restore_agent.c — vLLM checkpoint restore orchestrator
 *
 * Thin wrapper that:
 *   1. Verifies the checkpoint files exist (snap gate, base.img, delta.img).
 *   2. Creates /tmp/ckpt_gate with CKPT_CFG_RESTORE bit set so libckpt_vllm.so
 *      enters restore mode in its constructor.
 *   3. Forks + execs the user-supplied vLLM command with LD_PRELOAD pointing
 *      at libckpt_vllm.so and CKPT_RESTORE_* env vars pointing at the
 *      checkpoint files.
 *   4. Waits for the child to exit.
 *
 * Usage:
 *   sudo ./restore_agent \
 *       --snap  /tmp/snap_ckpt_gate \
 *       --base  /tmp/ckpt_inc_base.img \
 *       --delta /tmp/ckpt_inc_delta.img \
 *       [--gate /tmp/ckpt_gate] \
 *       [--lib  ./libckpt_vllm.so] \
 *       -- python run_vllm.py --model ~/huggingface/Llama-3.2-1B-Instruct ...
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <limits.h>

#include "ckpt_gate.h"

static int file_exists(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

static int create_restore_gate(const char *gate_path)
{
    /* Remove any stale gate file (so we own the new one cleanly) */
    unlink(gate_path);

    int fd = open(gate_path, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
        fprintf(stderr, "[restore_agent] open(%s) failed: %s\n",
                gate_path, strerror(errno));
        return -1;
    }
    if (ftruncate(fd, GATE_FILE_SIZE) < 0) {
        fprintf(stderr, "[restore_agent] ftruncate failed: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    void *m = mmap(NULL, GATE_FILE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        fprintf(stderr, "[restore_agent] mmap failed: %s\n", strerror(errno));
        return -1;
    }
    memset(m, 0, GATE_FILE_SIZE);
    ckpt_gate_t *g = (ckpt_gate_t *)m;
    g->phase = GATE_PHASE_INVALID;
    g->config_flags = CKPT_CFG_RESTORE;
    g->app_pid = 0;  /* child process will CAS this on attach */
    munmap(m, GATE_FILE_SIZE);
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --snap PATH --base IMG --delta IMG\n"
        "          [--state PATH] [--gate PATH] [--lib PATH]\n"
        "          -- <vllm command...>\n"
        "\n"
        "Required:\n"
        "  --snap   PATH    Snap gate file from checkpoint (e.g. /tmp/snap_ckpt_gate)\n"
        "  --base   IMG     Base image (e.g. /tmp/ckpt_inc_base.img)\n"
        "  --delta  IMG     Delta image (e.g. /tmp/ckpt_inc_delta.img)\n"
        "Optional:\n"
        "  --state  PATH    P5 engine state JSON (prefix cache + requests).\n"
        "                   Default: auto-detect <snap_dir>/snap_ckpt_engine_state.json\n"
        "  --gate   PATH    Live gate file (default /tmp/ckpt_gate)\n"
        "  --lib    PATH    libckpt_vllm.so to LD_PRELOAD (default ./libckpt_vllm.so)\n"
        "\n"
        "Example:\n"
        "  sudo %s --snap /tmp/snap_ckpt_gate \\\n"
        "      --base /tmp/ckpt_inc_base.img --delta /tmp/ckpt_inc_delta.img \\\n"
        "      -- python run_vllm.py --model ~/huggingface/Llama-3.2-1B-Instruct\n",
        prog, prog);
}

int main(int argc, char **argv)
{
    const char *snap   = NULL;
    const char *base   = NULL;
    const char *delta  = NULL;
    const char *state  = NULL;  /* P5 engine state JSON (optional) */
    const char *static_img = NULL;  /* P6 static.img (optional) */
    const char *gate   = "/tmp/ckpt_gate";
    const char *lib    = "./libckpt_vllm.so";

    int i = 1;
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--")) { i++; break; }
        else if (!strcmp(argv[i], "--snap")  && i + 1 < argc) snap  = argv[++i];
        else if (!strcmp(argv[i], "--base")  && i + 1 < argc) base  = argv[++i];
        else if (!strcmp(argv[i], "--delta") && i + 1 < argc) delta = argv[++i];
        else if (!strcmp(argv[i], "--state") && i + 1 < argc) state = argv[++i];
        else if (!strcmp(argv[i], "--static") && i + 1 < argc) static_img = argv[++i];
        else if (!strcmp(argv[i], "--gate")  && i + 1 < argc) gate  = argv[++i];
        else if (!strcmp(argv[i], "--lib")   && i + 1 < argc) lib   = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[restore_agent] unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!snap || !base || !delta || i >= argc) {
        usage(argv[0]);
        return 1;
    }

    if (!file_exists(snap)) {
        fprintf(stderr, "[restore_agent] snap not found: %s\n", snap);
        return 2;
    }
    if (!file_exists(base)) {
        fprintf(stderr, "[restore_agent] base image not found: %s\n", base);
        return 2;
    }
    if (!file_exists(delta)) {
        fprintf(stderr, "[restore_agent] delta image not found: %s\n", delta);
        return 2;
    }
    if (!file_exists(lib)) {
        fprintf(stderr, "[restore_agent] libckpt_vllm.so not found: %s\n", lib);
        return 2;
    }

    fprintf(stderr, "[restore_agent] snap  = %s\n", snap);
    fprintf(stderr, "[restore_agent] base  = %s\n", base);
    fprintf(stderr, "[restore_agent] delta = %s\n", delta);
    fprintf(stderr, "[restore_agent] gate  = %s\n", gate);
    fprintf(stderr, "[restore_agent] lib   = %s\n", lib);

    if (create_restore_gate(gate) != 0) return 3;

    /* Resolve absolute path of lib so LD_PRELOAD works regardless of child cwd */
    char lib_abs[PATH_MAX];
    if (!realpath(lib, lib_abs)) {
        fprintf(stderr, "[restore_agent] realpath(%s) failed: %s\n",
                lib, strerror(errno));
        return 3;
    }

    /* Build child env: keep parent env, override LD_PRELOAD + CKPT_RESTORE_* */
    setenv("LD_PRELOAD",         lib_abs, 1);
    setenv("CKPT_GATE_FILE",     gate,    1);
    setenv("CKPT_RESTORE_SNAP",  snap,    1);
    setenv("CKPT_RESTORE_BASE",  base,    1);
    setenv("CKPT_RESTORE_DELTA", delta,   1);

    /* P6: static.img — explicit --static wins; otherwise auto-detect
     * sibling of --base (typical convention: alongside ckpt_base.img).
     * Forwarded so libckpt_vllm.c restore mmaps it and applies the
     * captured H2D ranges before the base+delta passes. */
    {
        char static_path[PATH_MAX];
        static_path[0] = '\0';
        if (static_img) {
            snprintf(static_path, sizeof(static_path), "%s", static_img);
        } else {
            char base_copy[PATH_MAX];
            snprintf(base_copy, sizeof(base_copy), "%s", base);
            char *slash = strrchr(base_copy, '/');
            const char *base_dir = slash ? (slash[1] = '\0', base_copy) : ".";
            snprintf(static_path, sizeof(static_path),
                     "%s/ckpt_static.img", base_dir);
        }
        if (static_path[0] && file_exists(static_path)) {
            setenv("CKPT_RESTORE_STATIC", static_path, 1);
            fprintf(stderr, "[restore_agent] static = %s (P6 enabled)\n", static_path);
        } else if (static_img) {
            fprintf(stderr, "[restore_agent] static = %s (NOT FOUND, P6 disabled)\n",
                    static_path);
        } else {
            fprintf(stderr, "[restore_agent] static = %s (auto-detect miss, P6 disabled)\n",
                    static_path);
        }
    }

    /* P5: deterministic hash seed so NONE_HASH in vllm.v1.core.kv_cache_utils
     * matches across checkpoint and restore processes. Without this, the
     * restored prefix-cache hash map is unreachable from new requests. */
    if (!getenv("PYTHONHASHSEED"))
        setenv("PYTHONHASHSEED", "0", 1);

    /* P5: forward engine-state JSON to the child via CKPT_RESTORE_STATE.
     * Explicit --state flag wins; otherwise auto-locate next to the snap
     * gate (ckpt_core's snapshot_file emits <snap_dir>/snap_ckpt_engine_state.json). */
    char state_path[PATH_MAX];
    state_path[0] = '\0';
    if (state) {
        snprintf(state_path, sizeof(state_path), "%s", state);
    } else {
        char snap_copy[PATH_MAX];
        snprintf(snap_copy, sizeof(snap_copy), "%s", snap);
        char *slash = strrchr(snap_copy, '/');
        const char *snap_dir = slash ? (slash[1] = '\0', snap_copy) : ".";
        snprintf(state_path, sizeof(state_path),
                 "%s/snap_ckpt_engine_state.json", snap_dir);
    }
    if (state_path[0] && file_exists(state_path)) {
        setenv("CKPT_RESTORE_STATE", state_path, 1);
        fprintf(stderr, "[restore_agent] state = %s (P5 enabled)\n", state_path);
        /* vLLM V1 defaults to SyncMPClient (engine in subprocess). The P5
         * restore_patch walks llm.llm_engine.engine_core.scheduler in the
         * main process — that only works with InprocClient. Force it. */
        if (!getenv("VLLM_ENABLE_V1_MULTIPROCESSING")) {
            setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0", 1);
            fprintf(stderr, "[restore_agent] forced VLLM_ENABLE_V1_MULTIPROCESSING=0 for P5\n");
        }
    } else if (state) {
        fprintf(stderr, "[restore_agent] state = %s (NOT FOUND, P5 disabled)\n", state_path);
    } else {
        fprintf(stderr, "[restore_agent] state = %s (auto-detect miss, P5 disabled)\n", state_path);
    }

    /* Fork + exec the vLLM command */
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[restore_agent] fork failed: %s\n", strerror(errno));
        return 3;
    }
    if (pid == 0) {
        execvp(argv[i], &argv[i]);
        fprintf(stderr, "[restore_agent] execvp(%s) failed: %s\n",
                argv[i], strerror(errno));
        _exit(127);
    }

    fprintf(stderr, "[restore_agent] launched child pid=%d\n", pid);
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        fprintf(stderr, "[restore_agent] child exited with %d\n", WEXITSTATUS(status));
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        fprintf(stderr, "[restore_agent] child killed by signal %d\n", WTERMSIG(status));
        return 128 + WTERMSIG(status);
    }
    return 0;
}
