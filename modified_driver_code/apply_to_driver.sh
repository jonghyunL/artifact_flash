#!/usr/bin/env bash
#
# apply_to_driver.sh — overlay FLASH's modified NVIDIA UVM driver files onto a
# clean open-gpu-kernel-modules checkout at tag 550.163.01.
#
# Expected layout:
#
#   artifact_flash/
#   ├── modified_driver_code/
#   │   ├── apply_to_driver.sh        (this script)
#   │   └── kernel-open/nvidia-uvm/   (FLASH's files, mirroring the upstream tree)
#   └── open-gpu-kernel-modules/      (cloned by this script if absent)
#
# All FLASH changes live in kernel-open/nvidia-uvm/. The RM side (src/) is
# unmodified. Two files are new (uvm_live_migration.c/.h); the rest are
# upstream files with additions.
#
# Usage:
#   ./apply_to_driver.sh              clone if needed, verify, copy
#   DRIVER_DIR=/path ./apply_to_driver.sh   use an existing checkout elsewhere
#   FORCE=1 ./apply_to_driver.sh      proceed even if the checkout has
#                                     unrelated local modifications
#
# Then build:
#   cd ../open-gpu-kernel-modules && make modules -j"$(nproc)"

set -euo pipefail

UPSTREAM_URL="https://github.com/NVIDIA/open-gpu-kernel-modules.git"
UPSTREAM_TAG="550.163.01"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(dirname "$HERE")"
DRIVER_DIR="${DRIVER_DIR:-$ARTIFACT_ROOT/open-gpu-kernel-modules}"

# Every file FLASH adds or modifies, relative to the driver root.
FILES=(
    kernel-open/nvidia-uvm/uvm_live_migration.c        # new: live migration ioctls
    kernel-open/nvidia-uvm/uvm_live_migration.h        # new
    kernel-open/nvidia-uvm/nvidia-uvm-sources.Kbuild   # adds uvm_live_migration.c to the build
    kernel-open/nvidia-uvm/uvm.c                       # ioctl routing
    kernel-open/nvidia-uvm/uvm_api.h                   # ioctl handler declarations
    kernel-open/nvidia-uvm/uvm_ioctl.h                 # ioctl numbers and param structs
    kernel-open/nvidia-uvm/uvm_channel.c               # CE channel pool size
    kernel-open/nvidia-uvm/uvm_conf_computing.c        # CE encrypt/decrypt helpers
    kernel-open/nvidia-uvm/uvm_conf_computing.h
    kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c # fault path dirty marking
    kernel-open/nvidia-uvm/uvm_va_block.c              # dirty tracking, 2MB fault upgrade
    kernel-open/nvidia-uvm/uvm_va_block.h              # per-block migration state
    kernel-open/nvidia-uvm/uvm_va_range.c
    kernel-open/nvidia-uvm/uvm_va_space.h
)

die()  { echo "error: $*" >&2; exit 1; }
note() { echo "==> $*"; }

# ---------------------------------------------------------------------------
# 1. Every FLASH file must be present before touching the checkout.
# ---------------------------------------------------------------------------
missing=()
for f in "${FILES[@]}"; do
    [[ -f "$HERE/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
    echo "error: these FLASH files are missing under $HERE:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    die "copy them from the tree that built the evaluated module, then rerun"
fi

# Build artifacts must not ship as source.
if [[ -e "$HERE/kernel-open/nv_compiler.h" ]]; then
    die "kernel-open/nv_compiler.h is a generated build file; remove it from $HERE"
fi

# ---------------------------------------------------------------------------
# 2. Obtain a pristine 550.163.01 checkout.
# ---------------------------------------------------------------------------
if [[ ! -d "$DRIVER_DIR" ]]; then
    note "cloning $UPSTREAM_TAG into $DRIVER_DIR"
    git clone --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_URL" "$DRIVER_DIR"
fi

[[ -f "$DRIVER_DIR/version.mk" ]] || die "$DRIVER_DIR does not look like open-gpu-kernel-modules"

version="$(sed -n 's/^NVIDIA_VERSION *= *//p' "$DRIVER_DIR/version.mk")"
[[ "$version" == "$UPSTREAM_TAG" ]] \
    || die "checkout is $version, expected $UPSTREAM_TAG (FLASH's files are not portable across driver versions)"
note "checkout is $UPSTREAM_TAG"

# ---------------------------------------------------------------------------
# 3. Refuse to clobber unrelated local edits. Re-running this script is fine:
#    changes confined to FLASH's own files are treated as a previous apply.
# ---------------------------------------------------------------------------
if git -C "$DRIVER_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    unrelated=()
    while IFS= read -r path; do
        [[ -z "$path" ]] && continue
        match=0
        for f in "${FILES[@]}"; do [[ "$path" == "$f" ]] && { match=1; break; }; done
        (( match )) || unrelated+=("$path")
    done < <(git -C "$DRIVER_DIR" status --porcelain | awk '{print $NF}')

    if (( ${#unrelated[@]} )) && [[ "${FORCE:-0}" != 1 ]]; then
        echo "error: $DRIVER_DIR has local changes outside FLASH's files:" >&2
        printf '  %s\n' "${unrelated[@]}" >&2
        die "start from a clean checkout, or rerun with FORCE=1"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Copy.
# ---------------------------------------------------------------------------
for f in "${FILES[@]}"; do
    install -D -m 0644 "$HERE/$f" "$DRIVER_DIR/$f"
    echo "    $f"
done
note "copied ${#FILES[@]} files"

if git -C "$DRIVER_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo
    git -C "$DRIVER_DIR" add -N "${FILES[@]}" 2>/dev/null || true
    git -C "$DRIVER_DIR" --no-pager diff --stat -- "${FILES[@]}"
fi

echo
note "done. build with:  cd $DRIVER_DIR && make modules -j\$(nproc)"
