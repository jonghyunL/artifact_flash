#!/usr/bin/env bash
#
# verify_driver.sh — confirm that the nvidia-uvm module modprobe resolves, and
# the one currently in memory, are FLASH's modified build rather than the
# distro's.
#
# Run as root after step 6 of ../README_SETUP.md:   sudo ./verify_driver.sh
#
# Exit status is 0 only if every check passes.

set -uo pipefail

EXPECTED_VERSION="550.163.01"
EXPECTED_CE_CHANNELS=4

# Non-static ioctl handlers defined in uvm_live_migration.c. They exist only in
# FLASH's module, so finding them identifies which build is installed.
SYMBOLS=(
    uvm_api_live_migration_prepare_all
    uvm_api_live_migration_get_dirty_pages
    uvm_api_live_migration_finalize
    uvm_api_live_migration_read_pages_resident_encrypted_mt
    uvm_api_live_migration_read_dirty_delta_encrypted_mt
    uvm_api_live_migration_decrypt_encrypted_pages
)

npass=0; nfail=0; nwarn=0
pass() { echo "  PASS  $*"; npass=$((npass + 1)); }
fail() { echo "  FAIL  $*"; nfail=$((nfail + 1)); }
warn() { echo "  WARN  $*"; nwarn=$((nwarn + 1)); }

[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 1. The module file modprobe will load.
# ---------------------------------------------------------------------------
echo "[1] installed module (what modprobe resolves)"
ko="$(modinfo -n nvidia_uvm 2>/dev/null || true)"
if [[ -z "$ko" ]]; then
    fail "modinfo cannot find nvidia_uvm; did 'make modules_install' and 'depmod -a' run?"
else
    echo "        $ko"
    case "$ko" in
        */updates/dkms/*)
            fail "module is under updates/dkms: the distro DKMS build is shadowing FLASH's (see README_SETUP.md, Troubleshooting)" ;;
    esac

    tmp="$(mktemp --suffix=.ko)"
    trap 'rm -f "$tmp"' EXIT
    case "$ko" in
        *.zst) zstd -dcq "$ko" > "$tmp" ;;
        *.xz)  xz -dc "$ko" > "$tmp" ;;
        *)     cp "$ko" "$tmp" ;;
    esac

    if command -v nm >/dev/null 2>&1; then
        syms="$(nm "$tmp" 2>/dev/null | awk '{print $NF}')"
    else
        syms="$(strings "$tmp")"
    fi
    for s in "${SYMBOLS[@]}"; do
        if grep -qx "$s" <<<"$syms"; then pass "file contains $s"
        else fail "file lacks $s"; fi
    done
fi

# ---------------------------------------------------------------------------
# 2. The module actually loaded in the kernel.
# ---------------------------------------------------------------------------
echo "[2] loaded module"
if [[ ! -d /sys/module/nvidia_uvm ]]; then
    fail "nvidia_uvm is not loaded (sudo modprobe nvidia_uvm)"
else
    for s in "${SYMBOLS[@]}"; do
        if grep -qw "$s" /proc/kallsyms; then pass "loaded module has $s"
        else fail "loaded module lacks $s"; fi
    done
fi

# ---------------------------------------------------------------------------
# 3. Kernel module and userspace must be the exact same version.
# ---------------------------------------------------------------------------
echo "[3] driver version"
kver="$(cat /sys/module/nvidia/version 2>/dev/null || true)"
if [[ "$kver" == "$EXPECTED_VERSION" ]]; then pass "kernel module is $kver"
else fail "kernel module is '${kver:-not loaded}', expected $EXPECTED_VERSION"; fi

uver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
if [[ "$uver" == "$EXPECTED_VERSION" ]]; then pass "userspace (nvidia-smi) is $uver"
else fail "userspace reports '${uver:-nvidia-smi failed}', expected $EXPECTED_VERSION"; fi

# ---------------------------------------------------------------------------
# 4. CE channel pool size, printed by FLASH's uvm_channel.c at module init.
#    Confirms uvm_channel.c came from the tree used for the evaluation.
# ---------------------------------------------------------------------------
echo "[4] copy engine channel pool"
line="$(dmesg | grep 'UVM LivMig: pool_type=CE' | tail -1)"
if [[ -z "$line" ]]; then
    warn "no 'pool_type=CE' line in dmesg (ring buffer may have rotated; reload nvidia_uvm and rerun)"
else
    n="$(sed -n 's/.* num_channels=\([0-9]*\).*/\1/p' <<<"$line")"
    if [[ "$n" == "$EXPECTED_CE_CHANNELS" ]]; then pass "CE channel pool is $n"
    else fail "CE channel pool is ${n:-unknown}, expected $EXPECTED_CE_CHANNELS"; fi
fi

echo
echo "summary: $npass passed, $nfail failed, $nwarn warnings"
(( nfail == 0 ))
