# FLASH: Fault-Led, Attested, Secure Hybrid-Key Checkpoint/Restore for Confidential GPUs

Artifact for the USENIX ATC 2026 paper. FLASH checkpoints GPU state on an H100
in confidential computing mode, using UVM write-fault dirty tracking, a
hybrid-key scheme that stores the copy engine's ciphertext instead of
decrypting on the CPU, static allocation classification, and a concurrent-copy
phase overlapped with the running workload.

## What is here

| Path | Contents |
|---|---|
| `modified_driver_code/` | FLASH's changes to NVIDIA's UVM kernel module, 550.163.01, plus scripts to apply them to a clean checkout and to verify the build |
| `benchmarks/flash/` | FLASH's checkpoint agent and `LD_PRELOAD` library, and the vLLM serving benchmark |
| `benchmarks/phos_emul/` | PhoenixOS emulation, the first baseline |
| `benchmarks/gcr_emul/` | GCR emulation, the second baseline |
| `benchmarks/table1_bandwidth/` | Host-to-device and device-to-host bandwidth, Table 1 |
| `benchmarks/bandwidth_breakdown/` | Decomposition of the confidential-mode copy path |
| `benchmarks/prealloc/` | Preallocates the checkpoint images and staging buffer |
| `benchmarks/download.sh` | Fetches the three models and the ShareGPT dataset |

## Getting started

1. [README_SETUP.md](modified_driver_code/README_SETUP.md) — set up the TDX host and guest, build
   and install the modified driver, and verify it.
2. [README_BENCHMARK.md](benchmarks/README_BENCHMARK.md) — configure the system, build the
   benchmarks, and reproduce the figures and tables.

## Hardware and software required

- An Intel TDX 1.5 host running Ubuntu 25.04, with an NVIDIA H100 in
  confidential computing mode passed through to the guest.
- A guest running Ubuntu 24.04 with 300 GB of memory, 64 vCPUs and a 300 GB
  disk.
- CUDA 12.4 and NVIDIA driver 550.163.01, both pinned. Building the driver
  needs the kernel headers.

The driver changes are specific to confidential computing on an H100 with UVM.
They will not produce meaningful results on a GPU that is not in confidential
computing mode.

## License

MIT, except for the files derived from NVIDIA's open-gpu-kernel-modules, which
stay under their original dual MIT / GPL-2.0 license. See [LICENSE](LICENSE).
