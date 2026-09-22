# Installing FLASH's modified NVIDIA driver

FLASH modifies the NVIDIA UVM kernel module from
[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules)
**550.163.01**. All changes are in `kernel-open/nvidia-uvm/`: two new files
(`uvm_live_migration.c`, `uvm_live_migration.h`) and additions to twelve
upstream files. The rest of the driver, including the RM and GSP interface in
`src/`, is unmodified.

Step 0 sets up the host and the confidential VM. **Every step after that runs
inside the TDX guest (CVM) that has the H100 passed through**, since the guest
loads the driver, not the host. Commands assume you start in the directory
that contains `artifact_flash/`.

> [!IMPORTANT]
> The kernel module and the userspace driver must be the **exact same
> version**, 550.163.01. If they differ, `nvidia-smi` and CUDA fail with a
> version mismatch error. Step 2 pins the userspace version for this reason.

## 0. Prerequisites

### 0.1 Host: TDX 1.5 on Ubuntu 25.04

The host must run **Ubuntu 25.04** with **Intel TDX 1.5** enabled, and the H100
must be in confidential computing mode before it is passed through to the
guest. Follow NVIDIA's
[Confidential Computing Deployment Guide for TDX and SEV-SNP](https://docs.nvidia.com/cc-deployment-guide-tdx-snp.pdf)
to set up the host, enable CC mode on the GPU, and create the guest.

If you need help with the guide, contact <!-- TODO: contact email --> for our
setup instructions.

### 0.2 Guest VM: Ubuntu 24.04

The guest must run **Ubuntu 24.04** and be launched with at least these
resources:

| Resource | Value | QEMU / image setting |
|---|---|---|
| Memory | 300 GB | `-m 300G` |
| vCPUs | 64 | `-smp 64` |
| Disk | 300 GB | guest image size, e.g. `qemu-img create -f qcow2 <image> 300G` |

The memory is needed for the host-side checkpoint buffers and the in-memory
image staging used in the evaluation. Smaller guests fail the larger model and
transfer-size runs.

The guest launch command we used, following the deployment guide. Run it on
the host as root, after setting the three variables for your machine:

```bash
IMAGE=/path/to/tdx-guest.qcow2   # the 300 GB Ubuntu 24.04 guest image
GPU=0000:b0:00.0                 # PCI address of the H100 (lspci -D | grep -i nvidia)
SSH_PORT=10022                   # host port forwarded to the guest's port 22

qemu-system-x86_64 \
    -accel kvm \
    -m 300G -smp 64 \
    -name td,process=td,debug-threads=on \
    -cpu host,-avx10 \
    -object '{"qom-type":"tdx-guest","id":"tdx","quote-generation-socket":{"type":"vsock","cid":"3","port":"4051"}}' \
    -object memory-backend-ram,id=mem0,size=300G \
    -machine q35,kernel_irqchip=split,confidential-guest-support=tdx,memory-backend=mem0 \
    -bios /usr/share/ovmf/OVMF.fd \
    -nographic -daemonize -nodefaults -vga none \
    -device virtio-net-pci,netdev=nic0_td \
    -netdev user,id=nic0_td,hostfwd=tcp::${SSH_PORT}-:22 \
    -drive file=${IMAGE},if=none,id=virtio-disk0 \
    -device virtio-blk-pci,drive=virtio-disk0 \
    -pidfile /tmp/flash-td.pid \
    -device vhost-vsock-pci,guest-cid=5 \
    -object iommufd,id=iommufd0 \
    -device pcie-root-port,id=pci.0,bus=pcie.0 \
    -device vfio-pci,host=${GPU},bus=pci.0,iommufd=iommufd0
```

Then connect with `ssh -p ${SSH_PORT} <user>@localhost`. The VM runs in the
background (`-daemonize`); stop it with `sudo kill $(cat /tmp/flash-td.pid)`.

Notes on the command:

- `quote-generation-socket` points at the host's TDX Quote Generation Service
  on vsock port 4051, which the guest uses for attestation. It must be running
  on the host, as set up by the deployment guide.
- The GPU must already be bound to `vfio-pci` on the host, and CC mode enabled,
  before launch.
- The H100 has very large BARs, so boot can pause for several seconds while
  they are mapped. That is expected.

### 0.3 Guest packages

Inside the guest:

```bash
sudo apt-get update
sudo apt-get install -y build-essential linux-headers-$(uname -r) git zstd binutils
```

## 1. Install the CUDA toolkit (without its driver)

The runfile bundles an older driver, 550.54.14. Install the toolkit only.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit
```

Add the toolkit to your environment, for example in `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
```

## 2. Install the userspace driver, version 550.163.01 (do not reboot yet)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

apt-cache policy nvidia-driver-550-open          # find the 550.163.01 version string
sudo apt-get install -y nvidia-driver-550-open=<550.163.01 version string>
sudo apt-mark hold nvidia-driver-550-open
```

The version string looks like `550.163.01-0ubuntu1`; use exactly what
`apt-cache policy` lists. Without the pin, apt installs the newest 550.x, which
will not match the modules built in step 5. `apt-mark hold` stops a later
upgrade from breaking the match.

## 3. Stop Ubuntu from loading the distro kernel modules

```bash
sudo tee /etc/modprobe.d/blacklist-nvidia-distro.conf <<'EOF'
blacklist nvidia
blacklist nvidia_uvm
blacklist nvidia_modeset
blacklist nvidia_drm
EOF
sudo update-initramfs -u
sudo reboot
```

## 4. Fetch the driver source and apply FLASH's changes

```bash
cd artifact_flash/modified_driver_code
./apply_to_driver.sh
```

This clones open-gpu-kernel-modules at tag 550.163.01 into
`artifact_flash/open-gpu-kernel-modules/`, checks the version, and copies
FLASH's 14 files into place. It stops without changing anything if a file is
missing or the checkout is the wrong version.

## 5. Build and install the modified modules

```bash
cd ../open-gpu-kernel-modules
make modules -j$(nproc)
sudo make modules_install
sudo depmod -a

sudo rm /etc/modprobe.d/blacklist-nvidia-distro.conf
sudo update-initramfs -u
sudo reboot
```

## 6. Check the driver and confidential computing

```bash
sudo modprobe nvidia
sudo modprobe nvidia_uvm
nvidia-smi

sudo nvidia-smi conf-compute -f     # confidential computing feature: expect ON
sudo nvidia-smi conf-compute -d     # DevTools mode: expect OFF
```

If the feature reports OFF, the GPU's confidential computing mode has not been
enabled. That is set from the host before the GPU is passed through; see the
VM setup instructions.

**After every boot**, before running any workload:

```bash
sudo nvidia-persistenced --uvm-persistence-mode
sudo nvidia-smi conf-compute -srs 1
```

In confidential computing mode the GPU does not accept work until it is marked
ready.

## 7. Verify that FLASH's module is the one installed and loaded

```bash
cd artifact_flash/modified_driver_code
sudo ./verify_driver.sh
```

It checks:

1. **The installed file.** Finds the `nvidia-uvm.ko` (or `.ko.zst`) that
   modprobe resolves, decompresses it, and checks it contains ioctl handlers
   defined in `uvm_live_migration.c`, such as
   `uvm_api_live_migration_prepare_all` and
   `uvm_api_live_migration_read_pages_resident_encrypted_mt`. The distro
   module does not have these.
2. **The loaded module.** Checks that the same handlers appear in
   `/proc/kallsyms`, so the module in memory is FLASH's and not only the one
   on disk.
3. **Version match.** The kernel module and `nvidia-smi` must both report
   550.163.01.
4. **CE channel pool.** FLASH's `uvm_channel.c` logs the copy engine channel
   pool size at module load. The evaluation used 4.

Every check should print `PASS`. The equivalent manual commands are:

```bash
modinfo -n nvidia_uvm
zstd -dc "$(modinfo -n nvidia_uvm)" | strings | grep uvm_api_live_migration
grep uvm_api_live_migration /proc/kallsyms
sudo dmesg | grep 'pool_type=CE'
```

## Troubleshooting

**`Failed to initialize NVML: Driver/library version mismatch`**
The userspace driver is not 550.163.01. Reinstall it with the pinned version
from step 2.

**The module path is under `updates/dkms/`**
The distro package built its own module with DKMS. depmod searches `updates/`
before other locations, so it shadows FLASH's module. Remove it and rebuild the
module index:

```bash
dkms status                              # shows the name/version, e.g. nvidia/550.163.01
sudo dkms remove <name/version> --all
sudo depmod -a
sudo reboot
```

**The CE channel pool is not 4**
`uvm_channel.c` came from a tree other than the one used for the evaluation.
Recopy the files into `modified_driver_code/` and repeat steps 4 and 5.

**Additional Errors**
please reach out to the authors of the paper.