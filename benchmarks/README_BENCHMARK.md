# Running FLASH's benchmarks

Complete [README_SETUP.md](../modified_driver_code/README_SETUP.md) first, including step 7:
`verify_driver.sh` must report every check as `PASS`. All benchmarks run inside
the guest VM. Commands assume you start in the directory that contains
`artifact_flash/`.

## System settings

Apply these inside the guest after every boot, before running any benchmark.
None of them persist across a reboot, and skipping any of them changes the
results without an error or warning.

### Transparent huge pages

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Check that both report `[always]`:

```bash
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled
cat /sys/kernel/mm/transparent_hugepage/enabled
```

The benchmarks move tens of gigabytes through host memory. With the default
4 KB pages, the first touch of each page in a TDX guest is expensive, and 2 MB
pages cut the number of those operations by 512x.

### GPU clocks and confidential computing ready state

```bash
sudo nvidia-smi -lgc 1980
sudo nvidia-smi -lmc 2619

sudo nvidia-persistenced --uvm-persistence-mode
sudo nvidia-smi conf-compute -srs 1
```

When idle, the H100 drops to about 17% of its peak clock and does not ramp up
fast enough for short transfers, which lowers measured bandwidth by roughly
40%. Locking the clocks removes that. The last command marks the GPU ready to
accept work in confidential computing mode.

---

## Table 1

### Build

```bash
cd artifact_flash/benchmarks/table1_bandwidth
make
```

### Run

Run both binaries once in the TD (confidential computing on) and once in a
legacy VM (confidential computing off):

```bash
./memory_transfer_benchmark_sync  | tee table1_sync.log
./memory_transfer_benchmark_async | tee table1_async.log
```

---

## Generating Figures 3, 6, 7 and Table 4

These results come from the same experiment: serving LLM inference with
vLLM while each system (PhoS, GCR, FLASH-SC, FLASH) takes periodic
checkpoints. Each run produces the data for all four:

- **Figure 3:** dirty data rate
- **Figure 6:** concurrent-copy and stop-and-copy latency
- **Figure 7:** dirty memory size in the concurrent-copy phase
- **Table 4:** ratio of concurrent-copy time to total tracking time

### Build

```bash
cd artifact_flash/benchmarks/flash && make
cd artifact_flash/benchmarks/phos_emul && make
cd artifact_flash/benchmarks/gcr_emul && make
```

### Set up /dev/shm

After every boot, enlarge `/dev/shm` and back it with huge pages. The
checkpoint images live there.

```bash
sudo mount -o remount,size=200G,huge=always /dev/shm
```

### Preallocate checkpoint buffers

After every boot, and before any run, preallocate the staging buffer and
checkpoint images so their allocation is not part of the measurement. 
Static image size can be adjusted based on the mode size (i.e. 4GB, 16GB, 32GB ):

```bash
cd artifact_flash/benchmarks/prealloc && make
sudo ./ckpt_prealloc --gpu-size-gb 80 --out /dev/shm/ckpt_base.img --out-size-gb 90 --out
sudo ./ckpt_prealloc --out /dev/shm/ckpt_static.img --out-size-gb 17
```

### Setting up python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install vllm==0.8.5
pip install -r artifact_flash/benchmarks/requirements.txt
```

### Download the models and dataset

```bash
cd artifact_flash/benchmarks
./download.sh
```

Models go to `~/huggingface/`, and ShareGPT goes to `~/`, where
`bench_inference_vllm_async.py` looks for it. The Llama models are gated:
request access on Hugging Face and run `hf auth login` first.

### Configuration for each Model/Application

Flags for `bench_inference_vllm_async.py` (paper Table 2). Every run also uses
`--dataset sharegpt --ignore-eos --gpu-memory-utilization 0.85 --seed 0`.
Output tokens set both `--min-tokens` and `--max-tokens`.

| Application | Model | `--model` | Output tokens | `--max-concurrent` | `--rate` | `--max-model-len` |
|---|---|---|---|---|---|---|
| Chatbot (Short) | Llama-3.2-1B | `~/huggingface/Llama-3.2-1B-Instruct` | 512 | 512 | 30.0 | 4096 |
| Chatbot (Short) | Qwen-2.5-7B | `~/huggingface/Qwen2.5-7B-Instruct` | 512 | 256 | 20.0 | 4096 |
| Chatbot (Short) | Llama-2-13B | `~/huggingface/Llama-2-13b-chat-hf` | 512 | 64 | 5.0 | 4096 |
| Reasoning (Long) | Llama-3.2-1B | `~/huggingface/Llama-3.2-1B-Instruct` | 8192 | 200 | 1.0 | 16384 |
| Reasoning (Long) | Qwen-2.5-7B | `~/huggingface/Qwen2.5-7B-Instruct` | 8192 | 64 | 0.5 | 10000 |
| Reasoning (Long) | Llama-2-13B | `~/huggingface/Llama-2-13b-chat-hf` | 4096 | 32 | 0.5 | 4096 |

Checkpoint interval (1, 3 or 5 minutes):

| Interval | FLASH `ckpt_agent` | PhoS | GCR |
|---|---|---|---|
| 1 min | `--interval 60` | `PHOS_CKPT_INTERVAL_MS=60000` | `GCR_CKPT_INTERVAL_MS=60000` |
| 3 min | `--interval 180` | `PHOS_CKPT_INTERVAL_MS=180000` | `GCR_CKPT_INTERVAL_MS=180000` |
| 5 min | `--interval 300` | `PHOS_CKPT_INTERVAL_MS=300000` | `GCR_CKPT_INTERVAL_MS=300000` |

### Run FLASH

Example command running Llama-3.2-1B

#### Terminal 1
```bash
cd artifact_flash/benchmarks/flash && make
LD_PRELOAD=./artifact_flash/benchmarks/flash/libckpt_vllm.so python bench_inference_vllm_async.py --model ~/huggingface/Llama-3.2-1B-Instruct/ --dataset sharegpt  --rate 30 --duration 600 --ignore-eos --min-tokens 512 --max-tokens 512 --max-concurrent 512 --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 
```

#### Terminal 2 

mode: v3 (FLASH without Static Classification) v3-static (FLASH with static Classification)

```bash
sudo ./ckpt_agent --mode v3 --delay 400 --interval 60  --rounds 10
sudo ./ckpt_agent --mode v3-static --delay 400 --interval 60  --rounds 10
```

### Run_PhoenixOS_Emul

Example command running Llama-3.2-1B

```bash
cd artifact_flash/benchmarks/phos_emul && make
PHOS_SCAN_KERNELS=1 LD_PRELOAD=./home/tdx/artifact_flash/benchmarks/phos_emul/libphos_intercept.so  PHOS_CKPT_DELAY=400 PHOS_CKPT_INTERVAL_MS=60000 PHOS_CKPT_ROUNDS=10  PHOS_STAGING_MB=4096 python bench_inference_vllm_async.py --model ~/huggingface/Llama-3.2-1B-Instruct/ --dataset sharegpt  --rate 30 --duration 600 --ignore-eos --min-tokens 512 --max-tokens 512 --max-concurrent 512 --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 
```

### Run_GCR_Emul

Example command running Llama-3.2-1B

```bash
cd artifact_flash/benchmarks/gcr_emul && make
LD_PRELOAD=./home/tdx/artifact_flash/benchmarks/gcr_emul/libgcr_intercept.so CKPT_STAGING_MB=4096  GCR_CKPT_DELAY=400 GCR_CKPT_INTERVAL_MS=60000 GCR_CKPT_ROUNDS=10 python bench_inference_vllm_async.py --model ~/huggingface/Llama-3.2-1B-Instruct/ --dataset sharegpt  --rate 30 --duration 600 --ignore-eos --min-tokens 512 --max-tokens 512 --max-concurrent 512 --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 
```



