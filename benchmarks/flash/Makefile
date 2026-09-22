CC = gcc
CFLAGS = -O2 -Wall -I.
CUDA_PATH ?= /usr/local/cuda
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lssl -lcrypto -lpthread -ldl

all: libckpt_vllm.so libbaseline_uvm.so ckpt_agent ckpt_core restore_agent

libckpt_vllm.so: libckpt_vllm.c ckpt_baseline.c ckpt_baseline.h ckpt_gate.h ckpt_crypto.h ckpt_crypto.c librst_vllm.c librst_vllm.h
	$(CC) -shared -fPIC $(CFLAGS) -I$(CUDA_PATH)/include -o $@ libckpt_vllm.c ckpt_baseline.c ckpt_crypto.c librst_vllm.c $(LDFLAGS)

# UVM substrate-only baseline: cudaMalloc/cudaMallocAsync/cuMemAlloc_v2 -> managed.
# No checkpointing, no gate, no alloc table. Used to isolate UVM cost from ckpt cost.
libbaseline_uvm.so: libbaseline_uvm.c
	$(CC) -shared -fPIC $(CFLAGS) -I$(CUDA_PATH)/include -o $@ libbaseline_uvm.c -L$(CUDA_PATH)/lib64 -lcudart -lcuda -ldl -lpthread

ckpt_agent: ckpt_agent.c ckpt_gate.h
	$(CC) $(CFLAGS) -o $@ ckpt_agent.c

ckpt_core: ckpt_core.c ckpt_gate.h ckpt_crypto.h ckpt_crypto.c
	$(CC) $(CFLAGS) -o $@ ckpt_core.c ckpt_crypto.c -lssl -lcrypto -lpthread

restore_agent: restore_agent.c ckpt_gate.h
	$(CC) $(CFLAGS) -o $@ restore_agent.c

clean:
	rm -f libckpt_vllm.so libbaseline_uvm.so ckpt_agent ckpt_core restore_agent

.PHONY: all clean
