CC        = gcc
CUDA_PATH ?= /usr/local/cuda
CFLAGS    = -O2 -Wall -I$(CUDA_PATH)/include
LDFLAGS   = -L$(CUDA_PATH)/lib64 -lcudart

bench_decompose: bench_decompose.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f bench_decompose

.PHONY: clean
