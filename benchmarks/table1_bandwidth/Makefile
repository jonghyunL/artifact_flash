NVCC      ?= /usr/local/cuda/bin/nvcc
NVCCFLAGS  = -O2

# The two binaries differ only in the three copy calls: the async build uses
# cudaMemcpyAsync on the default stream (stream 0), still followed by
# cudaDeviceSynchronize. Run `diff` on the sources to see the change.
all: memory_transfer_benchmark_sync memory_transfer_benchmark_async

memory_transfer_benchmark_sync: memory_transfer_benchmark_sync.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

memory_transfer_benchmark_async: memory_transfer_benchmark_async.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f memory_transfer_benchmark_sync memory_transfer_benchmark_async

.PHONY: all clean
