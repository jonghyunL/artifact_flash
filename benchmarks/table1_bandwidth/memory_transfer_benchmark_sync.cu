#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple kernel to touch device memory
__global__ void touchMemory(char *data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1;
    }
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmarkTransfer(size_t sizeGB) {
    size_t sizeBytes = sizeGB * 1024ULL * 1024ULL * 1024ULL;
    char *h_data = NULL;
    char *d_data = NULL;

    printf("\n=== Benchmarking %lu GB (%lu bytes) ===\n", sizeGB, sizeBytes);

    // Allocate host memory (pinned for faster transfer)
    CUDA_CHECK(cudaMallocHost((void**)&h_data, sizeBytes));

    // Initialize host data
    printf("Initializing host data...\n");
    for (size_t i = 0; i < sizeBytes; i += 4096) {
        h_data[i] = (char)(i % 256);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_data, sizeBytes));

    // Warm-up transfer
    CUDA_CHECK(cudaMemcpy(d_data, h_data, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark Host to Device transfer
    double start = getTime();
    CUDA_CHECK(cudaMemcpy(d_data, h_data, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double end = getTime();
    double htodTime = end - start;
    double htodBandwidth = (sizeBytes / (1024.0 * 1024.0 * 1024.0)) / htodTime;

    printf("Host to Device: %.3f seconds, Bandwidth: %.2f GB/s\n", htodTime, htodBandwidth);

    // Touch device memory to ensure it's on GPU
    int blockSize = 256;
    int numBlocks = (sizeBytes + blockSize - 1) / blockSize;
    touchMemory<<<numBlocks, blockSize>>>(d_data, sizeBytes);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark Device to Host transfer
    start = getTime();
    CUDA_CHECK(cudaMemcpy(h_data, d_data, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    end = getTime();
    double dtohTime = end - start;
    double dtohBandwidth = (sizeBytes / (1024.0 * 1024.0 * 1024.0)) / dtohTime;

    printf("Device to Host: %.3f seconds, Bandwidth: %.2f GB/s\n", dtohTime, dtohBandwidth);

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));
}

int main(int argc, char **argv) {
    // Print GPU information
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return EXIT_FAILURE;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using Device 0: %s\n", prop.name);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    // Test data sizes in GB
    size_t dataSizes[] = {1, 10, 20, 30, 40, 50};
    int numSizes = sizeof(dataSizes) / sizeof(dataSizes[0]);

    printf("\n========================================\n");
    printf("Starting Memory Transfer Benchmarks\n");
    printf("========================================\n");

    for (int i = 0; i < numSizes; i++) {
        // Check if we have enough memory
        size_t freeMem, totalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        size_t requiredMem = dataSizes[i] * 1024ULL * 1024ULL * 1024ULL;

        if (freeMem < requiredMem) {
            printf("\n=== Skipping %lu GB (insufficient GPU memory: %.2f GB free) ===\n",
                   dataSizes[i], freeMem / (1024.0 * 1024.0 * 1024.0));
            continue;
        }

        benchmarkTransfer(dataSizes[i]);
    }

    printf("\n========================================\n");
    printf("Benchmark Complete\n");
    printf("========================================\n");

    return EXIT_SUCCESS;
}
