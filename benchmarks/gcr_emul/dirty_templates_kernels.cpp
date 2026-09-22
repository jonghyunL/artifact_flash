void dirty_templates__ZN4vllm30reshape_and_cache_flash_kernelIttLNS_18Fp8KVCacheDataTypeE0EEEvPKT_S4_PT0_S6_PKlllllliiiPKfSA_(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t param_0 = *(uint64_t*)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t param_1 = *(uint64_t*)*(void**)((char*)kernelParams + 1 * 8);
  uint64_t param_2 = *(uint64_t*)*(void**)((char*)kernelParams + 2 * 8);
  uint64_t param_3 = *(uint64_t*)*(void**)((char*)kernelParams + 3 * 8);
  uint64_t* param_4_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 4 * 8);
  uint64_t param_5 = *(uint64_t*)*(void**)((char*)kernelParams + 5 * 8);
  uint64_t param_6 = *(uint64_t*)*(void**)((char*)kernelParams + 6 * 8);
  uint64_t param_7 = *(uint64_t*)*(void**)((char*)kernelParams + 7 * 8);
  uint64_t param_8 = *(uint64_t*)*(void**)((char*)kernelParams + 8 * 8);
  uint64_t param_9 = *(uint64_t*)*(void**)((char*)kernelParams + 9 * 8);
  uint32_t param_10 = *(uint32_t*)*(void**)((char*)kernelParams + 10 * 8);
  uint32_t param_11 = *(uint32_t*)*(void**)((char*)kernelParams + 11 * 8);
  uint32_t param_12 = *(uint32_t*)*(void**)((char*)kernelParams + 12 * 8);
  uint64_t param_13 = *(uint64_t*)*(void**)((char*)kernelParams + 13 * 8);
  uint64_t param_14 = *(uint64_t*)*(void**)((char*)kernelParams + 14 * 8);

  // Copy array parameters from GPU to CPU
  static CUstream aux_stream = NULL;
  if (aux_stream == NULL) {
    CU_CHECK(real_cuStreamCreate(&aux_stream, CU_STREAM_NON_BLOCKING));
  }

  uint64_t* param_4_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_4_cpu, (CUdeviceptr)param_4_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_4 = param_4_cpu;

  CU_CHECK(real_cuStreamSynchronize(aux_stream));

  for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
    if (param_4[blockIdxX] < 0) {
      return;
    }
    if (param_7 == param_11) {
      // Optimized: merged consecutive addresses
      uint64_t min_addr_0 = (param_2 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((0) / param_11) * param_7)) + ((0) % param_11)) * 2));
      uint64_t max_addr_0 = (param_2 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((((param_11 * param_10) - 1)) / param_11) * param_7)) + ((((param_11 * param_10) - 1)) % param_11)) * 2));
      add_and_merge_dirty_address((void*)min_addr_0, (void*)((uint64_t)max_addr_0 + 2));
      uint64_t min_addr_1 = (param_3 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((0) / param_11) * param_7)) + ((0) % param_11)) * 2));
      uint64_t max_addr_1 = (param_3 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((((param_11 * param_10) - 1)) / param_11) * param_7)) + ((((param_11 * param_10) - 1)) % param_11)) * 2));
      add_and_merge_dirty_address((void*)min_addr_1, (void*)((uint64_t)max_addr_1 + 2));
    } else {
      for (int i = 0; i < (param_11 * param_10); i++) {
    uint64_t addr_0 = (param_2 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + ((i / param_11) * param_7)) + (i % param_11)) * 2));
    add_and_merge_dirty_address((void*)addr_0, (void*)((uint64_t)addr_0 + 2));
    uint64_t addr_1 = (param_3 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + ((i / param_11) * param_7)) + (i % param_11)) * 2));
    add_and_merge_dirty_address((void*)addr_1, (void*)((uint64_t)addr_1 + 2));
  }
      }
  }

  // Free allocated CPU memory for array parameters
  free(param_4_cpu);
}

void dirty_templates__ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t param_0 = *(uint64_t*)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t param_1 = *(uint64_t*)*(void**)((char*)kernelParams + 1 * 8);
  uint64_t param_2 = *(uint64_t*)*(void**)((char*)kernelParams + 2 * 8);
  uint64_t param_3 = *(uint64_t*)*(void**)((char*)kernelParams + 3 * 8);
  uint64_t* param_4_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 4 * 8);
  uint64_t param_5 = *(uint64_t*)*(void**)((char*)kernelParams + 5 * 8);
  uint64_t param_6 = *(uint64_t*)*(void**)((char*)kernelParams + 6 * 8);
  uint64_t param_7 = *(uint64_t*)*(void**)((char*)kernelParams + 7 * 8);
  uint64_t param_8 = *(uint64_t*)*(void**)((char*)kernelParams + 8 * 8);
  uint64_t param_9 = *(uint64_t*)*(void**)((char*)kernelParams + 9 * 8);
  uint32_t param_10 = *(uint32_t*)*(void**)((char*)kernelParams + 10 * 8);
  uint32_t param_11 = *(uint32_t*)*(void**)((char*)kernelParams + 11 * 8);
  uint32_t param_12 = *(uint32_t*)*(void**)((char*)kernelParams + 12 * 8);
  uint64_t param_13 = *(uint64_t*)*(void**)((char*)kernelParams + 13 * 8);
  uint64_t param_14 = *(uint64_t*)*(void**)((char*)kernelParams + 14 * 8);

  // Copy array parameters from GPU to CPU
  static CUstream aux_stream = NULL;
  if (aux_stream == NULL) {
    CU_CHECK(real_cuStreamCreate(&aux_stream, CU_STREAM_NON_BLOCKING));
  }

  uint64_t* param_4_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_4_cpu, (CUdeviceptr)param_4_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_4 = param_4_cpu;

  CU_CHECK(real_cuStreamSynchronize(aux_stream));

  for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
    if (param_4[blockIdxX] < 0) {
      return;
    }
    if (param_7 == param_11) {
      // Optimized: merged consecutive addresses
      uint64_t min_addr_0 = (param_2 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((0) / param_11) * param_7)) + ((0) % param_11)) * 2));
      uint64_t max_addr_0 = (param_2 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((((param_11 * param_10) - 1)) / param_11) * param_7)) + ((((param_11 * param_10) - 1)) % param_11)) * 2));
      add_and_merge_dirty_address((void*)min_addr_0, (void*)((uint64_t)max_addr_0 + 2));
      uint64_t min_addr_1 = (param_3 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((0) / param_11) * param_7)) + ((0) % param_11)) * 2));
      uint64_t max_addr_1 = (param_3 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + (((((param_11 * param_10) - 1)) / param_11) * param_7)) + ((((param_11 * param_10) - 1)) % param_11)) * 2));
      add_and_merge_dirty_address((void*)min_addr_1, (void*)((uint64_t)max_addr_1 + 2));
    } else {
      for (int i = 0; i < (param_11 * param_10); i++) {
    uint64_t addr_0 = (param_2 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + ((i / param_11) * param_7)) + (i % param_11)) * 2));
    add_and_merge_dirty_address((void*)addr_0, (void*)((uint64_t)addr_0 + 2));
    uint64_t addr_1 = (param_3 + ((((((param_4[blockIdxX] % param_12) * param_6) + ((param_4[blockIdxX] / param_12) * param_5)) + ((i / param_11) * param_7)) + (i % param_11)) * 2));
    add_and_merge_dirty_address((void*)addr_1, (void*)((uint64_t)addr_1 + 2));
  }
      }
  }

  // Free allocated CPU memory for array parameters
  free(param_4_cpu);
}

void dirty_templates__ZN4vllm18copy_blocks_kernelIN3c108BFloat16EEEvPlS3_PKli(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t* param_0_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t* param_1_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 1 * 8);
  uint64_t* param_2_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 2 * 8);
  uint32_t param_3 = *(uint32_t*)*(void**)((char*)kernelParams + 3 * 8);

  // Copy array parameters from GPU to CPU
  static CUstream aux_stream = NULL;
  if (aux_stream == NULL) {
    CU_CHECK(real_cuStreamCreate(&aux_stream, CU_STREAM_NON_BLOCKING));
  }

  uint64_t* param_0_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_0_cpu, (CUdeviceptr)param_0_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_0 = param_0_cpu;

  uint64_t* param_1_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_1_cpu, (CUdeviceptr)param_1_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_1 = param_1_cpu;

  uint64_t* param_2_cpu = (uint64_t*)malloc((gridDimY * 2) * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_2_cpu, (CUdeviceptr)param_2_gaddr, (gridDimY * 2) * sizeof(uint64_t), aux_stream));
  uint64_t* param_2 = param_2_cpu;

  CU_CHECK(real_cuStreamSynchronize(aux_stream));

  for (int blockIdxY = 0; blockIdxY < gridDimY; blockIdxY++) {
    for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
        // Optimized: merged consecutive addresses for addr_0
        uint64_t min_addr_0 = (param_0[blockIdxX] + (((param_2[((blockIdxY * 2) + 1)] * param_3) + (0)) * 2));
        uint64_t max_addr_0 = (param_0[blockIdxX] + (((param_2[((blockIdxY * 2) + 1)] * param_3) + ((param_3 - 1))) * 2));
        add_and_merge_dirty_address((void*)min_addr_0, (void*)((uint64_t)max_addr_0 + 2));
        // Optimized: merged consecutive addresses for addr_1
        uint64_t min_addr_1 = (param_1[blockIdxX] + (((param_2[((blockIdxY * 2) + 1)] * param_3) + (0)) * 2));
        uint64_t max_addr_1 = (param_1[blockIdxX] + (((param_2[((blockIdxY * 2) + 1)] * param_3) + ((param_3 - 1))) * 2));
        add_and_merge_dirty_address((void*)min_addr_1, (void*)((uint64_t)max_addr_1 + 2));
    }
  }

  // Free allocated CPU memory for array parameters
  free(param_0_cpu);
  free(param_1_cpu);
  free(param_2_cpu);
}

void dirty_templates__ZN4vllm22copy_blocks_mla_kernelIN3c108BFloat16EEEvPlPKli(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t* param_0_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t* param_1_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 1 * 8);
  uint32_t param_2 = *(uint32_t*)*(void**)((char*)kernelParams + 2 * 8);

  // Copy array parameters from GPU to CPU
  static CUstream aux_stream = NULL;
  if (aux_stream == NULL) {
    CU_CHECK(real_cuStreamCreate(&aux_stream, CU_STREAM_NON_BLOCKING));
  }

  uint64_t* param_0_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_0_cpu, (CUdeviceptr)param_0_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_0 = param_0_cpu;

  uint64_t* param_1_cpu = (uint64_t*)malloc((gridDimY * 2) * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_1_cpu, (CUdeviceptr)param_1_gaddr, (gridDimY * 2) * sizeof(uint64_t), aux_stream));
  uint64_t* param_1 = param_1_cpu;

  CU_CHECK(real_cuStreamSynchronize(aux_stream));

  for (int blockIdxY = 0; blockIdxY < gridDimY; blockIdxY++) {
    for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
        // Optimized: merged consecutive addresses for addr_0
        uint64_t min_addr_0 = (param_0[blockIdxX] + (((param_1[((blockIdxY * 2) + 1)] * param_2) + (0)) * 2));
        uint64_t max_addr_0 = (param_0[blockIdxX] + (((param_1[((blockIdxY * 2) + 1)] * param_2) + ((param_2 - 1))) * 2));
        add_and_merge_dirty_address((void*)min_addr_0, (void*)((uint64_t)max_addr_0 + 2));
    }
  }

  // Free allocated CPU memory for array parameters
  free(param_0_cpu);
  free(param_1_cpu);
}

void dirty_templates__ZN4vllm24reshape_and_cache_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKliiiiiiPKfSB_(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t param_0 = *(uint64_t*)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t param_1 = *(uint64_t*)*(void**)((char*)kernelParams + 1 * 8);
  uint64_t param_2 = *(uint64_t*)*(void**)((char*)kernelParams + 2 * 8);
  uint64_t param_3 = *(uint64_t*)*(void**)((char*)kernelParams + 3 * 8);
  uint64_t* param_4_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 4 * 8);
  uint32_t param_5 = *(uint32_t*)*(void**)((char*)kernelParams + 5 * 8);
  uint32_t param_6 = *(uint32_t*)*(void**)((char*)kernelParams + 6 * 8);
  uint32_t param_7 = *(uint32_t*)*(void**)((char*)kernelParams + 7 * 8);
  uint32_t param_8 = *(uint32_t*)*(void**)((char*)kernelParams + 8 * 8);
  uint32_t param_9 = *(uint32_t*)*(void**)((char*)kernelParams + 9 * 8);
  uint32_t param_10 = *(uint32_t*)*(void**)((char*)kernelParams + 10 * 8);
  uint64_t param_11 = *(uint64_t*)*(void**)((char*)kernelParams + 11 * 8);
  uint64_t param_12 = *(uint64_t*)*(void**)((char*)kernelParams + 12 * 8);

  // Copy array parameters from GPU to CPU
  static CUstream aux_stream = NULL;
  if (aux_stream == NULL) {
    CU_CHECK(real_cuStreamCreate(&aux_stream, CU_STREAM_NON_BLOCKING));
  }

  uint64_t* param_4_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_4_cpu, (CUdeviceptr)param_4_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_4 = param_4_cpu;

  CU_CHECK(real_cuStreamSynchronize(aux_stream));

  for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
    if (param_4[blockIdxX] < 0) {
      return;
    }
    for (int i = 0; i < (param_8 * param_7); i++) {
    uint64_t addr_0 = (param_2 + ((((((param_10 * param_9) * ((i % param_8) / param_10)) + ((i % param_8) % param_10)) + (((param_10 * param_9) * (i / param_8)) * (param_8 / param_10))) + ((((((param_4[blockIdxX] / param_9) * param_7) * param_9) * (param_8 / param_10)) + (param_4[blockIdxX] % param_9)) * param_10)) * 2));
    add_and_merge_dirty_address((void*)addr_0, (void*)((uint64_t)addr_0 + 2));
    uint64_t addr_1 = (param_3 + ((((((((param_4[blockIdxX] / param_9) * param_7) * param_9) * param_8) + (param_4[blockIdxX] % param_9)) + ((param_9 * param_8) * (i / param_8))) + ((i % param_8) * param_9)) * 2));
    add_and_merge_dirty_address((void*)addr_1, (void*)((uint64_t)addr_1 + 2));
  }
  }

  // Free allocated CPU memory for array parameters
  free(param_4_cpu);
}

void dirty_templates__ZN4vllm18convert_fp8_kernelIh13__nv_bfloat16LNS_18Fp8KVCacheDataTypeE1EEEvPKT0_PT_fl(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t param_0 = *(uint64_t*)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t param_1 = *(uint64_t*)*(void**)((char*)kernelParams + 1 * 8);
  float param_2 = *(float*)*(void**)((char*)kernelParams + 2 * 8);
  uint64_t param_3 = *(uint64_t*)*(void**)((char*)kernelParams + 3 * 8);
  for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
      // Optimized: merged consecutive addresses for addr_0
      uint64_t min_addr_0 = (param_1 + ((0) + (blockIdxX * param_3)));
      uint64_t max_addr_0 = (param_1 + (((param_3 - 1)) + (blockIdxX * param_3)));
      add_and_merge_dirty_address((void*)min_addr_0, (void*)((uint64_t)max_addr_0 + 1));
  }
}

void dirty_templates__ZN4vllm27concat_and_cache_mla_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_PKliiiiiiiPKf(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) {
  uint64_t param_0 = *(uint64_t*)*(void**)((char*)kernelParams + 0 * 8);
  uint64_t param_1 = *(uint64_t*)*(void**)((char*)kernelParams + 1 * 8);
  uint64_t param_2 = *(uint64_t*)*(void**)((char*)kernelParams + 2 * 8);
  uint64_t* param_3_gaddr = *(uint64_t**)*(void**)((char*)kernelParams + 3 * 8);
  uint32_t param_4 = *(uint32_t*)*(void**)((char*)kernelParams + 4 * 8);
  uint32_t param_5 = *(uint32_t*)*(void**)((char*)kernelParams + 5 * 8);
  uint32_t param_6 = *(uint32_t*)*(void**)((char*)kernelParams + 6 * 8);
  uint32_t param_7 = *(uint32_t*)*(void**)((char*)kernelParams + 7 * 8);
  uint32_t param_8 = *(uint32_t*)*(void**)((char*)kernelParams + 8 * 8);
  uint32_t param_9 = *(uint32_t*)*(void**)((char*)kernelParams + 9 * 8);
  uint32_t param_10 = *(uint32_t*)*(void**)((char*)kernelParams + 10 * 8);
  uint64_t param_11 = *(uint64_t*)*(void**)((char*)kernelParams + 11 * 8);

  // Copy array parameters from GPU to CPU
  static CUstream aux_stream = NULL;
  if (aux_stream == NULL) {
    CU_CHECK(real_cuStreamCreate(&aux_stream, CU_STREAM_NON_BLOCKING));
  }

  uint64_t* param_3_cpu = (uint64_t*)malloc(gridDimX * sizeof(uint64_t));
  CU_CHECK(real_cuMemcpyDtoHAsync_v2(param_3_cpu, (CUdeviceptr)param_3_gaddr, gridDimX * sizeof(uint64_t), aux_stream));
  uint64_t* param_3 = param_3_cpu;

  CU_CHECK(real_cuStreamSynchronize(aux_stream));

  for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
    if (param_3[blockIdxX] < 0) {
      return;
    }
      // Optimized: merged consecutive addresses for addr_0
      uint64_t min_addr_0 = (param_2 + (((((param_3[blockIdxX] % param_10) * param_5) + ((param_3[blockIdxX] / param_10) * param_4)) + (0)) * 2));
      uint64_t max_addr_0 = (param_2 + (((((param_3[blockIdxX] % param_10) * param_5) + ((param_3[blockIdxX] / param_10) * param_4)) + ((param_8 - 1))) * 2));
      add_and_merge_dirty_address((void*)min_addr_0, (void*)((uint64_t)max_addr_0 + 2));
      // Optimized: merged consecutive addresses for addr_1
      uint64_t min_addr_1 = (param_2 + ((((((param_3[blockIdxX] / param_10) * param_4) + param_8) + ((param_3[blockIdxX] % param_10) * param_5)) + (0)) * 2));
      uint64_t max_addr_1 = (param_2 + ((((((param_3[blockIdxX] / param_10) * param_4) + param_8) + ((param_3[blockIdxX] % param_10) * param_5)) + ((param_9 - 1))) * 2));
      add_and_merge_dirty_address((void*)min_addr_1, (void*)((uint64_t)max_addr_1 + 2));
  }

  // Free allocated CPU memory for array parameters
  free(param_3_cpu);
}