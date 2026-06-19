#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "experts_kernel.cuh"

__global__ void count_kernel(
    const int* expert_idx, // [B, K]
    int* counts, // [N]
    int total // BK
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total){
        atomicAdd(&counts[expert_idx[idx]], 1);
    }
}

__global__ void prefix_sum_kernel(
    const int* counts, // [N]
    uint32_t* expert_offsets, // [N + 1]
    int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0){
        expert_offsets[0] = 0;
        for (int i = 0; i < N; i++){
            expert_offsets[i + 1] = expert_offsets[i] + counts[i];
        }
    }
}
