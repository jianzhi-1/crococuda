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

template <typename T>
__global__ void combine_kernel(
    const T* expert_out, // [BK, D]
    const int* sorted_token_idx, // [BK] -> [0, B)
    const T* sorted_gate, // [BK]
    T* out, // [B, D]
    int BK, int D
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BK){
        int token_idx = sorted_token_idx[idx];
        T weight = sorted_gate[idx];
        for (int i = 0; i < D; i++){
            atomicAdd(out + token_idx * D + i, weight * expert_out[idx * D + i]);
        }
    }
}
