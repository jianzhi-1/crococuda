#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void count(
    const int* expert_idx, // [B, K]
    int* counts, // [N]
    int total // BK
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total){
        atomicAdd(&counts[expert_idx[idx]], 1);
    }
}

__global__ void prefix_sum(
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
__global__ void scatter(
    const int* expert_idx, // [B, K]
    const T* x, // [B, D]
    uint32_t* write_cursor, // [N]
    int* sorted_token_idx, // [BK]
    T* sorted_x, // [BK, D]
    int B, int D, int K, int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B*K){
        int token_idx = idx / K;
        int expert_id = expert_idx[idx];
        int write_idx = atomicAdd(&write_cursor[expert_id], 1);
        for (int d = 0; d < D; d++){
            sorted_x[write_idx * D + d] = x[token_idx * D + d];
        }
        sorted_token_idx[write_idx] = token_idx;
    }
}
