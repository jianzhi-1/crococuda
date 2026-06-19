# pragma once
#include <cstdint>
#include <cuda_runtime.h>

__global__ void count_kernel(
    const int* expert_idx, // [B, K]
    int* counts, // [N]
    int total // BK
);

__global__ void prefix_sum_kernel(
    const int* counts, // [N]
    uint32_t* expert_offsets, // [N + 1]
    int N
);

template <typename T>
__global__ void scatter_kernel(
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
