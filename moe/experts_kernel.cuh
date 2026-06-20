# pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void count_topk_kernel(
    // INPUT
    const int* topk_indices, // [B, K]
    // OUTPUT
    int* per_expert_counters, // [N]
    // CONST
    int BK // BK
);

__global__ void prefix_sum_kernel(
    // INPUT
    const int* per_expert_counters, // [N]
    // OUTPUT
    uint32_t* per_expert_offsets, // [N + 1]
    // CONST
    int N
);

template <typename T>
__global__ void partition_by_experts_kernel(
    // INPUT
    const int* topk_indices, // [B, K]
    const T* x, // [B, D]
    const T* weights, // [B, K]
    // IMM
    uint32_t* expert_partitions_cursor, // [N]
    // OUTPUT
    int* expert_partitions_to_x_idx, // [BK]
    T* expert_partitions_to_x, // [BK, D]
    T* expert_partitions_to_weight, // [BK]
    // CONST
    int B, int D, int K, int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B*K){
        int token_idx = idx / K;
        int expert_id = topk_indices[idx];
        int write_idx = atomicAdd(&expert_partitions_cursor[expert_id], 1);
        for (int d = 0; d < D; d++){
            expert_partitions_to_x[write_idx * D + d] = x[token_idx * D + d];
        }
        expert_partitions_to_weight[write_idx] = weights[idx];
        expert_partitions_to_x_idx[write_idx] = token_idx;
    }
}

template <typename T>
__global__ void weighted_experts_kernel(
    // INPUT
    const T* expert_partitions_to_output, // [BK, D]
    const int* expert_partitions_to_x_idx, // [BK] -> [0, B)
    const T* expert_partitions_to_weight, // [BK]
    // OUTPUT
    T* out, // [B, D]
    // CONST
    int BK, int D
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BK){
        int token_idx = expert_partitions_to_x_idx[idx];
        T weight = expert_partitions_to_weight[idx];
        for (int i = 0; i < D; i++){
            atomicAdd(out + token_idx * D + i, weight * expert_partitions_to_output[idx * D + i]);
        }
    }
}

template <typename T>
void partition_by_experts_launcher(
    // INPUT
    torch::Tensor topk_indices, // [B, K], int32
    torch::Tensor x, // [B, D], T
    torch::Tensor weights, // [B, K], T
    // OUTPUT
    torch::Tensor expert_partitions_to_x, // [BK, D], T
    torch::Tensor expert_partitions_to_x_idx, // [BK], int32, [0, B)
    torch::Tensor expert_partitions_to_weight, // [BK], T
    torch::Tensor per_expert_offsets, // [N + 1], uint32
    // CONST
    int N
){
    int B = x.size(0), K = topk_indices.size(1), D = x.size(1);
    int TOTAL = B * K;
    int* per_expert_counters;
    cudaMalloc(&per_expert_counters, N*sizeof(int));
    cudaMemset(per_expert_counters, 0, N*sizeof(int));
    int THREADS = 256;
    int BLOCKS = (B*K + THREADS - 1)/THREADS;
    count_topk_kernel<<<BLOCKS, THREADS>>>(topk_indices.data_ptr<int>(), per_expert_counters, TOTAL);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    prefix_sum_kernel<<<1,1>>>(per_expert_counters, per_expert_offsets.data_ptr<uint32_t>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    uint32_t* expert_partitions_cursor;
    cudaMalloc(&expert_partitions_cursor, N * sizeof(uint32_t));
    cudaMemcpy(expert_partitions_cursor, per_expert_offsets.data_ptr<uint32_t>(), N*sizeof(uint32_t), cudaMemcpyDeviceToDevice); // fill with per_expert_offsets[:N]
    partition_by_experts_kernel<T><<<BLOCKS, THREADS>>>(topk_indices.data_ptr<int>(), x.data_ptr<T>(), weights.data_ptr<T>(), expert_partitions_cursor, expert_partitions_to_x_idx.data_ptr<int>(), expert_partitions_to_x.data_ptr<T>(), expert_partitions_to_weight.data_ptr<T>(), B, D, K, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    cudaFree(per_expert_counters);
    cudaFree(expert_partitions_cursor);
}

template <typename T>
void weighted_experts_launcher(
    // INPUT
    torch::Tensor expert_partitions_to_output, // [BK, D]
    torch::Tensor expert_partitions_to_x_idx, // [BK]
    torch::Tensor expert_partitions_to_weight, // [BK], T
    // OUTPUT
    torch::Tensor out // [B, D]
){
    int BK = expert_partitions_to_output.size(0);
    int D = expert_partitions_to_output.size(1);
    int THREADS = 256;
    int BLOCKS = (BK + THREADS - 1) / THREADS;
    out.zero_();
    weighted_experts_kernel<T><<<BLOCKS, THREADS>>>(expert_partitions_to_output.data_ptr<T>(), expert_partitions_to_x_idx.data_ptr<int>(), expert_partitions_to_weight.data_ptr<T>(), out.data_ptr<T>(), BK, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}