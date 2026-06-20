# pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void count_kernel(
    // INPUT
    const int* expert_idx, // [B, K]
    // OUTPUT
    int* counts, // [N]
    // CONST
    int total // BK
);

__global__ void prefix_sum_kernel(
    // INPUT
    const int* counts, // [N]
    // OUTPUT
    uint32_t* expert_offsets, // [N + 1]
    // CONST
    int N
);

template <typename T>
__global__ void scatter_kernel(
    // INPUT
    const int* expert_idx, // [B, K]
    const T* x, // [B, D]
    const T* gate_weights, // [B, K]
    // IMM
    uint32_t* write_cursor, // [N]
    // OUTPUT
    int* sorted_token_idx, // [BK]
    T* sorted_x, // [BK, D]
    T* sorted_gate, // [BK]
    // CONST
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
        sorted_gate[write_idx] = gate_weights[idx];
        sorted_token_idx[write_idx] = token_idx;
    }
}

template <typename T>
__global__ void combine_kernel(
    // INPUT
    const T* expert_out, // [BK, D]
    const int* sorted_token_idx, // [BK] -> [0, B)
    const T* sorted_gate, // [BK]
    // OUTPUT
    T* out, // [B, D]
    // CONST
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

template <typename T>
void gather_launcher(
    // INPUT
    torch::Tensor expert_idx, // [B, K], int32
    torch::Tensor x, // [B, D], T
    torch::Tensor gate_weights, // [B, K], T
    // OUTPUT
    torch::Tensor sorted_x, // [BK, D], T
    torch::Tensor sorted_token_idx, // [BK], int32, [0, B)
    torch::Tensor sorted_gate, // [BK], T
    torch::Tensor expert_offsets, // [N + 1], uint32
    // CONST
    int N
){
    int B = x.size(0), K = expert_idx.size(1), D = x.size(1);
    int TOTAL = B * K;
    int* counts;
    cudaMalloc(&counts, N*sizeof(int));
    cudaMemset(counts, 0, N*sizeof(int));
    int THREADS = 256;
    int BLOCKS = (B*K + THREADS - 1)/THREADS;
    count_kernel<<<BLOCKS, THREADS>>>(expert_idx.data_ptr<int>(), counts, TOTAL);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    prefix_sum_kernel<<<1,1>>>(counts, expert_offsets.data_ptr<uint32_t>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    uint32_t* write_cursor;
    cudaMalloc(&write_cursor, N * sizeof(uint32_t));
    cudaMemcpy(write_cursor, expert_offsets.data_ptr<uint32_t>(), N*sizeof(uint32_t), cudaMemcpyDeviceToDevice); // fill with expert_offsets[:N]
    scatter_kernel<T><<<BLOCKS, THREADS>>>(expert_idx.data_ptr<int>(), x.data_ptr<T>(), gate_weights.data_ptr<T>(), write_cursor, sorted_token_idx.data_ptr<int>(), sorted_x.data_ptr<T>(), sorted_gate.data_ptr<T>(), B, D, K, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    cudaFree(counts);
    cudaFree(write_cursor);
}

template <typename T>
void combine_launcher(
    // INPUT
    torch::Tensor expert_out, // [BK, D]
    torch::Tensor sorted_token_idx, // [BK]
    torch::Tensor sorted_gate, // [BK], T
    // OUTPUT
    torch::Tensor out // [B, D]
){
    int BK = expert_out.size(0);
    int D = expert_out.size(1);
    int THREADS = 256;
    int BLOCKS = (BK + THREADS - 1) / THREADS;
    out.zero_();
    combine_kernel<T><<<BLOCKS, THREADS>>>(expert_out.data_ptr<T>(), sorted_token_idx.data_ptr<int>(), sorted_gate.data_ptr<T>(), out.data_ptr<T>(), BK, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}