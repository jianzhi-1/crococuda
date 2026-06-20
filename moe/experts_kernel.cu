#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "experts_kernel.cuh"

__global__ void count_topk_kernel(
    const int* topk_indices, // [B, K]
    int* per_expert_counters, // [N]
    int BK // BK
){
    /*
    Example: B = 5, K = 3, N = 4
    [
        [0, 1, 3],
        [1, 2, 3],
        [3, 1, 0],
        [2, 3, 1],
        [3, 2, 0]
    ]
    
    Expected output:
    [3, 4, 3, 5]
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BK){
        atomicAdd(&per_expert_counters[topk_indices[idx]], 1);
    }
}

__global__ void prefix_sum_kernel(
    const int* per_expert_counters, // [N]
    uint32_t* per_expert_offsets, // [N + 1]
    int N
){
    /*
    Example: N = 10
    [8, 2, 3, 5, 3, 2, 0, 10, 2, 3]

    Expected output:
    [0, 8, 10, 13, 18, 21, 23, 23, 33, 35, 38]
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0){
        per_expert_offsets[0] = 0;
        for (int i = 0; i < N; i++){
            per_expert_offsets[i + 1] = per_expert_offsets[i] + per_expert_counters[i];
        }
    }
}

// Specialized for PyBind
void partition_by_expert(
    // INPUT
    torch::Tensor topk_indices, torch::Tensor x, torch::Tensor weights,
    // OUTPUT
    torch::Tensor expert_partitions_to_x, torch::Tensor expert_partitions_to_x_idx, torch::Tensor expert_partitions_to_weight,
    torch::Tensor per_expert_offsets,
    // CONST
    int N
){
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "partition_by_experts_launcher", [&]{
        partition_by_experts_launcher<scalar_t>(topk_indices, x, weights, expert_partitions_to_x, expert_partitions_to_x_idx, expert_partitions_to_weight, per_expert_offsets, N);
    });
}

// Specialized for PyBind
void reduce_from_expert_partitions(
    // INPUT
    torch::Tensor expert_partitions_to_output,
    torch::Tensor expert_partitions_to_x_idx,
    torch::Tensor expert_partitions_to_weight,
    // OUTPUT
    torch::Tensor out
){
    AT_DISPATCH_FLOATING_TYPES(expert_partitions_to_output.scalar_type(), "weighted_experts_launcher", [&]{
        weighted_experts_launcher<scalar_t>(expert_partitions_to_output, expert_partitions_to_x_idx, expert_partitions_to_weight, out);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("partition_by_expert", &partition_by_expert, "MoE partition_by_expert");
    m.def("reduce_from_expert_partitions", &reduce_from_expert_partitions, "MoE reduce_from_expert_partitions");
}