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

// Specialized for PyBind
void gather(
    // INPUT
    torch::Tensor expert_idx, torch::Tensor x, torch::Tensor gate_weights,
    // OUTPUT
    torch::Tensor sorted_x, torch::Tensor sorted_token_idx, torch::Tensor sorted_gate,
    torch::Tensor expert_offsets,
    // CONST
    int N
){
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gather_launcher", [&]{
        gather_launcher<scalar_t>(expert_idx, x, gate_weights, sorted_x, sorted_token_idx, sorted_gate, expert_offsets, N);
    });
}

// Specialized for PyBind
void combine(
    // INPUT
    torch::Tensor expert_out,
    torch::Tensor sorted_token_idx,
    torch::Tensor sorted_gate,
    // OUTPUT
    torch::Tensor out
){
    AT_DISPATCH_FLOATING_TYPES(expert_out.scalar_type(), "combine_launcher", [&]{
        combine_launcher<scalar_t>(expert_out, sorted_token_idx, sorted_gate, out);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("gather", &gather, "MoE gather");
    m.def("combine", &combine, "MoE combine");
}