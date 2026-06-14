#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "checks.h"

template <typename T>
__global__ void silu_forward_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t n
){
    // x -> x * sigma(x)
    using acc_t = at::acc_type<T, true>;
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        const acc_t xf = static_cast<acc_t>(x[i]);
        const acc_t s = acc_t(1) / (acc_t(1) + exp(-xf));
        y[i] = static_cast<T>(xf * s);
    }
}

template <typename T>
__global__ void silu_backward_kernel(
    const T* __restrict__ grad_y,
    const T* __restrict__ x,
    T* __restrict__ grad_x,
    const int64_t n
){
    // grad_y -> grad_y * (sigma(x) + silu(x) - sigma(x) * silu(x))
    using acc_t = at::acc_type<T, true>;
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        const acc_t xf = static_cast<acc_t>(x[i]);
        const acc_t s = acc_t(1) / (acc_t(1) + exp(-xf));
        const acc_t dsilu = s * (acc_t(1) + xf * (acc_t(1) - s));
        grad_x[i] = static_cast<T>(static_cast<acc_t>(grad_y[i]) * dsilu);
    }
}

torch::Tensor silu_forward(torch::Tensor x){
    CHECK_INPUT(x);
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return y;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "silu_forward", ([&]{
            silu_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                n
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor silu_backward(torch::Tensor grad_y, torch::Tensor x){
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y and x must have the same shape");
    TORCH_CHECK(grad_y.scalar_type() == x.scalar_type(), "grad_y and x must have the same dtype");
    TORCH_CHECK(grad_y.device() == x.device(), "grad_y and x must be on the same device");
    auto grad_x = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return grad_x;

    const int threads = 256;
    const int blocks = (n + threads - 1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "silu_backward", ([&]{
            silu_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_y.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                n
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &silu_forward, "SiLU forward");
    m.def("backward", &silu_backward, "SiLU backward");
}
