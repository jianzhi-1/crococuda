#include <torch/extensions.h>

std::pair<torch::Tensor, torch::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double scale,
    int Br,
    int Bc
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attention_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L,
    double scale,
    int Br,
    int Bc
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &flash_attention_forward, "flash_attention_forward");
    m.def("backward", &flash_attention_backward, "flash_attention_backward");
}
