from torch.utils.cpp_extension import load
moe_kernels = load(name="moe_kernels", sources=["experts_kernel.cu"], verbose=True)
