TORCH=$(python -c 'import torch;print(torch.__path__[0]')
nvcc -std=c++17 -arch=sm_90 \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$GTEST_ROOT/include \
    forward_tests.cu kernel.cu \
    -L$TORCH/lib \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda # links library files \
    -L$GTEST_ROOT/lib64 -lgtest -lgtest_main \
    -o fa_forward && ./fa_forward

nvcc -std=c++17 -arch=sm_90 \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$GTEST_ROOT/include \
    backward_tests.cu kernel.cu \
    -L$TORCH/lib \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda # links library files \
    -L$GTEST_ROOT/lib64 -lgtest -lgtest_main \
    -o fa_backward && ./fa_backward