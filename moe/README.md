# MoE

### Testing
```shell
nvcc -O3 -std=c++20 -arch=sm_90 tests.cu experts_kernel.cu -o moe_tests
./moe_tests
```
