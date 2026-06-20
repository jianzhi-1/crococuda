#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "experts_kernel.cuh"
int main(){
    int B = 5;
    int K = 2;
    int N = 4;
    int D = 8;
    std::vector<int> topk_indices_data = {
        0, 2,
        3, 1,
        0, 3,
        0, 3,
        1, 0
    };
    std::vector<int> per_expert_counters_data(N);
    int* topk_indices;
    int* per_expert_counters;
    cudaMalloc(&topk_indices, B*K * sizeof(int));
    cudaMalloc(&per_expert_counters, N*sizeof(int));
    cudaMemset(per_expert_counters, 0, N*sizeof(int));
    cudaMemcpy(topk_indices, topk_indices_data.data(), B*K * sizeof(int), cudaMemcpyHostToDevice);
    
    int TOTAL = B*K;
    int THREADS = 256;
    int BLOCKS = (TOTAL + THREADS - 1) / THREADS;
    count_topk_kernel<<<BLOCKS, THREADS>>>(topk_indices, per_expert_counters, TOTAL);
    cudaMemcpy(per_expert_counters_data.data(), per_expert_counters, N*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> expected = {4, 2, 1, 3};
    assert(per_expert_counters_data.size() == expected.size());
    for (int i = 0; i < expected.size(); i++){
        assert(per_expert_counters_data[i] == expected[i]);
    }
    std::cout << "Test passed!" << std::endl;
    cudaFree(topk_indices);
    cudaFree(per_expert_counters);
}