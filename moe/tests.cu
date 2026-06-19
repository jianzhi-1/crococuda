#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include "experts_kernel.cuh"
int main(){
    int B = 5;
    int K = 2;
    int N = 4;
    int D = 8;
    std::vector<int> expert_idx_data = {
        0, 2,
        3, 1,
        0, 3,
        0, 3,
        1, 0
    };
    std::vector<int> counts_data(N);
    int* expert_idx;
    int* counts;
    cudaMalloc(&expert_idx, B*K * sizeof(int));
    cudaMalloc(&counts, N*sizeof(int));
    cudaMemset(counts, 0, N*sizeof(int));
    cudaMemcpy(expert_idx, expert_idx_data.data(), B*K * sizeof(int), cudaMemcpyHostToDevice);
    
    int TOTAL = B*K;
    int THREADS = 256;
    int BLOCKS = (TOTAL + THREADS - 1) / THREADS;
    count_kernel<<<BLOCKS, THREADS>>>(expert_idx, counts, TOTAL);
    cudaMemcpy(counts_data.data(), counts, N*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> expected = {4, 2, 1, 3};
    assert(counts_data.size() == expected.size());
    for (int i = 0; i < expected.size(); i++){
        assert(counts_data[i] == expected[i]);
    }
    
}