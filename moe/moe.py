
template <typename T>
__global__ void gather(
    const int* expert_idx, // [B, K]
    const T* x, // [B, D]
    int* sorted_token_idx, // [BK]
    uint32_t* expert_offsets, // [N+1]
    T* sorted_x, // [BK, D]
    int B, int D, int K, int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < )
    // 
}

template <typename T>
__global__ void experts_kernel(
    const T* x, // [B, D]
    const int* expert_idx, // [B, K]
    const T* gate_weights, // [B, K]
    const T* W1, // [N, D, D]
    const T* W2, // [N, D, D]
    T* out, // [B, D]
    int B, int D, int K, int N
){
    int idx = blockDim.x * blockDim.x + threadIdx.x;


}

int f(int x){
    return x;
}