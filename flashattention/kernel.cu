#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_D 128
#define MAX_BC 256
#define REP(i, a, b) for (int i = (int)a; i < (int)b; i++)
#define REPD(i, a, b, stride) for (int i = (int)a; i < (int)b; i+=stride)
/*
Variable definitions (my style):
- B: batch size
- H: number of heads
- S: sequence length
- d: model dimension per head
*/
__global__ void flash_attention_forward_kernel(
    // input
    const float* __restrict__ Q, // [B, H, S, d]
    const float* __restrict__ K,
    const float* __restrict__ V,
    // output
    float* __restrict__ O, // [B, H, S, d]
    float* __restrict__ L, // [B, H, S]
    // constant
    int H,
    int S,
    int d,
    int Br, // query tile size
    int Bc, // K, V tile size
    float scale // 1/sqrt(d)
){
    /*
    Tiled along the S dimension. Thread i handles a block of size Br.
    Think of a thread as indexed by (b, h, q_tile) x (i, ).
    It maps a query to output.
    */
    int q_tile = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    
    int idx = threadIdx.x;
    int q_idx = q_tile * Br + idx;
    if (q_idx >= S) return;

    // Q[b, h, q_idx, :]
    long base = (long)(b * H + h) * S * d;
    const float* q_row = Q + base + (long)q_idx * d;

    // Load Q[b, h, q_idx, :]
    float q_reg[MAX_D];
    REP(i, 0, d) q_reg[i] = q_row[i];

    float m = -FLT_MAX; // max qTkj / sqrt(d)
    float l = 0.f; // log sum e^qTkj
    float o_acc[MAX_D]; // sum aj * Vj
    REP(i, 0, d) o_acc[i] = 0.f;

    extern __shared__ float smem[]; // [2*Bc, d], first half K, second half V
    float* Ks = smem; // [Bc, d]
    float* Vs = smem + Bc * d; // [Bc, d]

    int n_kv = (S + Bc - 1) / Bc; // number of kv blocks
    REP(i, 0, n_kv){
        int kv_base = i * Bc;

        // Load K_j, V_j
        REPD(j, threadIdx.x, Bc * d, Br){
            int r = j / d;
            int c = j % d;
            int gk = kv_base + r;
            // Ks[r, c] = K[b, h, i*Bc + r, c]
            // Vs[r, c] = V[b, h, i*Bc + r, c]
            // r += Br / d
            Ks[j] = (gk < S) ? K[base + (long)gk * d + c] : 0.f;
            Vs[j] = (gk < S) ? V[base + (long)gk * d + c] : 0.f;
        }
        __syncthreads();

        int bc_eff = min(Bc, S - kv_base);
        float local_max = -FLT_MAX; // max qT k / sqrt(d) over the tile
        float s_block[MAX_BC]; // s_block[i] = qTKs[c, :]

        REP(j, 0, bc_eff){
            float acc = 0.f; // qT Ks / sqrt(d)
            REP(k, 0, d) acc += q_reg[k] * Ks[j*d + k];
            acc *= scale;
            s_block[j] = acc;
            local_max = fmaxf(local_max, acc);
        }

        float m_new = fmaxf(m, local_max);
        float alpha = expf(m - m_new); // rebase old accumulators
        float l_new = alpha * l;

        REP(j, 0, d) o_acc[j] *= alpha;
        REP(j, 0, bc_eff){
            float p = expf(s_block[j] - m_new);
            l_new += p;
            REP(k, 0, d) o_acc[k] += p * Vs[j*d + k];
        }
        m = m_new;
        l = l_new;
        __syncthreads();
    }

    // Store O[b, h, q_idx, :]
    float* o_row = O + base + (long)q_idx * d;
    REP(i, 0, d) o_row[i] = o_acc[i] / l;

    // Store L[b, h, q_idx]
    L[(long)(b * H + h) * S + q_idx] = m + logf(l);

}

std::pair<torch::Tensor, torch::Tensor> flash_attention_forward(
    torch::Tensor Q, // [B, H, S, d]
    torch::Tensor K, // [B, H, S, d]
    torch::Tensor V, // [B, H, S, d]
    double scale,
    int Br,
    int Bc
){
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    TORCH_CHECK(K.is_cuda() && K.dtype() == torch::kFloat32);
    TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kFloat32);
    TORCH_CHECK(Q.dim() == 4);
    TORCH_CHECK(K.dim() == 4);
    TORCH_CHECK(V.dim() == 4);

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == d);
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == d);

    torch::Tensor O = torch::empty_like(Q);
    torch::Tensor L = torch::empty({B, H, S}, Q.options());

    dim3 grid {(S + Br - 1) / Br, H, B};
    dim3 block {Br};

    size_t smem = 2 * (size_t)Bc * d * sizeof(float);

    flash_attention_forward_kernel<<<grid, block, smem>>>(
        // input
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        // output
        O.data_ptr<float>(),
        L.data_ptr<float>(),
        // constant
        H, S, d, Br, Bc, (float)scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O, L};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &flash_attention_forward, "flash_attention_forward");
}