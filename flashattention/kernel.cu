#include <torch/torch.h>
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

    dim3 grid {(unsigned)((S + Br - 1) / Br), (unsigned)H, (unsigned)B};
    dim3 block {(unsigned)Br};

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

/*
Math for backward computation:
- O = softmax(QK^T)V = softmax(T)V = PV
- dV = P^T dO
- Let F = ∂P/∂T, then:
- dK = F((dO)V^T)^TQ
- dQ = F((dO)V^T)K

The mathematical description of F is that:
- F: R^{n, m} -> R^{n, m} where n is number of queries, m is number of keys
- F(T) = Sum_{ij} Pij 1{i, j} - Pij Row{i}
  where 1{i, j} is the nxm matrix with 1 in the (i, j)th coordinate and 0 everywhere else
  and Row{i} is the nxm matrix with only row i present and 0 everywhere else

Algorithmically, F is implemented as:
- dTij = Pij dPij - Pij Sum_{k} Pik dPik
       = Pij (dPij - Oi . dOi) =: Pij (dPij - Di)
*/
__global__ void flash_attention_backward_kernel(
    // input
    const float* __restrict__ Q, // [B, H, S, d]
    const float* __restrict__ K, // [B, H, S, d]
    const float* __restrict__ V, // [B, H, S, d]
    const float* __restrict__ O, // [B, H, S, d]
    const float* __restrict__ dO, // [B, H, S, d]
    const float* __restrict__ L, // [B, H, S]
    // output
    float* __restrict__ dQ, // [B, H, S, d]
    float* __restrict__ dK, // [B, H, S, d]
    float* __restrict__ dV, // [B, H, S, d]
    // constant
    int H, int S, int d, int Br, int Bc, float scale
){
    int kv_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int col = threadIdx.x;
    int idx = kv_block * Bc + col;
    if (idx >= S) return;

    long base = (long)(b * H + h) * S * d;
    long lbase = (long)(b * H + h) * S;

    float k_reg[MAX_D];
    float v_reg[MAX_D];
    REP(i, 0, d){
        // Load K[b, h, idx, :]
        k_reg[i] = K[base + (long)idx * d + i];
        // Load K[b, h, idx, :]
        v_reg[i] = V[base + (long)idx * d + i];
    }

    float dk_acc[MAX_D];
    float dv_acc[MAX_D];
    memset(dk_acc, 0, sizeof(dk_acc));
    memset(dv_acc, 0, sizeof(dv_acc));
    
    extern __shared__ float smem[]; // [Br, d], [Br, d], [Br], [Br]
    float* Qs = smem; // [Br, d]
    float* dOs = smem + Br * d; // [Br, d]
    float* Ls = dOs + Br * d; // [Br]
    float* Ds = Ls + Br; // [Br]

    int n_q_tiles = (S + Br - 1) / Br;
    REP(i, 0, n_q_tiles){
        int q_base = i * Br;
        REPD(j, col, Br * d, Bc){
            int r = j / d;
            int c = j % d;
            int gq = q_base + r;
            // Load Q[b, h, i*Br + r, c]
            Qs[j] = (gq < S) ? Q[base + (long)gq * d + c] : 0.f;
            // Load dO[b, h, i*Br + r, c]
            dOs[j] = (gq < S) ? dO[base + (long)gq * d + c] : 0.f;
        }
        REPD(j, col, Br, Bc){
            int gq = q_base + j;
            if (gq < S){
                // Load L[b, h, j]
                Ls[j] = L[lbase + gq];

                // Load Oi . dOi
                float Dval = 0.f;
                // dO[b, h, i*Br + j, :] . O[b, h, i*Br + j, :]
                REP(k, 0, d){
                    Dval += dO[base + (long)gq * d + k] * O[base + (long)gq * d + k];
                }
                Ds[j] = Dval;
            }
        }

        __syncthreads();
        // Qs = Q[b, h, i*Br:(i+1)*Br, :] now fully materialised
        // dOs = dO[b, h, i*Br:(i+1)*Br, :] now fully materialised
        // Ls = L[b, h, i*Br:(i+1)*Br] now fully materialised
        // Ds = dO[b, h, i*Br:(i+1)*Br, :] . O[b, h, i*Br:(i+1)*Br, :] now fully materialised

        int br_eff = min(Br, S - q_base);
        REP(j, 0, br_eff){
            float s = 0.f;

            // Q[b, h, i*Br+j, :] . K[b, h, idx, :] / sqrt(d)
            REP(k, 0, d) s += Qs[j * d + k] * k_reg[k];
            s *= scale;
            
            float p = expf(s - Ls[j]);
            // dP = dO V^T
            float dp = 0.f;
            // dP[b, h, i*Br+j, idx] = dOs[b, h, i*Br+j, :] * V[b, h, idx, :]
            REP(k, 0, d) dp += dOs[j * d + k] * v_reg[k];

            // dTij = Pij (dPij - Di)
            float ds = p * (dp - Ds[j]) * scale;

            // dV = P^T dO
            // dK = F((dO)V^T)^TQ
            // dQ = F((dO)V^T)K
            REP(k, 0, d){
                // dV[b, h, idx, :] = Sum_{ii} p * dO[b, h, ii, :]
                dv_acc[k] += p * dOs[j * d + k];
                // dK[b, h, idx, :] = Sum_{ii} dS * Q[b, h, ii, :]
                dk_acc[k] += ds * Qs[j * d + k];
                // dQ[b, h, i*Br+j, k] = Sum_{idx} dS * K[b, h, idx, :]
                atomicAdd(&dQ[base + (long)(q_base + j) * d + k], ds * k_reg[k]);
            }
        }
        __syncthreads();
    }

    float* dk_row = dK + base + (long)idx * d;
    float* dv_row = dV + base + (long)idx * d;
    REP(i, 0, d){
        // Store dK[b, h, idx, :]
        dk_row[i] = dk_acc[i];
        // Store dV[b, h, idx, :]
        dv_row[i] = dv_acc[i];
    }
}

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
){
    // TODO: TORCH_CHECK
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    torch::Tensor dQ = torch::zeros_like(Q); // to support atomic adds
    torch::Tensor dK = torch::empty_like(K);
    torch::Tensor dV = torch::empty_like(V);

    dim3 grid{(unsigned)((S + Bc - 1) / Bc), (unsigned)H, (unsigned)B};
    dim3 block{(unsigned)Bc};
    size_t smem = (2 * (size_t)Br * d + 2 * (size_t)Br) * sizeof(float);

    flash_attention_backward_kernel<<<grid, block, smem>>>(
        // input
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        dO.data_ptr<float>(),
        L.data_ptr<float>(),
        // output
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        // constant
        H, S, d, Br, Bc, (float)scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return {dQ, dK, dV};
}
