// test_flash_attention_forward.cu
// Build:
//   nvcc -std=c++17 -O2 test_flash_attention_forward.cu \
//        -I$(python -c "import torch; print(torch.utils.cmake_prefix_path + '/../../../include')") \
//        -I$(python -c "import torch; print(torch.utils.cmake_prefix_path + '/../../../include/torch/csrc/api/include')") \
//        -L$(python -c "import torch; print(torch.utils.cmake_prefix_path + '/../../')") \
//        -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
//        -lgtest -lgtest_main \
//        -Wl,-rpath,$(python -c "import torch; print(torch.utils.cmake_prefix_path + '/../../')") \
//        -o test_flash_attn && ./test_flash_attn

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cfloat>

// ---------------------------------------------------------------------------
// Declaration of the host launcher defined in flash_attention.cu
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> flash_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale, int Br, int Bc);

// ---------------------------------------------------------------------------
// Reference: naive attention in libtorch, no tiling
// ---------------------------------------------------------------------------
static std::pair<torch::Tensor, torch::Tensor> naive_attention(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale)
{
    // S[b,h,i,j] = Q[b,h,i,:] . K[b,h,j,:] * scale
    auto S = torch::matmul(Q, K.transpose(-1, -2)) * scale; // [B,H,Sq,Sk]
    auto P = torch::softmax(S, /*dim=*/-1);                 // [B,H,Sq,Sk]
    auto O = torch::matmul(P, V);                           // [B,H,Sq,d]
    // L[b,h,i] = log sum_j exp(S[b,h,i,j])  =  logsumexp over last dim
    auto L = torch::logsumexp(S, /*dim=*/-1);               // [B,H,Sq]
    return {O, L};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static torch::Tensor make_qkv(int B, int H, int S, int d, float scale = 1.f) {
    return torch::randn({B, H, S, d}, torch::kFloat32).cuda() * scale;
}

static void assert_close(torch::Tensor a, torch::Tensor b,
                         float atol = 1e-3f, float rtol = 1e-3f,
                         const char* name = "") {
    ASSERT_TRUE(torch::allclose(a.cpu(), b.cpu(), rtol, atol))
        << name << " max abs diff: "
        << (a - b).abs().max().item<float>();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. Basic correctness: output O and logsumexp L match naive reference
TEST(FlashAttentionForward, OutputMatchesNaive) {
    torch::manual_seed(0);
    int B=2, H=2, S=64, d=32;
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    auto [O_ref, L_ref] = naive_attention(Q, K, V, scale);
    auto [O,     L    ] = flash_attention_forward(Q, K, V, scale, /*Br=*/32, /*Bc=*/32);

    assert_close(O, O_ref, 1e-3f, 1e-3f, "O");
    assert_close(L, L_ref, 1e-3f, 1e-3f, "L");
}

// 2. S not divisible by Br — tests the `if (q_idx >= S) return` guard
TEST(FlashAttentionForward, SeqlenNotDivisibleByBr) {
    torch::manual_seed(1);
    int B=1, H=1, S=70, d=32;   // 70 not divisible by 32
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    auto [O_ref, L_ref] = naive_attention(Q, K, V, scale);
    auto [O,     L    ] = flash_attention_forward(Q, K, V, scale, /*Br=*/32, /*Bc=*/32);

    assert_close(O, O_ref, 1e-3f, 1e-3f, "O");
    assert_close(L, L_ref, 1e-3f, 1e-3f, "L");
}

// 3. S not divisible by Bc — tests the `bc_eff = min(Bc, S - kv_base)` guard
TEST(FlashAttentionForward, SeqlenNotDivisibleByBc) {
    torch::manual_seed(2);
    int B=1, H=1, S=70, d=32;   // 70 not divisible by Bc=48
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    auto [O_ref, L_ref] = naive_attention(Q, K, V, scale);
    auto [O,     L    ] = flash_attention_forward(Q, K, V, scale, /*Br=*/32, /*Bc=*/48);

    assert_close(O, O_ref, 1e-3f, 1e-3f, "O");
}

// 4. S=1: single-token degenerate case — softmax over one element = 1, O = V
TEST(FlashAttentionForward, SingleToken) {
    torch::manual_seed(3);
    int B=2, H=4, S=1, d=32;
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    auto [O, L] = flash_attention_forward(Q, K, V, scale, /*Br=*/1, /*Bc=*/1);

    // softmax over a single score = 1, so O = V
    assert_close(O, V, 1e-5f, 1e-5f, "O == V for S=1");
}

// 5. Numerical stability: large logits should not produce NaN/Inf
// Standard softmax would overflow; flash attention shifts by max so should be fine
TEST(FlashAttentionForward, NumericalStabilityLargeLogits) {
    torch::manual_seed(4);
    int B=1, H=1, S=64, d=32;
    float scale = 1.f / sqrtf((float)d);

    // Large values -> raw dot products ~80/sqrt(32) ~ 14, exp(14) is fine,
    // but without the max shift exp(raw_dot) would overflow for scale=1
    auto Q = make_qkv(B, H, S, d, /*scale=*/10.f);
    auto K = make_qkv(B, H, S, d, /*scale=*/10.f);
    auto V = make_qkv(B, H, S, d, /*scale=*/ 1.f);

    auto [O, L] = flash_attention_forward(Q, K, V, scale, /*Br=*/32, /*Bc=*/32);

    EXPECT_FALSE(O.isnan().any().item<bool>()) << "O contains NaN";
    EXPECT_FALSE(O.isinf().any().item<bool>()) << "O contains Inf";
    EXPECT_FALSE(L.isnan().any().item<bool>()) << "L contains NaN";
}

// 6. Multiple batches and heads: make sure base offset arithmetic is right
TEST(FlashAttentionForward, MultipleBatchesAndHeads) {
    torch::manual_seed(5);
    int B=4, H=8, S=128, d=64;
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    auto [O_ref, L_ref] = naive_attention(Q, K, V, scale);
    auto [O,     L    ] = flash_attention_forward(Q, K, V, scale, /*Br=*/64, /*Bc=*/64);

    assert_close(O, O_ref, 1e-3f, 1e-3f, "O");
    assert_close(L, L_ref, 1e-3f, 1e-3f, "L");
}

// 7. Br > S: entire sequence fits in one query tile
TEST(FlashAttentionForward, BrLargerThanS) {
    torch::manual_seed(6);
    int B=1, H=1, S=16, d=32;
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    // Br=64 >> S=16, so only one query tile is ever launched
    auto [O_ref, L_ref] = naive_attention(Q, K, V, scale);
    auto [O,     L    ] = flash_attention_forward(Q, K, V, scale, /*Br=*/64, /*Bc=*/16);

    assert_close(O, O_ref, 1e-3f, 1e-3f, "O");
}

// 8. Output shape is correct
TEST(FlashAttentionForward, OutputShape) {
    torch::manual_seed(7);
    int B=3, H=5, S=48, d=32;
    float scale = 1.f / sqrtf((float)d);

    auto Q = make_qkv(B, H, S, d);
    auto K = make_qkv(B, H, S, d);
    auto V = make_qkv(B, H, S, d);

    auto [O, L] = flash_attention_forward(Q, K, V, scale, /*Br=*/16, /*Bc=*/16);

    ASSERT_EQ(O.sizes(), (std::vector<int64_t>{B, H, S, d}));
    ASSERT_EQ(L.sizes(), (std::vector<int64_t>{B, H, S}));
}
