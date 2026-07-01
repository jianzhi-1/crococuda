// test_flash_attention_backward.cu
// Same build command as forward tests, just add this file to the compilation.

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cfloat>

// ---------------------------------------------------------------------------
// Declarations of host launchers defined in flash_attention.cu
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> flash_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale, int Br, int Bc);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attention_backward(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O,  torch::Tensor L, double scale, int Br, int Bc);

// ---------------------------------------------------------------------------
// Reference: naive backward via libtorch autograd
// Returns (dQ, dK, dV)
// ---------------------------------------------------------------------------
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> naive_attention_backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor dO, float scale)
{
    auto Qg = Q.detach().requires_grad_(true);
    auto Kg = K.detach().requires_grad_(true);
    auto Vg = V.detach().requires_grad_(true);

    auto S = torch::matmul(Qg, Kg.transpose(-1, -2)) * scale;
    auto P = torch::softmax(S, /*dim=*/-1);
    auto O = torch::matmul(P, Vg);

    O.backward(dO);

    return {Qg.grad().clone(), Kg.grad().clone(), Vg.grad().clone()};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
struct FlashBwdFixture {
    torch::Tensor Q, K, V, dO, O, L;
    float scale;
    int Br, Bc;

    FlashBwdFixture(int B, int H, int S, int d, int Br_, int Bc_, float qkv_scale=1.f)
        : Br(Br_), Bc(Bc_)
    {
        scale = 1.f / sqrtf((float)d);
        Q  = torch::randn({B, H, S, d}, torch::kFloat32).cuda() * qkv_scale;
        K  = torch::randn({B, H, S, d}, torch::kFloat32).cuda() * qkv_scale;
        V  = torch::randn({B, H, S, d}, torch::kFloat32).cuda() * qkv_scale;
        dO = torch::randn({B, H, S, d}, torch::kFloat32).cuda();
        auto [O_, L_] = flash_attention_forward(Q, K, V, scale, Br, Bc);
        O = O_; L = L_;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_flash_bwd() {
        return flash_attention_backward(dO, Q, K, V, O, L, scale, Br, Bc);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_naive_bwd() {
        return naive_attention_backward(Q, K, V, dO, scale);
    }
};

static void assert_close(torch::Tensor a, torch::Tensor b,
                         float atol=1e-3f, float rtol=1e-3f,
                         const char* name="") {
    ASSERT_TRUE(torch::allclose(a.cpu(), b.cpu(), rtol, atol))
        << name << " max abs diff: "
        << (a.cpu() - b.cpu()).abs().max().item<float>();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. Basic correctness: dQ, dK, dV match autograd reference
TEST(FlashAttentionBackward, GradientsMatchNaive) {
    torch::manual_seed(0);
    FlashBwdFixture f(2, 2, 64, 32, /*Br=*/32, /*Bc=*/32);

    auto [dQ, dK, dV] = f.run_flash_bwd();
    auto [dQ_ref, dK_ref, dV_ref] = f.run_naive_bwd();

    assert_close(dQ, dQ_ref, 1e-3f, 1e-3f, "dQ");
    assert_close(dK, dK_ref, 1e-3f, 1e-3f, "dK");
    assert_close(dV, dV_ref, 1e-3f, 1e-3f, "dV");
}

// 2. S not divisible by Br
TEST(FlashAttentionBackward, SeqlenNotDivisibleByBr) {
    torch::manual_seed(1);
    FlashBwdFixture f(1, 1, 70, 32, /*Br=*/32, /*Bc=*/32);

    auto [dQ, dK, dV] = f.run_flash_bwd();
    auto [dQ_ref, dK_ref, dV_ref] = f.run_naive_bwd();

    assert_close(dQ, dQ_ref, 1e-3f, 1e-3f, "dQ");
    assert_close(dK, dK_ref, 1e-3f, 1e-3f, "dK");
    assert_close(dV, dV_ref, 1e-3f, 1e-3f, "dV");
}

// 3. S not divisible by Bc
TEST(FlashAttentionBackward, SeqlenNotDivisibleByBc) {
    torch::manual_seed(2);
    FlashBwdFixture f(1, 1, 70, 32, /*Br=*/32, /*Bc=*/48);

    auto [dQ, dK, dV] = f.run_flash_bwd();
    auto [dQ_ref, dK_ref, dV_ref] = f.run_naive_bwd();

    assert_close(dQ, dQ_ref, 1e-3f, 1e-3f, "dQ");
    assert_close(dK, dK_ref, 1e-3f, 1e-3f, "dK");
    assert_close(dV, dV_ref, 1e-3f, 1e-3f, "dV");
}

// 4. Single token: P=1 everywhere, so dV = dO, dQ = dK = scale * dO * V^T * Q (trivial)
TEST(FlashAttentionBackward, SingleToken) {
    torch::manual_seed(3);
    FlashBwdFixture f(2, 4, 1, 32, /*Br=*/1, /*Bc=*/1);

    auto [dQ, dK, dV] = f.run_flash_bwd();

    // P = [[1]], so O = V, dV = dO exactly
    assert_close(dV, f.dO, 1e-5f, 1e-5f, "dV == dO for S=1");

    // Also check dQ and dK match naive just to be safe
    auto [dQ_ref, dK_ref, dV_ref] = f.run_naive_bwd();
    assert_close(dQ, dQ_ref, 1e-5f, 1e-5f, "dQ");
    assert_close(dK, dK_ref, 1e-5f, 1e-5f, "dK");
}

// 5. Numerical stability: large logits must not produce NaN/Inf in gradients
TEST(FlashAttentionBackward, NumericalStabilityLargeLogits) {
    torch::manual_seed(4);
    FlashBwdFixture f(1, 1, 64, 32, /*Br=*/32, /*Bc=*/32, /*qkv_scale=*/10.f);

    auto [dQ, dK, dV] = f.run_flash_bwd();

    EXPECT_FALSE(dQ.isnan().any().item<bool>()) << "dQ contains NaN";
    EXPECT_FALSE(dK.isnan().any().item<bool>()) << "dK contains NaN";
    EXPECT_FALSE(dV.isnan().any().item<bool>()) << "dV contains NaN";
    EXPECT_FALSE(dQ.isinf().any().item<bool>()) << "dQ contains Inf";
    EXPECT_FALSE(dK.isinf().any().item<bool>()) << "dK contains Inf";
    EXPECT_FALSE(dV.isinf().any().item<bool>()) << "dV contains Inf";
}

// 6. Multiple batches and heads: base offset arithmetic
TEST(FlashAttentionBackward, MultipleBatchesAndHeads) {
    torch::manual_seed(5);
    FlashBwdFixture f(4, 8, 128, 64, /*Br=*/64, /*Bc=*/64);

    auto [dQ, dK, dV] = f.run_flash_bwd();
    auto [dQ_ref, dK_ref, dV_ref] = f.run_naive_bwd();

    assert_close(dQ, dQ_ref, 1e-3f, 1e-3f, "dQ");
    assert_close(dK, dK_ref, 1e-3f, 1e-3f, "dK");
    assert_close(dV, dV_ref, 1e-3f, 1e-3f, "dV");
}

// 7. Output shapes
TEST(FlashAttentionBackward, OutputShapes) {
    torch::manual_seed(6);
    int B=3, H=5, S=48, d=32;
    FlashBwdFixture f(B, H, S, d, /*Br=*/16, /*Bc=*/16);

    auto [dQ, dK, dV] = f.run_flash_bwd();

    ASSERT_EQ(dQ.sizes(), (std::vector<int64_t>{B, H, S, d}));
    ASSERT_EQ(dK.sizes(), (std::vector<int64_t>{B, H, S, d}));
    ASSERT_EQ(dV.sizes(), (std::vector<int64_t>{B, H, S, d}));
}

// 8. dQ atomicAdd correctness: many KV blocks all contribute to dQ.
// If any block double-counts or skips, dQ will be wrong.
// Force this by making S large relative to Bc so many KV tiles are launched.
TEST(FlashAttentionBackward, dQAtomicAddManyKVBlocks) {
    torch::manual_seed(7);
    // Bc=8 -> S/Bc = 256/8 = 32 KV blocks, all atomicAdding into dQ
    FlashBwdFixture f(1, 1, 256, 32, /*Br=*/32, /*Bc=*/8);

    auto [dQ, dK, dV] = f.run_flash_bwd();
    auto [dQ_ref, dK_ref, dV_ref] = f.run_naive_bwd();

    assert_close(dQ, dQ_ref, 1e-3f, 1e-3f, "dQ (many KV blocks)");
    assert_close(dK, dK_ref, 1e-3f, 1e-3f, "dK");
    assert_close(dV, dV_ref, 1e-3f, 1e-3f, "dV");
}

// 9. dO = zeros -> all gradients should be zero
TEST(FlashAttentionBackward, ZerodO) {
    torch::manual_seed(8);
    int B=1, H=1, S=64, d=32;
    float scale = 1.f / sqrtf((float)d);
    int Br=32, Bc=32;

    auto Q  = torch::randn({B, H, S, d}, torch::kFloat32).cuda();
    auto K  = torch::randn({B, H, S, d}, torch::kFloat32).cuda();
    auto V  = torch::randn({B, H, S, d}, torch::kFloat32).cuda();
    auto dO = torch::zeros({B, H, S, d}, torch::kFloat32).cuda();

    auto [O, L]     = flash_attention_forward(Q, K, V, scale, Br, Bc);
    auto [dQ, dK, dV] = flash_attention_backward(dO, Q, K, V, O, L, scale, Br, Bc);

    EXPECT_NEAR(dQ.abs().max().item<float>(), 0.f, 1e-6f) << "dQ should be zero";
    EXPECT_NEAR(dK.abs().max().item<float>(), 0.f, 1e-6f) << "dK should be zero";
    EXPECT_NEAR(dV.abs().max().item<float>(), 0.f, 1e-6f) << "dV should be zero";
}

// 10. D_i = rowsum(dO * O) term: uniform attention (all equal scores) simplifies
// P_ij = 1/S for all j, so we can derive dK analytically and cross-check.
TEST(FlashAttentionBackward, UniformAttentionAnalyticGradient) {
    torch::manual_seed(9);
    // To get uniform attention: set Q=0, K=0 -> all scores = 0 -> softmax = 1/S
    int B=1, H=1, S=4, d=4;
    float scale = 1.f / sqrtf((float)d);
    int Br=4, Bc=4;

    auto Q  = torch::zeros({B, H, S, d}, torch::kFloat32).cuda();
    auto K  = torch::zeros({B, H, S, d}, torch::kFloat32).cuda();
    auto V  = torch::randn({B, H, S, d}, torch::kFloat32).cuda();
    auto dO = torch::randn({B, H, S, d}, torch::kFloat32).cuda();

    auto [O, L]       = flash_attention_forward(Q, K, V, scale, Br, Bc);
    auto [dQ, dK, dV] = flash_attention_backward(dO, Q, K, V, O, L, scale, Br, Bc);
    auto [dQ_ref, dK_ref, dV_ref] = naive_attention_backward(Q, K, V, dO, scale);

    assert_close(dQ, dQ_ref, 1e-5f, 1e-5f, "dQ uniform");
    assert_close(dK, dK_ref, 1e-5f, 1e-5f, "dK uniform");
    assert_close(dV, dV_ref, 1e-5f, 1e-5f, "dV uniform");
}
