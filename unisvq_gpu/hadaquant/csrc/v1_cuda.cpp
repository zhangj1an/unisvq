// HadaQuant — PyTorch C++ Bindings
// Supports both 2-bit packed (4 values/byte) and ternary 5-trit packed (5 values/byte)
#include <torch/extension.h>

// Launchers (defined in v1_cuda_kernel.cu)
void fused_su_had128_launch(torch::Tensor data, torch::Tensor SU, torch::Tensor out);
void fused_had128_sv_launch(torch::Tensor data, torch::Tensor SV, torch::Tensor out);
void fused_dequant_gemv_packed_launch(torch::Tensor x, torch::Tensor Qint_packed,
                                       torch::Tensor out, float cb_scale, float cb_zero);
void dequant_weight_packed_launch(torch::Tensor Qint_packed, torch::Tensor w_out,
                                   float cb_scale, float cb_zero);
void fused_dequant_gemv_ternary_launch(torch::Tensor x, torch::Tensor Qint_ternary,
                                        torch::Tensor out, float cb_scale, float cb_zero,
                                        int K_orig);
void dequant_weight_ternary_launch(torch::Tensor Qint_ternary, torch::Tensor w_out,
                                    float cb_scale, float cb_zero, int K_orig);

// ── 2-bit packed ──

torch::Tensor fused_dequant_gemv_packed(
    torch::Tensor x,            // [M, K] bf16
    torch::Tensor Qint_packed,  // [N, K_packed] uint8, K_packed = K/4
    double cb_scale,
    double cb_zero
) {
    TORCH_CHECK(x.is_cuda() && Qint_packed.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(Qint_packed.dtype() == torch::kUInt8, "Qint_packed must be uint8");
    TORCH_CHECK(x.is_contiguous() && Qint_packed.is_contiguous(), "Tensors must be contiguous");

    int K = x.size(1);
    int K_packed = Qint_packed.size(1);
    TORCH_CHECK(K == K_packed * 4,
                "K (", K, ") must equal K_packed*4 (", K_packed * 4, ") for 2-bit packing");

    int M = x.size(0);
    int N = Qint_packed.size(0);
    auto out = torch::empty({M, N}, x.options());

    fused_dequant_gemv_packed_launch(x, Qint_packed, out, (float)cb_scale, (float)cb_zero);
    return out;
}

torch::Tensor dequant_weight_packed(
    torch::Tensor Qint_packed,  // [N, K_packed] uint8
    double cb_scale,
    double cb_zero
) {
    TORCH_CHECK(Qint_packed.is_cuda() && Qint_packed.dtype() == torch::kUInt8,
                "Qint_packed must be CUDA uint8");
    TORCH_CHECK(Qint_packed.is_contiguous(), "Qint_packed must be contiguous");

    int N = Qint_packed.size(0);
    int K_packed = Qint_packed.size(1);
    int K = K_packed * 4;
    auto w_out = torch::empty({N, K}, Qint_packed.options().dtype(torch::kBFloat16));
    dequant_weight_packed_launch(Qint_packed, w_out, (float)cb_scale, (float)cb_zero);
    return w_out;
}

// ── Ternary 5-trit packed ──

torch::Tensor fused_dequant_gemv_ternary(
    torch::Tensor x,              // [M, K] bf16
    torch::Tensor Qint_ternary,   // [N, K_trit] uint8
    double cb_scale,
    double cb_zero,
    int64_t K_orig
) {
    TORCH_CHECK(x.is_cuda() && Qint_ternary.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(Qint_ternary.dtype() == torch::kUInt8, "Qint_ternary must be uint8");
    TORCH_CHECK(x.is_contiguous() && Qint_ternary.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.size(1) == K_orig, "x.size(1) must equal K_orig");

    int M = x.size(0);
    int N = Qint_ternary.size(0);
    auto out = torch::empty({M, N}, x.options());

    fused_dequant_gemv_ternary_launch(x, Qint_ternary, out,
                                       (float)cb_scale, (float)cb_zero, (int)K_orig);
    return out;
}

torch::Tensor dequant_weight_ternary(
    torch::Tensor Qint_ternary,  // [N, K_trit] uint8
    double cb_scale,
    double cb_zero,
    int64_t K_orig
) {
    TORCH_CHECK(Qint_ternary.is_cuda() && Qint_ternary.dtype() == torch::kUInt8,
                "Qint_ternary must be CUDA uint8");
    TORCH_CHECK(Qint_ternary.is_contiguous(), "Qint_ternary must be contiguous");

    int N = Qint_ternary.size(0);
    auto w_out = torch::empty({N, (int)K_orig}, Qint_ternary.options().dtype(torch::kBFloat16));
    dequant_weight_ternary_launch(Qint_ternary, w_out,
                                   (float)cb_scale, (float)cb_zero, (int)K_orig);
    return w_out;
}

// ── Hadamard ──

void fused_su_had128(torch::Tensor data, torch::Tensor SU, torch::Tensor out) {
    TORCH_CHECK(data.is_cuda() && SU.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(data.dtype() == torch::kBFloat16, "data must be bf16");
    TORCH_CHECK(data.is_contiguous() && SU.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(data.size(-1) == SU.size(0), "Dimension mismatch");
    fused_su_had128_launch(data, SU, out);
}

void fused_had128_sv(torch::Tensor data, torch::Tensor SV, torch::Tensor out) {
    TORCH_CHECK(data.is_cuda() && SV.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(data.dtype() == torch::kBFloat16, "data must be bf16");
    TORCH_CHECK(data.is_contiguous() && SV.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(data.size(-1) == SV.size(0), "Dimension mismatch");
    fused_had128_sv_launch(data, SV, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_dequant_gemv_packed", &fused_dequant_gemv_packed,
          "Fused dequant GEMV (packed 2-bit): reads 4 values/byte");
    m.def("dequant_weight_packed", &dequant_weight_packed,
          "Dequant packed uint8 -> bf16, output [N, K_packed*4]");
    m.def("fused_dequant_gemv_ternary", &fused_dequant_gemv_ternary,
          "Fused dequant GEMV (ternary 5-trit): reads 5 values/byte");
    m.def("dequant_weight_ternary", &dequant_weight_ternary,
          "Dequant ternary 5-trit uint8 -> bf16, output [N, K_orig]");
    m.def("fused_su_had128", &fused_su_had128,
          "In-place SU multiply + block-128 Hadamard");
    m.def("fused_had128_sv", &fused_had128_sv,
          "In-place block-128 Hadamard + SV multiply");
}
