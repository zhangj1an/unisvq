/*
 * HadaQuant — Fused CUDA Kernels for Hadamard-based Quantized Inference
 *
 * Packed 2-bit format: [N, K/4] uint8, 4 values/byte (v0<<6|v1<<4|v2<<2|v3)
 * Packed ternary format: [N, K_trit] uint8, 5 trits/byte (v0*81+v1*27+v2*9+v3*3+v4)
 *
 * Kernel 1: FusedSUHad128              — SU multiply + block-128 Hadamard (in-place)
 * Kernel 2: FusedHad128SV              — block-128 Hadamard + SV multiply (in-place)
 * Kernel 3: FusedDequantGEMVPacked     — 2-bit dequant + GEMV (decode path)
 * Kernel 4: DequantWeightPacked        — 2-bit dequant to bf16 (GEMM path)
 * Kernel 5: FusedDequantGEMVTernary    — ternary 5-trit dequant + GEMV (decode path)
 * Kernel 6: DequantWeightTernary       — ternary 5-trit dequant to bf16 (GEMM path)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

// Reuse butterfly primitives from fast_hadamard_transform
#define FULL_MASK 0xffffffff

// ====================================================================
// Block-128 Hadamard helpers (log2(128) = 7 stages)
// 16 threads, 8 elements per thread.
// Stage 0-2: in-thread butterfly (hadamard_mult_thread<3>)
// Stage 3-6: warp shuffle butterfly (hadamard_mult_warp<4>)
// ====================================================================

template<int kLogN, int kNChunks>
__device__ __forceinline__ void had_mult_thread(float x[kNChunks][1 << kLogN]) {
    constexpr int N = 1 << kLogN;
    #pragma unroll
    for (int i = 0; i < kLogN; ++i) {
        const int stride = 1 << i;
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int c = 0; c < kNChunks; ++c) {
                const float a = x[c][idx];
                const float b = x[c][idx + stride];
                x[c][idx] = a + b;
                x[c][idx + stride] = a - b;
            }
        }
    }
}

template<int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
__device__ __forceinline__ void had_mult_warp(float x[kNChunks][kNItems]) {
    int lane_id = threadIdx.x % (1 << kLogWarpSize);
    #pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int lane_mask = 1 << step;
        const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float other = __shfl_xor_sync(FULL_MASK, x[c][i], lane_mask);
                x[c][i] = sign * x[c][i] + other;
            }
        }
    }
}

// Full block-128 Hadamard: 16 threads × 8 elements = 128
// kLogN=3 (in-thread), kLogWarpSize=4 (warp shuffle)
__device__ __forceinline__ void hadamard128(float vals[1][8]) {
    had_mult_thread<3, 1>(vals);     // 3 stages in-thread
    had_mult_warp<4, 0, 1, 8>(vals); // 4 stages warp shuffle
}

static constexpr float INV_SQRT_128 = 0.08838834764831845f; // 1/sqrt(128)

// ====================================================================
// Kernel 1: Fused SU × x + Block-128 Hadamard (in-place)
//   For each block of 128 elements: x_block = had128(x_block * SU_block)
//
// Grid:  (M * num_blocks128)
// Block: 16 threads (matching fast_hadamard_transform for dim=128)
// ====================================================================

__global__ void FusedSUHad128Kernel(
    const __nv_bfloat16* __restrict__ data,   // [M, K], read-only
    const __nv_bfloat16* __restrict__ SU,     // [K]
          __nv_bfloat16* __restrict__ out,    // [M, K], output  ← 新增
    int M, int K
) {
    int num_blocks = K / 128;
    int flat_id = blockIdx.x;
    int m = flat_id / num_blocks;
    int blk = flat_id % num_blocks;

    if (m >= M) return;

    int base = m * K + blk * 128;
    int lane = threadIdx.x;

    float vals[1][8];
    int su_base = blk * 128 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float x_val  = __bfloat162float(data[base + lane * 8 + i]);
        float su_val = __bfloat162float(SU[su_base + i]);
        vals[0][i] = x_val * su_val;
    }

    hadamard128(vals);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[base + lane * 8 + i] = __float2bfloat16(vals[0][i] * INV_SQRT_128);  // write out
    }
}

// ====================================================================
// Kernel 2: Block-128 Hadamard + SV multiply (in-place)
//   For each block of 128 elements: data_block = had128(data_block) * SV_block
//
// Grid:  (M * num_blocks128)
// Block: 16 threads
// ====================================================================

__global__ void FusedHad128SVKernel(
    const __nv_bfloat16* __restrict__ data,   // [M, N], read-only
    const __nv_bfloat16* __restrict__ SV,     // [N]
          __nv_bfloat16* __restrict__ out,    // [M, N], output
    int M, int N
) {
    int num_blocks = N / 128;
    int flat_id = blockIdx.x;
    int m = flat_id / num_blocks;
    int blk = flat_id % num_blocks;

    if (m >= M) return;

    int base = m * N + blk * 128;
    int lane = threadIdx.x;

    float vals[1][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        vals[0][i] = __bfloat162float(data[base + lane * 8 + i]);
    }

    hadamard128(vals);

    int sv_base = blk * 128 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float sv_val = __bfloat162float(SV[sv_base + i]);
        out[base + lane * 8 + i] = __float2bfloat16(vals[0][i] * INV_SQRT_128 * sv_val);  // ← 写 out
    }
}

// ####################################################################
//  PACKED 2-BIT VARIANTS
//  Each uint8 byte stores 4 quantized 2-bit values:
//    byte = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
//  Qint_packed shape: [N, K_packed] where K_packed = K / 4
// ####################################################################

// Helper: unpack one byte into 4 dequantized float values
__device__ __forceinline__ void unpack_dequant4(
    uint8_t byte_val, float cb_scale, float cb_zero, float out[4]
) {
    out[0] = cb_scale * (float)((byte_val >> 6) & 3) + cb_zero;
    out[1] = cb_scale * (float)((byte_val >> 4) & 3) + cb_zero;
    out[2] = cb_scale * (float)((byte_val >> 2) & 3) + cb_zero;
    out[3] = cb_scale * (float)( byte_val        & 3) + cb_zero;
}

// ====================================================================
// Kernel 3: Fused Dequant GEMV (PACKED 2-bit)
//   Reads packed uint8 (4 values/byte), dequant inline, GEMV.
//   Processes 16 quantized values per iteration (= 4 packed bytes).
//
// Grid:  (N, ceil(M / M_TILE))
// Block: 256 threads
// ====================================================================

template<int M_TILE>
__global__ void FusedDequantGEMVPackedKernel(
    const __nv_bfloat16* __restrict__ x,            // [M, K]
    const uint8_t*       __restrict__ Qint_packed,  // [N, K_packed]
    __nv_bfloat16*       __restrict__ out,          // [M, N]
    float cb_scale,
    float cb_zero,
    int N, int K, int K_packed, int M_global
) {
    int n_idx = blockIdx.x;
    int m_base = blockIdx.y * M_TILE;
    int tid = threadIdx.x;

    if (n_idx >= N) return;
    int m_limit = min(M_TILE, M_global - m_base);
    if (m_limit <= 0) return;

    float acc[M_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) acc[m] = 0.0f;

    const uint8_t* q_row = Qint_packed + (int64_t)n_idx * K_packed;

    // Process 16 quantized values per iteration = 4 packed bytes (uint32_t)
    int iters = K_packed / 4;
    for (int it = tid; it < iters; it += blockDim.x) {
        int p_base = it * 4;
        int k_base = it * 16;

        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(q_row + p_base);

        float w[16];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint8_t bv = (packed4 >> (b * 8)) & 0xFF;
            unpack_dequant4(bv, cb_scale, cb_zero, w + b * 4);
        }

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < m_limit) {
                const __nv_bfloat16* x_ptr = x + (int64_t)(m_base + m) * K + k_base;
                int4 x_lo = *reinterpret_cast<const int4*>(x_ptr);
                int4 x_hi = *reinterpret_cast<const int4*>(x_ptr + 8);
                __nv_bfloat16* xv_lo = reinterpret_cast<__nv_bfloat16*>(&x_lo);
                __nv_bfloat16* xv_hi = reinterpret_cast<__nv_bfloat16*>(&x_hi);

                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot += __bfloat162float(xv_lo[i]) * w[i];
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot += __bfloat162float(xv_hi[i]) * w[8 + i];
                }
                acc[m] += dot;
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    // Block-level reduction via shared memory
    __shared__ float smem_reduce[8 * M_TILE];
    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m)
            smem_reduce[wid * M_TILE + m] = acc[m];
    }
    __syncthreads();

    if (wid == 0) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m)
            acc[m] = (lane < 8) ? smem_reduce[lane * M_TILE + m] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m)
                acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }

        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (m < m_limit) {
                    out[(int64_t)(m_base + m) * N + n_idx] = __float2bfloat16(acc[m]);
                }
            }
        }
    }
}

// ====================================================================
// Kernel 4: Dequant packed weight to bf16 (for cuBLAS GEMM path)
//   Qint_packed [N, K_packed] → w_out [N, K]  where K = K_packed * 4
//
// Grid:  (N)
// Block: 256 threads
// ====================================================================

__global__ void DequantWeightPackedKernel(
    const uint8_t*       __restrict__ Qint_packed,  // [N, K_packed]
    __nv_bfloat16*       __restrict__ w_out,        // [N, K]
    float cb_scale,
    float cb_zero,
    int N, int K, int K_packed
) {
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    const uint8_t* q_row = Qint_packed + (int64_t)n * K_packed;
    __nv_bfloat16* w_row = w_out + (int64_t)n * K;

    // Process 4 packed bytes per iteration → 16 output bf16
    int iters = K_packed / 4;
    for (int it = tid; it < iters; it += blockDim.x) {
        int p_base = it * 4;
        int k_base = it * 16;

        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(q_row + p_base);
        __nv_bfloat16 w_vals[16];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint8_t bv = (packed4 >> (b * 8)) & 0xFF;
            w_vals[b*4+0] = __float2bfloat16(cb_scale * (float)((bv >> 6) & 3) + cb_zero);
            w_vals[b*4+1] = __float2bfloat16(cb_scale * (float)((bv >> 4) & 3) + cb_zero);
            w_vals[b*4+2] = __float2bfloat16(cb_scale * (float)((bv >> 2) & 3) + cb_zero);
            w_vals[b*4+3] = __float2bfloat16(cb_scale * (float)( bv        & 3) + cb_zero);
        }

        *reinterpret_cast<int4*>(w_row + k_base)     = *reinterpret_cast<int4*>(w_vals);
        *reinterpret_cast<int4*>(w_row + k_base + 8)  = *reinterpret_cast<int4*>(w_vals + 8);
    }

    // Remainder (when K_packed % 4 != 0)
    int p_rem = iters * 4;
    for (int p = p_rem + tid; p < K_packed; p += blockDim.x) {
        uint8_t bv = q_row[p];
        int k = p * 4;
        w_row[k+0] = __float2bfloat16(cb_scale * (float)((bv >> 6) & 3) + cb_zero);
        w_row[k+1] = __float2bfloat16(cb_scale * (float)((bv >> 4) & 3) + cb_zero);
        w_row[k+2] = __float2bfloat16(cb_scale * (float)((bv >> 2) & 3) + cb_zero);
        w_row[k+3] = __float2bfloat16(cb_scale * (float)( bv        & 3) + cb_zero);
    }
}

// ====================================================================
// Launch wrappers (called from C++ bindings)
// ====================================================================

void fused_su_had128_launch(
    torch::Tensor data,   // [M, K] bf16, 输入（只读）
    torch::Tensor SU,     // [K] bf16
    torch::Tensor out     // [M, K] bf16, 输出  ← 新增参数
) {
    int M = data.size(0);
    int K = data.size(1);
    TORCH_CHECK(K % 128 == 0, "K must be divisible by 128");
    TORCH_CHECK(out.sizes() == data.sizes(), "out shape must match data");  // 可选校验

    int num_blocks = K / 128;
    dim3 grid(M * num_blocks);
    dim3 block(16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    FusedSUHad128Kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(data.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(SU.data_ptr()),
        reinterpret_cast<      __nv_bfloat16*>(out.data_ptr()),  // ← 传入 out
        M, K
    );
}

void fused_had128_sv_launch(
    torch::Tensor data,   // [M, N] bf16, 输入（只读）
    torch::Tensor SV,     // [N] bf16
    torch::Tensor out     // [M, N] bf16, 输出  ← 新增参数
) {
    int M = data.size(0);
    int N = data.size(1);
    TORCH_CHECK(N % 128 == 0, "N must be divisible by 128");
    TORCH_CHECK(out.sizes() == data.sizes(), "out shape must match data");

    int num_blocks = N / 128;
    dim3 grid(M * num_blocks);
    dim3 block(16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    FusedHad128SVKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(data.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(SV.data_ptr()),
        reinterpret_cast<      __nv_bfloat16*>(out.data_ptr()),  // ← 传入 out
        M, N
    );
}

// ====================================================================
// Launch wrappers — PACKED variants
// ====================================================================

void fused_dequant_gemv_packed_launch(
    torch::Tensor x,            // [M, K] bf16
    torch::Tensor Qint_packed,  // [N, K_packed] uint8
    torch::Tensor out,          // [M, N] bf16
    float cb_scale,
    float cb_zero
) {
    int M = x.size(0);
    int K = x.size(1);
    int N = Qint_packed.size(0);
    int K_packed = Qint_packed.size(1);

    TORCH_CHECK(K == K_packed * 4, "K must equal K_packed * 4 for 2-bit packing");
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16 for packed GEMV");

    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (M <= 1) {
        dim3 grid(N, (M + 0) / 1);
        FusedDequantGEMVPackedKernel<1><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_packed.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_packed, M
        );
    } else if (M <= 2) {
        dim3 grid(N, (M + 1) / 2);
        FusedDequantGEMVPackedKernel<2><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_packed.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_packed, M
        );
    } else if (M <= 4) {
        dim3 grid(N, (M + 3) / 4);
        FusedDequantGEMVPackedKernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_packed.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_packed, M
        );
    } else {
        dim3 grid(N, (M + 7) / 8);
        FusedDequantGEMVPackedKernel<8><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_packed.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_packed, M
        );
    }
}

void dequant_weight_packed_launch(
    torch::Tensor Qint_packed,  // [N, K_packed] uint8
    torch::Tensor w_out,        // [N, K] bf16
    float cb_scale,
    float cb_zero
) {
    int N = Qint_packed.size(0);
    int K_packed = Qint_packed.size(1);
    int K = K_packed * 4;

    dim3 grid(N);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    DequantWeightPackedKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Qint_packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(w_out.data_ptr()),
        cb_scale, cb_zero, N, K, K_packed
    );
}


// ####################################################################
//  TERNARY 5-TRITS-PER-BYTE VARIANTS (Optimized)
//  Each uint8 byte stores 5 ternary values {0,1,2}:
//    byte = v0*81 + v1*27 + v2*9 + v3*3 + v4   (range 0..242)
//  Qint_ternary shape: [N, K_trit] where K_trit = ceil(K/5)
//
//  Optimizations over naive implementation:
//    1. Shared-memory LUT: byte → packed 5×2-bit trits (eliminates divisions)
//    2. Two-accumulator decomposition: sum(x*v) + sum(x) → cb_s*Σxv + cb_z*Σx
//    3. Batch 4-byte weight reads via uint32_t (20 values per iteration)
//    4. Pre-computed 3-value weight table for GEMM dequant path
// ####################################################################

// Build shared-memory LUT: 256 entries (0-242 valid), each is uint16_t
// packing 5 trit values × 2 bits: v0[9:8] v1[7:6] v2[5:4] v3[3:2] v4[1:0]
__device__ __forceinline__ void build_trit_lut(uint16_t* trit_lut, int tid) {
    if (tid < 243) {
        unsigned v = tid;
        unsigned t0 = v / 81u; v -= t0 * 81u;
        unsigned t1 = v / 27u; v -= t1 * 27u;
        unsigned t2 = v / 9u;  v -= t2 * 9u;
        unsigned t3 = v / 3u;
        unsigned t4 = v - t3 * 3u;
        trit_lut[tid] = (uint16_t)((t0 << 8) | (t1 << 6) | (t2 << 4) | (t3 << 2) | t4);
    } else if (tid < 256) {
        trit_lut[tid] = 0;
    }
}

// Extract 5 float trit values from a LUT entry (for dequant path)
__device__ __forceinline__ void extract_trits_dequant(
    uint16_t lut_val, const __nv_bfloat16 w3[3], __nv_bfloat16 out[5]
) {
    out[0] = w3[(lut_val >> 8) & 3];
    out[1] = w3[(lut_val >> 6) & 3];
    out[2] = w3[(lut_val >> 4) & 3];
    out[3] = w3[(lut_val >> 2) & 3];
    out[4] = w3[ lut_val       & 3];
}

// ====================================================================
// Kernel 5: Fused Dequant GEMV (Ternary 5-trit packed) — Optimized
//   LUT replaces integer division for trit extraction.
//   Pre-computed 3-value weight table eliminates per-element dequant math.
//   4-byte batched weight reads for reduced loop overhead.
//
// Grid:  (N, ceil(M / M_TILE))
// Block: 256 threads
// ====================================================================

template<int M_TILE>
__global__ void FusedDequantGEMVTernaryKernel(
    const __nv_bfloat16* __restrict__ x,             // [M, K]
    const uint8_t*       __restrict__ Qint_ternary,  // [N, K_trit]
    __nv_bfloat16*       __restrict__ out,           // [M, N]
    float cb_scale,
    float cb_zero,
    int N, int K, int K_trit, int M_global
) {
    __shared__ uint16_t trit_lut[256];
    int tid = threadIdx.x;
    build_trit_lut(trit_lut, tid);
    __syncthreads();

    int n_idx = blockIdx.x;
    int m_base = blockIdx.y * M_TILE;
    if (n_idx >= N) return;
    int m_limit = min(M_TILE, M_global - m_base);
    if (m_limit <= 0) return;

    // Pre-compute the only 3 possible dequantized weight values
    float w3[3];
    w3[0] = cb_zero;
    w3[1] = cb_scale + cb_zero;
    w3[2] = 2.0f * cb_scale + cb_zero;

    float acc[M_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) acc[m] = 0.0f;

    const uint8_t* q_row = Qint_ternary + (int64_t)n_idx * K_trit;

    // ── Main loop: 4 bytes (20 trit values) per iteration ──
    int iters4 = K_trit / 4;
    for (int it = tid; it < iters4; it += blockDim.x) {
        int q_base = it * 4;
        int k_base = it * 20;

        uint16_t lut0 = trit_lut[q_row[q_base    ]];
        uint16_t lut1 = trit_lut[q_row[q_base + 1]];
        uint16_t lut2 = trit_lut[q_row[q_base + 2]];
        uint16_t lut3 = trit_lut[q_row[q_base + 3]];

        // Dequant all 20 values using 3-value table lookup
        float w[20];
        #pragma unroll
        for (int i = 0; i < 5; ++i) w[i]      = w3[(lut0 >> (8 - i*2)) & 3];
        #pragma unroll
        for (int i = 0; i < 5; ++i) w[5 + i]  = w3[(lut1 >> (8 - i*2)) & 3];
        #pragma unroll
        for (int i = 0; i < 5; ++i) w[10 + i] = w3[(lut2 >> (8 - i*2)) & 3];
        #pragma unroll
        for (int i = 0; i < 5; ++i) w[15 + i] = w3[(lut3 >> (8 - i*2)) & 3];

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < m_limit) {
                const __nv_bfloat16* x_ptr = x + (int64_t)(m_base + m) * K + k_base;
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 20; ++i) {
                    dot += __bfloat162float(x_ptr[i]) * w[i];
                }
                acc[m] += dot;
            }
        }
    }

    // ── Remainder: one byte at a time ──
    int rem_start = iters4 * 4;
    for (int it = rem_start + tid; it < K_trit; it += blockDim.x) {
        int k_base = it * 5;
        if (k_base >= K) break;
        uint16_t lut_val = trit_lut[q_row[it]];
        int k_valid = min(5, K - k_base);

        float w[5];
        #pragma unroll
        for (int i = 0; i < 5; ++i) w[i] = w3[(lut_val >> (8 - i*2)) & 3];

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < m_limit) {
                const __nv_bfloat16* x_ptr = x + (int64_t)(m_base + m) * K + k_base;
                float dot = 0.0f;
                for (int i = 0; i < k_valid; ++i) {
                    dot += __bfloat162float(x_ptr[i]) * w[i];
                }
                acc[m] += dot;
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    // Block-level reduction via shared memory
    __shared__ float smem_reduce[8 * M_TILE];
    __syncthreads();
    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m)
            smem_reduce[wid * M_TILE + m] = acc[m];
    }
    __syncthreads();

    if (wid == 0) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m)
            acc[m] = (lane < 8) ? smem_reduce[lane * M_TILE + m] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m)
                acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }

        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (m < m_limit) {
                    out[(int64_t)(m_base + m) * N + n_idx] = __float2bfloat16(acc[m]);
                }
            }
        }
    }
}

// ====================================================================
// Kernel 6: Dequant ternary 5-trit packed weight to bf16 — Optimized
//   Uses LUT + pre-computed 3-value table for zero-arithmetic dequant.
//   Batch 4-byte weight reads.
//
// Grid:  (N)
// Block: 256 threads
// ====================================================================

__global__ void DequantWeightTernaryKernel(
    const uint8_t*       __restrict__ Qint_ternary,  // [N, K_trit]
    __nv_bfloat16*       __restrict__ w_out,         // [N, K]
    float cb_scale,
    float cb_zero,
    int N, int K, int K_trit
) {
    __shared__ uint16_t trit_lut[256];
    int tid = threadIdx.x;
    build_trit_lut(trit_lut, tid);
    __syncthreads();

    int n = blockIdx.x;
    if (n >= N) return;

    const uint8_t* q_row = Qint_ternary + (int64_t)n * K_trit;
    __nv_bfloat16* w_row = w_out + (int64_t)n * K;

    // Only 3 possible dequantized values — precompute in registers
    __nv_bfloat16 w3[3];
    w3[0] = __float2bfloat16(cb_zero);
    w3[1] = __float2bfloat16(cb_scale + cb_zero);
    w3[2] = __float2bfloat16(2.0f * cb_scale + cb_zero);

    // Main loop: 4 bytes → 20 output bf16 (write as 10 × bf16 pairs)
    int iters4 = K_trit / 4;
    for (int it = tid; it < iters4; it += blockDim.x) {
        int q_base = it * 4;
        int k_base = it * 20;

        __nv_bfloat16 vals[20];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            extract_trits_dequant(trit_lut[q_row[q_base + b]], w3, vals + b * 5);
        }

        // Write 20 bf16 as 10 pairs (uint32_t, always 4-byte aligned since k_base is even)
        uint32_t* dst = reinterpret_cast<uint32_t*>(w_row + k_base);
        uint32_t* src = reinterpret_cast<uint32_t*>(vals);
        #pragma unroll
        for (int j = 0; j < 10; ++j) {
            dst[j] = src[j];
        }
    }

    // Remainder
    int rem_start = iters4 * 4;
    for (int it = rem_start + tid; it < K_trit; it += blockDim.x) {
        int k_base = it * 5;
        if (k_base >= K) break;
        uint16_t lut_val = trit_lut[q_row[it]];
        int k_valid = min(5, K - k_base);
        __nv_bfloat16 vals[5];
        extract_trits_dequant(lut_val, w3, vals);
        for (int i = 0; i < k_valid; ++i)
            w_row[k_base + i] = vals[i];
    }
}

// ====================================================================
// Launch wrappers — TERNARY variants
// ====================================================================

void fused_dequant_gemv_ternary_launch(
    torch::Tensor x,              // [M, K] bf16
    torch::Tensor Qint_ternary,   // [N, K_trit] uint8
    torch::Tensor out,            // [M, N] bf16
    float cb_scale,
    float cb_zero,
    int K_orig
) {
    int M = x.size(0);
    int K = x.size(1);
    int N = Qint_ternary.size(0);
    int K_trit = Qint_ternary.size(1);

    TORCH_CHECK(K == K_orig, "K mismatch: x.size(1)=", K, " vs K_orig=", K_orig);

    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (M <= 1) {
        dim3 grid(N, 1);
        FusedDequantGEMVTernaryKernel<1><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_ternary.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_trit, M);
    } else if (M <= 2) {
        dim3 grid(N, (M + 1) / 2);
        FusedDequantGEMVTernaryKernel<2><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_ternary.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_trit, M);
    } else if (M <= 4) {
        dim3 grid(N, (M + 3) / 4);
        FusedDequantGEMVTernaryKernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_ternary.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_trit, M);
    } else {
        dim3 grid(N, (M + 7) / 8);
        FusedDequantGEMVTernaryKernel<8><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const uint8_t*>(Qint_ternary.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            cb_scale, cb_zero, N, K, K_trit, M);
    }
}

void dequant_weight_ternary_launch(
    torch::Tensor Qint_ternary,  // [N, K_trit] uint8
    torch::Tensor w_out,         // [N, K] bf16
    float cb_scale,
    float cb_zero,
    int K_orig
) {
    int N = Qint_ternary.size(0);
    int K_trit = Qint_ternary.size(1);

    dim3 grid(N);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    DequantWeightTernaryKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Qint_ternary.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(w_out.data_ptr()),
        cb_scale, cb_zero, N, K_orig, K_trit);
}
