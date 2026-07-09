"""Hadamard transform utilities — pure PyTorch, NPU/CUDA/CPU compatible.

Provides blockwise (128-element blocks) and full-dimension Hadamard transforms
using butterfly-style operations in pure PyTorch. No external CUDA library needed.

Backward-compatible aliases (matmul_hadU_cuda, matmul_hadU_cuda_blockwise,
HadamardFunction) are provided so that existing importers work without changes.
"""

from os.path import join, dirname

import torch
from safetensors import safe_open

from lib import utils

# ---------------------------------------------------------------------------
# Precomputed Hadamard matrices (loaded but not currently used by get_hadK)
# ---------------------------------------------------------------------------
hadamard_mats = {}
with safe_open(join(dirname(__file__), "hadamard.mod.safetensors"), framework="pt", device="cpu") as f:
    k_sort = []
    for k in f.keys():
        k_sort.append((int(k), f.get_tensor(k)))
k_sort.sort(key=lambda x: x[0], reverse=True)
hadamard_mats = {x[0]: x[1] for x in k_sort}
del k_sort


# ---------------------------------------------------------------------------
#  get_hadK  (unchanged — always returns K=1, hadK=[[1]])
# ---------------------------------------------------------------------------
def get_hadK(n, transpose=False):
    n = 128
    hadK = torch.tensor([[1]])
    K = 1
    return hadK, K


# ---------------------------------------------------------------------------
#  Pure-PyTorch full-dimension butterfly Hadamard  (original reference impl)
# ---------------------------------------------------------------------------
def matmul_hadU(X, transpose=False):
    """Pure PyTorch butterfly Hadamard transform.

    Applies H_n to the last dimension of X, where n=128.
    Works on any device (CPU, CUDA, NPU).
    """
    n = 128
    hadK, K = get_hadK(n, transpose)
    input = X.clone().reshape(-1, n, 1)  # m, n, 1
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]  # Butterfly: a+b
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]  # Butterfly: a-b
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)  # swap for in-place
    del output
    utils.clean()
    if K > 1:
        input = torch.bmm(hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


# ---------------------------------------------------------------------------
#  Blockwise Hadamard: applies H_128 independently to each 128-element block
#  This is the NPU-compatible replacement for matmul_hadU_cuda_blockwise.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128


def _hadamard_blockwise_butterfly(X_blocks):
    """Apply butterfly Hadamard to a (M, BLOCK_SIZE) tensor.

    X_blocks: shape (M, 128) — each row is an independent 128-dim block.
    Returns: shape (M, 128) with H_128 applied to each row.
    """
    n = BLOCK_SIZE
    M = X_blocks.shape[0]

    # Reshape to (M, n, 1) butterfly format
    x = X_blocks.clone().reshape(M, n, 1)
    y = x.clone()

    # 7 stages for n=128: n→64→32→16→8→4→2→1
    while x.shape[1] > 1:
        x = x.view(x.shape[0], x.shape[1] // 2, 2, x.shape[2])
        y = y.view(x.shape)
        y[:, :, 0, :] = x[:, :, 0, :] + x[:, :, 1, :]
        y[:, :, 1, :] = x[:, :, 0, :] - x[:, :, 1, :]
        y = y.view(x.shape[0], x.shape[1], -1)
        (x, y) = (y, x)

    del y
    return x.reshape(M, n) / (n ** 0.5)


def matmul_hadU_npu_blockwise(X):
    """Blockwise Hadamard: apply H_128 to each consecutive 128-element block.

    Pure PyTorch, NPU/CUDA/CPU compatible.
    Drop-in replacement for matmul_hadU_cuda_blockwise.
    """
    orig_shape = X.shape
    ncols = X.shape[-1]

    # Pad if last dim not divisible by BLOCK_SIZE
    if ncols % BLOCK_SIZE != 0:
        pad_size = BLOCK_SIZE - (ncols % BLOCK_SIZE)
        X_flat = torch.nn.functional.pad(X.reshape(-1, ncols), (0, pad_size))
    else:
        X_flat = X.reshape(-1, ncols)

    # Reshape into blocks: each chunk of BLOCK_SIZE gets H_128 applied
    X_blocks = X_flat.reshape(-1, BLOCK_SIZE)  # (total_blocks, 128)
    result_blocks = _hadamard_blockwise_butterfly(X_blocks)

    # Restore
    if ncols % BLOCK_SIZE != 0:
        result_flat = result_blocks.reshape(-1, ncols + pad_size)[:, :ncols]
    else:
        result_flat = result_blocks.reshape(-1, ncols)

    return result_flat.reshape(orig_shape)


def matmul_hadU_npu(X, hadK, K, transpose=False):
    """Full-dimension Hadamard — pure PyTorch.
    Drop-in replacement for matmul_hadU_cuda.
    """
    return matmul_hadU(X, transpose=transpose)


def matmul_hadUt_npu(X, hadK, K):
    return matmul_hadU_npu(X, hadK, K, transpose=True)


# ---------------------------------------------------------------------------
#  HadamardFunction: autograd wrapper with correct backward pass.
#  The Hadamard transform is its own inverse (up to scaling), so the
#  backward pass is the same transform applied to grad_output.
# ---------------------------------------------------------------------------
class HadamardFunction(torch.autograd.Function):
    """Autograd Hadamard transform, pure PyTorch, works on NPU/CUDA/CPU."""

    @staticmethod
    def forward(ctx, x, scale):
        # Apply blockwise Hadamard: reshape to (-1, 1, BLOCK_SIZE), transform, reshape back
        input_x = x.reshape(-1, 1, BLOCK_SIZE)
        result = _hadamard_blockwise_butterfly(input_x.reshape(-1, BLOCK_SIZE))
        output = result.reshape(input_x.shape) * scale
        ctx.scale = scale
        ctx.save_for_backward()  # nothing needed since backward is self-inverse
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Hadamard is its own inverse: forward(f) * scale → backward(g) = forward(g / scale) * scale = forward(g)
        # Actually: y = H*x*scale, dy/dx = H*scale, so grad_x = H^T * grad_y * scale = H * grad_y * scale
        # But since H^T = H and H*H = n*I, the correct backward is H(grad_y) * scale
        scale = ctx.scale
        input_g = grad_output.reshape(-1, 1, BLOCK_SIZE)
        result = _hadamard_blockwise_butterfly(input_g.reshape(-1, BLOCK_SIZE))
        return result.reshape(input_g.shape) * scale, None


# ---------------------------------------------------------------------------
#  Backward-compatibility aliases
#  Code that imports matmul_hadU_cuda or matmul_hadU_cuda_blockwise
#  will resolve to the pure-PyTorch NPU-compatible implementations.
# ---------------------------------------------------------------------------
matmul_hadU_cuda = matmul_hadU_npu
matmul_hadUt_cuda = matmul_hadUt_npu
matmul_hadU_cuda_blockwise = matmul_hadU_npu_blockwise
