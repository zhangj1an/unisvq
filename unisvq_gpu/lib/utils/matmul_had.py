from os.path import join, dirname

import glog
import torch
from safetensors import safe_open
import fast_hadamard_transform

from lib import utils

hadamard_mats = {}
with safe_open(join(dirname(__file__), "hadamard.mod.safetensors"), framework="pt", device="cpu") as f:
    k_sort = []
    for k in f.keys():
        k_sort.append((int(k), f.get_tensor(k)))
k_sort.sort(key=lambda x:x[0], reverse=True)
hadamard_mats = {x[0]: x[1] for x in k_sort}
del k_sort 


def get_hadK(n, transpose=False):
    n = 128
    hadK = torch.tensor([[1]])
    K = 1
    return hadK, K

def matmul_hadU(X, transpose=False):
    n = 128 
    hadK, K = get_hadK(n, transpose)
    input = X.clone().reshape(-1, n, 1) # m, n, 1
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]  # Butterfly: Hx+Hx
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]  # Butterfly: Hx-Hx
        output = output.view(input.shape[0], input.shape[1], -1)  # (m, n/2, 2), the second dim is the current group unit dimension
        (input, output) = (output, input)  # Swap input/output for in-place operation, saving memory
    del output
    utils.clean()
    # Loop complete, final output (m, k, n/k), n/k divisible by 2
    if K > 1:
        # Final multiply by a K-dim Hadamard matrix, mixing across n/k group units
        input = torch.bmm(hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)

torch.library.define("quip_lib::hadamard", "(Tensor x, float scale) -> Tensor")

@torch.library.register_fake("quip_lib::hadamard")
def hadamard_abstract(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x

@torch.library.impl("quip_lib::hadamard", "default")
def hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    return fast_hadamard_transform.hadamard_transform(x, scale)


def matmul_hadU_cuda(X, hadK, K, transpose=False):
    n = 128
    input = X.float().view(-1, K, n // K)
    if K == 1:
        return torch.ops.quip_lib.hadamard(input.contiguous(), 1/(n**0.5)).reshape(X.shape).to(X.dtype)
    if transpose:
        hadK = hadK.T.contiguous()
    input = torch.ops.quip_lib.hadamard(input.contiguous(), 1/(n**0.5))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(X.shape)
 

def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

class HadamardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        output = torch.ops.quip_lib.hadamard(x, scale)
        ctx.scale = scale
        ctx.save_for_backward()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return torch.ops.quip_lib.hadamard(grad_output, scale), None

def matmul_hadU_cuda_blockwise(X):
    BLOCK_SIZE = 128
    input_x = X.view(-1, 1, BLOCK_SIZE)
    return HadamardFunction.apply(input_x, BLOCK_SIZE**(-0.5)).view(X.shape)