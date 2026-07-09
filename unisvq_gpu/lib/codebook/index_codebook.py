import os
import math

import torch
from torch import nn

from ..utils.matmul_had import matmul_hadU_cuda

import glog

_LI_CODESZ = 4
_LI_CODEBIT = 2

class Index_codebook(nn.Module):
    
    def __init__(self, inference=False, codebook_bit=None, **kwargs):
        super().__init__()
        self.id = "index"
        self.codesz = _LI_CODESZ
        self.opt_scale = 1.0
        
        self.packsz = 1
        self.pack_out = False
        self.version = 0
        self.codebook_scale = 1.0

        if codebook_bit is None:
            self.codebook_bit = _LI_CODEBIT
        else:
            self.codebook_bit = codebook_bit

        if self.codebook_bit == 2:
            self.idx_dtype = torch.uint8
        elif self.codebook_bit == 3:
            self.idx_dtype = torch.int16
        else:
            raise NotImplementedError(f"codebook_bit={self.codebook_bit} is not implemented")

        self.min_var = 0
        self.max_var = 2**self.codebook_bit - 1
        
        linear_proj = nn.Linear(_LI_CODESZ, _LI_CODESZ, bias=True, dtype=torch.bfloat16)
        linear_proj.weight.requires_grad = False
        linear_proj.bias.requires_grad = False
        input_index_sample = torch.cartesian_prod(*([torch.arange(2**self.codebook_bit)]*_LI_CODESZ)).to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)

        ortho_mat = torch.eye(_LI_CODESZ)
        ortho_mat = ortho_mat.to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        proj_mean = (2**self.codebook_bit-1)/2
        proj_scale = (torch.pow(torch.arange(2**self.codebook_bit, dtype=torch.bfloat16), 2).mean().item() - proj_mean**2)
        proj_scale = 1/math.sqrt(proj_scale)
        linear_proj.weight.data = ortho_mat * proj_scale
        linear_proj.bias.data = -nn.functional.linear(torch.ones(1, _LI_CODESZ, device=linear_proj.weight.device, dtype=linear_proj.weight.dtype), ortho_mat).squeeze() * proj_scale * proj_mean
        fake_codebook = linear_proj(input_index_sample) 
        self.register_buffer('linear_proj_weight', linear_proj.weight.data.clone())
        self.register_buffer('linear_proj_bias', linear_proj.bias.data.clone())
        self.register_buffer('grid', fake_codebook)
        if not inference:
            self.register_buffer('grid_norm', fake_codebook.norm(dim=-1)**2)


    def quantize(self, w, return_idx=True, **kwargs):
        assert w.shape[-1] == self.codesz
        wqidx = (2 * w @ self.grid.T - self.grid_norm).argmax(1)
        wq = self.grid[wqidx, :]
        if return_idx:
            return wq, wqidx.to(self.idx_dtype)
        return wq
    
    def maybe_pack_idxs(self, idxs):
        return idxs

    def by_idxs(self, Qidxs, **kwargs):
        m = Qidxs.shape[0]
        Qidxs = Qidxs.to(torch.long).flatten()
        w = self.grid[Qidxs, :].reshape(m, -1)
        return w

class IndexLinear(nn.Module):
    
    def __init__(self, device, codebook_bit=None, **kwarg):
        super().__init__()
        self.codebook = Index_codebook(codebook_bit=codebook_bit).to(torch.bfloat16).to(device)
        self.scale = 32

    def maybe_unpack_idxs(self, idxs):
        return (idxs, )

    def cache_WH(self, **kwargs):
        return 
    
    def forward(self, input, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right, rescale_WH=False, scaleWH=None, train_mode=False, **kwargs):
        n, m = len(SU), len(SV)
        x = input.view(-1, n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU

        x = matmul_hadU_cuda(x, had_left_T, K_left) / self.scale
    
        W_decompressed = self.codebook.by_idxs(Qidxs_list[0])
        x = (x.to(W_decompressed.dtype) @ W_decompressed.T).to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV * self.scale

        output = x.view(*input.shape[:-1], m)
        
        return output
    
