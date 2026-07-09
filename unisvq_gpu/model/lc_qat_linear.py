import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributed as dist

from lib.utils.matmul_had import matmul_hadU_cuda_blockwise


def power_derivative(w, delta, k, power_clamp_max):
    abs_term = torch.abs(2*w/delta-1)
    val = abs_term.pow(1.0/k-1)/k
    return torch.clamp(val, max=power_clamp_max)


class DGEFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, w, qmax:int=3, k:float=5.0, power_clamp_max: float=3.0):
        ctx.save_for_backward(w)
        ctx.k = k
        ctx.power_clamp_max = power_clamp_max
        ctx.qmax = qmax
        q = torch.round(w).clamp(0, qmax) 
        return q

    @staticmethod
    def backward(ctx, grad_output):
        w = ctx.saved_tensors[0]

        k = ctx.k
        power_clamp_max = ctx.power_clamp_max
        qmax = ctx.qmax

        q_idx = torch.floor(w).clamp(0, qmax)
        q_center = q_idx
        rel = w-q_center
        dy = power_derivative(rel, 1.0, k, power_clamp_max=power_clamp_max)
        grad = grad_output * dy

        grad_update_rate = 1.0
        grad_shape = grad.shape
        num_blocks = 128
        block_size = grad.numel() // num_blocks
        block_mask = (torch.rand(num_blocks, device=grad.device) < grad_update_rate).to(grad.dtype)
        grad_input =  grad.view(num_blocks, block_size) * block_mask.unsqueeze(1)

        return grad_input.view(grad_shape), None, None, None, None, None


@torch.compile(fullgraph=False)
def quant_with_max_var(weight: torch.Tensor, max_var: int, weight_scale: float, codebook_scale: float, codebook_zero_point: float) -> torch.Tensor:
    ori_weight = weight_scale * weight + max_var/2
    w = DGEFunction.apply(ori_weight, max_var, 5.0, 3.0)
    w = codebook_scale*w + codebook_zero_point
    return w  


def is_meta_module(module: nn.Module):
    for p in module.parameters():
        if p.is_meta:
            return True
    return False


class CodeBookGenerator(nn.Module):
    
    def __init__(self, vec_size, vec_bit, in_features, out_features, device=None, dtype=None, proj_type='bypass', mixed_percision=False, linear_name="linear"):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.proj_type = proj_type
        self.min_var = 0
        if vec_bit < 2:
            max_var = 2
        else:
            max_var = 2**vec_bit-1

        if mixed_percision:
            self.register_buffer("max_var", torch.tensor(max_var, **factory_kwargs), persistent=True)
        else:
            self.max_var = max_var

        self.codebook_zero_point = nn.Parameter(torch.tensor(0.0, **factory_kwargs), requires_grad=False)
        self.codebook_scale = nn.Parameter(torch.tensor(1.0, **factory_kwargs), requires_grad=False)

        self.weight_scale = math.sqrt(out_features+in_features)/2
        self.grad_update_rate = 0
        self.linear_name = linear_name

    def quant_weight(self, ori_weight) -> torch.Tensor:
        # if not getattr(self, "_buf_synced", False):
        #     if dist.is_initialized():
        #         dist.broadcast(self.max_var, src=0)
        #     self._buf_synced = True

        if type(self.max_var) == nn.Parameter or type(self.max_var) == torch.Tensor:
            max_var = self.max_var.data.item()
        else:
            max_var = self.max_var
            
        return quant_with_max_var(
            weight=ori_weight,
            max_var=max_var,
            weight_scale=self.weight_scale,
            codebook_scale=self.codebook_scale,
            codebook_zero_point=self.codebook_zero_point
        )
    
    def forward(self, ori_weight) -> torch.Tensor:
        return self.quant_weight(ori_weight)
    
    def set_rate(self, grad_update_rate):
        self.grad_update_rate = grad_update_rate


class LCQATHadLinear(nn.Module):

    def __init__(
            self, 
            in_features, 
            out_features, 
            vec_size=4, 
            vec_bit=2, 
            bias=False, 
            device=None, 
            dtype=None,
            proj_type="bypass",
            mixed_percision=False,
            linear_name="linear",
        ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vec_size = vec_size
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.SU = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        self.SV = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.code_generator = CodeBookGenerator(
            vec_size=vec_size, 
            vec_bit=vec_bit, 
            in_features=in_features,
            out_features=out_features,
            proj_type=proj_type,
            mixed_percision=mixed_percision,
            linear_name=linear_name
        )   

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, inputs):
        inputs = inputs * self.SU
        inputs = matmul_hadU_cuda_blockwise(inputs)
        weight = self.code_generator.quant_weight(self.weight)
        outputs = F.linear(inputs, weight)
        outputs = matmul_hadU_cuda_blockwise(outputs)
        outputs = outputs * self.SV

        if self.bias is not None:
            outputs += self.bias
        
        return outputs
