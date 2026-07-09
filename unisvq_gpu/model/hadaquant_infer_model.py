import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn

from transformers.utils import logging
from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from safetensors.torch import load_file
logger = logging.get_logger(__name__)

import HadaQuant as _HadaQuant

_GEMV_THRESHOLD = int(os.environ.get("LCQAT_GEMV_THRESHOLD", "8"))

# Register ops via torch.library for torch.compile / CUDA Graph compatibility

# ── Hadamard ──
torch.library.define("hadaquant::fused_su_had128_", "(Tensor(a!) data, Tensor SU, Tensor out) -> ()")

@torch.library.impl("hadaquant::fused_su_had128_", "cuda")
def _su_had_impl(data, SU, out):
    _HadaQuant.fused_su_had128(data, SU, out)

@torch.library.register_fake("hadaquant::fused_su_had128_")
def _su_had_fake(data, SU, out):
    return

torch.library.define("hadaquant::fused_had128_sv_", "(Tensor(a!) data, Tensor SV, Tensor out) -> ()")

@torch.library.impl("hadaquant::fused_had128_sv_", "cuda")
def _had_sv_impl(data, SV, out):
    _HadaQuant.fused_had128_sv(data, SV, out)

@torch.library.register_fake("hadaquant::fused_had128_sv_")
def _had_sv_fake(data, SV, out):
    return

# ── 2-bit packed (4 values/byte) ──
torch.library.define("hadaquant::fused_dequant_gemv", "(Tensor x, Tensor Qint, float cb_s, float cb_z) -> Tensor")

@torch.library.impl("hadaquant::fused_dequant_gemv", "cuda")
def _gemv_impl(x, Qint, cb_s, cb_z):
    return _HadaQuant.fused_dequant_gemv_packed(x, Qint, cb_s, cb_z)

@torch.library.register_fake("hadaquant::fused_dequant_gemv")
def _gemv_fake(x, Qint, cb_s, cb_z):
    return torch.empty(x.shape[0], Qint.shape[0], dtype=x.dtype, device=x.device)

torch.library.define("hadaquant::dequant_weight_packed", "(Tensor Qint, float cb_s, float cb_z) -> Tensor")

@torch.library.impl("hadaquant::dequant_weight_packed", "cuda")
def _dequant_w_impl(Qint, cb_s, cb_z):
    return _HadaQuant.dequant_weight_packed(Qint, cb_s, cb_z)

@torch.library.register_fake("hadaquant::dequant_weight_packed")
def _dequant_w_fake(Qint, cb_s, cb_z):
    N, K_packed = Qint.shape
    return torch.empty(N, K_packed * 4, dtype=torch.bfloat16, device=Qint.device)


def _hq_fused_su_had128(data, SU):
    out = torch.empty_like(data)
    torch.ops.hadaquant.fused_su_had128_(data, SU, out)
    return out


def _hq_fused_had128_sv(data, SV):
    out = torch.empty_like(data)
    torch.ops.hadaquant.fused_had128_sv_(data, SV, out)
    return out


def _hq_fused_dequant_gemv(x, Qint, cb_s, cb_z):
    return torch.ops.hadaquant.fused_dequant_gemv(x, Qint, cb_s, cb_z)


def _hq_dequant_weight(Qint, cb_s, cb_z):
    return torch.ops.hadaquant.dequant_weight_packed(Qint, cb_s, cb_z)


def remap_checkpoint_keys(state_dict: dict, config=None) -> dict:

    if any(".code_generator.codebook_scale" in k for k in state_dict):
        new_sd = {}
        biases = {}
        for k, v in state_dict.items():
            if k.endswith("_proj.weight") and v.dtype == torch.uint8:
                new_sd[k[:-len("weight")] + "Qint"] = v
            elif ".code_generator.codebook_scale" in k:
                new_sd[k.replace(".code_generator.codebook_scale", ".cb_scale")] = v
            elif ".code_generator.codebook_zero_point" in k:
                new_sd[k.replace(".code_generator.codebook_zero_point", ".cb_zero")] = v
            elif k.endswith("_proj.bias"):
                biases[k[:-len("bias")] + "lcqat_bias"] = v
            else:
                new_sd[k] = v
        new_sd["__biases__"] = biases

        return new_sd

    return state_dict


def load_and_assign(model, state_dict):
    """Load state dict and assign biases that can't go through load_state_dict.
    Pre-resizes Qint buffers if the checkpoint has a different packed format.
    Call AFTER model.to(device) since biases are moved to the model's device.
    """
    biases = state_dict.pop("__biases__", {})
    # print("biases", biases)
    # Pre-resize Qint buffers to match checkpoint shapes (ternary ↔ 2-bit)
    model_sd = model.state_dict()
    for key in state_dict:
        if key.endswith(".Qint") and key in model_sd:
            if state_dict[key].shape != model_sd[key].shape:
                parts = key.rsplit(".", 1)
                mod = dict(model.named_modules()).get(parts[0])
                if mod is not None:
                    new_buf = torch.zeros_like(state_dict[key])
                    mod.Qint = new_buf

    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=True, assign=True)
    return biases


def assign_biases(model, biases):
    """Assign bias tensors to LCQATLinear modules. Call after model.to(device)."""
    if not biases:
        return
    named = dict(model.named_modules())
    for k, v in biases.items():
        parts = k.rsplit(".", 1)
        mod = named.get(parts[0])
        if mod is not None:
            ref = mod.SU if hasattr(mod, "SU") else next(mod.buffers())
            mod.lcqat_bias = v.to(device=ref.device, dtype=ref.dtype)
    print(f"[load] assigned {len(biases)} bias tensors")


# ---------------------------------------------------------------------------
# LCQATLinear — Block-128 Hadamard quantized linear
#   Supports two packed formats:
#     "2bit":    Qint [out, in//4] uint8, 4 values/byte (2-bit, for 4-value models)
#     "ternary": Qint [out, K_trit] uint8, 5 values/byte (5-trit, for 3-value models)
#   Decode (M≤threshold): fused dequant GEMV
#   Large M: dequant to bf16 + cuBLAS GEMM
# ---------------------------------------------------------------------------
class LCQATCudaLinear(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("Qint", torch.zeros(out_features, in_features // 4, dtype=torch.uint8))
        self.register_buffer("cb_scale", torch.tensor(0.0, dtype=dtype))
        self.register_buffer("cb_zero", torch.tensor(0.0, dtype=dtype))
        self.register_buffer("SU", torch.ones(in_features, dtype=dtype))
        self.register_buffer("SV", torch.ones(out_features, dtype=dtype))
        self.lcqat_bias: Optional[torch.Tensor] = None

        self._cached_w = None
        self._cb_s: float = 0.0
        self._cb_z: float = 0.0
        self._quant_mode: str = "2bit"

    def _get_dequant_weight(self):
        if self._cached_w is None:
            self._cached_w = _hq_dequant_weight(self.Qint, self._cb_s, self._cb_z)
        return self._cached_w

    def warm_weight_cache(self):
        """Pre-populate dequant weight cache (call before torch.compile)."""
        self._get_dequant_weight()

    def clear_weight_cache(self):
        self._cached_w = None

    def prepare_for_inference(self):
        """Pre-extract codebook scalars and detect quant mode."""
        self._cb_s = self.cb_scale.float().item()
        self._cb_z = self.cb_zero.float().item()
        self._quant_mode = "2bit"

    def forward(self, x):
        shape = x.shape
        dtype = self.SU.dtype
        if x.dtype != dtype:
            x = x.to(dtype)
        x = x.view(-1, self.in_features)
        M = x.shape[0]

        x = _hq_fused_su_had128(x.contiguous(), self.SU)

        if M <= _GEMV_THRESHOLD:
            out = _hq_fused_dequant_gemv(x, self.Qint, self._cb_s, self._cb_z)
        else:
            if self._cached_w is not None:
                out = F.linear(x, self._cached_w)
            else:
                w = _hq_dequant_weight(self.Qint, self._cb_s, self._cb_z)
                out = F.linear(x, w)

        out = _hq_fused_had128_sv(out.contiguous(), self.SV)

        out = out.view(*shape[:-1], self.out_features)
        if self.lcqat_bias is not None:
            out = out + self.lcqat_bias
        return out


class Qwen3LCQATCudaConfig(Qwen3Config):
    model_type = "qwen3_cuda_lcqat"
    def __init__(self, *args, vec_size=4, vec_bit=2, proj_type="bypass",  mixed_percision=False, **kwargs):
        self.vec_size = vec_size
        self.vec_bit = vec_bit
        self.proj_type = proj_type
        self.mixed_percision = mixed_percision
        super().__init__(*args, **kwargs)
    
class Qwen3LCQATCudaCudaForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3LCQATCudaConfig
    def __init__(self, config: Qwen3LCQATCudaConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size//num_heads)
        num_key_value_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size

        attention_bias = getattr(config, "attention_bias", False)
        for layer in self.model.layers:
            layer.self_attn.q_proj = LCQATCudaLinear(hidden_size, num_heads * head_dim)
            layer.self_attn.k_proj = LCQATCudaLinear(hidden_size, num_key_value_heads*head_dim)
            layer.self_attn.v_proj = LCQATCudaLinear(hidden_size, num_key_value_heads*head_dim)
            layer.self_attn.o_proj = LCQATCudaLinear(num_heads*head_dim, hidden_size)
            layer.mlp.gate_proj = LCQATCudaLinear(hidden_size, intermediate_size)
            layer.mlp.up_proj = LCQATCudaLinear(hidden_size, intermediate_size)
            layer.mlp.down_proj = LCQATCudaLinear(intermediate_size, hidden_size)                
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None) or Qwen3LCQATCudaConfig.from_pretrained(pretrained_model_name_or_path)
        config._attn_implementation = "sdpa"
        
        model = Qwen3LCQATCudaCudaForCausalLM(config)
        ckpt_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(pretrained_model_name_or_path, "model.compressed.safetensors")
        sd = load_file(ckpt_path, device="cpu")
        sd = remap_checkpoint_keys(sd, config=config)
        biases = load_and_assign(model, sd)

        model = model.to(kwargs.get("torch_dtype", torch.bfloat16)).eval()
        model.tie_weights()
        assign_biases(model, biases)
        for mod in model.modules():
            if isinstance(mod, LCQATCudaLinear):
                mod.prepare_for_inference()

        return model