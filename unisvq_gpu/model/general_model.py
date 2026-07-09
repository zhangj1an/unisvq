import os
import json
from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from lib.linear.quantized_linear import QuantizedLinear
from lib.codebook import codebook_id as codebook_id_dict


def find_layers(module: nn.Module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def format_quantize_model(model, config, register_susv=False, mixed_percision_rule={}):
    
    codesz = config.quip_params['codesz']
    packsz = config.quip_params['packsz']
    codebook_name=config.quip_params["codebook"]
    codebook_bit = config.quip_params["codebook_bit"]
    codebook_id=codebook_id_dict[codebook_name][0]
    pack_out = config.quip_params.get("pack_out", False)
    idx_dtype = config.quip_params['idx_dtype']
    codebook_version = config.quip_params.get('codebook_version', 0)
    rank = config.quip_params['lora_rank']
    rescale_WH = config.quip_params['rescale_WH']
    resid_scale_override=config.quip_params.get('resid_scale_override', -1)
    train_mode=config.quip_params.get('train_mode', False)
    grad_ckpt=False

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    num_key_value_heads = config.num_key_value_heads
    intermediate_size = config.intermediate_size

    if hasattr(model.model.layers[0].self_attn, "q_proj"):
        shape_dict = {
            "self_attn.q_proj": (hidden_size, num_heads*head_dim, model.model.layers[0].self_attn.q_proj.bias is not None),
            "self_attn.k_proj":(hidden_size, num_key_value_heads*head_dim, model.model.layers[0].self_attn.k_proj.bias is not None),
            "self_attn.v_proj": (hidden_size, num_key_value_heads*head_dim, model.model.layers[0].self_attn.v_proj.bias is not None),
            "self_attn.o_proj": (num_heads*head_dim, hidden_size, model.model.layers[0].self_attn.o_proj.bias is not None),
            "mlp.gate_proj": (hidden_size, intermediate_size, False),
            "mlp.up_proj": (hidden_size, intermediate_size, False),
            "mlp.down_proj": (intermediate_size, hidden_size, False),
        }
    else:
        shape_dict = {
            "self_attn.qkv_proj": (hidden_size, num_heads*head_dim+2*num_key_value_heads*head_dim, model.model.layers[0].self_attn.qkv_proj.bias is not None),
            "self_attn.o_proj": (num_heads*head_dim, hidden_size, model.model.layers[0].self_attn.o_proj.bias is not None),
            "mlp.gate_up_proj": (hidden_size, 2*intermediate_size, False),
            "mlp.down_proj": (intermediate_size, hidden_size, False),
        }
   
    for idx, layer in enumerate(model.model.layers):
        linears = find_layers(layer)
        for name, linear in linears.items():
            codebook_bit_for_linear = mixed_percision_rule.get(f"layers.{idx}.{name}", codebook_bit)
            if codebook_name == "identical":
                codebook_idx_dtype = 'torch.uint8' if codebook_bit_for_linear == 2 else 'torch.int16'
            else:
                codebook_idx_dtype = idx_dtype

            in_feat, out_feat, bias = shape_dict[name]
            new_linear = QuantizedLinear(
                in_features=in_feat,
                out_features=out_feat,
                codesz=codesz,
                packsz=packsz,
                pack_out=pack_out,
                idx_dtype=codebook_idx_dtype,
                codebook_version=codebook_version,
                rank=rank,
                rescale_WH=rescale_WH,
                bias=bias,
                resid_scale_override=resid_scale_override,
                train_mode=train_mode,
                grad_ckpt=grad_ckpt,
                codebook_id=codebook_id,
                codebook_bit=codebook_bit_for_linear
            )
            if register_susv:
                new_linear.SU = nn.Parameter(new_linear.SU.to(linear.weight.dtype), requires_grad=True)
                new_linear.SV = nn.Parameter(new_linear.SV.to(linear.weight.dtype), requires_grad=True)

            set_op_by_name(layer, name, new_linear)
    
    return model


def load_quantized_model(
    save_folder: str,
    torch_dtype: Optional[Union[str, torch.dtype]] = torch.float16,
    trust_remote_code: bool = True,
    device_map: Optional[dict] = None,
    max_mem_ratio: Optional[float] = 0.7,
    num_gpu: Optional[int] = -1,
    attn_implementation: Optional[str] = "sdpa",
    register_susv: Optional[bool] = False,
    ) -> PreTrainedModel:
    # print(save_folder, trust_remote_code)
    config = AutoConfig.from_pretrained(save_folder, trust_remote_code=trust_remote_code)
    # print(config)
    config._attn_implementation =attn_implementation
    assert hasattr(config, 'quip_params')
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

    if os.path.exists(os.path.join(save_folder, "reallocated_bit.json")):
        rule_file = os.path.join(save_folder, "reallocated_bit.json")
        with open(rule_file) as f:
            mixed_percision_rule = json.load(f)
    else:
        rule_file = None
        mixed_percision_rule = {}

    format_quantize_model(model=model, config=config, register_susv=register_susv, mixed_percision_rule=mixed_percision_rule)
    if config.tie_word_embeddings:
        model.tie_weights()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    mmap = {local_rank: f"{torch.cuda.mem_get_info(local_rank)[1]*0.8/(1 << 30)}GiB"}
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[model.model.layers[0].__class__.__name__],
        max_memory=mmap
    )

    load_checkpoint_and_dispatch(model, checkpoint=save_folder, device_map=device_map, offload_state_dict=True, dtype=torch_dtype, strict=True)
    
    return model