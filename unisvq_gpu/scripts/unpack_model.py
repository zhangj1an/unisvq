import os
import sys
path = __file__
for i in range(2):
    path = os.path.dirname(path)
sys.path.append(path)
import tqdm
import json
import shutil
import argparse

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model.lc_qat_linear import LCQATHadLinear, matmul_hadU_cuda_blockwise
from accelerate import init_empty_weights


def unpack_vq_blockwise_linear(linear: LCQATHadLinear):
    in_features = linear.in_features
    out_features = linear.out_features
    device = linear.weight.device
    dtype = linear.weight.dtype
    new_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=(linear.bias is not None))
    new_linear.to(device=device, dtype=dtype)

    W_decompressed = linear.code_generator.quant_weight(linear.weight)

    W_decompressed = matmul_hadU_cuda_blockwise(W_decompressed)
    W_decompressed = matmul_hadU_cuda_blockwise(W_decompressed.T.contiguous()).T
    W_decompressed = linear.SV.unsqueeze(1) * W_decompressed * linear.SU.unsqueeze(0)

    new_linear.weight.data = W_decompressed.to(device=device, dtype=new_linear.weight.dtype)
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.to(device=device, dtype=new_linear.bias.dtype)
    return new_linear


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


def del_op_by_name(layer, name):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        delattr(mod_, levels[-1]) 
    else:
        delattr(layer, name)


def split_layer(linear: nn.Linear, name: str, config):
    if "o_proj" in name or "down_proj" in name:
        return [(name, linear)], False
    else:
        if name == "self_attn.qkv_proj":
            num_query_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = getattr(config, "head_dim", config.hidden_size//num_query_heads)
            hidden_dim = config.hidden_size
            attention_bias = linear.bias is not None
            num_query_heads_per_group = num_query_heads // num_kv_heads
            q_proj = nn.Linear(
                in_features=hidden_dim, 
                out_features=head_dim*num_query_heads, 
                bias=attention_bias
            )
            k_proj = nn.Linear(
                in_features=hidden_dim, 
                out_features=head_dim*num_kv_heads, 
                bias=attention_bias
            )
            v_proj = nn.Linear(
                in_features=hidden_dim, 
                out_features=head_dim*num_kv_heads, 
                bias=attention_bias
            )
            q_weights = []
            q_biases = []
            k_weights = []
            k_biases = []
            v_weights = []
            v_biases = []

            qkv_proj_split = torch.split(linear.weight, split_size_or_sections=head_dim, dim=0)
            if attention_bias:
                qkv_bias_split = torch.split(linear.bias, split_size_or_sections=head_dim, dim=0)
            for i in range(num_kv_heads):
                start = (num_query_heads_per_group + 2) * i
                q_weights.extend(qkv_proj_split[start : start+num_query_heads_per_group])
                k_weights.append(qkv_proj_split[start+num_query_heads_per_group])
                v_weights.append(qkv_proj_split[start+num_query_heads_per_group+1])
                if attention_bias:
                    q_biases.extend(qkv_bias_split[start : start+num_query_heads_per_group])
                    k_biases.append(qkv_bias_split[start+num_query_heads_per_group])
                    v_biases.append(qkv_bias_split[start+num_query_heads_per_group+1])
                # print(i, start, start + num_query_heads_per_group, attention_bias)
            q_proj.weight.data = torch.cat(q_weights, dim=0).clone()
            k_proj.weight.data = torch.cat(k_weights, dim=0).clone()
            v_proj.weight.data = torch.cat(v_weights, dim=0).clone() 
            if attention_bias:
                q_proj.bias.data = torch.cat(q_biases, dim=0).clone()
                k_proj.bias.data = torch.cat(k_biases, dim=0).clone()
                v_proj.bias.data = torch.cat(v_biases, dim=0).clone()
            return [
                ("self_attn.q_proj", q_proj),
                ("self_attn.k_proj", k_proj),
                ("self_attn.v_proj", v_proj)
            ], True
        elif "gate_up_proj" in name:
            hidden_dim = config.hidden_size
            intermediate_size = config.intermediate_size
            gate_proj = nn.Linear(
                in_features=hidden_dim, 
                out_features=intermediate_size, 
                bias=False
            )
            up_proj = nn.Linear(
                in_features=hidden_dim, 
                out_features=intermediate_size, 
                bias=False
            )
            gate_up_weight = torch.split(linear.weight, dim=0, split_size_or_sections=intermediate_size)
            # print(gate_up_weight[0].shape, gate_up_weight[1].shape)
            gate_proj.weight.data = gate_up_weight[0].clone()
            up_proj.weight.data = gate_up_weight[1].clone()
            return [
                ("mlp.gate_proj", gate_proj),
                ("mlp.up_proj", up_proj),
            ], True


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--ref_model_path", type=str, default=None)
parser.add_argument("--overwrite_config_path", type=str, default=None)

args = parser.parse_args()
if args.overwrite_config_path is not None:
    assert os.path.exists(args.overwrite_config_path) and os.path.isfile(args.overwrite_config_path)
model_path = args.model_path
output_path = args.output_path

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32).to(device="cuda")
pbar = tqdm.tqdm(total=len(model.model.layers))

if args.ref_model_path is None:
    args.ref_model_path = os.path.dirname(args.overwrite_config_path)

hf_ref_config = AutoConfig.from_pretrained(args.ref_model_path, trust_remote_code=True)

for i, layer in enumerate(model.model.layers):
    linears = find_layers(layer, layers=[LCQATHadLinear])
    pbar.set_description_str(f"unpacking layer {i}")
    for name, q_linear in linears.items():
        new_linear = unpack_vq_blockwise_linear(q_linear)
        set_op_by_name(layer, name, new_linear)
    pbar.update(1)


model.to(dtype=torch.bfloat16)
with init_empty_weights():
    over_write_model = AutoModelForCausalLM.from_config(hf_ref_config, trust_remote_code=True)
over_write_model.to_empty(device="cuda:0")
over_write_model.load_state_dict(model.state_dict())
over_write_model.save_pretrained(output_path)

tokenizer.save_pretrained(output_path)

if args.overwrite_config_path is not None:
    custom_files = [args.overwrite_config_path]
    config_dir = os.path.dirname(args.overwrite_config_path)
else:
    custom_files = []
    config_dir = args.model_path
for config_name in ["config.json", "tokenizer_config.json"]:

    config_path = os.path.join(config_dir, config_name)
    print(config_path)
    if os.path.exists(config_path):
        with open(config_path) as f:
            automap = json.load(f).get('auto_map', {})
        for module_name in automap.values():
            if type(module_name) == list:
                module_name = module_name[0]
            module_name = module_name.split(".")[0]
            if not module_name in custom_files:
                custom_files.append(module_name + ".py")
                print(module_name)
for f in custom_files:
    shutil.copy(os.path.join(config_dir, f), output_path)