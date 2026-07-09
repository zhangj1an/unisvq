import os
import sys
import math
import json
import shutil
import argparse
import warnings
from tqdm import tqdm
path = __file__
for i in range(2):
    path = os.path.dirname(path)
sys.path.append(path)
warnings.simplefilter('once', UserWarning)

import torch
import torch.nn.functional as F
from safetensors import safe_open, SafetensorError
from transformers import AutoTokenizer
from model.modeling_qwen3_vq import Qwen3LCQATConfig, Qwen3LCQATForCausalLM, LCQATHadLinear
from lib.utils.matmul_had import matmul_hadU_cuda_blockwise

parser = argparse.ArgumentParser()
parser.add_argument("--ori_model_path", type=str)
parser.add_argument("--init_ckpt_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--attention_bias", action='store_true')
args = parser.parse_args()

ori_model_path = args.ori_model_path 
init_ckpt_path = args.init_ckpt_path 
output_path = args.output_path
attention_bias = args.attention_bias

def idx_to_quant_weight(idx: torch.Tensor, code_vec_size: int, code_possible_vars: int) -> torch.Tensor:
    quant_weight = idx.new_zeros(idx.shape[0], code_vec_size*idx.shape[1])
    for i in range(code_vec_size-1, -1, -1):
        quant_weight[:, i::code_vec_size] = idx % code_possible_vars
        idx = idx // code_possible_vars
    return quant_weight

def create_init_weight(
    fp16_weight: torch.Tensor,
    scale_su: torch.Tensor,
    scale_sv: torch.Tensor,
    proj_scale: torch.Tensor,
    proj_bias: torch.Tensor,
    vec_size: int=4
):
    out_features, in_features = fp16_weight.shape
    weight = 1./scale_sv.unsqueeze(1) * fp16_weight * 1./scale_su.unsqueeze(0)

    weight = matmul_hadU_cuda_blockwise(weight)
    weight = matmul_hadU_cuda_blockwise(weight.T.contiguous()).T
    weight = weight.reshape(-1, vec_size)
    weight = (weight-proj_bias)/proj_scale
    return weight.reshape(out_features, in_features)


def idx_to_quant_weight(idx: torch.Tensor, code_vec_size: int, code_possible_vars: int) -> torch.Tensor:
    quant_weight = idx.new_zeros(idx.shape[0], code_vec_size*idx.shape[1])
    for i in range(code_vec_size-1, -1, -1):
        quant_weight[:, i::code_vec_size] = idx % code_possible_vars
        idx = idx // code_possible_vars
    return quant_weight


def get_tensors_from_ckpt(quip_ckpt: safe_open, key: str, module: LCQATHadLinear, quip_cfgs: dict, attention_bias=False):

    if quip_cfgs["code_nbit"] <= 2:
        code_dtype = torch.uint8
    else:
        code_dtype = torch.int16

    max_var = quip_cfgs["max_var"]

    codes = quip_ckpt.get_tensor(key + ".Qidxs").to(device=device, dtype=code_dtype).to(torch.long)
    num_out_groups, num_in_groups = codes.shape
    out_features = num_out_groups
    in_features = num_in_groups * quip_cfgs["vec_size"]
    # Get scales
    su = quip_ckpt.get_tensor(key + ".SU").to(device=device, dtype=config.torch_dtype)
    sv = quip_ckpt.get_tensor(key + ".SV").to(device=device, dtype=config.torch_dtype)
    try:
        scale_wh = quip_ckpt.get_tensor(key + ".scaleWH")
    except SafetensorError as err:
        scale_wh = None
    if scale_wh is not None:
        sv = sv/scale_wh
    sv = quip_ckpt.get_tensor(key + ".Wscale") * sv
    su_norm = torch.norm(su, p=2)
    sv_norm = torch.norm(sv, p=2)
    scaling_norm = math.sqrt(su_norm*sv_norm + 1e-6)
    su = su * scaling_norm/su_norm
    sv = sv * scaling_norm/sv_norm
    module.SU.data = su.to(config.torch_dtype)
    module.SV.data = sv.to(config.torch_dtype)
    # bias
    if attention_bias:
        bias = quip_ckpt.get_tensor(key + ".bias")
        module.bias.data = bias.to(device=device, dtype=config.torch_dtype)

    # Generate weights, apply reference value and scaling
    weight = idx_to_quant_weight(
        codes, 
        code_vec_size=quip_cfgs["vec_size"], 
        code_possible_vars=max_var+1).to(dtype=config.torch_dtype)
    xavier_zero_point = (max_var)/2
    xavier_scale = math.sqrt(out_features+in_features)/2
    weight = (weight - xavier_zero_point)/xavier_scale

    if quip_cfgs["proj_type"] == "hadamard_linear":
        module.code_generator.g1.weight.data = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_weight").to(device=device, dtype=config.torch_dtype)
        module.code_generator.g1.bias.data = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_bias").to(device=device, dtype=config.torch_dtype)
    else:
        codebook_scale = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_weight")[0, 0].item()
        codebook_zero_point = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_bias")[0].item()
        module.code_generator.codebook_scale.data = torch.tensor(codebook_scale)
        module.code_generator.codebook_zero_point.data = torch.tensor(codebook_zero_point)
    
    module.weight.data = weight.to(config.torch_dtype)
    return module


def get_config_from_quip_param(quip_config: dict):
    code_vec_size = quip_config["codesz"]
    if "identical" in quip_config["codebook"]:
        proj_type = "hadamard_identical"
    else:
        proj_type = "hadamard_linear"

    if quip_config["idx_dtype"] == "torch.uint8":
        if "tri" in quip_config["codebook"]:
            code_nbit = 1.5
        else:
            code_nbit = 2
    elif quip_config["idx_dtype"] == "torch.int16":
        code_nbit = 3
    else:
        raise NotImplementedError("{} is not implemented".format(quip_config["idx_dtype"]))

    if abs(code_nbit-1.5) < 0.0001:
        max_var = 2
    else:
        max_var = 2**code_nbit-1

    print("codebook", code_vec_size, proj_type, code_nbit, max_var)

    return {
        "vec_size": code_vec_size,
        "proj_type": proj_type,
        "code_nbit": code_nbit,
        "max_var": max_var
    }


if __name__ == "__main__":

    with open(os.path.join(init_ckpt_path, "config.json")) as f:
        quip_params = json.load(f)["quip_params"]
    
    quant_cfg = get_config_from_quip_param(quip_params)

    config = Qwen3LCQATConfig.from_pretrained(
        init_ckpt_path,
        vec_size=quant_cfg["vec_size"],
        torch_dtype=torch.bfloat16,
        proj_type=quant_cfg["proj_type"],
        attention_bias=attention_bias,
        block_hadamard=True,
        mixed_percision=False
    )
    delattr(config, "quip_params")
    
    model = Qwen3LCQATForCausalLM.from_pretrained(ori_model_path, config=config, ignore_mismatched_sizes=True)
    device = torch.device("cuda:0")

    with safe_open(os.path.join(init_ckpt_path, "model.safetensors"), framework="pt") as quip_ckpt:
        # Generate regular codebook
        pbar = tqdm(total=len(model.model.layers))
        last_layer = 0
        for key, module in model.named_modules():
            module.to(device=device, dtype=config.torch_dtype)
            if not key.endswith("_proj"):  # TODO: collect non-proj params and copy after replacement
                continue
            pbar.set_description_str(f"processing module {key} {type(module)}")
            module = get_tensors_from_ckpt(
                quip_ckpt=quip_ckpt,
                key=key,
                module=module,
                quip_cfgs=quant_cfg,
                attention_bias=attention_bias
            )
            current_layer = int(key.split('.')[2])
            if current_layer > last_layer:
                pbar.update(1)
                last_layer = current_layer
        pbar.update(1)

    # Save
    pbar.set_description_str(f"saving to {output_path}")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(init_ckpt_path, trust_remote_code=True)
    model.to(torch.bfloat16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    custom_files = []
    for config_name in ["config.json", "tokenizer_config.json"]:
        config_path = os.path.join(ori_model_path, config_name)
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
        shutil.copy(
            os.path.join(ori_model_path, f),
            output_path)
