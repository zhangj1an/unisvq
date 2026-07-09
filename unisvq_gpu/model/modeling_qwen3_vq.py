import os
import math
import tqdm
import torch
from safetensors import safe_open
from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from model.lc_qat_linear import LCQATHadLinear
from model.compress_utils import pack_2bit, unpack_2bit

class Qwen3LCQATConfig(Qwen3Config):
    model_type = "qwen3_lcqat"
    def __init__(self, *args, vec_size=4, vec_bit=2, proj_type="bypass",  mixed_percision=False, **kwargs):
        self.vec_size = vec_size
        self.vec_bit = vec_bit
        self.proj_type = proj_type
        self.mixed_percision = mixed_percision
        super().__init__(*args, **kwargs)
    

class Qwen3LCQATForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3LCQATConfig
    def __init__(self, config: Qwen3LCQATConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size//num_heads)
        num_key_value_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size
        vec_size = config.vec_size
        vec_bit = config.vec_bit
        proj_type = config.proj_type
        code_book_args = {
            "vec_size": vec_size, 
            "vec_bit": vec_bit, 
            "proj_type": proj_type, 
            "mixed_percision": config.mixed_percision
        }
        attention_bias = getattr(config, "attention_bias", False)
        for layer in self.model.layers:
            layer.self_attn.q_proj = LCQATHadLinear(hidden_size, num_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.k_proj = LCQATHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.v_proj = LCQATHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.o_proj = LCQATHadLinear(num_heads * head_dim, hidden_size, bias=False, **code_book_args)
            layer.mlp.gate_proj = LCQATHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.up_proj = LCQATHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.down_proj = LCQATHadLinear(intermediate_size, hidden_size, bias=False, **code_book_args)                
        self.config = config


class Qwen3LCQATCompressionConfig(Qwen3Config):
    model_type = "qwen3_lcqat_compression"
    def __init__(self, *args, vec_size=4, vec_bit=2, proj_type="bypass",  mixed_percision=False, **kwargs):
        self.vec_size = vec_size
        self.vec_bit = vec_bit
        self.proj_type = proj_type
        self.mixed_percision = mixed_percision
        super().__init__(*args, **kwargs)


class Qwen3LCQATForCompression(Qwen3ForCausalLM):
    config_class = Qwen3LCQATCompressionConfig
    def __init__(self, config: Qwen3LCQATCompressionConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size//num_heads)
        num_key_value_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size
        vec_size = config.vec_size
        vec_bit = config.vec_bit
        proj_type = config.proj_type
        code_book_args = {
            "vec_size": vec_size, 
            "vec_bit": vec_bit, 
            "proj_type": proj_type, 
            "mixed_percision": config.mixed_percision
        }
        attention_bias = getattr(config, "attention_bias", False)
        for layer in self.model.layers:
            layer.self_attn.q_proj = LCQATHadLinear(hidden_size, num_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.k_proj = LCQATHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.v_proj = LCQATHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.o_proj = LCQATHadLinear(num_heads * head_dim, hidden_size, bias=False, **code_book_args)
            layer.mlp.gate_proj = LCQATHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.up_proj = LCQATHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.down_proj = LCQATHadLinear(intermediate_size, hidden_size, bias=False, **code_book_args)                
        self.config = config

    def save_pretrained(self, save_directory, **kwargs):
        common_max_var = 2**self.config.vec_bit-1
        state_dict = self.state_dict()
        compressed_state_dict = {}
        original_shapes = {}
        pbar = tqdm.tqdm(state_dict.items())
        for k, v in pbar:
            pbar.set_description_str(f"[saving compressed] processing {k}")
            if "_proj.weight" in k and v.dim() >= 2: 
                original_shapes[k] = list(v.shape)
                out_features, in_features = original_shapes[k]
                max_var = state_dict.get(k.replace(".weight", ".code_generator.max_var"), common_max_var)
                weight_scale = math.sqrt(out_features+in_features)/2
                ori_weight = weight_scale*v + max_var/2
                quant_weight = torch.round(ori_weight).clamp(0, max_var)
                compressed_state_dict[k] = pack_2bit(quant_weight)
            else:
                compressed_state_dict[k] = v
        self.config.update({"original_shapes": original_shapes, "is_packed": True})
        super().save_pretrained(save_directory, state_dict=compressed_state_dict, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None) or Qwen3LCQATCompressionConfig.from_pretrained(pretrained_model_name_or_path)
        common_max_var = 2**config.vec_bit-1   
        is_packed = getattr(config, "is_packed", False)
        if is_packed:
            packed_state_dict = {}
            if os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors")):
                with safe_open(os.path.join(pretrained_model_name_or_path, "model.safetensors"), framework="pt") as f:
                    for key in f.keys():
                        packed_state_dict[key] = f.get_tensor(key)
            elif os.path.exists(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")):
                archive_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
                packed_state_dict = torch.load(archive_file, map_location="cpu")
            else:
                raise FileNotFoundError(f"no ckpt file in {pretrained_model_name_or_path}")
            
            original_shapes = config.original_shapes
            unpacked_state_dict = {}
            pbar = tqdm.tqdm(packed_state_dict.items())
            for k, v in pbar:
                pbar.set_description_str(f"[loading compressed] processing {k}")
                if "_proj.weight" in k:
                    quant_weight = unpack_2bit(v, original_shapes[k])
                    out_features, in_features = original_shapes[k]
                    max_var = packed_state_dict.get(k.replace(".weight", ".code_generator.max_var"), common_max_var)
                    weight_scale = math.sqrt(out_features+in_features)/2
                    ori_weight = (quant_weight - max_var/2)/weight_scale
                    unpacked_state_dict[k] = ori_weight             
                else:
                    unpacked_state_dict[k] = v

            kwargs["state_dict"] = unpacked_state_dict
            
            return super(Qwen3LCQATForCompression, cls).from_pretrained(
                pretrained_model_name_or_path=None, 
                config=config,
                *model_args, 
                **kwargs)
        else:
            return super(Qwen3LCQATForCompression, cls).from_pretrained(
                pretrained_model_name_or_path, 
                config=config,
                *model_args, 
                **kwargs                
            )