import os
import random
import argparse

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--devset_size', default=256, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--save_path', default='hessians/llama2_70b', type=str)
parser.add_argument('--scratch_path', default=None, type=str)
parser.add_argument('--chunk_size', default=256, type=int)
parser.add_argument('--async_copy_speed', default=-1, type=int)
parser.add_argument('--act_save_rate', default=4, type=int)
parser.add_argument('--save_activations', action='store_true')
parser.add_argument('--sample_proc', default=4, type=int)
parser.add_argument('--dataset_path', default='/data', type=str)


def forward_layer(layer, position_ids, position_embeddings, attention_mask, bs, device, dev_emb):

    assert len(dev_emb) % bs == 0
    for i in range(len(dev_emb) // bs):
        if position_embeddings is not None:
            dev_emb[i * bs:(i + 1) * bs] = layer(
                dev_emb[i * bs:(i + 1) * bs].to(device),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False)[0].cpu()
        else:
            dev_emb[i * bs:(i + 1) * bs] = layer(
                dev_emb[i * bs:(i + 1) * bs].to(device),
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False)[0].cpu()


def save_result(activation_results, args, transformer_layer_index):
    for linear_name, linear_result in activation_results.items():
        mu = linear_result["mu"]
        H = linear_result["H"]
        ct = linear_result["ct"]
        mu.div_(ct)
        H.div_(ct)
        H.addmm_(-mu.unsqueeze(-1), mu.unsqueeze(0))
        save_path = f"{args.save_path}/{transformer_layer_index}_{linear_name}.pt"
        torch.save(
            {
                'flatH': utils.sym_to_flat(H),
                'mu': mu.to(torch.float32),
                'n': H.shape[0],
                'ct': ct
            }, save_path
        )


def main(args):
    print("loading model...")
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    print("loaded model!")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    if args.dataset_path == "pajama":
        devset = utils.sample_rp1t_concat(tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    elif ".jsonl" in args.dataset_path:
        devset = utils.sample_jsonl_concat(args.dataset_path, tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    else:
        not NotImplementedError(args.dataset_path)
    dev_emb = model.model.embed_tokens(devset)
    if hasattr(config, "scale_emb"):
        dev_emb = dev_emb * config.scale_emb
    after_layer = -1
    print("loaded dataset!")

    print(f"dev_emb dtype: {dev_emb.dtype}")
    dev_emb.share_memory_()

    device = torch.device("cuda")

    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
    if hasattr(model.model, "rotary_emb"):
        position_embeddings_cos, position_embeddings_sin = model.model.rotary_emb(dev_emb[0], position_ids)
    else:
        position_embeddings_cos, position_embeddings_sin = None, None
    if hasattr(model.config, 'sliding_window'):
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size],
            0,
            sliding_window=model.config.sliding_window)
    else:
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size], 0)

    pbar = tqdm(range(len(model.model.layers)))

    for transformer_layer_index in pbar:
        if (transformer_layer_index <= after_layer):
            pbar.write(f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}")
            continue

        transformer_layer = model.model.layers[transformer_layer_index]
        assert (len([m for m in transformer_layer.modules()if isinstance(m, torch.nn.Linear)]) in [4, 5, 7])

        chunk_size = min(args.chunk_size, len(dev_emb))
        assert len(dev_emb) % args.batch_size == 0 and chunk_size % args.batch_size == 0

        transformer_layer = transformer_layer.to(device)
        position_ids = position_ids.to(device)
        attention_mask = attention_mask.to(device)
        if position_embeddings_sin is not None:
            position_embeddings = (position_embeddings_cos.cuda(), position_embeddings_sin.cuda())
        else:
            position_embeddings = None
        layer_activations = {}
        def hook_generator(layer_name, device):
            def get_hessian_hook(module, x):
                n = module.in_features
                layer_hessian = layer_activations.get(layer_name, {})
                H = layer_hessian.get("H", torch.zeros(n, n, dtype=torch.float64, device=device))
                mu = layer_hessian.get("mu", torch.zeros(n, dtype=torch.float64, device=device))
                ct = layer_hessian.get("ct", 0)
                x = x[0].reshape(-1, n).to(torch.float64)
                H.addmm_(x.T, x)
                mu.add_(x.sum(dim=0))
                ct += len(x)
                layer_activations[layer_name] = {
                    "H": H,
                    "mu": mu,
                    "ct": ct
                }
            return get_hessian_hook

        if hasattr(transformer_layer.self_attn, "q_proj"):
            hook_qkv = transformer_layer.self_attn.q_proj.register_forward_pre_hook(hook_generator("qkv", device))
        else:
            hook_qkv = transformer_layer.self_attn.qkv_proj.register_forward_pre_hook(hook_generator("qkv", device))
        hook_o = transformer_layer.self_attn.o_proj.register_forward_pre_hook(hook_generator("o", device))
        if hasattr(transformer_layer.mlp, "up_proj"):
            hook_upgate = transformer_layer.mlp.up_proj.register_forward_pre_hook(hook_generator("upgate", device))
        else:
            hook_upgate = transformer_layer.mlp.gate_up_proj.register_forward_pre_hook(hook_generator("upgate", device))
        hook_down = transformer_layer.mlp.down_proj.register_forward_pre_hook(hook_generator("down", device))

        for i in range(0, len(dev_emb), chunk_size):
            pbar.write(f"block {transformer_layer_index}, batch {i}/{len(dev_emb)}")
            batch_end = min(i + chunk_size, len(dev_emb))
            batch_emb = dev_emb[i:batch_end]
            forward_layer(
                transformer_layer,
                position_ids,
                position_embeddings,
                attention_mask,
                args.batch_size,
                device=torch.device("cuda"),
                dev_emb=batch_emb
            )
        
        save_result(layer_activations, args=args, transformer_layer_index=transformer_layer_index)
        hook_qkv.remove()
        hook_o.remove()
        hook_upgate.remove()
        hook_down.remove()
        del layer_activations
        transformer_layer.cpu()
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        # pbar.write(f"done processing layer {transformer_layer_index}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
