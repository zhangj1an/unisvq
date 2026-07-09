"""
HadaQuant quantized inference benchmark

Usage:
  # Correctness + throughput
  python quantized_infer.py

  # Throughput only, different compile modes
  python quantized_infer.py --skip_phase1 --compile_mode reduce-overhead
  python quantized_infer.py --skip_phase1 --compile_mode default
  python quantized_infer.py --skip_phase1 --compile_mode none

  # Skip dequant weight caching (saves ~13GB VRAM, GEMM dequants on-the-fly)
  python quantized_infer.py --skip_phase1 --no_weight_cache

  # Force full GEMV (low VRAM, no weight caching)
  TVQ_GEMV_THRESHOLD=999 python quantized_infer.py --skip_phase1
"""
import time, json, argparse, os, sys

import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from safetensors.torch import load_file
from datasets import load_dataset

torch.set_grad_enabled(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model.hadaquant_infer_model as _mtv
# from model.hadaquant_infer_model import (
#     Qwen3ForCausalLM_LCQAT, LCQATLinear,
#     remap_checkpoint_keys, load_and_assign, assign_biases,
# )
from model.hadaquant_infer_model import Qwen3LCQATCudaCudaForCausalLM, LCQATCudaLinear

def log(msg=""):
    print(msg, flush=True)

def _sample(logits, temperature=1.0, top_k=None):
    logits = logits[:, -1] / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < v[:, -1:], -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(torch.int)


# def load_model(model_path, device="cuda:0"):
#     t0 = time.time()
#     config = AutoConfig.from_pretrained(model_path)
#     attn_impl = "sdpa"
#     # try:
#     #     import flash_attn
#     #     from packaging.version import Version
#     #     import transformers
#     #     if Version(transformers.__version__) >= Version("4.48.0"):
#     #         attn_impl = "flash_attention_2"
#     # except ImportError:
#     #     pass
#     config._attn_implementation = attn_impl
#     model = Qwen3ForCausalLM_LCQAT(config)

#     ckpt_path = os.path.join(model_path, "model.safetensors")
#     if not os.path.exists(ckpt_path):
#         ckpt_path = os.path.join(model_path, "model.compressed.safetensors")
#     sd = load_file(ckpt_path, device="cpu")
#     sd = remap_checkpoint_keys(sd, config=config)
#     biases = load_and_assign(model, sd)

#     model = model.to(device).to(torch.bfloat16).eval()
#     model.tie_weights()
#     assign_biases(model, biases)
#     for mod in model.modules():
#         if isinstance(mod, LCQATLinear):
#             mod.prepare_for_inference()

#     mem = torch.cuda.max_memory_allocated(device) / 1024**3
#     log(f"[load] {time.time()-t0:.1f}s, attn={attn_impl}, mem={mem:.2f}GB")
#     return model, config


def warm_weights(model):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    for mod in m.modules():
        if isinstance(mod, LCQATCudaLinear):
            mod.warm_weight_cache()

def clear_weights(model):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    for mod in m.modules():
        if isinstance(mod, LCQATCudaLinear):
            mod.clear_weight_cache()


@torch.no_grad()
def decode_one_token(model, cur_token, past_kv, cache_position):
    logits = model(cur_token, past_key_values=past_kv,
                   cache_position=cache_position, use_cache=True).logits
    return logits


@torch.no_grad()
def generate(model, tokenizer, texts, max_new_tokens, config,
             top_k=None, temperature=1.0, device="cuda:0",
             past_kv=None, pad_to=None):
    pad_kwargs = {"padding": True}
    if pad_to is not None:
        pad_kwargs = {"padding": "max_length", "max_length": pad_to, "truncation": True}
    inputs = tokenizer(texts, return_tensors="pt", **pad_kwargs).to(device)
    batch_size, seq_length = inputs["input_ids"].shape

    if past_kv is None:
        max_cache_len = seq_length + max_new_tokens + 16
        past_kv = StaticCache(config=config, max_batch_size=batch_size,
                              max_cache_len=max_cache_len, device=device,
                              dtype=torch.bfloat16)
    past_kv.reset()

    generated_ids = torch.zeros(batch_size, seq_length + max_new_tokens,
                                dtype=torch.int, device=device)
    generated_ids[:, :seq_length] = inputs["input_ids"].int()

    cache_position = torch.arange(seq_length, device=device)
    logits = model(**inputs, past_key_values=past_kv,
                   cache_position=cache_position, use_cache=True).logits
    next_token = _sample(logits, temperature, top_k)
    generated_ids[:, seq_length] = next_token.squeeze(-1)

    cache_position = torch.tensor([seq_length], device=device)
    for step in range(1, max_new_tokens):
        logits = decode_one_token(model, next_token, past_kv, cache_position)
        next_token = _sample(logits, temperature, top_k)
        generated_ids[:, seq_length + step] = next_token.squeeze(-1)
        cache_position += 1

    torch.cuda.synchronize(device)
    gen_tokens = generated_ids[:, seq_length:]
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True), max_new_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="YOUR_MODEL_PATH")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--trim_num", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile_mode", default="reduce-overhead",
                        choices=["none", "default", "reduce-overhead"])
    parser.add_argument("--skip_phase1", action="store_true")
    parser.add_argument("--no_weight_cache", action="store_true",
                        help="Skip dequant weight caching (saves ~13GB VRAM, GEMM dequants on-the-fly)")
    args = parser.parse_args()

    device = args.device
    threshold = _mtv._GEMV_THRESHOLD
    use_gemm = args.batch_size > threshold

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.compiler.allow_in_graph(flash_attn_2_cuda.PyCapsule.varlen_fwd)
    config = AutoConfig.from_pretrained(args.model_path)
    model = Qwen3LCQATCudaCudaForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    jinja_path = os.path.join(args.model_path, "chat_template.jinja")
    if not tokenizer.chat_template and os.path.exists(jinja_path):
        with open(jinja_path) as f:
            tokenizer.chat_template = f.read()

    def fmt(text):
        return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], 
                tokenize=False, 
                add_generation_prompt=True
            )

    # ── Phase 1: Correctness (eager) ──
    if not args.skip_phase1:
        log("\n" + "=" * 60)
        log("Phase 1: Correctness verification (eager, batch=1)")
        log("=" * 60)
        for p in ['介绍一下泰山。', '写一首关于冬天的诗。', '介绍下大模型预训练技术。']:
            text, _ = generate(model, tokenizer, [fmt(p)], args.max_new_tokens, config, device=device)
            log("  {}\n  → {}... (truncated)\n".format(
                fmt(p).replace('\n', '\\n'), text[0][:200]
            ))
        torch.cuda.empty_cache()

    # exit(0)

    # ── Phase 2: Throughput ──
    cache_weights = use_gemm and not args.no_weight_cache
    if use_gemm:
        path = "GEMM(cached)" if cache_weights else "GEMM(on-the-fly)"
    else:
        path = "GEMV(fused)"
    log(f"\n{'='*60}")
    log(f"Phase 2: batch={args.batch_size}, tokens={args.max_new_tokens}")
    log(f"  path={path}, threshold={threshold}, compile={args.compile_mode}")
    log(f"{'='*60}")

    if cache_weights:
        log("  Caching dequant weights ...")
        t0 = time.time()
        warm_weights(model)
        log(f"  Done ({time.time()-t0:.1f}s)")

    if args.compile_mode != "none":
        torch._dynamo.config.cache_size_limit = 64
        log(f"  torch.compile(mode='{args.compile_mode}') ...")
        t0 = time.time()
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
        log(f"  Done ({time.time()-t0:.1f}s, JIT on first call)")

    dataset = load_dataset("json", data_files={'test': "YOUR_DATASET_PATH/moss-en-quentions.jsonl"})['test']
    
    questions = [dataset[i]["question"] for i in range(args.trim_num*args.batch_size)]

    all_texts = [fmt(q) for q in questions]
    warmup_text = fmt("hello")
    max_prompt_len = max(
        len(tokenizer.encode(warmup_text)),
        max(len(tokenizer.encode(t)) for t in all_texts)
    )
    pad_to = max_prompt_len + 4

    max_cache_len = pad_to + args.max_new_tokens + 16
    past_kv = StaticCache(config=config, max_batch_size=args.batch_size,
                          max_cache_len=max_cache_len, device=device,
                          dtype=torch.bfloat16)

    log(f"  pad_to={pad_to}, max_cache_len={max_cache_len}")
    log("  Warmup ...")
    t0 = time.time()
    for _ in range(3):
        generate(model, tokenizer, [warmup_text] * args.batch_size,
                 args.max_new_tokens, config, top_k=3, temperature=0.6,
                 device=device, past_kv=past_kv, pad_to=pad_to)
    torch.cuda.synchronize(device)
    log(f"  Warmup done ({time.time()-t0:.1f}s)")
    torch.cuda.reset_peak_memory_stats(device)

    total_tok = total_t = 0
    for i in range(args.trim_num):
        torch.cuda.synchronize(device)
        t0 = time.time()
        prompt = [fmt(sample_question) for sample_question in questions[i*args.batch_size:(i+1)*args.batch_size]]
        texts, gen_len = generate(
            model, tokenizer, prompt,
            args.max_new_tokens, config, top_k=3, temperature=0.6,
            device=device, past_kv=past_kv, pad_to=pad_to)
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        n = args.batch_size * gen_len
        total_tok += n
        total_t += dt
        log(f"  [{i+1}/{len(questions)}] {n} tok / {dt:.2f}s | {total_tok/total_t:.0f} tok/s")

    mem = torch.cuda.max_memory_allocated(device) / 1024**3
    log(f"\n  RESULT: {total_tok/total_t:.0f} tok/s | {mem:.2f} GB peak")
    log(f"  Config: batch={args.batch_size}, path={path}, compile={args.compile_mode}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
