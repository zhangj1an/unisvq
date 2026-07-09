
import time
import os
import sys
import logging
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from safetensors.torch import load_file
from datasets import load_dataset

torch.set_grad_enabled(False)    

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

model_path = "YOUR_MODEL_PATH"

def _sample(logits, temperature=1.0, top_k=None):
    logits = logits[:, -1] / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < v[:, -1:], -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(torch.int)

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

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
jinja_path = os.path.join(model_path, "chat_template.jinja")
if not tokenizer.chat_template and os.path.exists(jinja_path):
    with open(jinja_path) as f:
        tokenizer.chat_template = f.read()

def fmt(text):
    return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], 
            tokenize=False, 
            add_generation_prompt=True
        )

compile_mode = "reduce-overhead"
trim_num = 3
batch_size = 1
max_new_tokens = 128
device = torch.device("cuda:0")

if compile_mode != "none":
    torch._dynamo.config.cache_size_limit = 64
    logger.info(f"  torch.compile(mode='{compile_mode}') ...")
    t0 = time.time()
    model = torch.compile(model, mode=compile_mode, fullgraph=False)
    logger.info(f"  Done ({time.time()-t0:.1f}s, JIT on first call)")

dataset = load_dataset("json", data_files={'test': "YOUR_DATASET_PATH/moss-en-quentions.jsonl"})['test']

questions = [dataset[i]["question"] for i in range(trim_num*batch_size)]

all_texts = [fmt(q) for q in questions]
warmup_text = fmt("hello")
max_prompt_len = max(
    len(tokenizer.encode(warmup_text)),
    max(len(tokenizer.encode(t)) for t in all_texts)
)
pad_to = max_prompt_len + 4

max_cache_len = pad_to + max_new_tokens + 16
past_kv = StaticCache(config=config, max_batch_size=batch_size,
                        max_cache_len=max_cache_len, device=device,
                        dtype=torch.bfloat16)

logger.info(f"  pad_to={pad_to}, max_cache_len={max_cache_len}")
logger.info("  Warmup ...")
t0 = time.time()
for _ in range(3):
    generate(model, tokenizer, [warmup_text] * batch_size,
                max_new_tokens, config, top_k=3, temperature=0.6,
                device=device, past_kv=past_kv, pad_to=pad_to)
torch.cuda.synchronize(device)
logger.info(f"  Warmup done ({time.time()-t0:.1f}s)")
torch.cuda.reset_peak_memory_stats(device)

total_tok = total_t = 0
for i in range(trim_num):
    torch.cuda.synchronize(device)
    t0 = time.time()
    prompt = [fmt(sample_question) for sample_question in questions[i*batch_size:(i+1)*batch_size]]
    texts, gen_len = generate(
        model, tokenizer, prompt,
        max_new_tokens, config, top_k=3, temperature=0.6,
        device=device, past_kv=past_kv, pad_to=pad_to)
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    n = batch_size * gen_len
    total_tok += n
    total_t += dt
    logger.info(f"  [{i+1}/{len(questions)}] {n} tok / {dt:.2f}s | {total_tok/total_t:.0f} tok/s")

mem = torch.cuda.max_memory_allocated(device) / 1024**3
logger.info(f"\n  RESULT: {total_tok/total_t:.0f} tok/s | {mem:.2f} GB peak")
logger.info(f"  Config: batch={batch_size}, path={model_path}, compile={compile_mode}")
logger.info(f"{'='*60}")

