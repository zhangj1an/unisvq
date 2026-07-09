import torch
from model import *
from model.general_model import load_quantized_model
from transformers import AutoTokenizer, AutoModelForCausalLM

model_type = "quip"
model_path = "/data0/wanghaoyu/exp/lc_qat/Qwen3-1.7B/hf"
if model_type == "quip":
    model = load_quantized_model(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

while True:
    prompt = input("input prompt (enter quit to exit): ")
    if prompt.strip() == "quit":
        break

    # prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        attention_mask=inputs['attention_mask'].cuda(),
        max_new_tokens=64,
        return_dict_in_generate=True,
    )
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    print(output_str)
    real_output = output_str[len(prompt):].replace("\n", "\\n")
