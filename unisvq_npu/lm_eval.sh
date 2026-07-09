#! /usr/bin/env bash

set -ue

exp_dir="YOUR_EXP_DIR"
origin_model_path="YOUR_ORIGIN_MODEL_PATH"

ckpt="checkpoint-$iter"
model_path="${exp_dir}/$ckpt"
unpack_path="${exp_dir}/unpacked_tmp"
log_path="logs/Qwen3-1.7B_pretrain/$ckpt"

python scripts/unpack_model.py \
    --model_path $model_path \
    --output_path $unpack_path \
    --overwrite_config_path $origin_model_path/config.json

lm_eval \
    --model hf \
    --model_args pretrained=${unpack_path},dtype=bfloat16,trust_remote_code=True \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,piqa,winogrande \
    --batch_size 16 \
    --output_path $log_path/result.json
