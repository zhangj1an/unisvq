#! /usr/bin/env bash

base_model_path=YOUR_BASE_MODEL_PATH
exp_dir=YOUR_EXP_DIR
exp_suffix=""
hess_suffix=""

python scripts/create_qwen3_ckpt_from_quip.py \
    --ori_model_path ${base_model_path} \
    --init_ckpt_path ${exp_dir}/hf${exp_suffix} \
    --output_path ${exp_dir}/init_ckpt