#! /usr/bin/env bash

export NPROC_PER_NODE=8
export PYTHONPATH=$PWD:$PYTHONPATH

set -ue

exp_dir=YOUR_EXP_DIR
exp_name=Qwen3-8B-NPU
init_ckpt_path=$exp_dir/$exp_name/init_ckpt

swift pt configs/8B.yaml \
    --model $init_ckpt_path \
    --model_type qwen3_lcqat \
    --custom_register_path model/register.py \
    --output_dir $exp_dir/$exp_name \
    --check_model False \
