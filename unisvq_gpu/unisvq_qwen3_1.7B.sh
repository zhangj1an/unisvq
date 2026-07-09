#! /usr/bin/env bash

export PYTHONPATH="$PWD:$PYTHONPATH"

set -ue

base_model_path=YOUR_BASE_MODEL_PATH
exp_dir=YOUR_EXP_DIR
exp_name="Qwen3-1.7B"
exp_suffix=""
hess_suffix=""
dataset_path="pajama.jsonl"
stage=1

ckpt_path=${exp_dir}/ckpt${exp_suffix}
hf_path=${exp_dir}/hf${exp_suffix}
log_path=logs/${exp_name}${exp_suffix}
hess_path=${exp_dir}/hess${hess_suffix}

mkdir -p $log_path

if [[ $stage -eq 1 ]]; then
    python quantize/hessian_general_model.py \
        --batch_size 64 \
        --devset_size 1024 \
        --ctx_size 4096 \
        --base_model $base_model_path \
        --save_path $hess_path \
        --dataset_path $dataset_path
fi


if [[ $stage -le 2 ]]; then
    mkdir -p $ckpt_path
    if ! [[ $(ls $ckpt_path | wc -w ) == "0" ]]; then
        echo "$ckpt_path already exists:"
        ls -l $ckpt_path
        read -p "Delete the content? [y/n]" confirm
        if [ "$confirm" = "y" ]; then
            rm $ckpt_path/*
            echo "'$ckpt_path' is cleaned."
        else
            echo "canceled"
        fi
    fi

    python quantize/quantize_finetune.py \
        --save_path $ckpt_path \
        --codebook identical \
        --batch_size 32 \
        --ft_bs 4 \
        --scale_override 0.83 \
        --base_model $base_model_path \
        --hessian_path $hess_path \
        --dataset_path $dataset_path \
        --devset_size 1024 \
        --ft_valid_size 128 \
        --scale_search_iters 3 \
        --blockwise_hadamard \
        --codebook_bit 2 \
        --ft_epoch 1 2>&1 | tee $log_path/log.txt
fi

if [[ $stage -le 3 ]]; then
    mkdir -p $hf_path
    python quantize/hfize.py \
        --base_model $base_model_path \
        --quantized_path $ckpt_path \
        --hf_output_path $hf_path
fi
