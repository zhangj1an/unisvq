#! /usr/bin/env bash

set -ue 

export CUDA_VISIBLE_DEVICES=1    # Set visible GPUs, e.g., "0,1,2,3" for 4 GPUs or "0-7" for 8 GPUs

python -W ignore quantized_infer.py \
    --model_path "YOUR_MODEL_PATH" \
    --batch_size 1
