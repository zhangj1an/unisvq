#! /usr/bin/env bash
"""
UniSVQ PTQ pipeline for Qwen3-8B on NPU.
Uses pre-tokenized RedPajama calibration data (1024 x 2048).
Matches paper spec: Section 4.1, arXiv:2606.10520.
"""
set -ue

export PYTHONPATH="$PWD:$PYTHONPATH"

BASE_MODEL=/run/z84450661/Qwen3-8B
CALIB_DATA=/home/ma-user/work/z84450661/UniSVQ/data/redpajama_1024x2048.pt
EXP_DIR=/run/z84450661/unisvq_8b_exp
LOG_DIR=${EXP_DIR}/logs

STAGE=${1:-1}

CKPT_PATH=${EXP_DIR}/ckpt
HF_PATH=${EXP_DIR}/hf
HESS_PATH=${EXP_DIR}/hess

mkdir -p ${LOG_DIR} ${CKPT_PATH} ${HF_PATH} ${HESS_PATH}

echo "=== UniSVQ PTQ: Qwen3-8B on NPU ==="
echo "Base model: ${BASE_MODEL}"
echo "Calib data: ${CALIB_DATA}"
echo "Exp dir:    ${EXP_DIR}"
echo "Stage:      ${STAGE}"
echo ""

# ---- Stage 1: Compute Hessians ----
if [[ ${STAGE} -eq 1 ]]; then
    echo "[Stage 1] Computing Hessians..."
    python -c "
import torch, sys, os
sys.path.insert(0, '.')

# Load pre-tokenized calibration data
data = torch.load('${CALIB_DATA}')
devset = data['input_ids']
print(f'Loaded calibration data: {devset.shape} (paper: 1024x2048)')

# Save as the format expected by the pipeline
torch.save(devset, '${HESS_PATH}/devset.pt')
print('Saved devset for hessian computation')
"
    echo "[Stage 1] Done. Run stage 2 for quantization."
fi

# ---- Stage 2: Quantize and Fine-tune ----
if [[ ${STAGE} -le 2 ]]; then
    echo "[Stage 2] Quantizing and fine-tuning..."

    python -u quantize/quantize_finetune.py \
        --save_path ${CKPT_PATH} \
        --codebook identical \
        --batch_size 16 \
        --ft_bs 4 \
        --scale_override 0.83 \
        --base_model ${BASE_MODEL} \
        --hessian_path ${HESS_PATH} \
        --dataset_path ${CALIB_DATA} \
        --devset_size 1024 \
        --ctx_size 2048 \
        --ft_valid_size 128 \
        --scale_search_iters 3 \
        --blockwise_hadamard \
        --codebook_bit 2 \
        --ft_epochs 5 \
        --ft_early_stop 3 \
        --seed 42 \
        2>&1 | tee ${LOG_DIR}/stage2_log.txt
fi

# ---- Stage 3: Convert to HuggingFace Format ----
if [[ ${STAGE} -le 3 ]]; then
    echo "[Stage 3] Converting to HuggingFace format..."
    python quantize/hfize.py \
        --base_model ${BASE_MODEL} \
        --quantized_path ${CKPT_PATH} \
        --hf_output_path ${HF_PATH} \
        2>&1 | tee ${LOG_DIR}/stage3_log.txt
    echo "[Stage 3] Done. HF model at: ${HF_PATH}"
fi

echo "=== Pipeline complete ==="
