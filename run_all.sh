#!/bin/sh
# Base eval -> LoRA SFT -> fine-tuned eval, back to back.
set -e
M=CohereLabs/tiny-aya-fire
T=artifacts/scenarios_test6.jsonl
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python evaluate.py --name base --model $M --scenarios $T
.venv/bin/python train.py --model $M --out out/lora --minutes ${TRAIN_MIN:-22}
.venv/bin/python evaluate.py --name sft --model $M --adapter out/lora --scenarios $T
echo ALL_DONE
