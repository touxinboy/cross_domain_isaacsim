#!/usr/bin/env bash
# Prepared only. Submit later after arranging GPU availability.
#SBATCH --job-name=l17-qwen-vl
#SBATCH --partition=epic-cluster-compute-a100-01
#SBATCH --nodelist=epic-cluster-compute-a100-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${QWEN_PYTHON:?Set QWEN_PYTHON to the verified model environment Python; no packages are installed}"
: "${QWEN_MODEL_DIR:?Set QWEN_MODEL_DIR to the existing Qwen3-VL-4B-Instruct checkpoint}"
export L17_QWEN_ROOT="${L17_QWEN_ROOT:-$HOME/L17/qwen_service}"
exec "$QWEN_PYTHON" -B "$L17_QWEN_ROOT/qwen_transformers_server.py" --serve
