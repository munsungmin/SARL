#!/bin/bash
# =============================================================================
# Environment setup for SARL training scripts
# Source this file or inline it into your SLURM scripts:
#   source scripts/setup_env.sh
#
# Customize the variables below for your cluster.
# =============================================================================

# ---------- Conda environment ----------
# Path to your conda environment with veRL installed.
# Update this to match your environment path.
CONDA_ENV="${SARL_CONDA_ENV:-/path/to/conda/envs/verl}"
conda activate "$CONDA_ENV"

# ---------- Cache directories ----------
# HuggingFace model cache (set to a directory with enough space)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"

# ---------- Ray temp directories ----------
# Use short paths to avoid Unix socket path length limit (107 bytes)
JOB_ID="${SLURM_JOB_ID:-local}"
export RAY_TMPDIR="/tmp/ray_${JOB_ID}"
export TMPDIR="/tmp/tmp_${JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# ---------- Misc ----------
export PYTHONNOUSERSITE=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ---------- Auto-detect GPUs ----------
n_gpus_per_node=""
nnodes=1

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($n_gpus_per_node GPUs)"
fi

if [ -z "$n_gpus_per_node" ] && command -v nvidia-smi &>/dev/null; then
    n_gpus_per_node=$(nvidia-smi --list-gpus | wc -l)
fi

if [ -z "$n_gpus_per_node" ] || [ "$n_gpus_per_node" -eq 0 ]; then
    echo "Warning: Could not detect GPUs, defaulting to 8"
    n_gpus_per_node=8
fi

echo "Config: nnodes=$nnodes, n_gpus_per_node=$n_gpus_per_node"

# ---------- Logger (wandb when running as a real SLURM job) ----------
if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_JOB_NAME" ] && [ "$SLURM_JOB_NAME" != "debug" ]; then
    logger='["console","wandb"]'
    echo "SLURM job $SLURM_JOB_ID - wandb logging enabled"
else
    logger='["console"]'
    echo "Local / debug run - console logging only"
fi
