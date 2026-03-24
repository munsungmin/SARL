#!/bin/bash
#SBATCH --job-name=sarl-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=3-00:00:00
#SBATCH --output=./eval_results/logs/%j.out
#SBATCH --error=./eval_results/logs/%j.err
# NOTE: Set --partition, --account, --qos to match your cluster.

# ============================================================
# SARL evaluation script (Qwen3 / think-mode models)
#
# Usage:
#   sbatch eval/run_eval.slurm
#   MODELS=("Qwen/Qwen3-4B" "/path/to/ckpt") sbatch eval/run_eval.slurm
#   DATASETS="aime25 math500" N_SAMPLES=16 sbatch eval/run_eval.slurm
# ============================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Load conda/CUDA modules as needed for your cluster, e.g.:
# module load conda/2024.09 cuda/12.6
# conda activate sarl

mkdir -p eval_results/logs

# ---- Configuration ----
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results/vllm}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VLLM_MEM="${VLLM_MEM:-0.85}"
DATASETS="${DATASETS:-math500 amc23 aime25 minervamath}"
N_SAMPLES="${N_SAMPLES:-8}"

# ---- Models to evaluate ----
MODELS=(
    "Qwen/Qwen3-4B"
    # "/path/to/SARL/checkpoints/qwen3-4b_aime_grpo_sarl"
)

# ---- Run evaluations ----
mkdir -p "$OUTPUT_DIR"
total=${#MODELS[@]}
counter=0

for model_path in "${MODELS[@]}"; do
    counter=$(( counter + 1 ))
    echo "[$counter/$total] Evaluating: $model_path"

    python eval/eval_all.py \
        --model        "$model_path" \
        --datasets     $DATASETS \
        --num-samples  "$N_SAMPLES" \
        --global-batch-size "$BATCH_SIZE" \
        --temperature  "$TEMPERATURE" \
        --top-p        "$TOP_P" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir   "$OUTPUT_DIR" \
        --enable-thinking \
        --format-agnostic \
        --use-vllm \
        --vllm-mem "$VLLM_MEM"

    [ $counter -lt $total ] && sleep 2
done

echo ""
echo "Done. Results saved to $OUTPUT_DIR"
