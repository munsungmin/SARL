#!/bin/bash
#SBATCH --job-name=sarl-grpo-aime-struct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=14
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
# NOTE: Set --partition, --account, --qos to match your cluster.

# ============================================================
# GRPO on AIME dataset — SARL
# ============================================================

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# Load conda/CUDA modules as needed for your cluster, e.g.:
# module load conda/2024.09 cuda/12.9.0
# conda activate /path/to/conda/envs/verl

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
JOB_ID="${SLURM_JOB_ID:-local}"
export RAY_TMPDIR="/tmp/ray_${JOB_ID}"
export TMPDIR="/tmp/tmp_${JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

n_gpus_per_node=""
nnodes=1
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
if [ -z "$n_gpus_per_node" ] && command -v nvidia-smi &>/dev/null; then
    n_gpus_per_node=$(nvidia-smi --list-gpus | wc -l)
fi
[ -z "$n_gpus_per_node" ] || [ "$n_gpus_per_node" -eq 0 ] && n_gpus_per_node=8
echo "Config: nnodes=$nnodes, n_gpus_per_node=$n_gpus_per_node"

if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_JOB_NAME" ] && [ "$SLURM_JOB_NAME" != "debug" ]; then
    logger='["console","wandb"]'
else
    logger='["console"]'
fi

# ---- Data ----
data_root="./data"
train_files="['$data_root/aime/train.parquet']"
test_files="['$data_root/math500/test.parquet','$data_root/amc23/test.parquet','$data_root/aime25/test.parquet']"

# ---- Model ----
model_name="Qwen/Qwen3-4B"
rollout_n=8
batch_size_per_gpu=32

# ---- Experiment ----
project_name="SARL"
experiment_name="qwen3-4b_aime_grpo_structure_only"

# ---- SARL reward configuration ----
embedding_model="Qwen/Qwen3-Embedding-0.6B"
embedding_server_port=8100
cluster_method="hdbscan"
is_think_model="true"

# ---- Start standalone vLLM embedding server ----
echo "Starting vLLM embedding server on port $embedding_server_port..."
python -m vllm.entrypoints.openai.api_server \
    --model $embedding_model \
    --task embed \
    --port $embedding_server_port \
    --gpu-memory-utilization 0.05 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --disable-log-stats \
    --tensor-parallel-size 1 \
    > ./logs/embedding_server_${JOB_ID}.log 2>&1 &
EMBEDDING_SERVER_PID=$!

echo "Waiting for embedding server..."
for i in {1..60}; do
    if curl -s http://localhost:$embedding_server_port/health >/dev/null 2>&1; then
        echo "Embedding server ready!"; break
    fi
    [ $i -eq 60 ] && echo "Warning: embedding server may not be ready, continuing..."
    sleep 2
done

cleanup() { kill $EMBEDDING_SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(( n_gpus_per_node * batch_size_per_gpu )) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.enable_activation_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="$logger" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    custom_reward_function.path=verl/verl/utils/reward_score/sarl.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.embedding_server_address="localhost:$embedding_server_port" \
    +custom_reward_function.reward_kwargs.is_think_model=$is_think_model \
    +custom_reward_function.reward_kwargs.cluster_method=$cluster_method \
    $@
