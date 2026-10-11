#!/bin/bash
#SBATCH --job-name=sarl-grpo-aime-struct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4 
#SBATCH --cpus-per-task=14
#SBATCH --time=1-00:00:00
# NOTE: Set --partition, --account, --qos to match your cluster.

# ============================================================
# GRPO on AIME dataset — SARL
# ============================================================

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# (add) ---- conda 환경에서 다른 저장소의 verl 이 아닌 이 저장소의 verl을 사용하기 위함 
export PYTHONPATH="$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

# Load conda/CUDA modules as needed for your cluster, e.g.:
# module load conda/2024.09 cuda/12.9.0
# conda activate /path/to/conda/envs/verl

export PYTHONNOUSERSITE=1
export VLLM_USE_V1=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
JOB_ID="${SLURM_JOB_ID:-local}"
export RAY_TMPDIR="/data/sungmin/RL/tmp/ray_${JOB_ID}"
export TMPDIR="/data/sungmin/RL/tmp/tmp_${JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# ---- (add) GPU configuration ----
# bash scripts/grpo/aime/sarl.slurm 로 실행
export CUDA_VISIBLE_DEVICES=0,1,2,3 #slurm 사용 안할 때 

n_gpus_per_node=""
nnodes=1
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
if [ -z "$n_gpus_per_node" ] && command -v nvidia-smi &>/dev/null; then
    n_gpus_per_node=$(nvidia-smi --list-gpus | wc -l)
fi
[ -z "$n_gpus_per_node" ] || [ "$n_gpus_per_node" -eq 0 ] && n_gpus_per_node=4
echo "Config: nnodes=$nnodes, n_gpus_per_node=$n_gpus_per_node"

if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_JOB_NAME" ] && [ "$SLURM_JOB_NAME" != "debug" ]; then
    logger='["console","wandb"]'
else
    logger='["console"]'
fi



# ---- Model ----
model_name="Qwen/Qwen3-4B"
rollout_n=8
batch_size_per_gpu=32

# ---- Experiment ----
project_name="SARL"
experiment_name="qwen3-4b_aime_grpo_structure_only"


# (add) ---- root directory for logs, checkpoints, and wandb ----
git_commit=$(git rev-parse --short HEAD)
root_dir="/data/sungmin/RL/${project_name}/${experiment_name}/${git_commit}" 

# ---- Data ----

data_root="/data/sungmin/RL/datasets"
train_files="['$data_root/aime/train.parquet']"
test_files="['$data_root/math500/test.parquet','$data_root/amc23/test.parquet','$data_root/aime25/test.parquet']"

# (add) ---- Checkpoint configuration ----

## checkpoint 저장 위치 변경 
## trainer.default_local_dir="$checkpoint_dir" 
checkpoint_dir="${root_dir}/checkpoints"

## checkpoint 저장 주기 변경 
## trainer.max_actor_ckpt_to_keep=2 (grpo)
## trainer.max_critic_ckpt_to_keep=2 (ppo라면 아래에 추가)
max_actor_ckpt_to_keep=2
max_critic_ckpt_to_keep=2

## checkpoint resume mode 설정 
## trainer.resume_mode = "auto" 
## trainer.resume_mode = "diable"  
## trainer.resume_mode = "resume_path" , trainer.resume_from_path = "${checkpoint_dir}/global_step_XXX.pt" 
resume_mode="auto"

# (add) ---- default log configuration ---- 
logger='["console","wandb"]'

# (add) ---- log(wandb,online) configuration ---- 
export WANDB_API_KEY="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"
export WANDB_ENTITY="RL_reasoning1"
export WANDB_MODE="online"
export WANDB_DIR="${root_dir}/wandb"
export WANDB_GIT_ROOT="$REPO_ROOT" # github commit 
mkdir -p "$WANDB_DIR"

# (add) ---- log(console,local) configuration ---- 
## out : 일반 출력과 console metric 
## err : error 출력 
## 명령어 정리 
### exec : 이 시점 이후의 실행되는 모든 명령에 다음 규칙을 적용한다. 
### 예 ) python train.py : 터미널에만 출력  
### python train.py > train.log : train.log 에만 출력
### python train.py > >(tee -a train.log) : 터미널과 train.log 에 동시에 출력 (-a는 append)
### 궁금증 : >&2 를 해야 하는 이유가 뭐임 ? 터미널에서 섞여 보일 수 있다 ? 이거 외에는 저걸 안하면 뭔 문제가 생김 ? 
### 문제가 , tee가 stderr를 받아서 , stdout 으로 출력을 해서 , 다시 tee stdout 으로 들어가서, log 파일이 실제로 분리가 안되는 문제 같은데, 
### 그런 메커니즘이라면 >&2를 붙였을 때도, 다시 들어가서 재귀처럼 작동하는거 아님 ? 
log_dir="${root_dir}/logs"
mkdir -p "$log_dir"
log_id="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}" 
exec 1> >(tee -a "${log_dir}/${log_id}.out") \
     2> >(tee -a "${log_dir}/${log_id}.err" 1>&2) 

# (add) ---- gpu issue ----

## actor_rollout_ref.actor.use_dynamic_bsz=True 기본적으로 얘가 토큰 수 제한을 기준으로 자동 분할
use_dynamic_bsz=True
### 1024 + 8192 = 9216 < 16384 (ppo_max_token_len_per_gpu) => batch_size_per_gpu=32로 설정 가능
ppo_micro_batch_size_per_gpu=32  # 여기에 gpu 개수 * rollout_n 개가 update 단위 , train_batch_size=256 이라면 , gpu 4개라면, 2번에 걸쳐서 update 수행 
ppo_max_token_len_per_gpu=16384              # forward -> loss 계산 -> backward -> gradient 누적 

rollout_log_prob_micro_batch_size_per_gpu=32
rollout_log_prob_max_token_len_per_gpu=16384 # rollout 생성 -> actor로 log_prob 계산  

ref_log_prob_micro_batch_size_per_gpu=32
ref_log_prob_max_token_len_per_gpu=16384     # rollout 생성 -> ref로 log_prob 계산 

## error1 : update_actor, backward 중 OOM  
param_offload=False 
optimizer_offload=False

## error2 : rollout/generate 중 OOM 
max_prompt_length=1024
max_response_length=8192
rollout_n=8
rollout_gpu_memory_utilization=0.10

## error3 : embedding server OOM => --gpu-memory-utilization 0.10 으로 변경 
embedding_gpu_memory_utilization=0.10

# (add) ---- reward module ----
## kwargs 같은것 들은 , 알아서 compute_score 부분에 kwargs 로 들어감, 어쨌든 저런식으로 넘겨주면 될 듯 



# ---- SARL reward configuration ----
embedding_model="Qwen/Qwen3-Embedding-0.6B"
embedding_server_port=8100
cluster_method="hdbscan"
is_think_model="true"

# ---- Start standalone vLLM embedding server ----

# (add) ---- embedding server 에서 나오는 과정, terminal 에서 볼 수 있도록 하기 ----
## > (tee ~) 이렇게 사용하면 됨 
## 위에 log(console) 부분에도 tee 로 바꿈 
echo "Starting vLLM embedding server on port $embedding_server_port..."
python -m vllm.entrypoints.openai.api_server \
    --model $embedding_model \
    --task embed \
    --port $embedding_server_port \
    --gpu-memory-utilization $embedding_gpu_memory_utilization \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --disable-log-stats \
    --tensor-parallel-size 1 \
    1> >(tee -a "${log_dir}/embedding_server_${JOB_ID}.log") 2>&1 &
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
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(( n_gpus_per_node * batch_size_per_gpu )) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$rollout_log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ref_log_prob_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
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
    trainer.default_local_dir="$checkpoint_dir" \
    trainer.resume_mode="$resume_mode" \
    trainer.max_actor_ckpt_to_keep=$max_actor_ckpt_to_keep \
    trainer.max_critic_ckpt_to_keep=$max_critic_ckpt_to_keep \
    reward_model.enable=False \
    custom_reward_function.path="verl/verl/utils/reward_score/sarl.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.embedding_server_address="localhost:$embedding_server_port" \
    +custom_reward_function.reward_kwargs.is_think_model=$is_think_model \
    +custom_reward_function.reward_kwargs.cluster_method=$cluster_method \
    $@
