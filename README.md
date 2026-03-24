# SARL: Structure-Aware Reinforcement Learning for Reasoning

> A label-free RL framework that improves LLM reasoning by optimizing the **topology of the reasoning process itself** — no ground-truth answers required, applicable to both verifiable and open-ended tasks.

## Overview

Current RL for reasoning (RLVR) is bottlenecked by verifiability: it requires ground-truth labels and automated verifiers, limiting applicability to domains like math and code. SARL breaks this constraint by shifting supervision from *what* the model produces to *how* it thinks.

Inspired by the **small-world organization of the human brain**, SARL constructs a per-response **Reasoning Map** from intermediate `<think>` steps and rewards its small-world topology — encouraging reasoning traces that are locally coherent within functional subproblems and globally efficient across reasoning modes.

**SARL pipeline:**
1. Extract reasoning steps from the `<think>` block.
2. Embed each step with a lightweight embedding model.
3. Cluster steps into latent reasoning types (nodes) and build a transition graph (edges).
4. Score the graph's small-world structure:

```
SR(G) = 0.5 × C(G)  +  1 / (1 + L(G))
```

where `C(G)` is the average clustering coefficient (local coherence) and `L(G)` is the average shortest path length (global efficiency).

**Results on Qwen3-4B:**
- Math (AIME25, AMC23, MATH500, Minerva): **+7.65% avg under GRPO**, **+5.87% under PPO** — matches or surpasses ground-truth RL without any labels.
- Open-ended (WildBench, OpenRubrics-v2): **+9.10% under GRPO**, **+10.35% under PPO** — substantially outperforms DPO and entropy-based baselines.
- Training dynamics exhibit lower KL divergence while preserving high policy entropy.


The embedding model runs as a **standalone vLLM server** (separate from the policy GPU workers), adding minimal overhead.

### Key components

| Component | Description |
|---|---|
| `verl/verl/utils/reward_score/sarl.py` | Label-free structure reward (plug-and-play with veRL's `custom_reward_function`) |
| `scripts/grpo/` | GRPO training scripts for Qwen3-4B |
| `scripts/ppo/` | PPO training scripts for Qwen3-4B |
| `data/aime_dataset.py` | Preprocesses AIME 1983–2024 CSV to veRL parquet format |
| `data/openrubrics_v2.py` | Preprocesses OpenRubrics-v2 dataset to veRL parquet format |
| `eval/eval_all.py` | Evaluation across MATH-500, AIME24/25, AMC23, and more |
| `verl/` | veRL framework (bundled and pre-patched) |

---

## Getting Started

### 1. Prerequisites

- Python 3.12
- CUDA 12.9

### 2. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/SARL.git
cd SARL
```

### 3. Create conda environment and install dependencies

```bash
conda create -n sarl python=3.12 -y
conda activate sarl
```

Install veRL with vLLM support (the bundled `verl/` directory is pre-patched):

```bash
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..
```

Install SARL-specific dependencies:

```bash
pip install -r requirements.txt
```

> `hdbscan` is used for density-based clustering of reasoning steps. The code automatically falls back to KMeans if it is unavailable.

### 4. Prepare training data

#### AIME dataset (1983–2024)

The raw CSV is already included at `data/AIME_Dataset_1983_2024.csv`. Convert it to parquet:

```bash
python data/aime_dataset.py \
    --csv_path data/AIME_Dataset_1983_2024.csv \
    --local_save_dir data/aime
```

Evaluation parquet files (AIME25, AMC23, MATH-500, MinervaMatch) are generated using the scripts in `data/`:

```bash
python data/aime25.py       --local_dir data/aime25
python data/amc23.py        --local_dir data/amc23
python data/math500.py      --local_dir data/math500
python data/minervamath.py  --local_dir data/minervamath
```

#### OpenRubrics-v2 dataset

```bash
python data/openrubrics_v2.py \
    --local_save_dir data/openrubrics_v2 \
    --tokenizer Qwen/Qwen3-4B \
    --min_response_tokens 512 \
    --max_response_tokens 4096
```

This downloads `OpenRubrics/OpenRubric-v2` from HuggingFace, filters by response length, and saves `train.parquet` + `eval.parquet`.

---

## Training

### Configure your cluster

Before running, open the relevant `.slurm` file and:
1. Set `#SBATCH --partition`, `--account`, `--qos` for your cluster.
2. Update the conda activation line to point to your environment.
3. Optionally set `HF_HOME` and `TRITON_CACHE_DIR` via environment variables.

### GRPO

| Script | Dataset | Description |
|---|---|---|
| `scripts/grpo/aime/sarl.slurm` | AIME 1983–2024 | GRPO + SARL on verifiable tasks|
| `scripts/grpo/openrubric/sarl.slurm` | OpenRubrics-v2 | GRPO + SARL on non-verifiable tasks |

Run example:
```bash
sbatch scripts/grpo/aime/sarl.slurm
```

### PPO

| Script | Dataset | Description |
|---|---|---|
| `scripts/ppo/aime/sarl.slurm` | AIME 1983–2024 | PPO + SARL on verifiable tasks |
| `scripts/ppo/openrubric/sarl.slurm` | OpenRubrics-v2 | PPO + SARL on non-verifiable tasks |

Run example:
```bash
sbatch scripts/ppo/aime/sarl.slurm
```

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model for structure reward |
| `cluster_method` | `kmeans` | Clustering algorithm (`kmeans` or `hdbscan`) |
| `is_think_model` | `true` | Whether model uses `<think>` tags (Qwen3-style) |

Override from the command line:

```bash
sbatch scripts/grpo/aime/sarl.slurm \
    +custom_reward_function.reward_kwargs.cluster_method=kmeans
```

### How the embedding server works

Each training script starts a standalone vLLM embedding server before launching the trainer:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task embed \
    --port 8100 \
    --gpu-memory-utilization 0.05
```

The SARL reward function calls this server via HTTP to embed reasoning steps. The server is automatically shut down when training ends.

---

## Evaluation

### Prepare eval datasets

Ensure the parquet files for your target datasets are under `data/` (see data prep above):

```
data/
  math500/test.parquet
  aime25/test.parquet
  amc23/test.parquet
  minervamath/test.parquet
```

### Run evaluation

```bash
# Evaluate a HuggingFace model
python eval/eval_all.py \
    --model Qwen/Qwen3-4B \
    --datasets math500 aime25 amc23 \
    --num-samples 8 \
    --enable-thinking \
    --format-agnostic \
    --use-vllm

# Evaluate a local checkpoint
python eval/eval_all.py \
    --model /path/to/checkpoint \
    --datasets math500 aime25 amc23 minervamath \
    --num-samples 32 \
    --enable-thinking \
    --format-agnostic \
    --use-vllm \
    --output-dir ./eval_results/my_run
```

### Via SLURM

Edit `eval/run_eval.slurm` to configure the `MODELS` array, then:

```bash
sbatch eval/run_eval.slurm
```

### Evaluation metrics

- **avg@n**: mean per-problem accuracy across `n` samples
- **pass@n**: fraction of problems with at least one correct solution among `n` samples
- Grader: `eval/verify.py` — cascades MathD → SymPy → Math-Verify for high-recall answer matching

Results are saved as JSON files in `--output-dir`.

---

## Repository Structure

```
SARL/
├── README.md
├── requirements.txt
├── scripts/
│   ├── setup_env.sh             # Cluster environment setup reference
│   ├── grpo/
│   │   ├── aime/sarl.slurm      
│   │   └── openrubric/sarl.slurm
│   └── ppo/
│       ├── aime/sarl.slurm      
│       └── openrubric/sarl.slurm
├── data/
│   ├── aime_dataset.py          # Preprocess AIME 1983-2024 CSV
│   ├── openrubrics_v2.py        # Preprocess OpenRubrics-v2
│   └── AIME_Dataset_1983_2024.csv
├── eval/
│   ├── eval_all.py              # Evaluation script
│   ├── verify.py                # Math answer grader (MathD → SymPy → Math-Verify)
│   └── run_eval.slurm           # SLURM eval launcher
└── verl/                        # veRL framework (bundled, minimally patched)
    ├── verl/
    │   ├── workers/reward_manager/batch.py      # patched (skip_special_tokens)
    │   ├── trainer/config/ppo_trainer.yaml      # patched (reward_kwargs)
    │   └── utils/reward_score/
    │       ├── __init__.py                      # patched (data sources)
    │       ├── math_dapo.py                     # patched (boxed extraction)
    │       └── sarl.py                          # SARL structure reward function
    ├── setup.py
    └── ...
```

### veRL modifications

This repo bundles [veRL](https://github.com/volcengine/verl) with **4 minimal patches** plus the SARL reward file. All other files are identical to upstream.

| File | Change | Why |
|---|---|---|
| `verl/utils/reward_score/sarl.py` | **New file** — SARL structure reward | Core contribution. |
| `verl/workers/reward_manager/batch.py` | `skip_special_tokens=False` in response decoding | **Critical.** `<think>` and `</think>` are special tokens in Qwen3. Without this, they are stripped before the reward function sees the response, making the structure reward always 0. |
| `verl/trainer/config/ppo_trainer.yaml` | Adds `reward_kwargs: {}` under `custom_reward_function` | **Critical.** Required for Hydra to accept `+custom_reward_function.reward_kwargs.*` overrides at the command line. |
| `verl/utils/reward_score/__init__.py` | Adds `"AIME_1983_2024"`, `"HuggingFaceH4/aime_2024"`, `"MathArena/aime_2025"`, `"numina_math"`, `"knoveleng/AMC-23"` to data-source routing | Needed for eval and baseline acc-only runs with these datasets. |
| `verl/utils/reward_score/math_dapo.py` | Tries `\boxed{}` extraction before `Answer:` pattern in `is_correct_minerva` | Improves answer extraction accuracy for AIME/MATH eval. |

---

## Using SARL with a Different Model

The reward function is model-agnostic. To use it with a different base model:

1. Edit the `model_name` variable in the training script.
2. Set `is_think_model=true` if the model produces `<think>...</think>` tags (e.g. Qwen3 series), `false` otherwise.

---

## Acknowledgements

- [veRL](https://github.com/volcengine/verl) — the RL training framework
- [TTRL](https://github.com/prime-rl/ttrl) — Test-Time Reinforcement Learning baseline
- [Qwen3](https://huggingface.co/Qwen/Qwen3-4B) — base model
- [OpenRubrics](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2) — preference dataset
