# Copyright 2024-2025 SARL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SARL Reward Function: Structure-based, label-free reward.

Rewards the topological diversity of the model's reasoning chain,
without requiring any ground-truth labels.

Algorithm:
  1. Extract the <think>...</think> block from the response.
  2. Segment it into reasoning steps.
  3. Embed each step with a lightweight vLLM embedding server.
  4. Cluster step embeddings and build a transition graph.
  5. Score the graph: 0.5 * avg_clustering_coeff + 1/(1 + avg_path_length)

The embedding model runs as a STANDALONE vLLM server (outside veRL workers):
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-Embedding-0.6B --task embed --port 8100 &

Configure in veRL:
    custom_reward_function.path=sarl/reward_score/sarl.py
    custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.embedding_server_address=localhost:8100
    +custom_reward_function.reward_kwargs.is_think_model=true
"""

import re
import sys
from typing import List, Optional

import numpy as np

# Counter for limiting log output (first N calls only)
_CALL_COUNT = 0
_MAX_LOG_CALLS = 5


def _log(msg: str, force: bool = False):
    """Write to stderr for visibility inside Ray workers."""
    global _CALL_COUNT
    if force or _CALL_COUNT < _MAX_LOG_CALLS:
        sys.stderr.write(f"{msg}\n")
        sys.stderr.flush()
        if not force:
            _CALL_COUNT += 1


# ---------------------------------------------------------------------------
# HTTP helper: embeddings via standalone vLLM server
# ---------------------------------------------------------------------------

def _get_embeddings(
    server_address: str,
    texts: List[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> Optional[List[np.ndarray]]:
    """POST to the vLLM /v1/embeddings endpoint and return normalized vectors."""
    import requests

    if not texts:
        return None

    try:
        resp = requests.post(
            f"http://{server_address}/v1/embeddings",
            json={"model": model_name, "input": texts},
            timeout=60,
        )
        if resp.status_code != 200:
            _log(f"[SARL] Embedding request failed: {resp.status_code}", force=True)
            return None

        embeddings = []
        for item in sorted(resp.json().get("data", []), key=lambda x: x.get("index", 0)):
            emb = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
        return embeddings
    except Exception as e:
        _log(f"[SARL] Embedding HTTP error: {e}", force=True)
        return None


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

_STEP_MARKER_RE = re.compile(
    r'(?:^|\n)'
    r'(?:'
    r'Wait[,. ]|Hmm[,. ]|Actually[,. ]|No[,. ]|'
    r'Let me (?:re|try|check|verify|reconsider|re-examine|think)|'
    r'I (?:made an error|was wrong|need to reconsider)|'
    r'Alternatively[,. ]|Another (?:approach|way|method)|'
    r'Let me verify|Let me check|Checking:|Indeed[,. ]|'
    r'This confirms|So the answer|The answer is|'
    r'Step \d+[:.)]|'
    r'First[,. ]|Second[,. ]|Third[,. ]|Next[,. ]|Finally[,. ]|'
    r'Therefore[,. ]|Thus[,. ]|Hence[,. ]|So[,. ]'
    r')',
    re.MULTILINE | re.IGNORECASE,
)

_MIN_STEP_LEN = 20


def _extract_think_block(text: str, is_think_model: bool) -> str:
    """Return the content of the <think>...</think> block (or full text for non-think models)."""
    if is_think_model:
        m = re.search(r"<think>\s*([\s\S]*?)\s*</think>", text, re.MULTILINE)
        if m:
            return m.group(1)
        # Ministral-style [THINK]...[/THINK]
        m = re.search(r"\[THINK\]\s*([\s\S]*?)\s*\[/THINK\]", text, re.MULTILINE)
        if m:
            return m.group(1)
        return ""
    return text.strip()


def _split_on_markers(text: str) -> List[str]:
    boundaries = [m.start() for m in _STEP_MARKER_RE.finditer(text)]
    if not boundaries:
        return [text]
    fragments, prev = [], 0
    for b in boundaries:
        chunk = text[prev:b].strip()
        if chunk:
            fragments.append(chunk)
        prev = b
    tail = text[prev:].strip()
    if tail:
        fragments.append(tail)
    return fragments


def _extract_steps(think_block: str) -> List[str]:
    """Segment the think block into reasoning steps."""
    text = think_block.strip()
    if not text:
        return []

    # Pass 1: double-newline paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    steps: List[str] = []
    for para in paragraphs:
        # Pass 2: split long paragraphs on semantic markers
        if len(para) > 200:
            fragments = _split_on_markers(para)
            if len(fragments) > 1:
                steps.extend(f for f in fragments if len(f) >= _MIN_STEP_LEN)
                continue
        if len(para) >= _MIN_STEP_LEN:
            steps.append(para)

    # Pass 3: fallback to single newlines
    if not steps:
        steps = [s.strip() for s in text.split("\n") if len(s.strip()) >= _MIN_STEP_LEN]

    return steps


# ---------------------------------------------------------------------------
# Graph construction and scoring
# ---------------------------------------------------------------------------

def _build_graph(step_embeddings: List[np.ndarray], cluster_method: str = "hdbscan"):
    """Build a transition graph where nodes are semantic clusters of steps."""
    import networkx as nx
    from sklearn.cluster import KMeans

    if len(step_embeddings) <= 1:
        return nx.Graph()

    X = np.array(step_embeddings, dtype=np.float32)

    if cluster_method == "hdbscan":
        try:
            import hdbscan
            min_cluster_size = max(2, min(5, len(X) // 4)) if len(X) >= 4 else 2
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=max(1, min_cluster_size - 1),
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(X)
            noise = labels == -1
            if noise.any():
                labels[noise] = 0 if noise.all() else labels.max() + 1
            cluster_seq = labels.tolist()
        except ImportError:
            _log("[SARL] HDBSCAN not found, falling back to KMeans", force=True)
            cluster_method = "kmeans"

    if cluster_method == "kmeans":
        k = min(max(2, int(np.sqrt(len(X)))), len(X))
        cluster_seq = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X).tolist()

    distances = [float(np.linalg.norm(X[i] - X[i + 1])) for i in range(len(X) - 1)]

    g = nx.Graph()
    for i in range(len(cluster_seq) - 1):
        u, v, w = cluster_seq[i], cluster_seq[i + 1], distances[i]
        if u != v:
            if g.has_edge(u, v):
                cnt = g[u][v].get("count", 1)
                g[u][v]["weight"] = (g[u][v]["weight"] * cnt + w) / (cnt + 1)
                g[u][v]["count"] = cnt + 1
            else:
                g.add_edge(u, v, weight=w, count=1)
    return g


def _score_graph(g) -> float:
    """
    Structure score in [0, 1]:
        0.5 * avg_clustering_coeff  +  1 / (1 + avg_shortest_path_length)

    Higher score = more diverse, interconnected reasoning topology.
    """
    import networkx as nx

    if g.number_of_nodes() <= 1:
        return 0.0

    clustering = nx.average_clustering(g)
    try:
        path_len_norm = min(1.0 / (1.0 + nx.average_shortest_path_length(g)), 0.5)
    except Exception:
        path_len_norm = 0.0

    return float(0.5 * clustering + path_len_norm)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    embedding_server_address: str = "localhost:8100",
    is_think_model: bool = True,
    cluster_method: str = "kmeans",
    **kwargs,
) -> dict:
    """
    SARL label-free structure reward.

    Scores the topological diversity of the model's reasoning chain
    without requiring ground-truth labels.

    Args:
        data_source:               Dataset identifier (passed by veRL, unused here).
        solution_str:              Full model response string.
        ground_truth:              Ground-truth answer (ignored — label-free).
        extra_info:                Optional extra info from veRL (unused).
        embedding_server_address:  Address of the standalone vLLM embedding server.
        is_think_model:            True if the model wraps reasoning in <think> tags.
        cluster_method:            "hdbscan" (recommended) or "kmeans".

    Returns:
        dict with keys: score, num_steps, num_clusters, num_edges
    """
    _log(f"[SARL] compute_score called (cluster_method={cluster_method})")

    try:
        think_block = _extract_think_block(solution_str, is_think_model)
        if not think_block:
            return {"score": 0.0, "num_steps": 0, "num_clusters": 0, "num_edges": 0}

        steps = _extract_steps(think_block)
        num_steps = len(steps)
        if num_steps <= 1:
            return {"score": 0.0, "num_steps": num_steps, "num_clusters": 0, "num_edges": 0}

        embeddings = _get_embeddings(embedding_server_address, steps)
        if not embeddings or len(embeddings) <= 1:
            return {"score": 0.0, "num_steps": num_steps, "num_clusters": 0, "num_edges": 0}

        g = _build_graph(embeddings, cluster_method=cluster_method)
        score = _score_graph(g)

        _log(
            f"[SARL] score={score:.4f}, num_steps={num_steps}, "
            f"num_clusters={g.number_of_nodes()}, num_edges={g.number_of_edges()}"
        )

        return {
            "score": score,
            "num_steps": num_steps,
            "num_clusters": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
        }

    except Exception as e:
        _log(f"[SARL] compute_score error: {e}", force=True)
        return {"score": 0.0, "num_steps": 0, "num_clusters": 0, "num_edges": 0}
