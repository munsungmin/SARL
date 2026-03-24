# Thin wrapper: all TTRL logic lives in verl/utils/reward_score/ttrl.py
from verl.utils.reward_score.ttrl import (
    apply_ttrl_gt,
    select_top_k_per_prompt,
    compute_ttrl_metrics,
)

__all__ = ["apply_ttrl_gt", "select_top_k_per_prompt", "compute_ttrl_metrics"]
