"""Non-exhaustive evaluation: does exploration actually help?

In the standard evaluation (20 candidates, 20 steps), every policy selects
every item — total reward is identical and exploration is structurally
penalized. This script uses the FULL candidate pool (~176 items per user)
with only T=20 steps, so the agent must *discover* which items are good.

Under this setup, exploration-based methods (Thompson Sampling) should
outperform pure exploitation (greedy CF) because:
  - The agent can't just pick everything; it must choose wisely
  - Exploring early reveals the user's taste, enabling better later picks
  - Greedy exploitation of a potentially wrong prior wastes slots on bad items

Run from repo root:
    python -m src.eval.evaluate_exploration
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import DataConfig
from src.data.env import ColdStartEnv
from src.methods.neural_linear_ts import NeuralLinearTS, compute_warm_prior
from src.methods.hybrid_neural_linear_ts import HybridNeuralLinearTS
from src.baselines.greedy_cf import GreedyCFBaseline
from src.baselines.random_baseline import RandomBaseline


# --- metrics ---

def dcg_at_k(rewards: List[float], k: int) -> float:
    k = min(k, len(rewards))
    gains = np.asarray(rewards[:k], dtype=np.float64)
    discounts = np.log2(np.arange(2, k + 2))
    return float((gains / discounts).sum())

def ndcg_at_k(rewards: List[float], k: int) -> float:
    ideal = sorted(rewards, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(rewards, k) / idcg if idcg > 0.0 else 0.0

def hit_at_k(rewards: List[float], k: int, threshold: float = 4.0) -> float:
    return float(any(r >= threshold for r in rewards[:k]))


def run_episodes(policy, env, cold_users, label):
    all_rewards = []
    for user_idx in tqdm(cold_users, desc=label, ncols=72):
        state = env.reset(user_idx=user_idx)
        policy.reset()
        ep_rewards = []
        done = False
        while not done:
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            policy.update(action, reward, state, done)
            ep_rewards.append(reward)
        all_rewards.append(ep_rewards)
    return all_rewards


def compute_metrics(all_rewards, k):
    cum   = [sum(r) for r in all_rewards]
    step1 = [r[0] if r else 0.0 for r in all_rewards]
    avg_k = [float(np.mean(r[:k])) for r in all_rewards]
    ndcg  = [ndcg_at_k(r, k) for r in all_rewards]
    hit   = [hit_at_k(r, k) for r in all_rewards]
    avg   = [float(np.mean(r)) for r in all_rewards]
    return {
        "avg_cum_reward":      float(np.mean(cum)),
        "std_cum_reward":      float(np.std(cum)),
        "avg_reward_per_step": float(np.mean(avg)),
        "std_reward_per_step": float(np.std(avg)),
        "avg_reward_step1":    float(np.mean(step1)),
        "std_reward_step1":    float(np.std(step1)),
        f"avg_reward_{k}":     float(np.mean(avg_k)),
        f"std_reward_{k}":     float(np.std(avg_k)),
        f"ndcg_{k}":           float(np.mean(ndcg)),
        f"std_ndcg_{k}":       float(np.std(ndcg)),
        f"hit_{k}":            float(np.mean(hit)),
    }


def main() -> None:
    config = DataConfig()
    processed = Path(config.data_dir) / "processed"

    required = ["ratings_by_user.pt", "user_split.pt", "item_emb.pt", "user_emb.pt"]
    missing = [f for f in required if not (processed / f).exists()]
    if missing:
        print(f"Missing artifacts: {missing}\nRun `python -m src.data.preprocess` first.")
        sys.exit(1)

    print("Loading artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split      = torch.load(processed / "user_split.pt",      weights_only=False)
    item_emb        = torch.load(processed / "item_emb.pt",        weights_only=False)
    user_emb        = torch.load(processed / "user_emb.pt",        weights_only=False)

    warm_users = user_split["warm"]
    cold_users = user_split["cold"]
    n_items    = item_emb.shape[0]
    T = config.cold_start_horizon_T
    K_EVAL = 5

    # report how big the candidate pools are
    pool_sizes = [len(ratings_by_user[u]) for u in cold_users]
    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | Items: {n_items} | T: {T}\n"
        f"Full candidate pool stats: mean={np.mean(pool_sizes):.0f}, "
        f"median={np.median(pool_sizes):.0f}, min={min(pool_sizes)}, max={max(pool_sizes)}"
    )

    # -- build the non-exhaustive environment --
    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=True,  # the key difference
    )

    # -- build all policies --

    # content-based NLTS (tuned hyperparameters from sweep)
    LAMBDA_NLTS = 50.0
    SIGMA_NLTS  = 0.5
    print(f"\nComputing warm prior for content-based NLTS...")
    mu_0, Lambda_0 = compute_warm_prior(
        ratings_by_user, warm_users, item_emb, reg=LAMBDA_NLTS,
    )
    nlts = NeuralLinearTS(
        item_emb=item_emb, lambda_prior=LAMBDA_NLTS,
        sigma_noise=SIGMA_NLTS, mu_0=mu_0, Lambda_0=Lambda_0,
    )

    # hybrid NLTS (CF features + Thompson Sampling)
    K_FACTORS = 50
    print(f"Fitting Hybrid TS (CF features, k={K_FACTORS})...")
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user, warm_users=warm_users,
        n_items=n_items, k=K_FACTORS, lambda_prior=1.0, sigma_noise=1.0,
    )

    # greedy CF baseline (same features as hybrid, but no exploration)
    print(f"Fitting Greedy CF (k={K_FACTORS})...")
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user, warm_users=warm_users,
        n_items=n_items, k=K_FACTORS, reg=1.0,
    )

    random_bl = RandomBaseline(seed=42)

    # -- run episodes --
    print(f"\n{'='*90}")
    print(f"  NON-EXHAUSTIVE EVALUATION: {T} selections from ~{int(np.mean(pool_sizes))} candidates")
    print(f"{'='*90}\n")

    policies = [
        (hybrid,    "HybridTS(CF)"),
        (nlts,      "NLTS(content)"),
        (greedy_cf, "GreedyCF"),
        (random_bl, "Random"),
    ]

    all_metrics = {}
    for policy, label in policies:
        rewards = run_episodes(policy, env, cold_users, label=f"{label:<14}")
        all_metrics[label] = compute_metrics(rewards, k=K_EVAL)

    # -- print results table --
    print(f"\n{'='*100}")
    print(f"  NON-EXHAUSTIVE EVALUATION RESULTS")
    print(f"  {T} steps from full candidate pool (~{int(np.mean(pool_sizes))} items/user)")
    print(f"{'='*100}")

    header_labels = [label for _, label in policies]
    header = f"  {'Metric':<22}" + "".join(f" {l:<22}" for l in header_labels)
    print(header)
    print("  " + "-" * 96)

    def row(metric_label, key, fmt=".4f", show_std=True):
        std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
        parts = []
        for _, label in policies:
            m = all_metrics[label]
            if show_std and std_key in m:
                parts.append(f"{m[key]:{fmt}} ± {m[std_key]:{fmt}}")
            else:
                parts.append(f"{m[key]:{fmt}}")
        line = f"  {metric_label:<22}" + "".join(f" {p:<22}" for p in parts)
        print(line)

    row("Avg reward @ 1",       "avg_reward_step1")
    row(f"Avg reward @ {K_EVAL}", f"avg_reward_{K_EVAL}")
    row("Avg reward / step",    "avg_reward_per_step")
    row(f"NDCG@{K_EVAL}",       f"ndcg_{K_EVAL}")
    row(f"Hit@{K_EVAL} (≥ 4)",  f"hit_{K_EVAL}", show_std=False)
    row("Avg cum. reward",      "avg_cum_reward")
    print("  " + "-" * 96)
    print(f"  Note: cum. reward now VARIES between policies (non-exhaustive selection)")
    print(f"{'='*100}")

    # -- save --
    out_path = _REPO_ROOT / "results" / "exploration_results.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "evaluation": {
            "mode": "non_exhaustive",
            "T": T,
            "mean_candidates_per_user": float(np.mean(pool_sizes)),
            "n_cold_users": len(cold_users),
            "n_warm_users": len(warm_users),
        },
    }
    for _, label in policies:
        payload[label] = {"metrics": all_metrics[label]}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
