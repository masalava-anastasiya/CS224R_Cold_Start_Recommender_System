"""Evaluate policies with a non-exhaustive candidate pool."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

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
from src.baselines.popularity_baseline import PopularityBaseline
from src.methods.constrained_bandit import ConstrainedLinearUCBBandit


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
    for user_idx in tqdm(cold_users, desc=label):
        state = env.reset(user_idx=user_idx)
        policy.reset(user_idx=user_idx)
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
    cum = [sum(r) for r in all_rewards]
    step1 = [r[0] if r else 0.0 for r in all_rewards]
    avg_k = [float(np.mean(r[:k])) for r in all_rewards]
    ndcg = [ndcg_at_k(r, k) for r in all_rewards]
    hit = [hit_at_k(r, k) for r in all_rewards]
    avg = [float(np.mean(r)) for r in all_rewards]
    return {
        "avg_cum_reward": float(np.mean(cum)),
        "std_cum_reward": float(np.std(cum)),
        "avg_reward_per_step": float(np.mean(avg)),
        "std_reward_per_step": float(np.std(avg)),
        "avg_reward_step1": float(np.mean(step1)),
        "std_reward_step1": float(np.std(step1)),
        f"avg_reward_{k}": float(np.mean(avg_k)),
        f"std_reward_{k}": float(np.std(avg_k)),
        f"ndcg_{k}": float(np.mean(ndcg)),
        f"std_ndcg_{k}": float(np.std(ndcg)),
        f"hit_{k}": float(np.mean(hit)),
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
    user_split = torch.load(processed / "user_split.pt", weights_only=False)
    item_emb = torch.load(processed / "item_emb.pt", weights_only=False)
    user_emb = torch.load(processed / "user_emb.pt", weights_only=False)

    warm_users = user_split["warm"]
    cold_users = user_split["cold"]
    n_items = item_emb.shape[0]
    T = config.cold_start_horizon_T
    K_EVAL = 5

    pool_sizes = [len(ratings_by_user[u]) for u in cold_users]
    mean_pool = int(np.mean(pool_sizes))
    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | Items: {n_items} | T: {T}"
    )
    print(
        f"Candidate pool: mean={np.mean(pool_sizes):.0f}, "
        f"median={np.median(pool_sizes):.0f}, min={min(pool_sizes)}, max={max(pool_sizes)}"
    )

    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=True,
    )

    LAMBDA_NLTS = 50.0
    SIGMA_NLTS = 0.5
    print("Computing warm prior for NLTS...")
    mu_0, Lambda_0 = compute_warm_prior(
        ratings_by_user, warm_users, item_emb, reg=LAMBDA_NLTS,
    )
    nlts = NeuralLinearTS(
        item_emb=item_emb,
        lambda_prior=LAMBDA_NLTS,
        sigma_noise=SIGMA_NLTS,
        mu_0=mu_0,
        Lambda_0=Lambda_0,
    )

    K_FACTORS = 50
    print(f"Fitting Hybrid TS (k={K_FACTORS})...")
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )

    print(f"Fitting Greedy CF (k={K_FACTORS})...")
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=1.0,
    )

    random_bl = RandomBaseline(seed=42)

    print("Computing warm prior for constrained bandit...")
    mu_0_cb, _ = compute_warm_prior(ratings_by_user, warm_users, item_emb, reg=1.0)
    popularity_cb = PopularityBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        shrinkage=10.0,
    )
    constrained = ConstrainedLinearUCBBandit(
        item_emb=item_emb,
        baseline_policy=popularity_cb,
        lambda_reg=1.0,
        sigma2=1.0,
        beta=1.0,
        beta_safe=2.0,
        alpha=0.90,
        prior_mean=mu_0_cb,
        constraint_mode="cumulative",
    )

    print(f"\nNon-exhaustive eval: {T} selections from ~{mean_pool} candidates")
    policies = [
        (hybrid, "HybridTS(CF)"),
        (nlts, "NLTS(content)"),
        (constrained, "Constrained"),
        (greedy_cf, "GreedyCF"),
        (random_bl, "Random"),
    ]

    all_metrics = {}
    for policy, label in policies:
        rewards = run_episodes(policy, env, cold_users, label=label)
        all_metrics[label] = compute_metrics(rewards, k=K_EVAL)

    def row(metric_label, key, show_std=True):
        std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
        parts = []
        for _, label in policies:
            m = all_metrics[label]
            if show_std and std_key in m:
                parts.append(f"{label}: {m[key]:.4f} ± {m[std_key]:.4f}")
            else:
                parts.append(f"{label}: {m[key]:.4f}")
        print(f"{metric_label}: {', '.join(parts)}")

    print("\nResults")
    row("Avg reward @ 1", "avg_reward_step1")
    row(f"Avg reward @ {K_EVAL}", f"avg_reward_{K_EVAL}")
    row("Avg reward / step", "avg_reward_per_step")
    row(f"NDCG@{K_EVAL}", f"ndcg_{K_EVAL}")
    row(f"Hit@{K_EVAL} (rating >= 4)", f"hit_{K_EVAL}", show_std=False)
    row("Avg cum. reward", "avg_cum_reward")

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
    print(f"Results saved to {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
