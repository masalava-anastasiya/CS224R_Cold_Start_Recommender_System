"""Evaluate demographic-conditioned Thompson Sampling."""

from __future__ import annotations

import argparse
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
from src.methods.hybrid_neural_linear_ts import HybridNeuralLinearTS
from src.methods.demographic_prior_ts import DemographicPriorTS
from src.baselines.greedy_cf import GreedyCFBaseline
from src.baselines.random_baseline import RandomBaseline


def dcg_at_k(rewards, k):
    k = min(k, len(rewards))
    gains = np.asarray(rewards[:k], dtype=np.float64)
    return float((gains / np.log2(np.arange(2, k + 2))).sum())


def ndcg_at_k(rewards, k):
    idcg = dcg_at_k(sorted(rewards, reverse=True), k)
    return dcg_at_k(rewards, k) / idcg if idcg > 0.0 else 0.0


def hit_at_k(rewards, k, threshold=4.0):
    return float(any(r >= threshold for r in rewards[:k]))


def run_episodes(policy, env, users, label):
    all_rewards = []
    for u in tqdm(users, desc=label):
        state = env.reset(user_idx=u)
        policy.reset(user_idx=u)
        ep, done = [], False
        while not done:
            a = policy.select_action(state)
            state, r, done, _ = env.step(a)
            policy.update(a, r, state, done)
            ep.append(r)
        all_rewards.append(ep)
    return all_rewards


def compute_metrics(all_rewards, k):
    cum = [sum(r) for r in all_rewards]
    step1 = [r[0] if r else 0.0 for r in all_rewards]
    avg_k = [float(np.mean(r[:k])) for r in all_rewards]
    ndcg = [ndcg_at_k(r, k) for r in all_rewards]
    hit = [hit_at_k(r, k) for r in all_rewards]
    return {
        "avg_cum_reward": float(np.mean(cum)),
        "std_cum_reward": float(np.std(cum)),
        "avg_reward_step1": float(np.mean(step1)),
        "std_reward_step1": float(np.std(step1)),
        f"avg_reward_{k}": float(np.mean(avg_k)),
        f"std_reward_{k}": float(np.std(avg_k)),
        f"ndcg_{k}": float(np.mean(ndcg)),
        f"std_ndcg_{k}": float(np.std(ndcg)),
        f"hit_{k}": float(np.mean(hit)),
    }


def main(warm_frac: float = 1.0, out_name: str = "demographic_prior_results.json") -> None:
    config = DataConfig()
    processed = Path(config.data_dir) / "processed"

    print("Loading artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split = torch.load(processed / "user_split.pt", weights_only=False)
    item_emb = torch.load(processed / "item_emb.pt", weights_only=False)
    user_emb = torch.load(processed / "user_emb.pt", weights_only=False)

    warm_users = user_split["warm"]
    cold_users = user_split["cold"]
    if warm_frac < 1.0:
        n_keep = int(len(warm_users) * warm_frac)
        warm_users = warm_users[:n_keep]
        print(f"Using {warm_frac:.0%} of warm users ({len(warm_users)})")
    n_items = item_emb.shape[0]
    T = config.cold_start_horizon_T
    K_EVAL = 5
    K_FACTORS = 50

    print(f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | Items: {n_items} | T: {T}")

    print("Building policies...")
    t0 = time.time()
    demo = DemographicPriorTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        user_emb=user_emb,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
        ridge_alpha=10.0,
    )
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )
    greedy = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=1.0,
    )
    random_bl = RandomBaseline(seed=42)
    print(f"Built in {time.time() - t0:.1f}s")

    policies = [
        ("DemographicTS", demo),
        ("HybridTS(CF)", hybrid),
        ("GreedyCF", greedy),
        ("Random", random_bl),
    ]
    labels = [name for name, _ in policies]

    all_results: Dict = {
        "experiment": {
            "method": "demographic_prior_ts",
            "k_factors": K_FACTORS,
            "ridge_alpha": 10.0,
            "T": T,
            "k_eval": K_EVAL,
            "n_cold_users": len(cold_users),
            "n_warm_users": len(warm_users),
        },
    }

    for protocol, use_full in [("selection", True), ("ranking", False)]:
        pool_desc = "full pool" if use_full else f"pool={T}"
        print(f"\n{protocol} ({pool_desc})")
        env = ColdStartEnv(
            ratings_by_user=ratings_by_user,
            item_emb=item_emb,
            config=config,
            user_pool=cold_users,
            user_emb=user_emb,
            warm_users=set(warm_users),
            use_full_candidate_pool=use_full,
            reward_noise_std=0.0,
        )
        metrics = {}
        for lab, pol in policies:
            rewards = run_episodes(pol, env, cold_users, label=lab)
            metrics[lab] = compute_metrics(rewards, k=K_EVAL)

        def row(metric_label, key, show_std=True):
            std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
            parts = []
            for name in labels:
                m = metrics[name]
                if show_std and std_key in m:
                    parts.append(f"{name}: {m[key]:.4f} ± {m[std_key]:.4f}")
                else:
                    parts.append(f"{name}: {m[key]:.4f}")
            print(f"{metric_label}: {', '.join(parts)}")

        row("Avg reward @ 1", "avg_reward_step1")
        row(f"Avg reward @ {K_EVAL}", f"avg_reward_{K_EVAL}")
        row(f"NDCG@{K_EVAL}", f"ndcg_{K_EVAL}")
        row(f"Hit@{K_EVAL}", f"hit_{K_EVAL}", show_std=False)
        row("Avg cum. reward", "avg_cum_reward")

        d1 = metrics["DemographicTS"]["avg_reward_step1"] - metrics["HybridTS(CF)"]["avg_reward_step1"]
        dn = metrics["DemographicTS"][f"ndcg_{K_EVAL}"] - metrics["HybridTS(CF)"][f"ndcg_{K_EVAL}"]
        print(f"Demographic - Hybrid: step1={d1:+.4f}, NDCG@{K_EVAL}={dn:+.4f}")

        all_results[protocol] = {name: {"metrics": m} for name, m in metrics.items()}

    all_results["experiment"]["warm_frac"] = warm_frac
    all_results["experiment"]["n_warm_used"] = len(warm_users)
    out_path = _REPO_ROOT / "results" / out_name
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Results saved to {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--warm_frac", type=float, default=1.0)
    p.add_argument("--out_name", type=str, default="demographic_prior_results.json")
    args = p.parse_args()
    main(warm_frac=args.warm_frac, out_name=args.out_name)
