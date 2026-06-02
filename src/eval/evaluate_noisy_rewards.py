"""Gaussian reward noise experiments: does stochasticity help exploration?

Runs all 4 methods (Random, Greedy CF, NLTS, Hybrid TS) under multiple
noise levels across both exhaustive and non-exhaustive evaluation protocols.
Thompson Sampling's uncertainty modeling should show a clearer advantage
as reward noise increases, since greedy exploitation of noisy signals
leads to worse decisions while TS naturally accounts for uncertainty.

Run from repo root:
    python -m src.eval.evaluate_noisy_rewards
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# ── metrics ──────────────────────────────────────────────────────────

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


def run_episodes(policy, env: ColdStartEnv, cold_users: List[int], label: str) -> List[List[float]]:
    all_rewards: List[List[float]] = []
    for user_idx in tqdm(cold_users, desc=label, ncols=72):
        state = env.reset(user_idx=user_idx)
        policy.reset()
        ep_rewards: List[float] = []
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            policy.update(action, reward, next_state, done)
            state = next_state
            ep_rewards.append(reward)
        all_rewards.append(ep_rewards)
    return all_rewards


def compute_metrics(all_rewards: List[List[float]], k: int) -> Dict[str, float]:
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


# ── policy constructors ─────────────────────────────────────────────

def build_policies(
    ratings_by_user: Dict,
    warm_users: List[int],
    item_emb: torch.Tensor,
    n_items: int,
    mu_0: np.ndarray,
    Lambda_0: np.ndarray,
) -> List[Tuple[str, Any]]:
    """Build all 4 policies. Returns (label, policy) pairs."""

    nlts = NeuralLinearTS(
        item_emb=item_emb,
        lambda_prior=50.0,
        sigma_noise=0.5,
        mu_0=mu_0,
        Lambda_0=Lambda_0,
    )

    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=50,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )

    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=50,
        reg=1.0,
    )

    random_bl = RandomBaseline(seed=42)

    return [
        ("HybridTS(CF)", hybrid),
        ("NLTS(content)", nlts),
        ("GreedyCF", greedy_cf),
        ("Random", random_bl),
    ]


# ── printing ─────────────────────────────────────────────────────────

def print_table(
    title: str,
    all_metrics: Dict[str, Dict[str, float]],
    labels: List[str],
    k: int,
) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = f"  {'Metric':<22}" + "".join(f" {l:<22}" for l in labels)
    print(header)
    print("  " + "-" * 96)

    def row(metric_label: str, key: str, fmt: str = ".4f", show_std: bool = True) -> None:
        std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
        parts = []
        for label in labels:
            m = all_metrics[label]
            if show_std and std_key in m:
                parts.append(f"{m[key]:{fmt}} +/- {m[std_key]:{fmt}}")
            else:
                parts.append(f"{m[key]:{fmt}}")
        line = f"  {metric_label:<22}" + "".join(f" {p:<22}" for p in parts)
        print(line)

    row("Avg reward @ 1",        "avg_reward_step1")
    row(f"Avg reward @ {k}",     f"avg_reward_{k}")
    row("Avg reward / step",     "avg_reward_per_step")
    row(f"NDCG@{k}",             f"ndcg_{k}")
    row(f"Hit@{k} (>= 4)",       f"hit_{k}", show_std=False)
    row("Avg cum. reward",       "avg_cum_reward")
    print("  " + "-" * 96)
    print(f"{'=' * 100}")


# ── main ─────────────────────────────────────────────────────────────

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

    NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5]

    pool_sizes = [len(ratings_by_user[u]) for u in cold_users]
    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | Items: {n_items} | T: {T}\n"
        f"Full candidate pool: mean={np.mean(pool_sizes):.0f}, "
        f"median={np.median(pool_sizes):.0f}\n"
        f"Noise levels: {NOISE_LEVELS}\n"
        f"Warm user fraction: {config.warm_user_fraction}"
    )

    # precompute warm prior for NLTS (shared across noise levels)
    print("Computing warm prior for NLTS...")
    t0 = time.time()
    mu_0, Lambda_0 = compute_warm_prior(
        ratings_by_user, warm_users, item_emb, reg=50.0,
    )
    print(f"Done in {time.time() - t0:.1f}s")

    all_results: Dict[str, Any] = {
        "experiment": {
            "noise_levels": NOISE_LEVELS,
            "T": T,
            "k_eval": K_EVAL,
            "n_cold_users": len(cold_users),
            "n_warm_users": len(warm_users),
            "warm_user_fraction": config.warm_user_fraction,
            "reward_clipping": [1.0, 5.0],
            "mean_candidates_per_user": float(np.mean(pool_sizes)),
        },
    }

    for noise_std in NOISE_LEVELS:
        print(f"\n{'#' * 100}")
        print(f"  NOISE STD = {noise_std}")
        print(f"{'#' * 100}")

        noise_key = f"noise_{noise_std}"
        all_results[noise_key] = {}

        for protocol, use_full_pool in [("exhaustive", False), ("non_exhaustive", True)]:
            protocol_label = "NON-EXHAUSTIVE" if use_full_pool else "EXHAUSTIVE"
            print(f"\n  --- {protocol_label} (noise_std={noise_std}) ---")

            env = ColdStartEnv(
                ratings_by_user=ratings_by_user,
                item_emb=item_emb,
                config=config,
                user_pool=cold_users,
                user_emb=user_emb,
                warm_users=set(warm_users),
                use_full_candidate_pool=use_full_pool,
                reward_noise_std=noise_std,
            )

            policies = build_policies(
                ratings_by_user, warm_users, item_emb, n_items, mu_0, Lambda_0,
            )
            labels = [label for label, _ in policies]

            metrics_by_policy: Dict[str, Dict[str, float]] = {}
            for label, policy in policies:
                rewards = run_episodes(
                    policy, env, cold_users,
                    label=f"  {label:<14} s={noise_std}",
                )
                metrics_by_policy[label] = compute_metrics(rewards, k=K_EVAL)

            cand_desc = f"~{int(np.mean(pool_sizes))}" if use_full_pool else str(T)
            print_table(
                f"{protocol_label} | noise_std={noise_std} | "
                f"T={T} steps from {cand_desc} candidates | "
                f"warm_frac={config.warm_user_fraction}",
                metrics_by_policy,
                labels,
                K_EVAL,
            )

            all_results[noise_key][protocol] = {
                label: {"metrics": m} for label, m in metrics_by_policy.items()
            }

    # ── summary table: NDCG@5 across noise levels ────────────────────

    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY: NDCG@{K_EVAL} across noise levels")
    print(f"{'=' * 100}")

    policy_labels = ["HybridTS(CF)", "NLTS(content)", "GreedyCF", "Random"]

    for protocol in ["exhaustive", "non_exhaustive"]:
        protocol_label = "NON-EXHAUSTIVE" if protocol == "non_exhaustive" else "EXHAUSTIVE"
        print(f"\n  {protocol_label}:")
        header = f"  {'noise_std':<12}" + "".join(f" {l:<18}" for l in policy_labels)
        print(header)
        print("  " + "-" * 86)
        for noise_std in NOISE_LEVELS:
            noise_key = f"noise_{noise_std}"
            parts = []
            for label in policy_labels:
                val = all_results[noise_key][protocol][label]["metrics"][f"ndcg_{K_EVAL}"]
                parts.append(f"{val:.4f}")
            line = f"  {noise_std:<12}" + "".join(f" {p:<18}" for p in parts)
            print(line)
        print("  " + "-" * 86)

    print()

    # ── save ─────────────────────────────────────────────────────────

    out_path = _REPO_ROOT / "results" / "noisy_rewards_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Results saved -> {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
