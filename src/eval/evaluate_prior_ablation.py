"""Prior quality ablation across warm-user fractions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def build_policies(
    ratings_by_user: Dict,
    warm_subset: List[int],
    item_emb: torch.Tensor,
    n_items: int,
) -> List[Tuple[str, Any]]:
    mu_0, Lambda_0 = compute_warm_prior(
        ratings_by_user, warm_subset, item_emb, reg=50.0,
    )
    k = min(50, len(warm_subset) - 1)
    nlts = NeuralLinearTS(
        item_emb=item_emb,
        lambda_prior=50.0,
        sigma_noise=0.5,
        mu_0=mu_0,
        Lambda_0=Lambda_0,
    )
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_subset,
        n_items=n_items,
        k=k,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_subset,
        n_items=n_items,
        k=k,
        reg=1.0,
    )
    random_bl = RandomBaseline(seed=42)
    return [
        ("HybridTS(CF)", hybrid),
        ("NLTS(content)", nlts),
        ("GreedyCF", greedy_cf),
        ("Random", random_bl),
    ]


def print_results(
    title: str,
    all_metrics: Dict[str, Dict[str, float]],
    labels: List[str],
    k: int,
) -> None:
    print(f"\n{title}")

    def row(metric_label, key, show_std=True):
        parts = []
        for name in labels:
            m = all_metrics[name]
            std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
            if show_std and std_key in m:
                parts.append(f"{name}: {m[key]:.4f} ± {m[std_key]:.4f}")
            else:
                parts.append(f"{name}: {m[key]:.4f}")
        print(f"{metric_label}: {', '.join(parts)}")

    row("Avg reward @ 1", "avg_reward_step1")
    row(f"Avg reward @ {k}", f"avg_reward_{k}")
    row("Avg reward / step", "avg_reward_per_step")
    row(f"NDCG@{k}", f"ndcg_{k}")
    row(f"Hit@{k} (rating >= 4)", f"hit_{k}", show_std=False)
    row("Avg cum. reward", "avg_cum_reward")


def main(
    warm_fractions: Optional[List[float]] = None,
    out_name: str = "prior_ablation_results.json",
) -> None:
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

    warm_users_full = user_split["warm"]
    cold_users = user_split["cold"]
    n_users_total = len(warm_users_full) + len(cold_users)
    n_items = item_emb.shape[0]
    T = config.cold_start_horizon_T
    K_EVAL = 5

    WARM_FRACTIONS = warm_fractions if warm_fractions is not None else [0.10, 0.20, 0.30, 0.50, 0.70]
    NOISE_LEVELS = [0.0, 0.5]

    pool_sizes = [len(ratings_by_user[u]) for u in cold_users]
    print(
        f"Total users: {n_users_total}, cold eval set: {len(cold_users)}, "
        f"items: {n_items}, T: {T}"
    )
    print(
        f"Candidate pool mean={np.mean(pool_sizes):.0f}, "
        f"warm fractions={WARM_FRACTIONS}, noise levels={NOISE_LEVELS}"
    )

    all_results: Dict[str, Any] = {
        "experiment": {
            "warm_fractions": WARM_FRACTIONS,
            "noise_levels": NOISE_LEVELS,
            "T": T,
            "k_eval": K_EVAL,
            "n_cold_users": len(cold_users),
            "n_warm_users_full": len(warm_users_full),
            "n_users_total": n_users_total,
            "mean_candidates_per_user": float(np.mean(pool_sizes)),
            "design": "subsample warm users, fixed cold eval set",
        },
    }

    for warm_frac in WARM_FRACTIONS:
        n_warm = min(int(n_users_total * warm_frac), len(warm_users_full))
        warm_subset = warm_users_full[:n_warm]

        frac_key = f"warm_{warm_frac}"
        all_results[frac_key] = {"n_warm_used": n_warm}

        print(f"\nWarm fraction = {warm_frac} ({n_warm} users)")
        t0 = time.time()
        policies = build_policies(ratings_by_user, warm_subset, item_emb, n_items)
        labels = [name for name, _ in policies]
        print(f"Policies built in {time.time() - t0:.1f}s")

        for noise_std in NOISE_LEVELS:
            for protocol, use_full_pool in [("exhaustive", False), ("non_exhaustive", True)]:
                pool_desc = "full pool" if use_full_pool else f"pool={T}"
                print(f"warm={warm_frac}, noise={noise_std}, {protocol} ({pool_desc})")

                env = ColdStartEnv(
                    ratings_by_user=ratings_by_user,
                    item_emb=item_emb,
                    config=config,
                    user_pool=cold_users,
                    user_emb=user_emb,
                    warm_users=set(warm_subset),
                    use_full_candidate_pool=use_full_pool,
                    reward_noise_std=noise_std,
                )

                metrics_by_policy: Dict[str, Dict[str, float]] = {}
                for name, policy in policies:
                    rewards = run_episodes(
                        policy, env, cold_users,
                        label=f"{name} warm={warm_frac}",
                    )
                    metrics_by_policy[name] = compute_metrics(rewards, k=K_EVAL)

                print_results(
                    f"warm={warm_frac}, noise={noise_std}, {protocol}",
                    metrics_by_policy,
                    labels,
                    K_EVAL,
                )

                noise_key = f"noise_{noise_std}"
                if noise_key not in all_results[frac_key]:
                    all_results[frac_key][noise_key] = {}
                all_results[frac_key][noise_key][protocol] = {
                    name: {"metrics": m} for name, m in metrics_by_policy.items()
                }

    policy_labels = ["HybridTS(CF)", "NLTS(content)", "GreedyCF", "Random"]
    print(f"\nNDCG@{K_EVAL} summary")
    for noise_std in NOISE_LEVELS:
        noise_key = f"noise_{noise_std}"
        for protocol in ["exhaustive", "non_exhaustive"]:
            print(f"{protocol}, noise={noise_std}")
            for warm_frac in WARM_FRACTIONS:
                frac_key = f"warm_{warm_frac}"
                n_warm = all_results[frac_key]["n_warm_used"]
                parts = [
                    f"{name}: {all_results[frac_key][noise_key][protocol][name]['metrics'][f'ndcg_{K_EVAL}']:.4f}"
                    for name in policy_labels
                ]
                print(f"warm={warm_frac} (n={n_warm}): {', '.join(parts)}")

    print("\nGreedyCF - HybridTS NDCG gap (positive = Greedy wins)")
    for noise_std in NOISE_LEVELS:
        noise_key = f"noise_{noise_std}"
        for protocol in ["exhaustive", "non_exhaustive"]:
            print(f"{protocol}, noise={noise_std}")
            for warm_frac in WARM_FRACTIONS:
                frac_key = f"warm_{warm_frac}"
                gcf_ndcg = all_results[frac_key][noise_key][protocol]["GreedyCF"]["metrics"][f"ndcg_{K_EVAL}"]
                hyb_ndcg = all_results[frac_key][noise_key][protocol]["HybridTS(CF)"]["metrics"][f"ndcg_{K_EVAL}"]
                gap = gcf_ndcg - hyb_ndcg
                winner = "GreedyCF" if gap > 0.001 else ("HybridTS" if gap < -0.001 else "tie")
                print(f"warm={warm_frac}: gap={gap:+.4f}, winner={winner}")

    out_path = _REPO_ROOT / "results" / out_name
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Results saved to {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--warm_fractions", type=float, nargs="+", default=None)
    p.add_argument("--out_name", type=str, default="prior_ablation_results.json")
    args = p.parse_args()
    main(warm_fractions=args.warm_fractions, out_name=args.out_name)
