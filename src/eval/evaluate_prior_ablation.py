"""Prior quality ablation: when does exploration beat exploitation?

Varies the number of warm users available for training the prior (10-70%
of total users) while keeping the cold evaluation set fixed. Tests whether
Thompson Sampling overtakes Greedy CF when the prior degrades.

Evaluated under both exhaustive and non-exhaustive protocols, at noise
levels 0.0 and 0.5.

Run from repo root:
    python -m src.eval.evaluate_prior_ablation
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


# ── policy builders ──────────────────────────────────────────────────

def build_policies(
    ratings_by_user: Dict,
    warm_subset: List[int],
    item_emb: torch.Tensor,
    n_items: int,
) -> List[Tuple[str, Any]]:
    """Build all 4 policies using the given warm user subset."""

    mu_0, Lambda_0 = compute_warm_prior(
        ratings_by_user, warm_subset, item_emb, reg=50.0,
    )

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
        k=min(50, len(warm_subset) - 1),
        lambda_prior=1.0,
        sigma_noise=1.0,
    )

    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_subset,
        n_items=n_items,
        k=min(50, len(warm_subset) - 1),
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
    user_split      = torch.load(processed / "user_split.pt",      weights_only=False)
    item_emb        = torch.load(processed / "item_emb.pt",        weights_only=False)
    user_emb        = torch.load(processed / "user_emb.pt",        weights_only=False)

    warm_users_full = user_split["warm"]   # 3936 users (70%)
    cold_users      = user_split["cold"]   # 1688 users (30%) — always the eval set
    n_users_total   = len(warm_users_full) + len(cold_users)
    n_items         = item_emb.shape[0]
    T = config.cold_start_horizon_T
    K_EVAL = 5

    WARM_FRACTIONS = warm_fractions if warm_fractions is not None else [0.10, 0.20, 0.30, 0.50, 0.70]
    NOISE_LEVELS = [0.0, 0.5]

    pool_sizes = [len(ratings_by_user[u]) for u in cold_users]
    print(
        f"Total users: {n_users_total} | Cold (fixed eval set): {len(cold_users)} | "
        f"Items: {n_items} | T: {T}\n"
        f"Full candidate pool: mean={np.mean(pool_sizes):.0f}\n"
        f"Warm fractions to test: {WARM_FRACTIONS}\n"
        f"Noise levels: {NOISE_LEVELS}"
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
            "design": "subsample warm users while keeping cold eval set fixed",
        },
    }

    for warm_frac in WARM_FRACTIONS:
        n_warm = int(n_users_total * warm_frac)
        n_warm = min(n_warm, len(warm_users_full))
        warm_subset = warm_users_full[:n_warm]

        frac_key = f"warm_{warm_frac}"
        all_results[frac_key] = {"n_warm_used": n_warm}

        print(f"\n{'#' * 100}")
        print(f"  WARM FRACTION = {warm_frac} ({n_warm} warm users)")
        print(f"{'#' * 100}")

        print(f"  Building policies with {n_warm} warm users...")
        t0 = time.time()
        policies = build_policies(ratings_by_user, warm_subset, item_emb, n_items)
        labels = [label for label, _ in policies]
        print(f"  Policies built in {time.time() - t0:.1f}s")

        for noise_std in NOISE_LEVELS:
            for protocol, use_full_pool in [("exhaustive", False), ("non_exhaustive", True)]:
                protocol_label = "NON-EXHAUSTIVE" if use_full_pool else "EXHAUSTIVE"
                desc = f"warm={warm_frac} noise={noise_std} {protocol_label}"
                print(f"\n  --- {desc} ---")

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
                for label, policy in policies:
                    rewards = run_episodes(
                        policy, env, cold_users,
                        label=f"  {label:<14} w={warm_frac}",
                    )
                    metrics_by_policy[label] = compute_metrics(rewards, k=K_EVAL)

                cand_desc = f"~{int(np.mean(pool_sizes))}" if use_full_pool else str(T)
                print_table(
                    f"{protocol_label} | warm_frac={warm_frac} ({n_warm} users) | "
                    f"noise={noise_std} | T={T} from {cand_desc} candidates",
                    metrics_by_policy,
                    labels,
                    K_EVAL,
                )

                noise_key = f"noise_{noise_std}"
                if noise_key not in all_results[frac_key]:
                    all_results[frac_key][noise_key] = {}
                all_results[frac_key][noise_key][protocol] = {
                    label: {"metrics": m} for label, m in metrics_by_policy.items()
                }

    # ── summary tables ───────────────────────────────────────────────

    policy_labels = ["HybridTS(CF)", "NLTS(content)", "GreedyCF", "Random"]

    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY: NDCG@{K_EVAL} across warm fractions")
    print(f"{'=' * 100}")

    for noise_std in NOISE_LEVELS:
        noise_key = f"noise_{noise_std}"
        for protocol in ["exhaustive", "non_exhaustive"]:
            protocol_label = "NON-EXHAUSTIVE" if protocol == "non_exhaustive" else "EXHAUSTIVE"
            print(f"\n  {protocol_label} | noise={noise_std}:")
            header = f"  {'warm_frac':<12} {'n_warm':<8}" + "".join(f" {l:<18}" for l in policy_labels)
            print(header)
            print("  " + "-" * 92)
            for warm_frac in WARM_FRACTIONS:
                frac_key = f"warm_{warm_frac}"
                n_warm = all_results[frac_key]["n_warm_used"]
                parts = []
                for label in policy_labels:
                    val = all_results[frac_key][noise_key][protocol][label]["metrics"][f"ndcg_{K_EVAL}"]
                    parts.append(f"{val:.4f}")
                line = f"  {warm_frac:<12} {n_warm:<8}" + "".join(f" {p:<18}" for p in parts)
                print(line)
            print("  " + "-" * 92)

    # ── gap analysis ─────────────────────────────────────────────────

    print(f"\n\n{'=' * 100}")
    print(f"  GAP ANALYSIS: GreedyCF minus HybridTS(CF) NDCG@{K_EVAL}")
    print(f"  (positive = Greedy wins, negative = HybridTS wins)")
    print(f"{'=' * 100}")

    for noise_std in NOISE_LEVELS:
        noise_key = f"noise_{noise_std}"
        for protocol in ["exhaustive", "non_exhaustive"]:
            protocol_label = "NON-EXHAUSTIVE" if protocol == "non_exhaustive" else "EXHAUSTIVE"
            print(f"\n  {protocol_label} | noise={noise_std}:")
            print(f"  {'warm_frac':<12} {'n_warm':<8} {'gap':<12} {'winner'}")
            print("  " + "-" * 50)
            for warm_frac in WARM_FRACTIONS:
                frac_key = f"warm_{warm_frac}"
                n_warm = all_results[frac_key]["n_warm_used"]
                gcf_ndcg = all_results[frac_key][noise_key][protocol]["GreedyCF"]["metrics"][f"ndcg_{K_EVAL}"]
                hyb_ndcg = all_results[frac_key][noise_key][protocol]["HybridTS(CF)"]["metrics"][f"ndcg_{K_EVAL}"]
                gap = gcf_ndcg - hyb_ndcg
                winner = "GreedyCF" if gap > 0.001 else ("HybridTS" if gap < -0.001 else "~TIE")
                print(f"  {warm_frac:<12} {n_warm:<8} {gap:+.4f}      {winner}")
            print("  " + "-" * 50)

    print()

    # ── save ─────────────────────────────────────────────────────────

    out_path = _REPO_ROOT / "results" / out_name
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Results saved -> {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Prior quality ablation.")
    p.add_argument(
        "--warm_fractions", type=float, nargs="+", default=None,
        help="Warm fractions to test (default: 0.1 0.2 0.3 0.5 0.7).",
    )
    p.add_argument(
        "--out_name", type=str, default="prior_ablation_results.json",
        help="Output JSON filename under results/.",
    )
    args = p.parse_args()
    main(warm_fractions=args.warm_fractions, out_name=args.out_name)
