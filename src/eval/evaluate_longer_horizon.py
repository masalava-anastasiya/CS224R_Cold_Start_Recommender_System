"""Longer-horizon evaluation: does exploration compound over more steps?

At T=20, Greedy CF dominates because the SVD prior is well-calibrated
and 20 steps is not enough for exploration to pay off. This script
extends the horizon to T_max (default 100) and records per-step rewards,
letting us see whether exploration methods catch up or overtake greedy
as their posteriors sharpen with more feedback.

Uses the selection protocol (full candidate pool, ~176 items per user)
so exploration has genuine value. Only cold users with >= T_max ratings
are included, ensuring every episode can run to completion.

Run from repo root:
    python -m src.eval.evaluate_longer_horizon
    python -m src.eval.evaluate_longer_horizon --t_max 100 --include_rl2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

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
from src.baselines.nonpersonalized_baseline import NonPersonalizedBaseline
from src.methods.constrained_bandit import ConstrainedLinearUCBBandit


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg_at_k(rewards: List[float], k: int) -> float:
    """Discounted cumulative gain at position k."""
    k = min(k, len(rewards))
    gains = np.asarray(rewards[:k], dtype=np.float64)
    discounts = np.log2(np.arange(2, k + 2))
    return float((gains / discounts).sum())


def ndcg_at_k(rewards: List[float], k: int) -> float:
    """Normalized DCG at position k."""
    ideal = sorted(rewards, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg_at_k(rewards, k) / ideal_dcg


def compute_metrics_at_horizon(all_step_rewards: List[List[float]], T: int, k: int = 5) -> Dict:
    """Compute summary metrics using only the first T steps of each episode."""
    clipped = [rewards[:T] for rewards in all_step_rewards]
    cumulative = [sum(r) for r in clipped]
    avg_per_step = [float(np.mean(r)) for r in clipped]
    ndcg_scores = [ndcg_at_k(r, k) for r in clipped]
    step1 = [r[0] if r else 0.0 for r in clipped]

    return {
        "T": T,
        "avg_cum_reward": float(np.mean(cumulative)),
        "std_cum_reward": float(np.std(cumulative)),
        "avg_reward_per_step": float(np.mean(avg_per_step)),
        "std_reward_per_step": float(np.std(avg_per_step)),
        f"ndcg_{k}": float(np.mean(ndcg_scores)),
        f"std_ndcg_{k}": float(np.std(ndcg_scores)),
        "avg_reward_step1": float(np.mean(step1)),
    }


def compute_per_step_curve(all_step_rewards: List[List[float]], t_max: int) -> Dict:
    """Compute mean and std of reward at each step t = 1..t_max."""
    step_means = []
    step_stds = []
    for t in range(t_max):
        rewards_at_t = [ep[t] for ep in all_step_rewards if len(ep) > t]
        step_means.append(float(np.mean(rewards_at_t)))
        step_stds.append(float(np.std(rewards_at_t)))
    return {"mean": step_means, "std": step_stds}


# ---------------------------------------------------------------------------
# Episode runner (records per-step rewards)
# ---------------------------------------------------------------------------

def run_episodes(policy, env, cold_users: List[int], label: str) -> List[List[float]]:
    """Run one episode per cold user, returning per-step reward lists."""
    all_step_rewards = []
    for user_idx in tqdm(cold_users, desc=label, ncols=72):
        state = env.reset(user_idx=user_idx)
        policy.reset(user_idx=user_idx)
        episode_rewards = []
        done = False
        while not done:
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            policy.update(action, reward, state, done)
            episode_rewards.append(reward)
        all_step_rewards.append(episode_rewards)
    return all_step_rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_max = args.t_max
    k_eval = 5

    # Checkpoint horizons for summary metrics
    checkpoints = [t for t in [5, 10, 20, 30, 50, 60, 70, 80, 90, 100] if t <= t_max]

    # --- Load data ---
    config = DataConfig()
    # Override the horizon so the environment runs for t_max steps
    config.cold_start_horizon_T = t_max
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
    all_cold_users = user_split["cold"]
    n_items = item_emb.shape[0]

    # Filter cold users to those with enough ratings for the full horizon
    eligible_cold_users = [
        u for u in all_cold_users
        if len(ratings_by_user[u]) >= t_max
    ]
    print(
        f"\nCold users with >= {t_max} ratings: "
        f"{len(eligible_cold_users)} / {len(all_cold_users)}"
    )
    if len(eligible_cold_users) == 0:
        print("No cold users have enough ratings for this horizon. Try a smaller --t_max.")
        sys.exit(1)

    # Report candidate pool stats for eligible users
    pool_sizes = [len(ratings_by_user[u]) for u in eligible_cold_users]
    print(
        f"Candidate pool stats: mean={np.mean(pool_sizes):.0f}, "
        f"median={np.median(pool_sizes):.0f}, "
        f"min={min(pool_sizes)}, max={max(pool_sizes)}"
    )

    # --- Build the environment (selection protocol, full candidate pool) ---
    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=eligible_cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=True,
    )

    # --- Build policies ---
    K_FACTORS = 50
    LAMBDA_NLTS = 50.0
    SIGMA_NLTS = 0.5

    print(f"\nFitting policies...")
    start_time = time.time()

    # Greedy CF
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=1.0,
    )

    # Hybrid Thompson Sampling (CF features)
    hybrid_ts = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )

    # Content-based NLTS
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

    # Constrained LinUCB
    mu_0_cb, _ = compute_warm_prior(ratings_by_user, warm_users, item_emb, reg=1.0)
    nonpers = NonPersonalizedBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        shrinkage=10.0,
    )
    constrained = ConstrainedLinearUCBBandit(
        item_emb=item_emb,
        baseline_policy=nonpers,
        lambda_reg=1.0,
        sigma2=1.0,
        beta=1.0,
        beta_safe=2.0,
        alpha=0.90,
        prior_mean=mu_0_cb,
        constraint_mode="cumulative",
    )

    # Random baseline
    random_bl = RandomBaseline(seed=42)

    print(f"Policies fitted in {time.time() - start_time:.1f}s")

    # Optionally load RL2 (trained at T=t_max)
    rl2_policy = None
    if args.include_rl2:
        rl2_path = Path(args.rl2_checkpoint).resolve()
        if not rl2_path.exists():
            print(f"\n  WARNING: RL2 checkpoint not found at {rl2_path}")
            print("  Skipping RL2. Train first with:")
            print(f"    python -m src.train.train_rl2 --explore --horizon {t_max} "
                  f"--n_epochs 50 --device mps --save_path results/rl2_checkpoint_t{t_max}.pt")
        else:
            from src.methods.rl2_policy import RL2Policy
            device = torch.device(args.device)
            ckpt = torch.load(rl2_path, map_location=device, weights_only=False)
            hparams = ckpt.get("hparams", {})
            hidden_dim = hparams.get("hidden_dim", 256)
            rl2_policy = RL2Policy(
                item_emb=item_emb.to(device),
                hidden_dim=hidden_dim,
            ).to(device)
            rl2_policy.load_state_dict(ckpt["state_dict"])
            rl2_policy.eval()
            trained_horizon = hparams.get("horizon", 20)
            print(f"\n  Loaded RL2 checkpoint: epoch {ckpt.get('epoch', '?')}, "
                  f"trained at T={trained_horizon}")

    # --- Build the list of (policy, label) pairs ---
    policies = [
        (greedy_cf, "GreedyCF"),
        (hybrid_ts, "HybridTS(CF)"),
        (nlts, "NLTS(content)"),
        (constrained, "Constrained"),
        (random_bl, "Random"),
    ]
    if rl2_policy is not None:
        policies.append((rl2_policy, "RL2"))

    # --- Run episodes ---
    print(f"\n{'=' * 80}")
    print(f"  LONGER HORIZON EVALUATION")
    print(f"  T_max = {t_max} steps | {len(eligible_cold_users)} cold users")
    print(f"  Selection protocol (full candidate pool)")
    print(f"{'=' * 80}\n")

    method_rewards = {}
    for policy, label in policies:
        start = time.time()
        step_rewards = run_episodes(
            policy, env, eligible_cold_users, label=f"{label:<14}"
        )
        elapsed = time.time() - start
        method_rewards[label] = step_rewards
        avg_cum = np.mean([sum(r) for r in step_rewards])
        print(f"  {label}: cum_reward={avg_cum:.2f}, {elapsed:.1f}s\n")

    # --- Compute and print results at each checkpoint ---
    print(f"\n{'=' * 100}")
    print(f"  RESULTS BY HORIZON")
    print(f"{'=' * 100}")

    method_labels = [label for _, label in policies]
    header = f"  {'T':<6}" + "".join(f"  {l:<16}" for l in method_labels)
    print(header)
    print("  " + "-" * (6 + 18 * len(method_labels)))

    # Table: avg reward per step at each checkpoint
    all_checkpoint_metrics = {}
    for T in checkpoints:
        row_parts = [f"  {T:<6}"]
        all_checkpoint_metrics[T] = {}
        for label in method_labels:
            metrics = compute_metrics_at_horizon(method_rewards[label], T, k=k_eval)
            all_checkpoint_metrics[T][label] = metrics
            row_parts.append(f"  {metrics['avg_reward_per_step']:.4f}          ")
        print("".join(row_parts))

    print("  " + "-" * (6 + 18 * len(method_labels)))
    print("  (Values shown: avg reward per step)")
    print(f"{'=' * 100}\n")

    # Also print NDCG@5 table
    print(f"  NDCG@5 at each horizon checkpoint:")
    print(f"  {'T':<6}" + "".join(f"  {l:<16}" for l in method_labels))
    print("  " + "-" * (6 + 18 * len(method_labels)))
    for T in checkpoints:
        row_parts = [f"  {T:<6}"]
        for label in method_labels:
            ndcg = all_checkpoint_metrics[T][label][f"ndcg_{k_eval}"]
            row_parts.append(f"  {ndcg:.4f}          ")
        print("".join(row_parts))
    print()

    # --- Per-step reward curves ---
    per_step_curves = {}
    for label in method_labels:
        per_step_curves[label] = compute_per_step_curve(
            method_rewards[label], t_max
        )

    # --- Save results ---
    results = {
        "experiment": {
            "t_max": t_max,
            "k_eval": k_eval,
            "n_cold_users": len(eligible_cold_users),
            "n_cold_users_total": len(all_cold_users),
            "n_warm_users": len(warm_users),
            "checkpoints": checkpoints,
            "methods": method_labels,
            "protocol": "selection (full candidate pool)",
        },
        "per_step_curves": per_step_curves,
        "checkpoint_metrics": {
            str(T): all_checkpoint_metrics[T] for T in checkpoints
        },
    }

    out_path = _REPO_ROOT / "results" / args.out_name
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved -> {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Longer-horizon evaluation: does exploration compound?"
    )
    parser.add_argument(
        "--t_max", type=int, default=100,
        help="Maximum episode horizon (default: 100).",
    )
    parser.add_argument(
        "--include_rl2", action="store_true",
        help="Include RL2 policy (requires a checkpoint trained at the target horizon).",
    )
    parser.add_argument(
        "--rl2_checkpoint", type=str,
        default=str(_REPO_ROOT / "results" / "rl2_checkpoint_t100.pt"),
        help="Path to RL2 checkpoint trained at the target horizon.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for RL2 inference (cpu / mps).",
    )
    parser.add_argument(
        "--out_name", type=str, default="longer_horizon_results.json",
        help="Output filename within results/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
