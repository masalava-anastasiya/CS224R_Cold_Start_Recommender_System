"""Longer-horizon evaluation with a full candidate pool."""

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
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg_at_k(rewards, k) / ideal_dcg


def compute_metrics_at_horizon(all_step_rewards: List[List[float]], T: int, k: int = 5) -> Dict:
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
    step_means = []
    step_stds = []
    for t in range(t_max):
        rewards_at_t = [ep[t] for ep in all_step_rewards if len(ep) > t]
        step_means.append(float(np.mean(rewards_at_t)))
        step_stds.append(float(np.std(rewards_at_t)))
    return {"mean": step_means, "std": step_stds}


def run_episodes(policy, env, cold_users: List[int], label: str) -> List[List[float]]:
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


def main() -> None:
    args = parse_args()
    t_max = args.t_max
    k_eval = 5

    checkpoints = [t for t in [5, 10, 20, 30, 50, 60, 70, 80, 90, 100] if t <= t_max]

    config = DataConfig()
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

    eligible_cold_users = [
        u for u in all_cold_users
        if len(ratings_by_user[u]) >= t_max
    ]
    print(f"Cold users with >= {t_max} ratings: {len(eligible_cold_users)} / {len(all_cold_users)}")
    if len(eligible_cold_users) == 0:
        print("No cold users have enough ratings for this horizon. Try a smaller --t_max.")
        sys.exit(1)

    pool_sizes = [len(ratings_by_user[u]) for u in eligible_cold_users]
    print(
        f"Candidate pool: mean={np.mean(pool_sizes):.0f}, "
        f"median={np.median(pool_sizes):.0f}, min={min(pool_sizes)}, max={max(pool_sizes)}"
    )

    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=eligible_cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=True,
    )

    K_FACTORS = 50
    LAMBDA_NLTS = 50.0
    SIGMA_NLTS = 0.5

    print("Fitting policies...")
    start_time = time.time()

    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=1.0,
    )

    hybrid_ts = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )

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

    mu_0_cb, _ = compute_warm_prior(ratings_by_user, warm_users, item_emb, reg=1.0)
    popularity = PopularityBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        shrinkage=10.0,
    )
    constrained = ConstrainedLinearUCBBandit(
        item_emb=item_emb,
        baseline_policy=popularity,
        lambda_reg=1.0,
        sigma2=1.0,
        beta=1.0,
        beta_safe=2.0,
        alpha=0.90,
        prior_mean=mu_0_cb,
        constraint_mode="cumulative",
    )

    random_bl = RandomBaseline(seed=42)

    print(f"Policies fitted in {time.time() - start_time:.1f}s")

    rl2_policy = None
    if args.include_rl2:
        rl2_path = Path(args.rl2_checkpoint).resolve()
        if not rl2_path.exists():
            print(f"RL2 checkpoint not found: {rl2_path}")
            print(f"Train with: python -m src.train.train_rl2 --explore --horizon {t_max}")
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
            print(f"Loaded RL2 checkpoint: epoch {ckpt.get('epoch', '?')}, trained at T={trained_horizon}")

    policies = [
        (greedy_cf, "GreedyCF"),
        (hybrid_ts, "HybridTS(CF)"),
        (nlts, "NLTS(content)"),
        (constrained, "Constrained"),
        (random_bl, "Random"),
    ]
    if rl2_policy is not None:
        policies.append((rl2_policy, "RL2"))

    print(f"\nLonger horizon eval: T_max={t_max}, users={len(eligible_cold_users)}, full candidate pool")

    method_rewards = {}
    for policy, label in policies:
        start = time.time()
        step_rewards = run_episodes(policy, env, eligible_cold_users, label=label)
        elapsed = time.time() - start
        method_rewards[label] = step_rewards
        avg_cum = np.mean([sum(r) for r in step_rewards])
        print(f"{label}: cum_reward={avg_cum:.2f}, {elapsed:.1f}s")

    method_labels = [label for _, label in policies]
    all_checkpoint_metrics = {}

    print("\nAvg reward per step by horizon")
    for T in checkpoints:
        parts = []
        for label in method_labels:
            metrics = compute_metrics_at_horizon(method_rewards[label], T, k=k_eval)
            all_checkpoint_metrics.setdefault(T, {})[label] = metrics
            parts.append(f"{label}: {metrics['avg_reward_per_step']:.4f}")
        print(f"T={T}: {', '.join(parts)}")

    print("\nNDCG@5 by horizon")
    for T in checkpoints:
        parts = []
        for label in method_labels:
            ndcg = all_checkpoint_metrics[T][label][f"ndcg_{k_eval}"]
            parts.append(f"{label}: {ndcg:.4f}")
        print(f"T={T}: {', '.join(parts)}")

    per_step_curves = {}
    for label in method_labels:
        per_step_curves[label] = compute_per_step_curve(method_rewards[label], t_max)

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
    print(f"Results saved to {out_path.relative_to(_REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--include_rl2", action="store_true")
    parser.add_argument(
        "--rl2_checkpoint",
        type=str,
        default=str(_REPO_ROOT / "results" / "rl2_checkpoint_t100.pt"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_name", type=str, default="longer_horizon_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
