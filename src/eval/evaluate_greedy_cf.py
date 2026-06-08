"""Evaluate Greedy CF against Random."""

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
    for user_idx in tqdm(cold_users, desc=label):
        state = env.reset(user_idx=user_idx)
        policy.reset()
        episode_rewards: List[float] = []
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            policy.update(action, reward, next_state, done)
            state = next_state
            episode_rewards.append(reward)
        all_rewards.append(episode_rewards)
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

    K_FACTORS = 50
    REG = 1.0
    K_EVAL = 5

    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | "
        f"Items: {n_items} | T: {config.cold_start_horizon_T} | "
        f"Reward: {config.reward_mode}"
    )

    print(f"Fitting GreedyCFBaseline (k={K_FACTORS}, reg={REG})...")
    t0 = time.time()
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=REG,
    )
    print(f"Fit done in {time.time() - t0:.1f}s")

    random_bl = RandomBaseline(seed=42)

    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
    )

    gcf_rewards = run_episodes(greedy_cf, env, cold_users, label="GreedyCF")
    rnd_rewards = run_episodes(random_bl, env, cold_users, label="Random")

    gcf_m = compute_metrics(gcf_rewards, k=K_EVAL)
    rnd_m = compute_metrics(rnd_rewards, k=K_EVAL)

    T = config.cold_start_horizon_T

    def row(label, key, fmt=".4f", show_std=True):
        std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
        if show_std and std_key in gcf_m:
            gcf_str = f"{gcf_m[key]:{fmt}} ± {gcf_m[std_key]:{fmt}}"
            rnd_str = f"{rnd_m[key]:{fmt}} ± {rnd_m[std_key]:{fmt}}"
        else:
            gcf_str = f"{gcf_m[key]:{fmt}}"
            rnd_str = f"{rnd_m[key]:{fmt}}"
        print(f"{label}: Greedy CF {gcf_str}, Random {rnd_str}")

    print("\nGreedy CF vs Random")
    print(f"cold users: {len(cold_users)}, T: {T}, k: {K_FACTORS}, reg: {REG}, eval@{K_EVAL}")
    row("Avg reward @ step 1", "avg_reward_step1")
    row(f"Avg reward @ {K_EVAL}", f"avg_reward_{K_EVAL}")
    row("Avg reward / step", "avg_reward_per_step")
    row(f"NDCG@{K_EVAL}", f"ndcg_{K_EVAL}")
    row(f"Hit@{K_EVAL} (rating >= 4)", f"hit_{K_EVAL}", show_std=False)
    row("Avg cum. reward", "avg_cum_reward")

    out_path = _REPO_ROOT / "results" / "greedy_cf_results.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "greedy_cf": {
            "config": {
                "k": K_FACTORS,
                "reg": REG,
                "reward_mode": config.reward_mode,
                "T": T,
                "n_cold_users": len(cold_users),
                "n_warm_users": len(warm_users),
            },
            "metrics": gcf_m,
        },
        "random": {
            "config": {"seed": 42, "reward_mode": config.reward_mode, "T": T},
            "metrics": rnd_m,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results saved to {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
