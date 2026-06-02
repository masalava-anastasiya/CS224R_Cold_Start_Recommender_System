"""Evaluate the trained RL² policy on cold users.

Compares RL² against HybridTS, GreedyCF, and Random baselines using the
same metrics and environment as the other evaluation scripts.

Run from repo root:
    python -m src.eval.evaluate_rl2 [--checkpoint results/rl2_checkpoint.pt]
                                     [--device cpu]
                                     [--explore]
"""

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
from src.methods.rl2_policy import RL2Policy
from src.methods.neural_linear_ts import NeuralLinearTS, compute_warm_prior
from src.methods.hybrid_neural_linear_ts import HybridNeuralLinearTS
from src.baselines.greedy_cf import GreedyCFBaseline
from src.baselines.random_baseline import RandomBaseline


# ---------------------------------------------------------------------------
# Metrics  (identical to other eval scripts)
# ---------------------------------------------------------------------------

def dcg_at_k(rewards: List[float], k: int) -> float:
    k = min(k, len(rewards))
    gains     = np.asarray(rewards[:k], dtype=np.float64)
    discounts = np.log2(np.arange(2, k + 2))
    return float((gains / discounts).sum())


def ndcg_at_k(rewards: List[float], k: int) -> float:
    ideal = sorted(rewards, reverse=True)
    idcg  = dcg_at_k(ideal, k)
    return dcg_at_k(rewards, k) / idcg if idcg > 0.0 else 0.0


def hit_at_k(rewards: List[float], k: int, threshold: float = 4.0) -> float:
    return float(any(r >= threshold for r in rewards[:k]))


def run_episodes(policy, env, cold_users, label: str) -> List[List[float]]:
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


def compute_metrics(all_rewards: List[List[float]], k: int) -> Dict:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    config    = DataConfig()
    processed = Path(config.data_dir) / "processed"

    print("Loading artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split      = torch.load(processed / "user_split.pt",      weights_only=False)
    item_emb        = torch.load(processed / "item_emb.pt",        weights_only=False)
    user_emb        = torch.load(processed / "user_emb.pt",        weights_only=False)

    warm_users = user_split["warm"]
    cold_users = user_split["cold"]
    n_items    = item_emb.shape[0]
    T          = config.cold_start_horizon_T

    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | "
        f"Items: {n_items} | T: {T} | Reward: {config.reward_mode}"
        + (" | EXPLORE mode" if args.explore else "")
    )

    device   = torch.device(args.device)
    item_emb = item_emb.to(device)

    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=args.explore,
    )

    # ---- Load RL² checkpoint ----
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(
            f"Checkpoint not found at {ckpt_path}.\n"
            "Train first with:  python -m src.train.train_rl2"
        )
        sys.exit(1)

    print(f"\nLoading RL² checkpoint from {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hparams", {})
    hidden_dim = hparams.get("hidden_dim", 256)

    rl2 = RL2Policy(item_emb=item_emb, hidden_dim=hidden_dim).to(device)
    rl2.load_state_dict(ckpt["state_dict"])
    rl2.eval()
    print(
        f"  Loaded epoch {ckpt.get('epoch', '?')} | "
        f"train avg_cum_reward={ckpt.get('avg_cum_reward', '?'):.3f} | "
        f"hidden_dim={hidden_dim}"
    )

    # ---- Build comparison policies ----
    K_FACTORS = 50
    LAMBDA_PRIOR = 50.0
    SIGMA_NOISE  = 0.5
    print(f"\nFitting comparison policies (k={K_FACTORS})...")

    t0 = time.time()
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=1.0,
    )
    random_bl = RandomBaseline(seed=42)

    mu_0, Lambda_0 = compute_warm_prior(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        item_emb=item_emb.cpu(),
        reg=LAMBDA_PRIOR,
    )
    nlts = NeuralLinearTS(
        item_emb=item_emb.cpu(),
        lambda_prior=LAMBDA_PRIOR,
        sigma_noise=SIGMA_NOISE,
        mu_0=mu_0,
        Lambda_0=Lambda_0,
    )
    print(f"  Done in {time.time() - t0:.1f}s.")

    # ---- Run episodes ----
    K_EVAL = 5
    print()
    rl2_rewards  = run_episodes(rl2,       env, cold_users, label="RL²       ")
    nlts_rewards = run_episodes(nlts,      env, cold_users, label="NLTS      ")
    hyb_rewards  = run_episodes(hybrid,    env, cold_users, label="HybridTS  ")
    gcf_rewards  = run_episodes(greedy_cf, env, cold_users, label="GreedyCF  ")
    rnd_rewards  = run_episodes(random_bl, env, cold_users, label="Random    ")

    rl2_m  = compute_metrics(rl2_rewards,  k=K_EVAL)
    nlts_m = compute_metrics(nlts_rewards, k=K_EVAL)
    hyb_m  = compute_metrics(hyb_rewards,  k=K_EVAL)
    gcf_m  = compute_metrics(gcf_rewards,  k=K_EVAL)
    rnd_m  = compute_metrics(rnd_rewards,  k=K_EVAL)

    # ---- Print table ----
    def row(label, key, fmt=".4f", show_std=True):
        std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
        cols = []
        for m in [rl2_m, nlts_m, hyb_m, gcf_m, rnd_m]:
            if show_std and std_key in m:
                cols.append(f"{m[key]:{fmt}} ± {m[std_key]:{fmt}}")
            else:
                cols.append(f"{m[key]:{fmt}}")
        print(f"  {label:<24} {cols[0]:<24} {cols[1]:<24} {cols[2]:<24} {cols[3]:<24} {cols[4]}")

    print("\n" + "=" * 130)
    print("  RL² vs NLTS vs Hybrid TS vs Greedy CF vs Random")
    print("=" * 130)
    print(f"  {'Metric':<24} {'RL²':<24} {'NLTS':<24} {'Hybrid TS':<24} {'Greedy CF':<24} Random")
    print("  " + "-" * 126)
    row("Avg reward @ step 1",    "avg_reward_step1")
    row(f"Avg reward @ {K_EVAL}", f"avg_reward_{K_EVAL}")
    row("Avg reward / step",      "avg_reward_per_step")
    row(f"NDCG@{K_EVAL}",         f"ndcg_{K_EVAL}")
    row(f"Hit@{K_EVAL}  (≥ 4)",   f"hit_{K_EVAL}", show_std=False)
    row("Avg cum. reward",        "avg_cum_reward")
    print("  " + "-" * 126)
    if not args.explore:
        print(f"  (* order-invariant in exhaustive mode: all {T} candidates always selected)")
    print("=" * 130)

    # ---- Save results ----
    suffix = "_explore" if args.explore else ""
    out_path = _REPO_ROOT / "results" / f"rl2_results{suffix}.json"
    out_path.parent.mkdir(exist_ok=True)

    payload = {
        "rl2": {
            "config": {
                **hparams,
                "reward_mode": config.reward_mode,
                "T": T,
                "n_cold_users": len(cold_users),
                "n_warm_users": len(warm_users),
                "explore_mode": args.explore,
                "checkpoint_epoch": ckpt.get("epoch"),
            },
            "metrics": rl2_m,
        },
        "nlts":      {"metrics": nlts_m},
        "hybrid_ts": {"metrics": hyb_m},
        "greedy_cf": {"metrics": gcf_m},
        "random":    {"metrics": rnd_m},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the trained RL² policy.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / "results" / "rl2_checkpoint.pt"),
        help="Path to saved RL² checkpoint.",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu / mps / cuda).",
    )
    p.add_argument(
        "--explore", action="store_true",
        help="Use full candidate pool (exploration mode, T=20 from ~176 candidates).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
