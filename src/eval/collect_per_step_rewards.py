"""Collect per-step rewards for explore-mode evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def run_episodes(policy, env, cold_users, label: str):
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


def main(args):
    config = DataConfig()
    processed = Path(config.data_dir) / "processed"

    print("Loading artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split = torch.load(processed / "user_split.pt", weights_only=False)
    item_emb = torch.load(processed / "item_emb.pt", weights_only=False)
    user_emb = torch.load(processed / "user_emb.pt", weights_only=False)

    warm_users = user_split["warm"]
    cold_users = user_split["cold"]

    device = torch.device(args.device)
    item_emb = item_emb.to(device)

    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=True,
    )

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hparams", {}).get("hidden_dim", 256)
    rl2 = RL2Policy(item_emb=item_emb, hidden_dim=hidden_dim).to(device)
    rl2.load_state_dict(ckpt["state_dict"])
    rl2.eval()
    print(f"Loaded RL2 checkpoint: epoch {ckpt.get('epoch')}, hidden={hidden_dim}")

    K_FACTORS = 50
    LAMBDA_PRIOR = 50.0
    SIGMA_NOISE = 0.5
    print("Fitting comparison policies...")
    t0 = time.time()
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=item_emb.shape[0],
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=item_emb.shape[0],
        k=K_FACTORS,
        reg=1.0,
    )
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
    random_bl = RandomBaseline(seed=42)
    print(f"Done in {time.time() - t0:.1f}s")

    results = {
        "RL²": run_episodes(rl2, env, cold_users, "RL2"),
        "Hybrid TS": run_episodes(hybrid, env, cold_users, "Hybrid TS"),
        "NLTS": run_episodes(nlts, env, cold_users, "NLTS"),
        "Greedy CF": run_episodes(greedy_cf, env, cold_users, "Greedy CF"),
        "Random": run_episodes(random_bl, env, cold_users, "Random"),
    }

    out_path = _REPO_ROOT / "results" / "per_step_rewards_explore.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh)
    print(f"Saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / "results" / "rl2_explore_checkpoint.pt"),
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
