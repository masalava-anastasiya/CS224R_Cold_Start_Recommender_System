"""Meta-train the RL2 policy on warm users."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.config import DataConfig
from src.data.env import ColdStartEnv
from src.methods.rl2_policy import RL2Policy


def run_episode(
    policy: RL2Policy,
    env: ColdStartEnv,
    user_idx: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    state = env.reset(user_idx=user_idx)
    hidden = policy.init_hidden()
    log_probs, entropies, rewards = [], [], []
    done = False

    while not done:
        action, lp, ent, hidden = policy.forward_train(state, hidden)
        state, reward, done, _ = env.step(action)
        log_probs.append(lp)
        entropies.append(ent)
        rewards.append(reward)

    return log_probs, entropies, rewards


def compute_returns(
    rewards: List[float],
    gamma: float,
    dcg_shaping: bool = False,
) -> List[float]:
    if dcg_shaping:
        return [r / math.log2(t + 2) for t, r in enumerate(rewards)]

    T = len(rewards)
    G: List[float] = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def train(args: argparse.Namespace) -> None:
    config = DataConfig()
    processed = Path(config.data_dir) / "processed"

    if args.horizon is not None:
        config.cold_start_horizon_T = args.horizon

    print("Loading artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split = torch.load(processed / "user_split.pt", weights_only=False)
    item_emb = torch.load(processed / "item_emb.pt", weights_only=False)

    warm_users = user_split["warm"]
    n_items = item_emb.shape[0]
    T = config.cold_start_horizon_T

    if args.horizon is not None and args.explore:
        original_count = len(warm_users)
        warm_users = [u for u in warm_users if len(ratings_by_user[u]) >= T]
        print(
            f"filtered warm users for T={T}: "
            f"{original_count} -> {len(warm_users)} (need >={T} ratings)"
        )

    print(
        f"warm={len(warm_users)}, items={n_items}, T={T}, "
        f"reward_mode={config.reward_mode}"
    )

    device = torch.device(args.device)
    item_emb = item_emb.to(device)

    train_env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=warm_users,
        rng=np.random.default_rng(config.random_seed),
        use_full_candidate_pool=args.explore,
    )

    policy = RL2Policy(
        item_emb=item_emb,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rng = np.random.default_rng(config.random_seed)
    history_log = []

    if args.explore and args.dcg_shaping:
        reward_mode = "dcg+explore"
    elif args.explore:
        reward_mode = "explore"
    elif args.dcg_shaping:
        reward_mode = "dcg"
    else:
        reward_mode = "cumulative"

    print(
        f"training RL2: hidden={args.hidden_dim}, lr={args.lr}, "
        f"batch={args.batch_size}, entropy={args.entropy_coef}, "
        f"epochs={args.n_epochs}, returns={reward_mode}"
    )

    for epoch in range(1, args.n_epochs + 1):
        epoch_start = time.time()
        policy.train()

        shuffled = rng.permutation(warm_users).tolist()
        batches = [
            shuffled[i:i + args.batch_size]
            for i in range(0, len(shuffled), args.batch_size)
        ]

        epoch_losses, epoch_rewards = [], []

        for batch in tqdm(batches, desc=f"epoch {epoch}/{args.n_epochs}", leave=False):
            all_lps: List[torch.Tensor] = []
            all_ents: List[torch.Tensor] = []
            all_G: List[float] = []
            batch_cum: List[float] = []

            for user_idx in batch:
                lps, ents, rwds = run_episode(policy, train_env, int(user_idx))
                G = compute_returns(rwds, args.gamma, dcg_shaping=args.dcg_shaping)
                all_lps.extend(lps)
                all_ents.extend(ents)
                all_G.extend(G)
                batch_cum.append(sum(rwds))

            G_arr = np.array(all_G, dtype=np.float32)
            G_arr = (G_arr - G_arr.mean()) / (G_arr.std() + 1e-8)
            G_t = torch.tensor(G_arr, dtype=torch.float32, device=device)

            pg_loss = -(G_t * torch.stack(all_lps)).mean()
            ent_loss = -args.entropy_coef * torch.stack(all_ents).mean()
            loss = pg_loss + ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_rewards.extend(batch_cum)

        avg_loss = float(np.mean(epoch_losses))
        avg_reward = float(np.mean(epoch_rewards))
        elapsed = time.time() - epoch_start

        log_entry = {
            "epoch": epoch,
            "avg_loss": round(avg_loss, 4),
            "avg_cum_reward": round(avg_reward, 4),
            "elapsed_s": round(elapsed, 1),
        }
        history_log.append(log_entry)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch}/{args.n_epochs}, loss={avg_loss:.4f}, "
                f"avg_cum_reward={avg_reward:.3f}, {elapsed:.1f}s"
            )

        if epoch % args.save_every == 0 or epoch == args.n_epochs:
            _save_checkpoint(policy, optimizer, args, epoch, avg_reward)

    _save_checkpoint(policy, optimizer, args, args.n_epochs, avg_reward)

    curve_path = Path(args.save_path).resolve().parent / "rl2_training_curve_explore.json"
    with open(curve_path, "w") as fh:
        json.dump(history_log, fh, indent=2)
    print(f"wrote {curve_path}")


def _save_checkpoint(
    policy: RL2Policy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    avg_reward: float,
) -> None:
    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hparams": {
                "hidden_dim": args.hidden_dim,
                "horizon": args.horizon,
                "gamma": args.gamma,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "entropy_coef": args.entropy_coef,
                "n_epochs": args.n_epochs,
                "dcg_shaping": args.dcg_shaping,
                "explore": args.explore,
            },
            "avg_cum_reward": avg_reward,
        },
        save_path,
    )
    print(
        f"saved {save_path.relative_to(_REPO_ROOT)} "
        f"(epoch {epoch}, reward={avg_reward:.3f})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no_dcg_shaping", dest="dcg_shaping", action="store_false")
    p.set_defaults(dcg_shaping=True)
    p.add_argument("--explore", action="store_true")
    p.add_argument(
        "--save_path",
        type=str,
        default=str(_REPO_ROOT / "results" / "rl2_checkpoint.pt"),
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
