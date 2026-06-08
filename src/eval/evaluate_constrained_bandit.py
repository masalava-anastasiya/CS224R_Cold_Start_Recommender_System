"""Evaluate the constrained LinUCB bandit."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import DataConfig
from src.data.env import ColdStartEnv
from src.methods.neural_linear_ts import compute_warm_prior
from src.baselines.random_baseline import RandomBaseline
from src.baselines.popularity_baseline import PopularityBaseline
from src.methods.constrained_bandit import ConstrainedLinearUCBBandit


def _assert_no_leakage(warm_users, cold_users):
    assert set(warm_users).isdisjoint(set(cold_users)), (
        "Warm/cold user sets overlap"
    )


def dcg_at_k(rewards: List[float], k: int) -> float:
    k = min(k, len(rewards))
    gains = np.asarray(rewards[:k], dtype=np.float64)
    return float((gains / np.log2(np.arange(2, k + 2))).sum())


def ndcg_at_k(rewards: List[float], k: int) -> float:
    ideal = sorted(rewards, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(rewards, k) / idcg if idcg > 0.0 else 0.0


def hit_at_k(rewards: List[float], k: int, threshold: float = 4.0) -> float:
    return float(any(r >= threshold for r in rewards[:k]))


def run_episodes(
    policy,
    env: ColdStartEnv,
    cold_users: List[int],
    label: str,
) -> Tuple[List[List[float]], List[float]]:
    all_rewards: List[List[float]] = []
    fallback_rates: List[float] = []

    for user_idx in tqdm(cold_users, desc=label):
        state = env.reset(user_idx=user_idx)
        policy.reset(user_idx=user_idx)
        episode_rewards: List[float] = []
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            policy.update(action, reward, next_state, done)
            state = next_state
            episode_rewards.append(reward)
        all_rewards.append(episode_rewards)
        if hasattr(policy, "fallback_rate"):
            fallback_rates.append(policy.fallback_rate)

    return all_rewards, fallback_rates


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


def run_debug_episode(policy: ConstrainedLinearUCBBandit, env: ColdStartEnv, user_idx: int) -> None:
    state = env.reset(user_idx=user_idx)
    policy.reset()

    print(f"\nDebug episode, user {user_idx}, mode={policy.constraint_mode}")
    done = False
    t = 0
    while not done:
        diag = policy.step_diagnostics(state)
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        policy.update(action, reward, next_state, done)
        state = next_state
        t += 1
        print(
            f"t={t}: proposed={diag['proposed_action']}, fallback={diag['fallback_action']}, "
            f"lcb={diag['proposed_lcb']:.3f}, baseline={diag['baseline_score']:.3f}, "
            f"feasible={diag['feasible']}, chosen={action}, reward={reward:.1f}"
        )
    print(f"Episode fallback rate: {policy.fallback_rate:.1%}")


def _load_json_metrics(path: Path, model_key: str, k: int) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("model") == model_key and "metrics" in data:
            return data["metrics"]
        if model_key in data and "metrics" in data[model_key]:
            return data[model_key]["metrics"]
        return None
    except Exception:
        return None


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

    _assert_no_leakage(warm_users, cold_users)

    LAMBDA_REG = 1.0
    SIGMA2 = 1.0
    BETA = 1.0
    BETA_SAFE = 2.0
    ALPHA = 0.90
    CONSTRAINT_MODE = "cumulative"
    SHRINKAGE = 10.0
    K_EVAL = 5

    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | "
        f"Items: {n_items} | T: {config.cold_start_horizon_T} | "
        f"Reward: {config.reward_mode}"
    )

    print(f"Building PopularityBaseline (shrinkage={SHRINKAGE})...")
    popularity = PopularityBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        shrinkage=SHRINKAGE,
    )

    print(f"Computing warm-user prior (lambda={LAMBDA_REG})...")
    t0 = time.time()
    mu_0, _ = compute_warm_prior(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        item_emb=item_emb,
        reg=LAMBDA_REG,
    )
    print(f"Prior computed in {time.time() - t0:.1f}s (||mu_0|| = {np.linalg.norm(mu_0):.4f})")

    random_bl = RandomBaseline(seed=42)

    popularity_for_bandit = PopularityBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        shrinkage=SHRINKAGE,
    )
    bandit = ConstrainedLinearUCBBandit(
        item_emb=item_emb,
        baseline_policy=popularity_for_bandit,
        lambda_reg=LAMBDA_REG,
        sigma2=SIGMA2,
        beta=BETA,
        beta_safe=BETA_SAFE,
        alpha=ALPHA,
        prior_mean=mu_0,
        constraint_mode=CONSTRAINT_MODE,
    )

    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
    )

    run_debug_episode(bandit, env, cold_users[0])

    T = config.cold_start_horizon_T
    print(f"Evaluating on {len(cold_users)} cold users...")

    rnd_rewards, _ = run_episodes(random_bl, env, cold_users, "Random")
    pop_rewards, _ = run_episodes(popularity, env, cold_users, "Popularity")
    bandit_rewards, bandit_fbrs = run_episodes(bandit, env, cold_users, "Constrained")

    rnd_m = compute_metrics(rnd_rewards, k=K_EVAL)
    pop_m = compute_metrics(pop_rewards, k=K_EVAL)
    bandit_m = compute_metrics(bandit_rewards, k=K_EVAL)
    bandit_m["fallback_rate"] = float(np.mean(bandit_fbrs)) if bandit_fbrs else 0.0

    results_dir = _REPO_ROOT / "results"
    gcf_m = _load_json_metrics(results_dir / "greedy_cf_results.json", "greedy_cf", K_EVAL)
    nlts_m = _load_json_metrics(results_dir / "neural_linear_results.json", "neural_linear_ts", K_EVAL)

    methods = [
        ("Random", rnd_m, False),
        ("Popularity", pop_m, False),
        ("GreedyCF*", gcf_m, False),
        ("NLTS*", nlts_m, False),
        ("Constrained", bandit_m, True),
    ]

    def fmt_val(m: Optional[Dict], key: str, std_key: Optional[str] = None) -> str:
        if m is None:
            return "—"
        v = m.get(key)
        if v is None:
            return "—"
        s = m.get(std_key) if std_key else None
        if s is not None:
            return f"{v:.4f} ± {s:.4f}"
        return f"{v:.4f}"

    def row(label, key, std_key=None):
        parts = [f"{name}: {fmt_val(m, key, std_key)}" for name, m, _ in methods]
        print(f"{label}: {', '.join(parts)}")

    print("\nConstrained LinUCB comparison")
    print(
        f"cold users: {len(cold_users)}, T: {T}, alpha: {ALPHA}, beta: {BETA}, "
        f"beta_safe: {BETA_SAFE}, lambda: {LAMBDA_REG}, mode: {CONSTRAINT_MODE}"
    )
    print("GreedyCF* and NLTS* loaded from saved JSON when available")
    row("Avg reward @ step 1", "avg_reward_step1", "std_reward_step1")
    row(f"Avg reward @ {K_EVAL}", f"avg_reward_{K_EVAL}", f"std_reward_{K_EVAL}")
    row("Avg reward / step", "avg_reward_per_step", "std_reward_per_step")
    row(f"NDCG@{K_EVAL}", f"ndcg_{K_EVAL}", f"std_ndcg_{K_EVAL}")
    row(f"Hit@{K_EVAL} (rating >= 4)", f"hit_{K_EVAL}")
    row("Avg cum. reward", "avg_cum_reward", "std_cum_reward")
    print(f"Fallback rate: Constrained {bandit_m['fallback_rate']:.1%}")

    out_path = results_dir / "constrained_bandit_results.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "model": "constrained_linucb",
        "config": {
            "lambda_reg": LAMBDA_REG,
            "sigma2": SIGMA2,
            "beta": BETA,
            "beta_safe": BETA_SAFE,
            "alpha": ALPHA,
            "constraint_mode": CONSTRAINT_MODE,
            "shrinkage": SHRINKAGE,
            "reward_mode": config.reward_mode,
            "T": T,
            "n_cold_users": len(cold_users),
            "n_warm_users": len(warm_users),
        },
        "metrics": bandit_m,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results saved to {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
