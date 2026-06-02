"""Evaluate the Constrained Linear UCB Bandit and compare to baselines.

Run from the repo root:
    python -m src.eval.evaluate_constrained_bandit

The script:
  1. Trains NonPersonalizedBaseline from warm-user ratings.
  2. Computes a warm-user prior (mu_0) using the same helper as NLTS.
  3. Builds ConstrainedLinearUCBBandit with the non-personalized fallback.
  4. Runs a single debug episode to show per-step constraint decisions.
  5. Evaluates all three freshly-run methods (Random, NonPersonalized,
     ConstrainedBandit) over all cold users.
  6. Loads Greedy-CF and NLTS numbers from saved JSON files if present.
  7. Prints a unified comparison table and saves JSON results.
"""

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
from src.baselines.nonpersonalized_baseline import NonPersonalizedBaseline
from src.methods.constrained_bandit import ConstrainedLinearUCBBandit


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _assert_no_leakage(warm_users, cold_users):
    assert set(warm_users).isdisjoint(set(cold_users)), (
        "Warm/cold user sets overlap — data leakage!"
    )


# ---------------------------------------------------------------------------
# Ranking metrics (shared across all eval scripts)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(
    policy,
    env: ColdStartEnv,
    cold_users: List[int],
    label: str,
) -> Tuple[List[List[float]], List[float]]:
    """Run one episode per cold user.

    Returns
    -------
    all_rewards : list of per-episode reward lists
    fallback_rates : list of per-episode fallback rates (empty if policy
                     has no fallback_rate attribute)
    """
    all_rewards: List[List[float]] = []
    fallback_rates: List[float] = []

    for user_idx in tqdm(cold_users, desc=label, ncols=72):
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
    cum   = [sum(r)              for r in all_rewards]
    step1 = [r[0] if r else 0.0 for r in all_rewards]
    avg_k = [float(np.mean(r[:k])) for r in all_rewards]
    ndcg  = [ndcg_at_k(r, k)    for r in all_rewards]
    hit   = [hit_at_k(r, k)     for r in all_rewards]
    avg   = [float(np.mean(r))  for r in all_rewards]
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
# Debug episode
# ---------------------------------------------------------------------------

def run_debug_episode(policy: ConstrainedLinearUCBBandit, env: ColdStartEnv, user_idx: int) -> None:
    state = env.reset(user_idx=user_idx)
    policy.reset()

    print(f"\nDebug episode — cold user {user_idx} | constraint: {policy.constraint_mode}")
    hdr = f"{'t':>3}  {'proposed':>9}  {'fallback':>9}  {'p_lcb':>8}  {'b_score':>8}  {'ok':>5}  {'chosen':>9}  {'reward':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    done = False
    t = 0
    while not done:
        diag   = policy.step_diagnostics(state)          # read-only
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        policy.update(action, reward, next_state, done)
        state = next_state
        t += 1
        print(
            f"{t:3d}  {diag['proposed_action']:9d}  {diag['fallback_action']:9d}  "
            f"{diag['proposed_lcb']:8.3f}  {diag['baseline_score']:8.3f}  "
            f"{'YES' if diag['feasible'] else 'NO ':>5}  "
            f"{action:9d}  {reward:7.1f}"
        )
    print(f"  Episode fallback rate: {policy.fallback_rate:.1%}")


# ---------------------------------------------------------------------------
# Load saved results for the comparison table
# ---------------------------------------------------------------------------

def _load_json_metrics(path: Path, model_key: str, k: int) -> Optional[Dict]:
    """Return the metrics dict from a saved result JSON, or None.

    Handles two formats:
      Flat:   {"model": "greedy_cf", "metrics": {...}}   (greedy_cf_results.json)
      Nested: {"greedy_cf": {"metrics": {...}}}          (neural_linear_results.json)
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # flat format
        if data.get("model") == model_key and "metrics" in data:
            return data["metrics"]
        # nested format
        if model_key in data and "metrics" in data[model_key]:
            return data[model_key]["metrics"]
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    _assert_no_leakage(warm_users, cold_users)

    # --- hyperparameters ---
    LAMBDA_REG       = 1.0
    SIGMA2           = 1.0
    BETA             = 1.0
    BETA_SAFE        = 2.0
    ALPHA            = 0.90
    CONSTRAINT_MODE  = "cumulative"
    SHRINKAGE        = 10.0
    K_EVAL           = 5

    print(
        f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | "
        f"Items: {n_items} | T: {config.cold_start_horizon_T} | "
        f"Reward: {config.reward_mode}"
    )

    # ------------------------------------------------------------------
    # Build baselines
    # ------------------------------------------------------------------
    print(f"\nBuilding NonPersonalizedBaseline (shrinkage={SHRINKAGE})...")
    nonpers = NonPersonalizedBaseline(
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
    print(f"Prior computed in {time.time() - t0:.1f}s  (||mu_0|| = {np.linalg.norm(mu_0):.4f})")

    random_bl = RandomBaseline(seed=42)

    # ------------------------------------------------------------------
    # Build constrained bandit — needs a fresh baseline instance per run
    # so the per-episode selected-set is managed correctly
    # ------------------------------------------------------------------
    nonpers_for_bandit = NonPersonalizedBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        shrinkage=SHRINKAGE,
    )
    bandit = ConstrainedLinearUCBBandit(
        item_emb=item_emb,
        baseline_policy=nonpers_for_bandit,
        lambda_reg=LAMBDA_REG,
        sigma2=SIGMA2,
        beta=BETA,
        beta_safe=BETA_SAFE,
        alpha=ALPHA,
        prior_mean=mu_0,
        constraint_mode=CONSTRAINT_MODE,
    )

    # ------------------------------------------------------------------
    # Shared environment
    # ------------------------------------------------------------------
    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
    )

    # ------------------------------------------------------------------
    # Debug episode (first cold user)
    # ------------------------------------------------------------------
    run_debug_episode(bandit, env, cold_users[0])

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    T = config.cold_start_horizon_T
    print(f"\nEvaluating on {len(cold_users)} cold users...")

    rnd_rewards,    _             = run_episodes(random_bl, env, cold_users, "Random     ")
    nonp_rewards,   _             = run_episodes(nonpers,   env, cold_users, "NonPers    ")
    bandit_rewards, bandit_fbrs   = run_episodes(bandit,    env, cold_users, "Constrained")

    rnd_m   = compute_metrics(rnd_rewards,    k=K_EVAL)
    nonp_m  = compute_metrics(nonp_rewards,   k=K_EVAL)
    bandit_m = compute_metrics(bandit_rewards, k=K_EVAL)
    bandit_m["fallback_rate"] = float(np.mean(bandit_fbrs)) if bandit_fbrs else 0.0

    # load previously saved results if available
    results_dir = _REPO_ROOT / "results"
    gcf_m  = _load_json_metrics(results_dir / "greedy_cf_results.json",      "greedy_cf",        K_EVAL)
    nlts_m = _load_json_metrics(results_dir / "neural_linear_results.json",  "neural_linear_ts", K_EVAL)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    def fmt(m: Optional[Dict], key: str, std_key: Optional[str] = None) -> str:
        if m is None:
            return "  —  "
        v = m.get(key)
        if v is None:
            return "  —  "
        s = m.get(std_key) if std_key else None
        if s is not None:
            return f"{v:.4f} ± {s:.4f}"
        return f"{v:.4f}"

    methods = [
        ("Random",       rnd_m,    False),
        ("NonPers",      nonp_m,   False),
        ("GreedyCF*",    gcf_m,    False),
        ("NLTS*",        nlts_m,   False),
        ("Constrained",  bandit_m, True),
    ]
    col_w = 24

    print("\n" + "=" * 80)
    print("  Constrained LinUCB Bandit — Comparison Table")
    print("=" * 80)
    print(
        f"  Cold users: {len(cold_users)}  |  T: {T}  |  "
        f"alpha={ALPHA}  beta={BETA}  beta_safe={BETA_SAFE}  "
        f"lambda={LAMBDA_REG}  mode={CONSTRAINT_MODE}"
    )
    print(f"  (* loaded from saved JSON)")
    print("=" * 80)

    hdr = f"  {'Metric':<22}"
    for name, _, _ in methods:
        hdr += f"  {name:<{col_w}}"
    print(hdr)
    print("  " + "-" * (22 + len(methods) * (col_w + 2)))

    def table_row(label, key, std_key=None):
        line = f"  {label:<22}"
        for _, m, _ in methods:
            line += f"  {fmt(m, key, std_key):<{col_w}}"
        print(line)

    table_row("Avg reward @ step 1",   "avg_reward_step1",   "std_reward_step1")
    table_row(f"Avg reward @ {K_EVAL}",     f"avg_reward_{K_EVAL}",  f"std_reward_{K_EVAL}")
    table_row("Avg reward / step",     "avg_reward_per_step", "std_reward_per_step")
    table_row(f"NDCG@{K_EVAL}",             f"ndcg_{K_EVAL}",        f"std_ndcg_{K_EVAL}")
    table_row(f"Hit@{K_EVAL}  (≥4)",        f"hit_{K_EVAL}")
    table_row("Avg cum. reward",       "avg_cum_reward",      "std_cum_reward")

    # fallback rate row — only constrained has this
    fb_line = f"  {'Fallback rate':<22}"
    for _, m, has_fb in methods:
        if has_fb and m is not None and "fallback_rate" in m:
            fb_line += f"  {m['fallback_rate']:.1%}{'':<{col_w - 6}}"
        else:
            fb_line += f"  {'—':<{col_w}}"
    print(fb_line)

    print("  " + "-" * (22 + len(methods) * (col_w + 2)))
    print(f"  (cum. reward is order-invariant when all {T} candidates are exhausted)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Monotonicity sanity checks (printed, not asserted)
    # ------------------------------------------------------------------
    print("\nSanity checks:")
    print(f"  bandit NDCG@5 ({bandit_m[f'ndcg_{K_EVAL}']:.4f}) >= random NDCG@5 ({rnd_m[f'ndcg_{K_EVAL}']:.4f})? "
          f"{'PASS' if bandit_m[f'ndcg_{K_EVAL}'] >= rnd_m[f'ndcg_{K_EVAL}'] else 'FAIL'}")
    print(f"  fallback_rate ({bandit_m['fallback_rate']:.1%}) > 0 at alpha={ALPHA}? "
          f"{'PASS' if bandit_m['fallback_rate'] > 0 else 'NOTE: all steps were feasible'}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = results_dir / "constrained_bandit_results.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "model": "constrained_linucb",
        "config": {
            "lambda_reg":      LAMBDA_REG,
            "sigma2":          SIGMA2,
            "beta":            BETA,
            "beta_safe":       BETA_SAFE,
            "alpha":           ALPHA,
            "constraint_mode": CONSTRAINT_MODE,
            "shrinkage":       SHRINKAGE,
            "reward_mode":     config.reward_mode,
            "T":               T,
            "n_cold_users":    len(cold_users),
            "n_warm_users":    len(warm_users),
        },
        "metrics": bandit_m,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
