"""Method #1: does a WRONG prior let exploration beat exploitation?

Real-world cold start is not just "few warm users" --- it is a warm
population that does not match new users. Early adopters skew young/tech;
new sign-ups are broader. A prior trained on the warm crowd is then
systematically wrong for the newcomer.

We simulate this with a demographic mismatch. We split users by age into
YOUNG (codes 1, 18) and OLD (codes 50, 56), then build the prior from one
group and evaluate on cold users from the other:

  MATCHED    prior trained on the SAME age group as the eval users
  MISMATCHED prior trained on the OPPOSITE age group

Both warm subsets are capped to the same size (CAP), so the only thing
that changes is WHO the prior was trained on --- not how many. Our prior
ablation already showed that a random subset of this size keeps Greedy CF
ahead, so any flip here is caused by bias, not by a smaller prior.

Run from repo root:
    python -m src.eval.evaluate_mismatch_prior
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

# MovieLens age-bucket codes, in user_emb one-hot order
AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]
YOUNG_CODES = {1, 18}
OLD_CODES = {50, 56}


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


def run_episodes(policy, env: ColdStartEnv, users: List[int], label: str) -> List[List[float]]:
    all_rewards: List[List[float]] = []
    for user_idx in tqdm(users, desc=label, ncols=72):
        state = env.reset(user_idx=user_idx)
        policy.reset()
        ep: List[float] = []
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            policy.update(action, reward, next_state, done)
            state = next_state
            ep.append(reward)
        all_rewards.append(ep)
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
    ratings_by_user: Dict, warm_subset: List[int], item_emb: torch.Tensor, n_items: int,
) -> List[Tuple[str, Any]]:
    mu_0, Lambda_0 = compute_warm_prior(ratings_by_user, warm_subset, item_emb, reg=50.0)
    k = min(50, len(warm_subset) - 1)
    nlts = NeuralLinearTS(item_emb=item_emb, lambda_prior=50.0, sigma_noise=0.5,
                          mu_0=mu_0, Lambda_0=Lambda_0)
    hybrid = HybridNeuralLinearTS(ratings_by_user=ratings_by_user, warm_users=warm_subset,
                                  n_items=n_items, k=k, lambda_prior=1.0, sigma_noise=1.0)
    greedy = GreedyCFBaseline(ratings_by_user=ratings_by_user, warm_users=warm_subset,
                              n_items=n_items, k=k, reg=1.0)
    random_bl = RandomBaseline(seed=42)
    return [("HybridTS(CF)", hybrid), ("NLTS(content)", nlts),
            ("GreedyCF", greedy), ("Random", random_bl)]


# ── printing ─────────────────────────────────────────────────────────

def print_table(title: str, metrics: Dict[str, Dict[str, float]], labels: List[str], k: int) -> None:
    print(f"\n{'=' * 96}")
    print(f"  {title}")
    print(f"{'=' * 96}")
    print(f"  {'Metric':<20}" + "".join(f" {l:<20}" for l in labels))
    print("  " + "-" * 92)

    def row(lab, key, fmt=".4f", show_std=True):
        std_key = key.replace("avg_", "std_").replace("ndcg_", "std_ndcg_")
        parts = []
        for l in labels:
            m = metrics[l]
            parts.append(f"{m[key]:{fmt}} +/- {m[std_key]:{fmt}}" if (show_std and std_key in m) else f"{m[key]:{fmt}}")
        print(f"  {lab:<20}" + "".join(f" {p:<20}" for p in parts))

    row("Avg reward @ 1",   "avg_reward_step1")
    row(f"Avg reward @ {k}", f"avg_reward_{k}")
    row(f"NDCG@{k}",        f"ndcg_{k}")
    row(f"Hit@{k} (>=4)",   f"hit_{k}", show_std=False)
    row("Avg cum. reward",  "avg_cum_reward")
    print("  " + "-" * 92)
    print(f"{'=' * 96}")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    config = DataConfig()
    processed = Path(config.data_dir) / "processed"

    print("Loading artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split      = torch.load(processed / "user_split.pt",      weights_only=False)
    item_emb        = torch.load(processed / "item_emb.pt",        weights_only=False)
    user_emb        = torch.load(processed / "user_emb.pt",        weights_only=False)

    warm_users = user_split["warm"]
    cold_users = user_split["cold"]
    n_items    = item_emb.shape[0]
    T = config.cold_start_horizon_T
    K_EVAL = 5
    ue = user_emb.numpy()

    def age_code(u: int) -> int:
        return AGE_BUCKETS[int(np.argmax(ue[u][2:9]))]

    def group(u: int) -> str:
        c = age_code(u)
        if c in YOUNG_CODES:
            return "young"
        if c in OLD_CODES:
            return "old"
        return "mid"

    warm_by = {"young": [u for u in warm_users if group(u) == "young"],
               "old":   [u for u in warm_users if group(u) == "old"]}
    cold_by = {"young": [u for u in cold_users if group(u) == "young"],
               "old":   [u for u in cold_users if group(u) == "old"]}

    CAP = min(len(warm_by["young"]), len(warm_by["old"]))  # equal-size warm subsets
    print(f"warm young={len(warm_by['young'])} old={len(warm_by['old'])} -> CAP={CAP}")
    print(f"cold young={len(cold_by['young'])} old={len(cold_by['old'])}")
    print("YOUNG = ages <25 (codes 1,18) | OLD = ages 50+ (codes 50,56)\n")

    opposite = {"young": "old", "old": "young"}
    all_results: Dict[str, Any] = {
        "experiment": {
            "design": "demographic age mismatch; warm subsets capped to equal size",
            "young_codes": sorted(YOUNG_CODES), "old_codes": sorted(OLD_CODES),
            "cap_warm": CAP, "T": T, "k_eval": K_EVAL,
            "n_warm_young": len(warm_by["young"]), "n_warm_old": len(warm_by["old"]),
            "n_cold_young": len(cold_by["young"]), "n_cold_old": len(cold_by["old"]),
        },
    }

    labels = ["HybridTS(CF)", "NLTS(content)", "GreedyCF", "Random"]

    for target in ["old", "young"]:
        eval_cold = cold_by[target]
        conditions = {
            "matched":    warm_by[target][:CAP],            # same group as eval
            "mismatched": warm_by[opposite[target]][:CAP],  # opposite group
        }
        tgt_key = f"target_{target}"
        all_results[tgt_key] = {"n_eval_cold": len(eval_cold)}

        for cond, warm_subset in conditions.items():
            print(f"\n{'#'*96}\n  TARGET cold={target} ({len(eval_cold)} users) | "
                  f"PRIOR={cond} (trained on {group(warm_subset[0])}, n={len(warm_subset)})\n{'#'*96}")
            t0 = time.time()
            policies = build_policies(ratings_by_user, warm_subset, item_emb, n_items)
            print(f"  policies built in {time.time()-t0:.1f}s")

            all_results[tgt_key][cond] = {}
            for protocol, use_full in [("selection", True), ("ranking", False)]:
                env = ColdStartEnv(
                    ratings_by_user=ratings_by_user, item_emb=item_emb, config=config,
                    user_pool=eval_cold, user_emb=user_emb, warm_users=set(warm_subset),
                    use_full_candidate_pool=use_full, reward_noise_std=0.0,
                )
                metrics = {}
                for lab, pol in policies:
                    rewards = run_episodes(pol, env, eval_cold, label=f"  {lab:<13} {cond[:5]}")
                    metrics[lab] = compute_metrics(rewards, k=K_EVAL)
                print_table(
                    f"cold={target} | prior={cond} | {protocol} | "
                    f"NDCG flip: Greedy-Hybrid = "
                    f"{metrics['GreedyCF'][f'ndcg_{K_EVAL}'] - metrics['HybridTS(CF)'][f'ndcg_{K_EVAL}']:+.4f}",
                    metrics, labels, K_EVAL,
                )
                all_results[tgt_key][cond][protocol] = {l: {"metrics": m} for l, m in metrics.items()}

    # ── summary: matched vs mismatched gap (Greedy - best TS) ───────────
    print(f"\n\n{'='*96}\n  SUMMARY: NDCG@{K_EVAL} gap = GreedyCF - HybridTS  (selection)")
    print(f"  positive = Greedy wins (good prior) | negative = exploration wins (wrong prior)\n{'='*96}")
    print(f"  {'target':<10} {'matched':<14} {'mismatched':<14} {'swing'}")
    print("  " + "-" * 52)
    for target in ["old", "young"]:
        tk = f"target_{target}"
        def gap(cond):
            b = all_results[tk][cond]["selection"]
            return b["GreedyCF"]["metrics"][f"ndcg_{K_EVAL}"] - b["HybridTS(CF)"]["metrics"][f"ndcg_{K_EVAL}"]
        gm, gx = gap("matched"), gap("mismatched")
        print(f"  {target:<10} {gm:+.4f}        {gx:+.4f}        {gx-gm:+.4f}")
    print("  " + "-" * 52)

    out_path = _REPO_ROOT / "results" / "mismatch_prior_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved -> {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
