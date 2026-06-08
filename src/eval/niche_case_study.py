"""Prior quality case study: niche-taste cold users.

Identifies cold users where Greedy CF fails by computing a prior-mismatch
niche score, then analyses whether exploration methods recover more effectively.

Niche score = 1 − Spearman(item_means, user_ratings).
Because the warm-user SVD prior collapses to a zero vector, Greedy CF is
effectively a popularity ranker.  Users whose taste anti-correlates with
warm-user popularity are "niche" and will be poorly served by Greedy CF.

Evaluation uses the non-exhaustive protocol (full candidate pool, T=20
selections) so that exploration actually matters.

Run from repo root:
    python -m src.eval.niche_case_study [--checkpoint PATH] [--n-vignettes 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.methods.rl2_policy import RL2Policy
from src.methods.constrained_bandit import ConstrainedLinearUCBBandit
from src.baselines.greedy_cf import GreedyCFBaseline
from src.baselines.random_baseline import RandomBaseline
from src.baselines.nonpersonalized_baseline import NonPersonalizedBaseline

# MovieLens-1M genre ordering (matches the multi-hot last 18 dims of item_emb)
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

METHOD_COLORS: Dict[str, str] = {
    "Random":      "#aec7e8",
    "Greedy CF":   "#e15759",
    "NLTS":        "#f28e2b",
    "Hybrid TS":   "#4e79a7",
    "RL²":         "#59a14f",
    "Constrained": "#b07aa1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via numpy (avoids scipy type-annotation issues)."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Niche score
# ---------------------------------------------------------------------------

def compute_niche_scores(
    greedy_cf: GreedyCFBaseline,
    ratings_by_user: Dict,
    cold_users: List[int],
) -> np.ndarray:
    """1 − Spearman(item_means, user_ratings) for each cold user.

    Since the SVD prior is numerically zero, Greedy CF recommends by item
    popularity (item_means).  Users whose liked items have low popularity
    get a high niche score and will be poorly served by Greedy CF.
    """
    scores = []
    for u in cold_users:
        items   = np.array([itm for itm, _, _ in ratings_by_user[u]])
        ratings = np.array([float(r) for _, r, _ in ratings_by_user[u]])
        preds   = greedy_cf.item_means[items]
        if len(items) < 3 or ratings.std() < 0.1 or preds.std() < 1e-6:
            scores.append(0.5)
            continue
        corr = _spearman_r(preds, ratings)
        scores.append(1.0 - corr)
    return np.array(scores)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(policy, env, cold_users: List[int], label: str) -> List[List[float]]:
    all_rewards: List[List[float]] = []
    for user_idx in tqdm(cold_users, desc=label, ncols=72):
        state = env.reset(user_idx=user_idx)
        policy.reset(user_idx=user_idx)
        ep_rewards: List[float] = []
        done = False
        while not done:
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            policy.update(action, reward, state, done)
            ep_rewards.append(reward)
        all_rewards.append(ep_rewards)
    return all_rewards


# ---------------------------------------------------------------------------
# Genre profile helper
# ---------------------------------------------------------------------------

def user_genre_profile(
    user_idx: int,
    ratings_by_user: Dict,
    item_emb: torch.Tensor,
    threshold: float = 4.0,
) -> np.ndarray:
    genre_counts = np.zeros(18, dtype=np.float64)
    for item_idx, rating, _ in ratings_by_user[user_idx]:
        if rating >= threshold:
            genre_counts += item_emb[item_idx, -18:].cpu().numpy()
    total = genre_counts.sum()
    return genre_counts / total if total > 0 else genre_counts


# ---------------------------------------------------------------------------
# Quartile analysis
# ---------------------------------------------------------------------------

def quartile_analysis(
    niche_scores: np.ndarray,
    rewards_by_method: Dict[str, List[List[float]]],
    early_steps: int = 5,
) -> Dict:
    boundaries = np.percentile(niche_scores, [25, 50, 75])

    def _q(s: float) -> int:
        return int(np.searchsorted(boundaries, s, side="right"))

    assignments = [_q(s) for s in niche_scores]

    per_method: Dict[str, Dict] = {}
    for method, all_rewards in rewards_by_method.items():
        cum    = [sum(r) for r in all_rewards]
        early  = [float(np.mean(r[:early_steps])) if r else 0.0 for r in all_rewards]
        late   = [float(np.mean(r[early_steps:])) if len(r) > early_steps else 0.0
                  for r in all_rewards]

        cum_b:   List[List[float]] = [[] for _ in range(4)]
        early_b: List[List[float]] = [[] for _ in range(4)]
        late_b:  List[List[float]] = [[] for _ in range(4)]
        for c, e, la, q in zip(cum, early, late, assignments):
            cum_b[q].append(c)
            early_b[q].append(e)
            late_b[q].append(la)

        per_method[method] = {
            "per_quartile_mean":       [float(np.mean(b)) for b in cum_b],
            "per_quartile_std":        [float(np.std(b))  for b in cum_b],
            "per_quartile_n":          [len(b) for b in cum_b],
            "per_quartile_early_mean": [float(np.mean(b)) for b in early_b],
            "per_quartile_late_mean":  [float(np.mean(b)) for b in late_b],
        }

    return {
        "quartile_labels":      ["Q1 (mainstream)", "Q2", "Q3", "Q4 (niche)"],
        "quartile_boundaries":  boundaries.tolist(),
        "quartile_assignments": assignments,
        "niche_scores":         niche_scores.tolist(),
        "early_steps":          early_steps,
        "per_method":           per_method,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_quartile_table(analysis: Dict, method_order: List[str]) -> None:
    qlabels   = analysis["quartile_labels"]
    gcf_means = analysis["per_method"]["Greedy CF"]["per_quartile_mean"]
    n_per_q   = analysis["per_method"][method_order[0]]["per_quartile_n"]

    print("\n" + "=" * 110)
    print("  NICHE QUARTILE ANALYSIS — Avg Cumulative Reward  (non-exhaustive protocol)")
    print("  Q1 = mainstream (taste aligns with popularity), Q4 = most niche")
    print("=" * 110)
    col_w  = 22
    header = f"  {'Method':<16}" + "".join(f"  {q:<{col_w}}" for q in qlabels) + "  Gap Q4−Q1"
    print(header)
    print("  " + "-" * 106)
    for method in method_order:
        d     = analysis["per_method"][method]
        cells = "".join(
            f"  {d['per_quartile_mean'][q]:.3f}±{d['per_quartile_std'][q]:.2f}  "
            for q in range(4)
        )
        gap = d["per_quartile_mean"][3] - d["per_quartile_mean"][0]
        print(f"  {method:<16}{cells}  {gap:+.3f}")
    print("  " + "-" * 106)
    print("  Advantage over Greedy CF at Q4:")
    for method in method_order:
        if method == "Greedy CF":
            continue
        gain = analysis["per_method"][method]["per_quartile_mean"][3] - gcf_means[3]
        print(f"    {method}: {gain:+.3f}")
    print(f"  (N per quartile: {n_per_q})")
    print("=" * 110)


def print_vignette(
    user_idx: int,
    niche_score: float,
    niche_rank: int,
    n_cold: int,
    ratings_by_user: Dict,
    item_emb: torch.Tensor,
    rewards_by_method: Dict[str, List[float]],
) -> None:
    genre_dist = user_genre_profile(user_idx, ratings_by_user, item_emb)
    top_genres = sorted(enumerate(genre_dist), key=lambda x: -x[1])[:5]
    genre_str  = ", ".join(
        f"{GENRES[g]}({v*100:.0f}%)" for g, v in top_genres if v > 0.0
    )
    print(f"\n  -- User {user_idx}  |  niche_score={niche_score:.3f}  "
          f"|  rank {niche_rank + 1}/{n_cold}  --")
    print(f"     Top liked genres: {genre_str}")
    print(f"     Total ratings: {len(ratings_by_user[user_idx])}")
    print(f"     {'Method':<16}  cum  avg/step  step1  last5-avg")
    for method, rewards in rewards_by_method.items():
        if not rewards:
            continue
        cum   = sum(rewards)
        step1 = rewards[0]
        avg   = float(np.mean(rewards))
        last5 = float(np.mean(rewards[-5:])) if len(rewards) >= 5 else avg
        print(f"     {method:<16}  {cum:.1f}  {avg:.3f}     {step1:.1f}    {last5:.3f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_quartile_analysis(
    analysis: Dict,
    method_order: List[str],
    out_path: Path,
) -> None:
    qlabels   = ["Q1\n(mainstream)", "Q2", "Q3", "Q4\n(niche)"]
    x         = np.arange(4)
    n_methods = len(method_order)
    width     = 0.13
    offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width
    gcf_means = analysis["per_method"]["Greedy CF"]["per_quartile_mean"]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for i, method in enumerate(method_order):
        d = analysis["per_method"][method]
        ax1.bar(
            x + offsets[i],
            d["per_quartile_mean"],
            width,
            yerr=d["per_quartile_std"],
            label=method,
            color=METHOD_COLORS.get(method, f"C{i}"),
            capsize=2,
            alpha=0.85,
        )

    ax1.set_xlabel("Niche Quartile")
    ax1.set_ylabel("Avg Cumulative Reward")
    ax1.set_title("Avg Cumulative Reward by Niche Quartile")
    ax1.set_xticks(x)
    ax1.set_xticklabels(qlabels)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    for i, method in enumerate(method_order):
        if method == "Greedy CF":
            continue
        d    = analysis["per_method"][method]
        gaps = [d["per_quartile_mean"][q] - gcf_means[q] for q in range(4)]
        ax2.plot(
            x, gaps, marker="o",
            label=method,
            color=METHOD_COLORS.get(method, f"C{i}"),
            linewidth=2,
        )

    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Niche Quartile")
    ax2.set_ylabel("Reward Gap vs Greedy CF")
    ax2.set_title(
        "Exploration Advantage by Niche Quartile\n"
        "(positive = beats Greedy CF; upward trend = exploits niche recovery)"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(qlabels)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Figure saved → {out_path.relative_to(_REPO_ROOT)}")


def plot_relative_degradation(
    analysis: Dict,
    method_order: List[str],
    out_path: Path,
) -> None:
    """Normalize each method by its Q1 reward to show how much it degrades for niche users.

    A method that drops steeply relies heavily on the warm-user prior.
    A method that stays flat is robust to niche taste.
    """
    qlabels = ["Q1\n(mainstream)", "Q2", "Q3", "Q4\n(niche)"]
    x       = np.arange(4)

    _, ax = plt.subplots(figsize=(7, 5))

    for i, method in enumerate(method_order):
        d      = analysis["per_method"][method]
        q1_val = d["per_quartile_mean"][0]
        rel    = [d["per_quartile_mean"][q] / q1_val for q in range(4)]
        ax.plot(
            x, rel, marker="o",
            label=method,
            color=METHOD_COLORS.get(method, f"C{i}"),
            linewidth=2,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Niche Quartile")
    ax.set_ylabel("Relative Reward (normalised by Q1)")
    ax.set_title(
        "Relative Performance Degradation by Niche Quartile\n"
        "(1.0 = same as mainstream; steep drop = relies on popularity prior)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(qlabels)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure saved → {out_path.relative_to(_REPO_ROOT)}")


def plot_vignette_curves(
    vignette_info: List[Tuple[int, float, str]],
    user_idx_to_rewards: Dict[int, Dict[str, List[float]]],
    method_order: List[str],
    out_path: Path,
) -> None:
    n = len(vignette_info)
    _, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (user_idx, niche_score, label) in zip(axes, vignette_info):
        rewards_dict = user_idx_to_rewards[user_idx]
        for i, method in enumerate(method_order):
            rewards = rewards_dict.get(method, [])
            if rewards:
                ax.plot(
                    range(1, len(rewards) + 1),
                    rewards,
                    marker=".",
                    label=method,
                    color=METHOD_COLORS.get(method, f"C{i}"),
                    alpha=0.85,
                    linewidth=1.5,
                )
        ax.set_title(f"User {user_idx}\n{label}\nniche={niche_score:.3f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward (rating)")
        ax.set_ylim(0.5, 5.5)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure saved → {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Recovery curve plot
# ---------------------------------------------------------------------------

def plot_recovery_curves(
    analysis: Dict,
    rewards_by_method: Dict[str, List[List[float]]],
    method_order: List[str],
    out_path: Path,
    early_window: Tuple[int, int] = (0, 5),    # half-open: steps 1–5
    late_window:  Tuple[int, int] = (15, 20),  # half-open: steps 16–20
) -> None:
    """Two-panel figure for Q4 (niche) users only.

    Left  — full per-step avg reward curve (steps 1–20), one line per method.
             Shaded bands highlight the early and late windows.
    Right — delta bar: avg(late window) − avg(early window) per method.
             Positive delta = method improves over the episode (recovery);
             negative = method degrades (prior exploitation with no update).
    """
    assignments = analysis["quartile_assignments"]
    q4_indices  = [i for i, q in enumerate(assignments) if q == 3]
    n_q4        = len(q4_indices)

    # Determine T from data
    T = 20
    for m in method_order:
        sample = rewards_by_method[m][q4_indices[0]] if q4_indices else []
        if sample:
            T = len(sample)
            break

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left: per-step learning curves ----
    for i, method in enumerate(method_order):
        rewards_q4   = [rewards_by_method[method][idx] for idx in q4_indices]
        avg_per_step = [float(np.mean([r[t] for r in rewards_q4]))
                        for t in range(T)]
        ax1.plot(
            range(1, T + 1), avg_per_step,
            label=method,
            color=METHOD_COLORS.get(method, f"C{i}"),
            linewidth=2,
        )

    # Shade early / late windows after lines so y-limits are set
    ylo, yhi = ax1.get_ylim()
    ax1.axvspan(early_window[0] + 1, early_window[1], alpha=0.10,
                color="gray", zorder=0)
    ax1.axvspan(late_window[0] + 1,  late_window[1],  alpha=0.10,
                color="steelblue", zorder=0)
    ax1.text((early_window[0] + early_window[1]) / 2 + 0.5, ylo + 0.05 * (yhi - ylo),
             f"early\n(1–{early_window[1]})", ha="center", va="bottom",
             fontsize=7, color="dimgray")
    ax1.text((late_window[0] + late_window[1]) / 2 + 0.5, ylo + 0.05 * (yhi - ylo),
             f"late\n({late_window[0] + 1}–{late_window[1]})", ha="center", va="bottom",
             fontsize=7, color="steelblue")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Avg Reward")
    ax1.set_title(f"Per-Step Avg Reward — Q4 Niche Users (n={n_q4})\n"
                  "Recovery = line trends upward over steps")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # ---- Right: delta (late − early) bar chart ----
    early_avgs: List[float] = []
    late_avgs:  List[float] = []
    deltas:     List[float] = []
    for method in method_order:
        rewards_q4 = [rewards_by_method[method][idx] for idx in q4_indices]
        ea = float(np.mean([
            np.mean(r[early_window[0]:early_window[1]]) for r in rewards_q4
        ]))
        la = float(np.mean([
            np.mean(r[late_window[0]:late_window[1]]) for r in rewards_q4
        ]))
        early_avgs.append(ea)
        late_avgs.append(la)
        deltas.append(la - ea)

    x      = np.arange(len(method_order))
    colors = [METHOD_COLORS.get(m, f"C{i}") for i, m in enumerate(method_order)]
    bars   = ax2.bar(x, deltas, color=colors, alpha=0.85)

    # Annotate each bar with "early → late" values
    for bar, ea, la, delta in zip(bars, early_avgs, late_avgs, deltas):
        va     = "bottom" if delta >= 0 else "top"
        offset = 0.02 if delta >= 0 else -0.02
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            delta + offset,
            f"{ea:.2f}→{la:.2f}",
            ha="center", va=va, fontsize=7,
        )

    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_order, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel(
        f"Δ Avg Reward  (steps {late_window[0]+1}–{late_window[1]}  −  "
        f"steps {early_window[0]+1}–{early_window[1]})"
    )
    ax2.set_title(
        "Recovery for Q4 Niche Users\n"
        "(positive = method improves late vs early; negative = degrades)"
    )
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure saved → {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    config    = DataConfig()
    processed = Path(config.data_dir) / "processed"

    required = ["ratings_by_user.pt", "user_split.pt", "item_emb.pt", "user_emb.pt"]
    missing  = [f for f in required if not (processed / f).exists()]
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
    T          = config.cold_start_horizon_T
    device     = torch.device(args.device)

    print(f"Warm: {len(warm_users)} | Cold: {len(cold_users)} | Items: {n_items} | T: {T}")

    # --- Greedy CF (also provides item_means for niche scores) ---
    K_FACTORS = 50
    print(f"\nFitting GreedyCFBaseline (k={K_FACTORS})...")
    t0 = time.time()
    greedy_cf = GreedyCFBaseline(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        reg=1.0,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Niche scores ---
    print("\nComputing niche scores...")
    t0 = time.time()
    niche_scores = compute_niche_scores(greedy_cf, ratings_by_user, cold_users)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(
        f"  mean={niche_scores.mean():.3f}  std={niche_scores.std():.3f}  "
        f"min={niche_scores.min():.3f}  max={niche_scores.max():.3f}"
    )

    # --- Non-exhaustive environment ---
    pool_sizes = [len(ratings_by_user[u]) for u in cold_users]
    print(
        f"\nCandidate pool: mean={np.mean(pool_sizes):.0f}, "
        f"median={np.median(pool_sizes):.0f}, "
        f"min={min(pool_sizes)}, max={max(pool_sizes)}"
    )
    env = ColdStartEnv(
        ratings_by_user=ratings_by_user,
        item_emb=item_emb,
        config=config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=set(warm_users),
        use_full_candidate_pool=True,
    )

    # --- Build all policies ---
    LAMBDA_NLTS = 50.0
    SIGMA_NLTS  = 0.5
    print("\nFitting all policies...")
    t0 = time.time()

    mu_0_nlts, Lambda_0 = compute_warm_prior(
        ratings_by_user, warm_users, item_emb, reg=LAMBDA_NLTS,
    )
    nlts = NeuralLinearTS(
        item_emb=item_emb,
        lambda_prior=LAMBDA_NLTS,
        sigma_noise=SIGMA_NLTS,
        mu_0=mu_0_nlts,
        Lambda_0=Lambda_0,
    )
    hybrid = HybridNeuralLinearTS(
        ratings_by_user=ratings_by_user,
        warm_users=warm_users,
        n_items=n_items,
        k=K_FACTORS,
        lambda_prior=1.0,
        sigma_noise=1.0,
    )
    random_bl = RandomBaseline(seed=42)

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

    # RL²
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"  WARNING: RL² checkpoint not found at {ckpt_path}. Skipping RL².")
        rl2 = None
    else:
        item_emb_dev = item_emb.to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        hidden_dim = ckpt.get("hparams", {}).get("hidden_dim", 256)
        rl2 = RL2Policy(item_emb=item_emb_dev, hidden_dim=hidden_dim).to(device)
        rl2.load_state_dict(ckpt["state_dict"])
        rl2.eval()
        print(f"  Loaded RL² checkpoint: epoch {ckpt.get('epoch')}, hidden={hidden_dim}")

    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Run all episodes ---
    METHOD_ORDER = ["Random", "Greedy CF", "NLTS", "Hybrid TS", "Constrained", "RL²"]
    print()
    rewards_by_method: Dict[str, List[List[float]]] = {
        "Random":    run_episodes(random_bl,  env, cold_users, "Random   "),
        "Greedy CF": run_episodes(greedy_cf,  env, cold_users, "GreedyCF "),
        "NLTS":      run_episodes(nlts,        env, cold_users, "NLTS     "),
        "Hybrid TS": run_episodes(hybrid,     env, cold_users, "HybridTS "),
        "Constrained": run_episodes(constrained, env, cold_users, "Constrain"),
    }
    if rl2 is not None:
        rewards_by_method["RL²"] = run_episodes(rl2, env, cold_users, "RL²      ")
    else:
        METHOD_ORDER = [m for m in METHOD_ORDER if m != "RL²"]

    # --- Quartile analysis ---
    analysis = quartile_analysis(niche_scores, rewards_by_method)
    print_quartile_table(analysis, METHOD_ORDER)

    # --- Vignettes: 1 mainstream (Q1, GCF wins), n-1 niche (Q4, Hybrid TS wins) ---
    assignments  = analysis["quartile_assignments"]
    gcf_cum      = [sum(r) for r in rewards_by_method["Greedy CF"]]
    hybrid_cum   = [sum(r) for r in rewards_by_method["Hybrid TS"]]
    hybrid_delta = [hybrid_cum[i] - gcf_cum[i] for i in range(len(cold_users))]

    q1_idx = [i for i, q in enumerate(assignments) if q == 0]
    q4_idx = [i for i, q in enumerate(assignments) if q == 3]

    best_q1 = max(q1_idx, key=lambda i: gcf_cum[i] - hybrid_cum[i])
    top_q4  = sorted(q4_idx, key=lambda i: -hybrid_delta[i])[: args.n_vignettes - 1]
    vignette_indices = [best_q1] + top_q4

    sorted_by_niche = np.argsort(-niche_scores)
    rank_of = {int(idx): rank for rank, idx in enumerate(sorted_by_niche)}

    vignette_info: List[Tuple[int, float, str]] = []
    for pos, vi in enumerate(vignette_indices):
        uid    = cold_users[vi]
        nscore = float(niche_scores[vi])
        label  = ("Q1 (mainstream)\nGreedy CF wins" if pos == 0
                  else f"Q4 (niche)\nExploration +{hybrid_delta[vi]:.1f}")
        vignette_info.append((uid, nscore, label))

    user_idx_to_rewards: Dict[int, Dict[str, List[float]]] = {}
    for vi, (uid, _, _) in zip(vignette_indices, vignette_info):
        user_idx_to_rewards[uid] = {m: rewards_by_method[m][vi] for m in METHOD_ORDER}

    print("\n\n  CASE STUDY VIGNETTES")
    print("  " + "=" * 70)
    for vi, (uid, nscore, _) in zip(vignette_indices, vignette_info):
        print_vignette(
            user_idx=uid,
            niche_score=nscore,
            niche_rank=rank_of[vi],
            n_cold=len(cold_users),
            ratings_by_user=ratings_by_user,
            item_emb=item_emb,
            rewards_by_method={m: rewards_by_method[m][vi] for m in METHOD_ORDER},
        )

    # --- Save JSON ---
    out_json = _REPO_ROOT / "results" / "niche_case_study.json"
    payload: Dict = {
        "experiment": {
            "T": T, "n_cold_users": len(cold_users),
            "n_warm_users": len(warm_users),
            "mode": "non_exhaustive", "k_svd": K_FACTORS,
        },
        "niche_scores": {
            "mean": float(niche_scores.mean()),
            "std":  float(niche_scores.std()),
            "quartile_boundaries": analysis["quartile_boundaries"],
            "per_user": [
                {"user_idx": int(cold_users[i]), "niche_score": float(niche_scores[i])}
                for i in range(len(cold_users))
            ],
        },
        "quartile_analysis": {m: analysis["per_method"][m] for m in METHOD_ORDER},
        "vignettes": [
            {
                "user_idx": int(uid),
                "niche_score": float(nscore),
                "label": label,
                "genre_profile": {
                    GENRES[g]: float(v)
                    for g, v in enumerate(user_genre_profile(uid, ratings_by_user, item_emb))
                    if v > 0.0
                },
                "per_method_rewards": {m: rewards_by_method[m][vi] for m in METHOD_ORDER},
            }
            for vi, (uid, nscore, label) in zip(vignette_indices, vignette_info)
        ],
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved → {out_json.relative_to(_REPO_ROOT)}")

    # --- Figures ---
    figures_dir = _REPO_ROOT / "results" / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_quartile_analysis(
        analysis, METHOD_ORDER,
        out_path=figures_dir / "niche_quartile_analysis.png",
    )
    plot_relative_degradation(
        analysis, METHOD_ORDER,
        out_path=figures_dir / "niche_relative_degradation.png",
    )
    plot_vignette_curves(
        vignette_info, user_idx_to_rewards, METHOD_ORDER,
        out_path=figures_dir / "niche_vignette_curves.png",
    )
    plot_recovery_curves(
        analysis, rewards_by_method, METHOD_ORDER,
        out_path=figures_dir / "niche_recovery_curves.png",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Niche-user case study.")
    p.add_argument(
        "--checkpoint", type=str,
        default=str(_REPO_ROOT / "results" / "rl2_explore_checkpoint.pt"),
        help="Path to RL² explore checkpoint",
    )
    p.add_argument(
        "--n-vignettes", type=int, default=3,
        help="Total vignette users (1 mainstream + n-1 niche, default=3)",
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
