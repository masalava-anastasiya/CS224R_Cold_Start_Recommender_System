"""Result plots for the CS224R poster (noise sweep + prior ablation).

Reads the JSON result files produced by the evaluation scripts and writes
PNGs into results/figures/, matching the dataset-figure style:
  07_noise_sweep.png       NDCG@5 vs reward noise, ranking + selection
  08_prior_ablation.png    NDCG@5 vs warm-user fraction, selection
  09_robustness_heatmap.png  (GreedyCF - HybridTS) NDCG@5 over noise x warm

Run from repo root:
    python -m src.eval.plot_results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULTS_DIR = _REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Stanford-inspired palette (matches src/visualize_dataset.py) ----
CARDINAL   = "#8C1515"
SANDSTONE  = "#D2C295"
STONE      = "#544948"
COOL_GREY  = "#4D4F53"
PALO_ALTO  = "#175E54"
TEAL       = "#00505C"
SKY        = "#4298B5"
POPPY      = "#E98300"
PURPLE     = "#53284F"
LIGHT_RED  = "#B83A4B"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# consistent per-method styling across all plots
METHOD_STYLE = {
    "GreedyCF":      {"color": CARDINAL,  "marker": "o", "label": "Greedy CF"},
    "HybridTS(CF)":  {"color": SKY,       "marker": "s", "label": "Hybrid TS"},
    "NLTS(content)": {"color": PALO_ALTO, "marker": "^", "label": "NLTS"},
    "Random":        {"color": COOL_GREY, "marker": "D", "label": "Random"},
}
METHOD_ORDER = ["GreedyCF", "HybridTS(CF)", "NLTS(content)", "Random"]


def _load(name: str) -> Optional[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  [skip] {name} not found")
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 07: NDCG@5 vs reward noise (ranking + selection panels)
# ---------------------------------------------------------------------------

def plot_noise_sweep(noisy: dict, rl2_rank: Optional[dict], rl2_sel: Optional[dict]) -> None:
    noise_levels: List[float] = noisy["experiment"]["noise_levels"]
    k = noisy["experiment"]["k_eval"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    panels = [("exhaustive", "Ranking  (pool = 20)", rl2_rank),
              ("non_exhaustive", "Selection  (pool ~ 176)", rl2_sel)]

    for ax, (protocol, title, rl2_data) in zip(axes, panels):
        for method in METHOD_ORDER:
            ys = [
                noisy[f"noise_{s}"][protocol][method]["metrics"][f"ndcg_{k}"]
                for s in noise_levels
            ]
            st = METHOD_STYLE[method]
            ax.plot(noise_levels, ys, marker=st["marker"], color=st["color"],
                    label=st["label"], linewidth=2, markersize=7)

        # RL2 was only evaluated at sigma=0; show as a reference line
        if rl2_data is not None:
            rl2_ndcg = rl2_data["rl2"]["metrics"][f"ndcg_{k}"]
            ax.axhline(rl2_ndcg, color=POPPY, linestyle="--", linewidth=2,
                       label="RL$^2$ ($\\sigma$=0)")

        ax.set_title(title)
        ax.set_xlabel("Reward noise $\\sigma$")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"NDCG@{k}")
    axes[1].legend(loc="lower left", framealpha=0.9, fontsize=11)

    fig.suptitle("Exploration vs. exploitation under reward noise",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "07_noise_sweep.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# 08: NDCG@5 vs warm-user fraction (selection, noise=0)
# ---------------------------------------------------------------------------

def plot_prior_ablation(ablation: dict, extra: Optional[dict] = None) -> None:
    k = ablation["experiment"]["k_eval"]

    # merge in extra low-warm fractions if available
    frac_to_key: Dict[float, str] = {}
    for src in [extra, ablation]:
        if src is None:
            continue
        for fr in src["experiment"]["warm_fractions"]:
            frac_to_key[fr] = "__extra__" if src is extra else "__main__"

    fracs = sorted(frac_to_key)
    sources = {"__main__": ablation, "__extra__": extra}

    def get_ndcg(frac: float, method: str) -> float:
        src = sources[frac_to_key[frac]]
        return src[f"warm_{frac}"]["noise_0.0"]["non_exhaustive"][method]["metrics"][f"ndcg_{k}"]

    def get_nwarm(frac: float) -> int:
        src = sources[frac_to_key[frac]]
        return src[f"warm_{frac}"]["n_warm_used"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method in METHOD_ORDER:
        ys = [get_ndcg(fr, method) for fr in fracs]
        st = METHOD_STYLE[method]
        ax.plot([f * 100 for f in fracs], ys, marker=st["marker"], color=st["color"],
                label=st["label"], linewidth=2, markersize=7)

    ax.set_title("Does a weaker prior let exploration win?")
    ax.set_xlabel("Warm-user fraction (%)")
    ax.set_ylabel(f"NDCG@{k}  (selection, $\\sigma$=0)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)

    # annotate warm-user counts on the x ticks
    ax.set_xticks([f * 100 for f in fracs])

    fig.tight_layout()
    path = OUT_DIR / "08_prior_ablation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# 09: heatmap of (GreedyCF - HybridTS) NDCG@5 over noise x warm fraction
# ---------------------------------------------------------------------------

def plot_robustness_heatmap(ablation: dict, extra: Optional[dict] = None) -> None:
    k = ablation["experiment"]["k_eval"]
    noise_levels = ablation["experiment"]["noise_levels"]  # [0.0, 0.5]

    # collect all warm fractions across both sources
    frac_to_src: Dict[float, dict] = {}
    if extra is not None:
        for fr in extra["experiment"]["warm_fractions"]:
            frac_to_src[fr] = extra
    for fr in ablation["experiment"]["warm_fractions"]:
        frac_to_src[fr] = ablation
    fracs = sorted(frac_to_src)

    # gap matrix: rows = noise, cols = warm fraction, value = Greedy - Hybrid
    gap = np.zeros((len(noise_levels), len(fracs)))
    for i, s in enumerate(noise_levels):
        for j, fr in enumerate(fracs):
            src = frac_to_src[fr]
            block = src[f"warm_{fr}"][f"noise_{s}"]["non_exhaustive"]
            g = block["GreedyCF"]["metrics"][f"ndcg_{k}"]
            h = block["HybridTS(CF)"]["metrics"][f"ndcg_{k}"]
            gap[i, j] = g - h

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    vmax = float(np.abs(gap).max())
    im = ax.imshow(gap, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.set_yticks(range(len(noise_levels)))
    ax.set_yticklabels([f"$\\sigma$={s}" for s in noise_levels])
    ax.set_xlabel("Warm-user fraction")
    ax.set_title("Greedy CF minus Hybrid TS  (NDCG@5, selection)")

    # annotate each cell
    for i in range(len(noise_levels)):
        for j in range(len(fracs)):
            ax.text(j, i, f"{gap[i, j]:+.3f}", ha="center", va="center",
                    color="black", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDCG gap  (red = Greedy wins)")

    fig.tight_layout()
    path = OUT_DIR / "09_robustness_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# 10: mismatched prior -- matched vs mismatched, per target group
# ---------------------------------------------------------------------------

def plot_mismatch(mismatch: dict) -> None:
    k = mismatch["experiment"]["k_eval"]
    targets = [("old", "New user = senior (50+)"),
               ("young", "New user = young (<25)")]
    conds = ["matched", "mismatched"]
    methods = ["GreedyCF", "HybridTS(CF)", "NLTS(content)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (target, title) in zip(axes, targets):
        block = mismatch[f"target_{target}"]
        for method in methods:
            ys = [block[c]["selection"][method]["metrics"][f"ndcg_{k}"] for c in conds]
            st = METHOD_STYLE[method]
            ax.plot([0, 1], ys, marker=st["marker"], color=st["color"],
                    label=st["label"], linewidth=2.5, markersize=9)
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Matched\nprior", "Wrong\nprior"])
        ax.set_xlim(-0.3, 1.3)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(f"NDCG@{k}  (selection)")
    axes[1].legend(loc="lower left", framealpha=0.9, fontsize=11)
    fig.suptitle("A wrong-population prior erodes Greedy CF's edge",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "10_mismatch_prior.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# 11: method upgrades vs the strong prior (NDCG@5, selection)
# ---------------------------------------------------------------------------

def plot_method_upgrades(neural_factor: dict, demographic: dict) -> None:
    k = neural_factor["experiment"]["k_eval"]
    sel_nf = neural_factor["selection"]
    sel_dm = demographic["selection"]

    # (label, value, color) — shared baselines pulled from neural_factor file
    bars = [
        ("Greedy CF",      sel_nf["GreedyCF"]["metrics"][f"ndcg_{k}"],       CARDINAL),
        ("Hybrid TS",      sel_nf["HybridTS(CF)"]["metrics"][f"ndcg_{k}"],   SKY),
        ("Neural-Factor TS\n(#2 learned feats)", sel_nf["NeuralFactorTS"]["metrics"][f"ndcg_{k}"], TEAL),
        ("Demographic TS\n(#3 psi-prior)", sel_dm["DemographicTS"]["metrics"][f"ndcg_{k}"], PURPLE),
        ("NLTS",           sel_nf["NLTS(content)"]["metrics"][f"ndcg_{k}"],  PALO_ALTO),
        ("Random",         sel_nf["Random"]["metrics"][f"ndcg_{k}"],         COOL_GREY),
    ]
    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = np.arange(len(bars))
    ax.bar(xs, vals, color=colors, width=0.66)
    greedy_val = bars[0][1]
    ax.axhline(greedy_val, color=CARDINAL, linestyle="--", linewidth=1.5, alpha=0.7)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(f"NDCG@{k}  (selection)")
    ax.set_ylim(0.74, 0.91)
    ax.set_title("At full (70%) warm data, no exploration upgrade beats Greedy CF")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = OUT_DIR / "11_method_upgrades.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading result files...")
    noisy       = _load("noisy_rewards_results.json")
    ablation    = _load("prior_ablation_results.json")
    ablation_lo = _load("prior_ablation_lowwarm_results.json")
    mismatch    = _load("mismatch_prior_results.json")
    neural_fac  = _load("neural_factor_results.json")
    demographic = _load("demographic_prior_results.json")
    rl2_rank    = _load("rl2_results.json")
    rl2_sel     = _load("rl2_results_explore.json")

    print("Building figures...")
    if noisy is not None:
        plot_noise_sweep(noisy, rl2_rank, rl2_sel)
    if ablation is not None:
        plot_prior_ablation(ablation, extra=ablation_lo)
        plot_robustness_heatmap(ablation, extra=ablation_lo)
    if mismatch is not None:
        plot_mismatch(mismatch)
    if neural_fac is not None and demographic is not None:
        plot_method_upgrades(neural_fac, demographic)

    print("Done.")


if __name__ == "__main__":
    main()
