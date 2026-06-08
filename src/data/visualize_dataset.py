"""Dataset summary figures for MovieLens-1M."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OUT_DIR = _REPO_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARDINAL = "#8C1515"
SANDSTONE = "#D2C295"
STONE = "#544948"
COOL_GREY = "#4D4F53"
PALO_ALTO = "#175E54"
TEAL = "#00505C"
SKY = "#4298B5"
POPPY = "#E98300"
PURPLE = "#53284F"
LIGHT_RED = "#B83A4B"

PALETTE = [
    CARDINAL, SKY, PALO_ALTO, POPPY, PURPLE,
    LIGHT_RED, TEAL, SANDSTONE, COOL_GREY, STONE,
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_data():
    processed = _REPO_ROOT / "data" / "processed"
    raw = _REPO_ROOT / "data" / "raw"

    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split = torch.load(processed / "user_split.pt", weights_only=False)

    users_df = pd.read_csv(
        raw / "users.dat", sep="::", engine="python", encoding="latin-1",
        names=["UserID", "Gender", "Age", "Occupation", "Zip"],
    )
    ratings_df = pd.read_csv(
        raw / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    rating_counts = ratings_df.groupby("UserID")["Rating"].count()
    surviving_raw_ids = set(rating_counts[rating_counts >= 25].index)
    users_df = users_df[users_df["UserID"].isin(surviving_raw_ids)].copy()

    all_ratings = [r for history in ratings_by_user.values() for _, r, _ in history]
    ratings_per_user = [len(h) for h in ratings_by_user.values()]

    return ratings_by_user, user_split, users_df, all_ratings, ratings_per_user


def plot_rating_distribution(all_ratings: list) -> None:
    counts = Counter(all_ratings)
    stars = [1, 2, 3, 4, 5]
    values = [counts[s] / 1000 for s in stars]
    pcts = [100 * counts[s] / len(all_ratings) for s in stars]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stars, values, color=CARDINAL, width=0.6, edgecolor="white", linewidth=1.2)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=11, color=STONE,
        )

    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Count (thousands)")
    ax.set_title("Rating Distribution")
    ax.set_xticks(stars)
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.set_ylim(0, max(values) * 1.25)

    total = len(all_ratings)
    ax.text(
        0.97, 0.95, f"n={total:,}", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color=COOL_GREY,
    )

    fig.tight_layout()
    path = OUT_DIR / "01_rating_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def plot_ratings_per_user(ratings_per_user: list) -> None:
    rpu = np.array(ratings_per_user)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(np.log10(25), np.log10(2400), 25)
    ax.hist(rpu, bins=bins, color=SKY, edgecolor="white", linewidth=0.8)
    ax.axvline(rpu.mean(), color=CARDINAL, linewidth=2, linestyle="--",
               label=f"mean={rpu.mean():.0f}")
    ax.axvline(np.median(rpu), color=POPPY, linewidth=2, linestyle=":",
               label=f"median={np.median(rpu):.0f}")

    ax.set_xscale("log")
    ax.set_xlabel("Ratings per user (log scale)")
    ax.set_ylabel("Users")
    ax.set_title("User activity")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = OUT_DIR / "02_ratings_per_user.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def plot_gender(users_df: pd.DataFrame) -> None:
    counts = users_df["Gender"].value_counts()
    labels = ["Male", "Female"]
    sizes = [counts.get("M", 0), counts.get("F", 0)]
    colors = [SKY, CARDINAL]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=12),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title("Gender")
    fig.tight_layout()
    path = OUT_DIR / "03_gender_split.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def plot_age(users_df: pd.DataFrame) -> None:
    age_labels = {
        1: "Under 18", 18: "18-24", 25: "25-34",
        35: "35-44", 45: "45-49", 50: "50-55", 56: "56+",
    }
    counts = users_df["Age"].value_counts().sort_index()
    labels = [age_labels[k] for k in counts.index]
    values = counts.values

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        range(len(labels)), values,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
        edgecolor="white", linewidth=1.0,
    )

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(v), ha="center", va="bottom", fontsize=10, color=STONE,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Users")
    ax.set_title("Age")
    ax.set_ylim(0, max(values) * 1.18)

    fig.tight_layout()
    path = OUT_DIR / "04_age_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def plot_occupation(users_df: pd.DataFrame) -> None:
    occ_labels = {
        0: "Other/Not specified", 1: "Academic/Educator",
        2: "Artist", 3: "Clerical/Admin",
        4: "College/Grad student", 5: "Customer service",
        6: "Doctor/Health care", 7: "Executive/Managerial",
        8: "Farmer", 9: "Homemaker",
        10: "K-12 student", 11: "Lawyer",
        12: "Programmer", 13: "Retired",
        14: "Sales/Marketing", 15: "Scientist",
        16: "Self-employed", 17: "Technician/Engineer",
        18: "Tradesman/Craftsman", 19: "Unemployed",
        20: "Writer",
    }
    counts = users_df["Occupation"].value_counts().sort_values(ascending=True)
    labels = [occ_labels[k] for k in counts.index]
    values = counts.values
    colors = [CARDINAL if v == values.max() else SKY for v in values]

    fig, ax = plt.subplots(figsize=(7, 8))
    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor="white", linewidth=0.8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Users")
    ax.set_title("Occupation")

    for bar, v in zip(bars, values):
        ax.text(v + 10, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9, color=STONE)

    fig.tight_layout()
    path = OUT_DIR / "05_occupation_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def plot_warm_cold(user_split: dict, ratings_by_user: dict) -> None:
    warm = set(user_split["warm"])
    cold = set(user_split["cold"])

    warm_ratings = sum(len(ratings_by_user[u]) for u in warm)
    cold_ratings = sum(len(ratings_by_user[u]) for u in cold)

    categories = ["Users", "Ratings"]
    warm_vals = [len(warm), warm_ratings / 1000]
    cold_vals = [len(cold), cold_ratings / 1000]

    x = np.arange(len(categories))
    width = 0.45

    fig, ax = plt.subplots(figsize=(5, 4))
    b1 = ax.bar(x, warm_vals, width, label="Warm",
                color=PALO_ALTO, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x, cold_vals, width, bottom=warm_vals,
                label="Cold", color=SANDSTONE, edgecolor="white", linewidth=1.2)

    for bar, v in zip(b1, warm_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{v:,.0f}" if v > 100 else f"{v:.0f}k",
            ha="center", va="center", fontsize=11,
            color="white", fontweight="bold",
        )

    for bar, bv, v in zip(b2, warm_vals, cold_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bv + v / 2,
            f"{v:,.0f}" if v > 100 else f"{v:.0f}k",
            ha="center", va="center", fontsize=11,
            color=STONE, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Users / ratings (thousands)")
    ax.set_title("Warm/cold split")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = OUT_DIR / "06_warm_cold_split.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def main() -> None:
    ratings_by_user, user_split, users_df, all_ratings, ratings_per_user = load_data()
    print(f"users={len(ratings_by_user):,}, ratings={len(all_ratings):,}")

    plot_rating_distribution(all_ratings)
    plot_ratings_per_user(ratings_per_user)
    plot_gender(users_df)
    plot_age(users_df)
    plot_occupation(users_df)
    plot_warm_cold(user_split, ratings_by_user)


if __name__ == "__main__":
    main()
