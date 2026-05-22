"""One-time preprocessing pipeline for MovieLens-1M.

Run as:
    python -m src.data.preprocess

Produces (in data/processed/):
    id_maps.pt          – user/item ID remapping dicts + n_users, n_items
    ratings_by_user.pt  – dict[user_idx -> list[(item_idx, rating, timestamp)]]
    user_split.pt       – {'warm': [...], 'cold': [...]}
    item_emb.pt         – (n_items, emb_dim) tensor
    config_hash.txt     – hex digest of the DataConfig used to build the cache
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Allow `python -m src.data.preprocess` from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import DataConfig


# ---------------------------------------------------------------------------
# Step 1: Load raw .dat files
# ---------------------------------------------------------------------------

def load_raw(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three MovieLens-1M .dat files and return (ratings, movies, users).

    latin-1 encoding is required because several movie titles contain
    non-UTF-8 characters that will crash with the default utf-8 codec.
    """
    raw = Path(data_dir) / "raw"

    ratings = pd.read_csv(
        raw / "ratings.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )
    movies = pd.read_csv(
        raw / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["MovieID", "Title", "Genres"],
    )
    users = pd.read_csv(
        raw / "users.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["UserID", "Gender", "Age", "Occupation", "Zip"],
    )

    return ratings, movies, users


# ---------------------------------------------------------------------------
# Step 2: Build contiguous ID maps
# ---------------------------------------------------------------------------

def build_id_maps(
    ratings_df: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
    """Map raw UserIDs and MovieIDs to contiguous 0-based indices.

    Only IDs that appear in ratings_df are included, so the index space
    perfectly covers the embedding matrices.

    Returns (user_id_map, item_id_map, n_users, n_items).
    """
    unique_users = sorted(ratings_df["UserID"].unique())
    unique_items = sorted(ratings_df["MovieID"].unique())

    user_id_map: Dict[int, int] = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_map: Dict[int, int] = {mid: idx for idx, mid in enumerate(unique_items)}

    return user_id_map, item_id_map, len(unique_users), len(unique_items)


# ---------------------------------------------------------------------------
# Step 3: Apply ID maps to all DataFrames
# ---------------------------------------------------------------------------

def apply_id_maps(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
    user_id_map: Dict[int, int],
    item_id_map: Dict[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings_df = ratings_df.copy()
    movies_df = movies_df.copy()
    users_df = users_df.copy()

    ratings_df["user_idx"] = ratings_df["UserID"].map(user_id_map)
    ratings_df["item_idx"] = ratings_df["MovieID"].map(item_id_map)

    # Movies may contain entries never rated; map to -1 for those
    movies_df["item_idx"] = movies_df["MovieID"].map(item_id_map).fillna(-1).astype(int)

    users_df["user_idx"] = users_df["UserID"].map(user_id_map)

    return ratings_df, movies_df, users_df


# ---------------------------------------------------------------------------
# Step 4: Filter sparse users and sort chronologically
# ---------------------------------------------------------------------------

def filter_and_sort(
    ratings_df: pd.DataFrame,
    config: DataConfig,
) -> pd.DataFrame:
    """Drop users with < min_ratings_per_user and sort by (user_idx, Timestamp).

    Chronological order within each user is the foundation of the cold-start
    "reveal first k" protocol — never shuffle within a user.
    """
    counts = ratings_df.groupby("user_idx")["item_idx"].count()
    active_users = counts[counts >= config.min_ratings_per_user].index
    filtered = ratings_df[ratings_df["user_idx"].isin(active_users)].copy()
    filtered.sort_values(["user_idx", "Timestamp"], ascending=True, inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


# ---------------------------------------------------------------------------
# Step 5: Warm / cold user split
# ---------------------------------------------------------------------------

def split_users(
    user_indices: List[int],
    config: DataConfig,
) -> Tuple[List[int], List[int]]:
    """Deterministically split user_indices into warm and cold sets.

    Warm users are used to train any baseline model (e.g. matrix factorization)
    and compute µ_base. Cold users are held out for episode evaluation — we
    simulate having zero prior data for them.
    """
    rng = np.random.default_rng(config.random_seed)
    shuffled = np.array(sorted(user_indices), dtype=np.int64)
    rng.shuffle(shuffled)

    n_warm = int(len(shuffled) * config.warm_user_fraction)
    warm = sorted(shuffled[:n_warm].tolist())
    cold = sorted(shuffled[n_warm:].tolist())
    return warm, cold


# ---------------------------------------------------------------------------
# Step 6: Build per-user rating lists
# ---------------------------------------------------------------------------

def build_ratings_by_user(
    ratings_df: pd.DataFrame,
) -> Dict[int, List[Tuple[int, float, int]]]:
    """Return dict mapping user_idx -> [(item_idx, rating, timestamp), ...].

    Assumes ratings_df is already sorted by (user_idx, Timestamp).
    """
    result: Dict[int, List[Tuple[int, float, int]]] = {}
    cols = ratings_df[["user_idx", "item_idx", "Rating", "Timestamp"]]
    for uid_raw, iid_raw, rating_raw, ts_raw in cols.itertuples(index=False):
        uid = int(uid_raw)  # type: ignore[arg-type]
        entry: Tuple[int, float, int] = (
            int(iid_raw),   # type: ignore[arg-type]
            float(rating_raw),  # type: ignore[arg-type]
            int(ts_raw),    # type: ignore[arg-type]
        )
        if uid not in result:
            result[uid] = []
        result[uid].append(entry)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: DataConfig | None = None) -> None:
    if config is None:
        config = DataConfig()

    processed_dir = Path(config.data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check cache validity
    hash_file = processed_dir / "config_hash.txt"
    current_hash = config.config_hash()
    artifacts = [
        processed_dir / name
        for name in ("id_maps.pt", "ratings_by_user.pt", "user_split.pt", "item_emb.pt")
    ]
    if (
        hash_file.exists()
        and hash_file.read_text().strip() == current_hash
        and all(a.exists() for a in artifacts)
    ):
        print("All artifacts are up-to-date (config hash matches). Skipping recomputation.")
        return

    print("Loading raw MovieLens-1M files...")
    ratings_df, movies_df, users_df = load_raw(config.data_dir)

    print("Building ID maps...")
    user_id_map, item_id_map, n_users_raw, n_items_raw = build_id_maps(ratings_df)

    print("Applying ID maps...")
    ratings_df, movies_df, users_df = apply_id_maps(
        ratings_df, movies_df, users_df, user_id_map, item_id_map
    )

    print(f"Filtering users with < {config.min_ratings_per_user} ratings and sorting...")
    ratings_df = filter_and_sort(ratings_df, config)

    surviving_users = sorted(ratings_df["user_idx"].unique().tolist())
    n_users = len(surviving_users)
    n_items = len(item_id_map)

    print(f"Splitting {n_users} users into warm/cold sets...")
    warm_users, cold_users = split_users(surviving_users, config)

    print("Building per-user rating lists...")
    ratings_by_user = build_ratings_by_user(ratings_df)

    # Only keep the movies that survived the ratings filter
    surviving_items = set(ratings_df["item_idx"].unique())
    movies_df = movies_df[movies_df["item_idx"].isin(surviving_items)].copy()

    print("Building item embeddings (this may take a while on first run)...")
    from src.data.encoders import build_item_embeddings  # local import to avoid circular

    item_emb = build_item_embeddings(movies_df, item_id_map, config)

    print("Saving artifacts to", processed_dir)
    torch.save(
        {
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "n_users": n_users,
            "n_items": n_items,
        },
        processed_dir / "id_maps.pt",
    )
    torch.save(ratings_by_user, processed_dir / "ratings_by_user.pt")
    torch.save({"warm": warm_users, "cold": cold_users}, processed_dir / "user_split.pt")
    torch.save(item_emb, processed_dir / "item_emb.pt")
    hash_file.write_text(current_hash)

    print("\n=== Preprocessing complete ===")
    print(f"  Users (after filter): {n_users}  (raw: {n_users_raw})")
    print(f"  Items (after filter): {n_items}  (raw: {n_items_raw})")
    print(f"  Warm users:           {len(warm_users)}")
    print(f"  Cold users:           {len(cold_users)}")
    print(f"  Item embedding shape: {tuple(item_emb.shape)}")
    print(f"  Config hash:          {current_hash}")


if __name__ == "__main__":
    run()
