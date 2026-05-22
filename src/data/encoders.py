"""Item feature encoder φ(x).

Builds a (n_items, emb_dim) tensor of L2-normalised item embeddings by:
  1. Encoding a text string per item with a sentence-transformers model.
  2. Optionally concatenating a multi-hot genre vector.
  3. L2-normalising the final vector.

The result is cached by preprocess.py to data/processed/item_emb.pt and
shared across all three methods (Neural Linear, Meta-RL, Constrained Bandit).
Swapping encoder_name and re-running preprocess.py is the only change needed
for the ablation on "how much representation quality affects sample efficiency."
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


# Fixed MovieLens-1M genre vocabulary (order matters for the multi-hot vector)
GENRE_VOCAB: List[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
GENRE_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(GENRE_VOCAB)}


def _parse_year(title: str) -> str:
    """Extract the 4-digit year from MovieLens title format 'Movie Name (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else "unknown"


def _build_text(row: pd.Series) -> str:
    """Construct a descriptive text string for a single movie row."""
    title = row["Title"]
    year = _parse_year(title)
    # Strip the trailing year from the title for a cleaner sentence
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    genres = row["Genres"].replace("|", ", ")
    return f"{clean_title} ({year}). Genres: {genres}"


def _build_genre_multihot(genres_str: str) -> np.ndarray:
    vec = np.zeros(len(GENRE_VOCAB), dtype=np.float32)
    for g in genres_str.split("|"):
        if g in GENRE_TO_IDX:
            vec[GENRE_TO_IDX[g]] = 1.0
    return vec


def build_item_embeddings(
    movies_df: pd.DataFrame,
    item_id_map: Dict[int, int],
    config,  # DataConfig; avoid circular import by not annotating here
) -> torch.Tensor:
    """Encode all items in movies_df and return a (n_items, final_dim) tensor.

    Items absent from movies_df (e.g. filtered out) receive a zero vector.
    The tensor is indexed by item_idx (contiguous), not raw MovieID.

    Args:
        movies_df:   DataFrame with columns [MovieID, Title, Genres, item_idx].
        item_id_map: raw MovieID -> item_idx mapping.
        config:      DataConfig instance.

    Returns:
        Tensor of shape (n_items, final_dim) where final_dim = embedding_dim
        (+ len(GENRE_VOCAB) if use_genre_features is True).
    """
    from sentence_transformers import SentenceTransformer

    n_items = len(item_id_map)

    # Only encode movies that have a valid item_idx (survived filtering)
    valid_movies = movies_df[movies_df["item_idx"] >= 0].copy()

    print(f"  Encoding {len(valid_movies)} items with '{config.encoder_name}'...")
    model = SentenceTransformer(config.encoder_name)

    texts = valid_movies.apply(_build_text, axis=1).tolist()
    text_embs = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalise after optional concatenation
    )  # (n_valid, encoder_dim)

    actual_dim = text_embs.shape[1]
    assert actual_dim == config.embedding_dim, (
        f"Encoder '{config.encoder_name}' produced dim={actual_dim}, "
        f"but config.embedding_dim={config.embedding_dim}. "
        f"Update config.embedding_dim to match."
    )

    if config.use_genre_features:
        genre_vecs = np.stack(
            [_build_genre_multihot(row["Genres"]) for _, row in valid_movies.iterrows()],
            axis=0,
        )  # (n_valid, n_genres)
        combined = np.concatenate([text_embs, genre_vecs], axis=1)
    else:
        combined = text_embs  # (n_valid, embedding_dim)

    final_dim = combined.shape[1]

    # Place each item's embedding at its contiguous index
    emb_matrix = np.zeros((n_items, final_dim), dtype=np.float32)
    for emb, (_, row) in zip(combined, valid_movies.iterrows()):
        emb_matrix[int(row["item_idx"])] = emb

    # L2-normalise each row (zero-vectors stay zero)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    result = torch.tensor(emb_matrix, dtype=torch.float32)
    print(f"  Item embedding shape: {tuple(result.shape)}")
    return result
