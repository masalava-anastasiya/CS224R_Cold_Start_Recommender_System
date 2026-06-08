"""Item and user feature encoders."""

from __future__ import annotations
import re
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

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

GENDER_VOCAB: List[str] = ["M", "F"]
GENDER_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(GENDER_VOCAB)}

AGE_BUCKETS: List[int] = [1, 18, 25, 35, 45, 50, 56]
AGE_TO_IDX: Dict[int, int] = {code: i for i, code in enumerate(AGE_BUCKETS)}

OCCUPATION_VOCAB: List[int] = list(range(21))

USER_FEATURE_DIM: int = len(GENDER_VOCAB) + len(AGE_BUCKETS) + len(OCCUPATION_VOCAB)


def _parse_year(title: str) -> str:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else "unknown"


def _build_text(row: pd.Series) -> str:
    title = row["Title"]
    year = _parse_year(title)
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
    config,
) -> torch.Tensor:
    """Encode all items and return an L2-normalised (n_items, emb_dim) tensor."""
    n_items = len(item_id_map)
    valid_movies = movies_df[movies_df["item_idx"] >= 0].copy()

    print(f"Encoding {len(valid_movies)} items with '{config.encoder_name}'...")
    model = SentenceTransformer(config.encoder_name)

    texts = valid_movies.apply(_build_text, axis=1).tolist()
    text_embs = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    actual_dim = text_embs.shape[1]
    assert actual_dim == config.embedding_dim, (
        f"Encoder '{config.encoder_name}' produced dim={actual_dim}, "
        f"expected {config.embedding_dim}"
    )

    if config.use_genre_features:
        genre_vecs = np.stack(
            [_build_genre_multihot(row["Genres"]) for _, row in valid_movies.iterrows()],
            axis=0,
        )
        combined = np.concatenate([text_embs, genre_vecs], axis=1)
    else:
        combined = text_embs

    final_dim = combined.shape[1]
    emb_matrix = np.zeros((n_items, final_dim), dtype=np.float32)
    for emb, (_, row) in zip(combined, valid_movies.iterrows()):
        emb_matrix[int(row["item_idx"])] = emb

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    result = torch.tensor(emb_matrix, dtype=torch.float32)
    print(f"Item embedding shape: {tuple(result.shape)}")
    return result


def _encode_user_row(row: pd.Series) -> np.ndarray:
    vec = np.zeros(USER_FEATURE_DIM, dtype=np.float32)

    gender = row["Gender"]
    if gender in GENDER_TO_IDX:
        vec[GENDER_TO_IDX[gender]] = 1.0

    age_offset = len(GENDER_VOCAB)
    age = int(row["Age"])
    if age in AGE_TO_IDX:
        vec[age_offset + AGE_TO_IDX[age]] = 1.0

    occ_offset = age_offset + len(AGE_BUCKETS)
    occupation = int(row["Occupation"])
    if 0 <= occupation < len(OCCUPATION_VOCAB):
        vec[occ_offset + occupation] = 1.0

    return vec


def build_user_embeddings(
    users_df: pd.DataFrame,
    user_id_map: Dict[int, int],
    config,
) -> torch.Tensor:
    """Encode all users and return an L2-normalised (n_users, USER_FEATURE_DIM) tensor."""
    n_users = len(user_id_map)
    if not config.use_user_features:
        return torch.zeros(n_users, USER_FEATURE_DIM, dtype=torch.float32)

    valid_users = users_df[users_df["user_idx"].notna()].copy()
    assert len(valid_users) == n_users, (
        f"Expected {n_users} users in users_df, found {len(valid_users)}"
    )

    print(f"Encoding {n_users} users...")
    emb_matrix = np.zeros((n_users, USER_FEATURE_DIM), dtype=np.float32)
    for _, row in valid_users.iterrows():
        emb_matrix[int(row["user_idx"])] = _encode_user_row(row)

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    result = torch.tensor(emb_matrix, dtype=torch.float32)
    print(f"User embedding shape: {tuple(result.shape)}")
    return result
