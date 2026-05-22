This project studies cold-start recommendation on **MovieLens-1M** using a shared preprocessing pipeline and a Gym-like evaluation environment.

## Data layout

```
data/
  raw/          # MovieLens-1M .dat files (ratings, movies, users)
  processed/    # Artifacts produced by the pipeline 
```

Place `ratings.dat`, `movies.dat`, and `users.dat` in `data/raw/` before running preprocessing.

## Running preprocessing

From the repo root:

```bash
pip install -r requirements.txt
python -m src.data.preprocess
```

## Pipeline overview

| Step | What it does |
|------|--------------|
| **Load** | Read the three `.dat` files (`latin-1` encoding). |
| **ID maps** | Remap raw user/movie IDs to contiguous 0-based indices. |
| **Filter & sort** | Drop users with fewer than `min_ratings_per_user` ratings; sort each user's history by timestamp (chronological order is required to test cold-start). |
| **Warm/cold split** | Randomly assign users to warm (training/diagnostics) and cold (evaluation) sets. |
| **Rating lists** | Build `user_idx → [(item_idx, rating, timestamp), …]` dicts. |
| **Item embeddings** | Encode movie title + genres with a sentence-transformer; optionally append a multi-hot genre vector; L2-normalize. |

## Output artifacts

All files are written to `data/processed/`:

| File | Contents |
|------|----------|
| `id_maps.pt` | Raw→index maps, `n_users`, `n_items` |
| `ratings_by_user.pt` | Per-user chronological rating histories |
| `user_split.pt` | `{'warm': [...], 'cold': [...]}` user index lists |
| `item_emb.pt` | `(n_items, emb_dim)` item feature matrix φ(x) |
| `config_hash.txt` | Hash of `DataConfig` used to build the cache |

## Configuration

Key settings live in `src/config.py` (`DataConfig`):

- **`min_ratings_per_user`** (default 25) — minimum interactions to keep a user
- **`warm_user_fraction`** (default 0.7) — fraction of users assigned to the warm set
- **`encoder_name`** / **`embedding_dim`** — sentence-transformer model (default `all-MiniLM-L6-v2`, dim 384)
- **`use_genre_features`** — whether to concatenate an 18-dim multi-hot genre vector
- **`cold_start_horizon_T`** — number of interactions revealed per cold-start episode (used at runtime)
- **`random_seed`** — controls the warm/cold split

Change any of these and re-run `python -m src.data.preprocess` to rebuild the cache.

## Runtime usage

Processed artifacts feed `ColdStartEnv` (`src/data/env.py`), which simulates cold-start episodes: for a held-out user, the agent recommends from items in that user's first `cold_start_horizon_T` interactions and receives the logged rating as reward.
