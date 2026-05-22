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

The pipeline is idempotent: if `data/processed/` already contains consistent artifacts and the cache key matches, it skips recomputation.

## Dataset statistics

Default pipeline output on MovieLens-1M (`min_ratings_per_user=25`, `warm_user_fraction=0.7`):

### Raw vs processed

| Metric | Raw | After preprocessing |
|--------|-----|---------------------|
| Ratings | 1,000,209 | 991,077 |
| Users | 6,040 | 5,624 (416 dropped for < 25 ratings) |
| Items (rated movies) | 3,706 | 3,702 appear in filtered ratings |
| Movies in `movies.dat` | 3,883 | — |
| User index range | raw `UserID` | contiguous `0 … 5623` |
| Item index range | raw `MovieID` | contiguous `0 … 3705` |

### Per-user interaction counts (after filter)

| Stat | Value |
|------|-------|
| Minimum ratings per user | 25 |
| Maximum ratings per user | 2,314 |
| Mean ratings per user | 176.2 |

### Train / eval split (user-level)

| Set | # Users | Fraction | Use |
|-----|---------|----------|-----|
| Warm | 3,936 | 70% | Training, baselines, diagnostics |
| Cold | 1,688 | 30% | Cold-start evaluation |
| **Total** | **5,624** | 100% | — |

There is no row-level train/test split — warm and cold are disjoint **user** sets. See [Runtime usage](#runtime-usage) for how cold-start episodes use the first `cold_start_horizon_T=20` interactions per cold user.

### Feature dimensions

| Entity | Source fields | Encoding | Dim | Matrix shape |
|--------|---------------|----------|-----|--------------|
| Item φ(x) | Title, year, genres | Sentence-transformer (`all-MiniLM-L6-v2`) | 384 | `(3706, 402)` |
| Item φ(x) — genres | 18 MovieLens genres | Multi-hot (optional, `use_genre_features=True`) | +18 | |
| User ψ(u) | Gender | One-hot (`M`, `F`) | 2 | `(5624, 30)` |
| User ψ(u) | Age | One-hot over 7 buckets (1, 18, 25, 35, 45, 50, 56) | 7 | |
| User ψ(u) | Occupation | One-hot over 21 codes (0–20) | 21 | |

Notes:
- `item_emb.pt` has 3,706 rows; 3,702 are non-zero (4 movies were only rated by filtered-out users).
- `user_emb.pt` has one row per surviving user; all 5,624 rows are non-zero.
- Zip codes from `users.dat` are not used.

## Pipeline overview

| Step | What it does |
|------|--------------|
| **Load** | Read the three `.dat` files (`latin-1` encoding). |
| **ID maps** | Remap raw user/movie IDs to contiguous 0-based indices. |
| **Filter & sort** | Drop users with fewer than `min_ratings_per_user` ratings; sort each user's history by timestamp (chronological order is required for the cold-start protocol). |
| **User ID remap** | Remap surviving users to contiguous indices `0 … n_users-1`. |
| **Warm/cold split** | Randomly assign users to warm (training/diagnostics) and cold (evaluation) sets. |
| **Rating lists** | Build `user_idx → [(item_idx, rating, timestamp), …]` dicts. |
| **Item embeddings** | Encode movie title + genres with a sentence-transformer; optionally append a multi-hot genre vector; L2-normalize. |
| **User embeddings** | One-hot encode Gender, Age bucket, and Occupation; L2-normalize. |

## Output artifacts

All files are written to `data/processed/`:

| File | Contents |
|------|----------|
| `id_maps.pt` | Raw→index maps, `n_users`, `n_items` |
| `ratings_by_user.pt` | Per-user chronological rating histories |
| `user_split.pt` | `{'warm': [...], 'cold': [...]}` user index lists |
| `item_emb.pt` | `(n_items, emb_dim)` item feature matrix φ(x) |
| `user_emb.pt` | `(n_users, 30)` user feature matrix ψ(u); `n_users` = surviving users after filter |
| `config_hash.txt` | Cache key: preprocess version + `DataConfig` hash |

## Features

### Item features (`item_emb.pt`)

Built from `movies.dat`:

- **Title + year + genres** → sentence-transformer embedding (default 384-dim)
- **Optional genre multi-hot** → 18-dim vector appended when `use_genre_features=True`
- Final vectors are L2-normalized (default dim: 402)

### User features (`user_emb.pt`)

Built from `users.dat`:

| Field | Encoding | Dim |
|-------|----------|-----|
| Gender | one-hot (`M`, `F`) | 2 |
| Age | one-hot over MovieLens buckets (1, 18, 25, 35, 45, 50, 56) | 7 |
| Occupation | one-hot over codes 0–20 | 21 |

Zip codes are excluded (too sparse). Final vectors are L2-normalized (dim: **30**).

At runtime, `ColdStartEnv` exposes user features via `state['user_emb']` according to `expose_user_features` (see below). Interaction history is still exposed separately via `state['revealed_emb']`.

## Configuration

Key settings live in `src/config.py` (`DataConfig`):

- **`min_ratings_per_user`** (default 25) — minimum interactions to keep a user
- **`warm_user_fraction`** (default 0.7) — fraction of users assigned to the warm set
- **`encoder_name`** / **`embedding_dim`** — sentence-transformer model (default `all-MiniLM-L6-v2`, dim 384)
- **`use_genre_features`** — whether to concatenate an 18-dim multi-hot genre vector to item embeddings
- **`use_user_features`** — whether to build `user_emb.pt` from demographics (default `True`)
- **`expose_user_features`** — when the env includes `user_emb` in the state dict:
  - `"never"` (default) — always return a zero vector (pure cold-start)
  - `"always"` — expose demographics for every user
  - `"warm_only"` — expose demographics only for warm users (requires passing `warm_users` to the env)
- **`cold_start_horizon_T`** — number of interactions per cold-start episode (default 20)
- **`random_seed`** — controls the warm/cold split

Change preprocessing-related settings and re-run `python -m src.data.preprocess` to rebuild the cache. The cache is invalidated when `DataConfig` changes, when preprocessing code changes (pipeline version bump), or when saved artifacts fail consistency checks (e.g. `user_emb` rows ≠ `n_users`).

## Runtime usage

Processed artifacts feed `ColdStartEnv` (`src/data/env.py`), which simulates cold-start episodes: for a held-out user, the agent recommends from items in that user's first `cold_start_horizon_T` interactions and receives the logged rating as reward.

```python
ratings_by_user = torch.load("data/processed/ratings_by_user.pt")
user_split = torch.load("data/processed/user_split.pt")
item_emb = torch.load("data/processed/item_emb.pt")
user_emb = torch.load("data/processed/user_emb.pt")

env = ColdStartEnv(
    ratings_by_user,
    item_emb,
    config,
    user_pool=user_split["cold"],
    user_emb=user_emb,
    warm_users=set(user_split["warm"]),
)
state = env.reset()
# state keys: history, revealed_emb, user_emb, candidates, t
```
