# Cold-Start Movie Recommendation with Contextual Bandits

CS224R project on **MovieLens-1M**. Warm users (70%) fit priors and baselines; cold users (30%) are evaluated in a 20-step bandit episode. Each policy picks items from a candidate pool and receives the logged rating as reward. Methods include Greedy CF, Neural/Hybrid Thompson Sampling, a constrained LinUCB bandit, demographic priors, and RL².

## Project layout

```
data/
  raw/          MovieLens-1M .dat files (not in repo)
  processed/    tensors written by preprocessing
results/        eval JSON, checkpoints, figures
src/            code (see src/README.md)
```

## Setup

1. Download [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) and place `ratings.dat`, `movies.dat`, and `users.dat` in `data/raw/`.
2. From the repo root:

```bash
pip install -r src/requirements.txt
python -m src.data.preprocess
```

Preprocessing is cached in `data/processed/` and skips recomputation when config and artifacts match.

## Quick start

```bash
python -m src.data.visualize_dataset          # dataset figures -> results/figures/
python -m src.eval.evaluate_greedy_cf         # baseline eval
python -m src.train.train_rl2 --explore       # train RL2 (explore mode)
python -m src.eval.evaluate_rl2 --explore     # eval RL2
python -m src.eval.plot_results               # poster figures from JSON
```

Other experiments live under `src/eval/` (`evaluate_hybrid.py`, `evaluate_noisy_rewards.py`, `evaluate_prior_ablation.py`, etc.). Each script writes JSON to `results/`.

## Data pipeline

| Step | Output |
|------|--------|
| Filter users (< 25 ratings dropped), sort by time | `ratings_by_user.pt` |
| 70/30 warm/cold user split | `user_split.pt` |
| Sentence-transformer + genre features | `item_emb.pt` (402-d) |
| Gender / age / occupation one-hot | `user_emb.pt` (30-d) |
| ID maps + cache key | `id_maps.pt`, `config_hash.txt` |

Default processed stats: **5,624 users**, **991k ratings**, **3,702 items**, **~176 ratings/user** (mean). Episodes use each cold user's first `T=20` chronologically ordered interactions.

## Configuration

`src/config.py` (`DataConfig`) controls filtering, warm fraction, embedding model, genre/demographic features, episode length (`cold_start_horizon_T=20`), and reward mode. Change preprocessing settings and re-run `python -m src.data.preprocess` to rebuild artifacts.

## Environment

`ColdStartEnv` (`src/data/env.py`) loads processed tensors and runs cold-start episodes. Policies implement `reset`, `select_action`, and `update`. Evaluation scripts share metric helpers (NDCG@5, cumulative reward) and support ranking vs selection protocols.
