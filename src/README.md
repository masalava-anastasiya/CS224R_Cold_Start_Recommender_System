# Cold-Start Movie Recommendation with Contextual Bandits

This project studies **cold-start recommendation** on MovieLens-1M. Warm users train priors and baselines; cold users are evaluated in a fixed-horizon bandit setting where each policy selects items from a candidate pool and receives logged ratings as rewards. The codebase implements greedy CF, Thompson Sampling variants, a constrained LinUCB bandit, and an RL² meta-policy, with scripts for preprocessing, training, evaluation, and plotting.

## Data (not included)

Place the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) files next to `src/` at the repo root:

```
data/
  raw/       ratings.dat, movies.dat, users.dat
  processed/ written by src.data.preprocess
```

Run preprocessing from the parent directory: `python -m src.data.preprocess`

## `src/` layout

```
src/
  config.py                 # DataConfig
  data/
    preprocess.py           # build processed artifacts
    encoders.py             # item/user feature encoders
    env.py                  # ColdStartEnv
    visualize_dataset.py    # dataset summary figures
  baselines/
    greedy_cf.py            # Greedy CF (SVD + ridge update)
    random_baseline.py
    popularity_baseline.py
  methods/
    neural_linear_ts.py         # NLTS on frozen embeddings
    hybrid_neural_linear_ts.py  # TS on CF latent factors
    neural_factor_ts.py         # reward-trained item tower + TS
    demographic_prior_ts.py       # demographic prior mean
    constrained_bandit.py       # LinUCB with popularity fallback
    rl2_policy.py                 # LSTM meta-policy
  train/
    train_rl2.py              # meta-train RL2 on warm users
  eval/
    evaluate_*.py             # per-experiment eval scripts
    collect_per_step_rewards.py
    niche_case_study.py
    plot_results.py           # figures from saved JSON
```
