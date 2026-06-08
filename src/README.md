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
  config.py                      # paths, split, embeddings, rewards
  data/
    preprocess.py                # MovieLens -> processed tensors
    encoders.py                  # sentence-transformer + demographic encoders
    env.py                       # cold-start episode simulator
    visualize_dataset.py         # dataset summary plots
  baselines/
    greedy_cf.py                 # SVD collaborative filtering, greedy pick
    random_baseline.py           # uniform random recommendations
    popularity_baseline.py       # warm-user popularity ranking
  methods/
    neural_linear_ts.py          # Thompson Sampling on item embeddings
    hybrid_neural_linear_ts.py   # TS on SVD latent factors
    neural_factor_ts.py          # learned item tower + TS
    demographic_prior_ts.py      # TS with demographic prior mean
    constrained_bandit.py        # safe LinUCB with popularity fallback
    rl2_policy.py                # LSTM in-context meta-policy
  train/
    train_rl2.py                 # REINFORCE meta-training on warm users
  eval/
    evaluate_greedy_cf.py        # Greedy CF vs Random
    evaluate_hybrid.py           # Hybrid TS vs baselines
    evaluate_neural_linear.py    # NLTS vs baselines
    evaluate_neural_factor.py    # neural-factor TS vs baselines
    evaluate_demographic_prior.py  # demographic TS vs baselines
    evaluate_constrained_bandit.py # constrained LinUCB eval
    evaluate_exploration.py      # selection-protocol comparison
    evaluate_noisy_rewards.py    # reward-noise robustness sweep
    evaluate_prior_ablation.py   # warm-fraction prior ablation
    evaluate_mismatch_prior.py   # matched vs mismatched demographics
    evaluate_longer_horizon.py   # T > 20 horizon study
    evaluate_rl2.py              # RL2 vs baselines
    collect_per_step_rewards.py  # per-step reward curves for RL2
    niche_case_study.py          # niche-user quartile analysis
    plot_results.py              # poster figures from result JSON
```

