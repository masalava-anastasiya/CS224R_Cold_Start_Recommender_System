"""Hybrid Neural Linear Thompson Sampling with CF latent features.

Instead of raw 402-dim content embeddings, this variant uses the
k-dim SVD latent factors learned by collaborative filtering on warm
users. This gives Thompson Sampling the same expressive feature space
as the greedy CF baseline, plus principled exploration.

The intuition: CF latent factors already encode "which types of users
like which types of movies" from actual rating patterns. TS on top of
these features explores in the *right* directions — trying movies that
are informative about the cold user's position in latent space.

This is the "best of both worlds" — CF's informative features + TS's
exploration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.decomposition import TruncatedSVD


class HybridNeuralLinearTS:
    """Thompson Sampling using CF latent factors as features.

    Fits item factors Q (n_items, k) via truncated SVD on warm users,
    then runs Bayesian linear regression in the k-dim latent space.
    Much smaller feature space than raw embeddings → posterior updates
    are stronger and convergence is faster.

    Parameters
    ----------
    ratings_by_user:
        Full interaction dict from ratings_by_user.pt.
    warm_users:
        List of warm user indices (for SVD training).
    n_items:
        Total number of items.
    k:
        Number of latent factors for SVD (default 50, same as greedy CF).
    lambda_prior:
        Prior precision scale for the Bayesian regression.
    sigma_noise:
        Observation noise std dev.
    """

    def __init__(
        self,
        ratings_by_user: Dict,
        warm_users: List[int],
        n_items: int,
        k: int = 50,
        lambda_prior: float = 1.0,
        sigma_noise: float = 1.0,
    ) -> None:
        self.k = k
        self.n_items = n_items
        self.lambda_prior = lambda_prior
        self.sigma_noise = sigma_noise
        self.noise_precision = 1.0 / (sigma_noise ** 2)

        # learn the latent factors and prior from warm users
        self._fit(ratings_by_user, warm_users)

        # posterior state (set properly in reset)
        self._Lambda: np.ndarray = np.eye(k)
        self._b: np.ndarray = np.zeros(k)
        self._mu: np.ndarray = np.zeros(k)
        self._selected: Set[int] = set()
        self._rng = np.random.default_rng(42)

    def _fit(self, ratings_by_user: Dict, warm_users: List[int]) -> None:
        """Fit item factors Q and compute the warm-user prior."""
        n_warm = len(warm_users)
        user_to_row = {u: i for i, u in enumerate(warm_users)}

        # build the warm-user rating matrix
        R = np.zeros((n_warm, self.n_items), dtype=np.float32)
        for u in warm_users:
            row = user_to_row[u]
            for item_idx, rating, _ in ratings_by_user[u]:
                R[row, item_idx] = float(rating)

        # item means (for centering predictions)
        rated = R > 0
        counts = rated.sum(axis=0)
        self.item_means: np.ndarray = np.where(
            counts > 0, R.sum(axis=0) / np.maximum(counts, 1), 0.0
        ).astype(np.float64)

        # mean-center only rated entries
        R_centered = R.astype(np.float64)
        r_idx, c_idx = np.where(rated)
        R_centered[r_idx, c_idx] -= self.item_means[c_idx]

        # truncated SVD: R ≈ U Σ V^T, item factors Q = V^T transposed
        svd = TruncatedSVD(n_components=self.k, random_state=42)
        U_sigma = svd.fit_transform(R_centered)  # (n_warm, k)
        self.Q: np.ndarray = svd.components_.T    # (n_items, k)

        # warm-user prior: average latent user vector from warm data
        self._mu_0: np.ndarray = U_sigma.mean(axis=0).astype(np.float64)

        # empirical prior precision from the variance of warm-user vectors
        # this captures where users agree (tight) vs disagree (loose)
        user_cov = np.cov(U_sigma.T)  # (k, k)
        # regularize slightly to ensure invertibility
        user_cov += 1e-6 * np.eye(self.k)
        self._Lambda_0: np.ndarray = np.linalg.inv(user_cov) * self.lambda_prior

    def reset(self, user_idx: Optional[int] = None) -> None:
        """Reset posterior to the warm-user prior for a new episode."""
        self._Lambda = self._Lambda_0.copy()
        self._b = self._Lambda @ self._mu_0
        self._mu = self._mu_0.copy()
        self._selected = set()

    def select_action(self, state: Dict) -> int:
        """Sample theta from posterior, pick the best candidate."""
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates

        # Thompson sample: theta ~ N(mu, Lambda^{-1})
        try:
            L = np.linalg.cholesky(self._Lambda)
            z = self._rng.standard_normal(self.k)
            x = np.linalg.solve(L.T, z)
            theta_sample = self._mu + x
        except np.linalg.LinAlgError:
            theta_sample = self._mu

        # score candidates in latent space + item means
        Q_avail = self.Q[available]
        scores = Q_avail @ theta_sample + self.item_means[available]

        best_idx = int(np.argmax(scores))
        return available[best_idx]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        """Bayesian update in the k-dim latent space."""
        self._selected.add(action)

        q_a = self.Q[action]  # (k,) — much smaller than 402-dim phi!

        # center the reward before regression (same as greedy CF)
        centered_reward = reward - self.item_means[action]

        # rank-1 precision update
        self._Lambda += self.noise_precision * np.outer(q_a, q_a)

        # information vector update
        self._b += self.noise_precision * centered_reward * q_a

        # recompute posterior mean
        self._mu = np.linalg.solve(self._Lambda, self._b)
