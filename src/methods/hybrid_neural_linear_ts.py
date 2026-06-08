"""Hybrid Thompson Sampling on CF latent factors."""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import numpy as np
from sklearn.decomposition import TruncatedSVD


class HybridNeuralLinearTS:

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

        self._fit(ratings_by_user, warm_users)

        self._Lambda: np.ndarray = np.eye(k)
        self._b: np.ndarray = np.zeros(k)
        self._mu: np.ndarray = np.zeros(k)
        self._selected: Set[int] = set()
        self._rng = np.random.default_rng(42)

    def _fit(self, ratings_by_user: Dict, warm_users: List[int]) -> None:
        n_warm = len(warm_users)
        user_to_row = {u: i for i, u in enumerate(warm_users)}

        R = np.zeros((n_warm, self.n_items), dtype=np.float32)
        for u in warm_users:
            row = user_to_row[u]
            for item_idx, rating, _ in ratings_by_user[u]:
                R[row, item_idx] = float(rating)

        rated = R > 0
        counts = rated.sum(axis=0)
        self.item_means: np.ndarray = np.where(
            counts > 0, R.sum(axis=0) / np.maximum(counts, 1), 0.0
        ).astype(np.float64)

        R_centered = R.astype(np.float64)
        r_idx, c_idx = np.where(rated)
        R_centered[r_idx, c_idx] -= self.item_means[c_idx]

        svd = TruncatedSVD(n_components=self.k, random_state=42)
        U_sigma = svd.fit_transform(R_centered)
        self.Q: np.ndarray = svd.components_.T

        self._mu_0: np.ndarray = U_sigma.mean(axis=0).astype(np.float64)

        user_cov = np.cov(U_sigma.T)
        user_cov += 1e-6 * np.eye(self.k)
        self._Lambda_0: np.ndarray = np.linalg.inv(user_cov) * self.lambda_prior

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._Lambda = self._Lambda_0.copy()
        self._b = self._Lambda @ self._mu_0
        self._mu = self._mu_0.copy()
        self._selected = set()

    def select_action(self, state: Dict) -> int:
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates

        try:
            L = np.linalg.cholesky(self._Lambda)
            z = self._rng.standard_normal(self.k)
            x = np.linalg.solve(L.T, z)
            theta_sample = self._mu + x
        except np.linalg.LinAlgError:
            theta_sample = self._mu

        Q_avail = self.Q[available]
        scores = Q_avail @ theta_sample + self.item_means[available]

        best_idx = int(np.argmax(scores))
        return available[best_idx]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)

        q_a = self.Q[action]
        centered_reward = reward - self.item_means[action]

        self._Lambda += self.noise_precision * np.outer(q_a, q_a)
        self._b += self.noise_precision * centered_reward * q_a
        self._mu = np.linalg.solve(self._Lambda, self._b)
