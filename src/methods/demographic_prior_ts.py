"""Thompson Sampling with a demographic-conditioned prior mean."""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


class DemographicPriorTS:

    def __init__(
        self,
        ratings_by_user: Dict,
        warm_users: List[int],
        n_items: int,
        user_emb: torch.Tensor,
        k: int = 50,
        lambda_prior: float = 1.0,
        sigma_noise: float = 1.0,
        ridge_alpha: float = 10.0,
    ) -> None:
        self.k = k
        self.n_items = n_items
        self.noise_precision = 1.0 / (sigma_noise ** 2)
        self.user_emb = user_emb.numpy().astype(np.float64)
        self.ridge_alpha = ridge_alpha
        self._fit(ratings_by_user, warm_users, lambda_prior)

        self._Lambda = np.eye(k)
        self._b = np.zeros(k)
        self._mu = np.zeros(k)
        self._mu_0_user = self._mu_0_global.copy()
        self._selected: Set[int] = set()
        self._rng = np.random.default_rng(42)

    def _fit(self, ratings_by_user: Dict, warm_users: List[int], lambda_prior: float) -> None:
        n_warm = len(warm_users)
        user_to_row = {u: i for i, u in enumerate(warm_users)}

        R = np.zeros((n_warm, self.n_items), dtype=np.float32)
        for u in warm_users:
            for item_idx, rating, _ in ratings_by_user[u]:
                R[user_to_row[u], item_idx] = float(rating)

        rated = R > 0
        counts = rated.sum(axis=0)
        self.item_means = np.where(
            counts > 0, R.sum(axis=0) / np.maximum(counts, 1), 0.0
        ).astype(np.float64)

        R_centered = R.astype(np.float64)
        r_idx, c_idx = np.where(rated)
        R_centered[r_idx, c_idx] -= self.item_means[c_idx]

        svd = TruncatedSVD(n_components=self.k, random_state=42)
        U_sigma = svd.fit_transform(R_centered)
        self.Q = svd.components_.T

        self._mu_0_global = U_sigma.mean(axis=0).astype(np.float64)

        cov = np.cov(U_sigma.T) + 1e-6 * np.eye(self.k)
        self._Lambda_0 = np.linalg.inv(cov) * lambda_prior

        Psi = self.user_emb[warm_users]
        d = Psi.shape[1]
        A = Psi.T @ Psi + self.ridge_alpha * np.eye(d)
        B = Psi.T @ U_sigma
        self._W = np.linalg.solve(A, B)

    def _prior_mean_for(self, user_idx: Optional[int]) -> np.ndarray:
        if user_idx is None:
            return self._mu_0_global.copy()
        psi = self.user_emb[user_idx]
        if not np.any(psi):
            return self._mu_0_global.copy()
        return psi @ self._W

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._mu_0_user = self._prior_mean_for(user_idx)
        self._Lambda = self._Lambda_0.copy()
        self._b = self._Lambda @ self._mu_0_user
        self._mu = self._mu_0_user.copy()
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
        scores = self.Q[available] @ theta_sample + self.item_means[available]
        return available[int(np.argmax(scores))]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)
        q_a = self.Q[action]
        centered = reward - self.item_means[action]
        self._Lambda += self.noise_precision * np.outer(q_a, q_a)
        self._b += self.noise_precision * centered * q_a
        self._mu = np.linalg.solve(self._Lambda, self._b)
