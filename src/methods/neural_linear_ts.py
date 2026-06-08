"""Neural Linear Thompson Sampling on frozen item embeddings."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import torch


def compute_warm_prior(
    ratings_by_user: Dict,
    warm_users: List[int],
    item_emb: torch.Tensor,
    reg: float = 1.0,
) -> tuple:
    phi = item_emb.numpy().astype(np.float64)
    d = phi.shape[1]

    A = reg * np.eye(d, dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)

    for u in warm_users:
        interactions = ratings_by_user[u]
        item_idxs = [iid for iid, _, _ in interactions]
        rewards = np.array([r for _, r, _ in interactions], dtype=np.float64)

        Phi_u = phi[item_idxs]
        A += Phi_u.T @ Phi_u
        b += Phi_u.T @ rewards

    mu_0 = np.linalg.solve(A, b)

    scale = (reg * d) / np.trace(A)
    Lambda_0 = A * scale

    return mu_0, Lambda_0


class NeuralLinearTS:

    def __init__(
        self,
        item_emb: torch.Tensor,
        lambda_prior: float = 1.0,
        sigma_noise: float = 1.0,
        mu_0: Optional[np.ndarray] = None,
        Lambda_0: Optional[np.ndarray] = None,
    ) -> None:
        self.phi = item_emb.numpy().astype(np.float64)
        self.d = self.phi.shape[1]
        self.lambda_prior = lambda_prior
        self.sigma_noise = sigma_noise
        self.noise_precision = 1.0 / (sigma_noise ** 2)

        if mu_0 is not None:
            self._mu_0 = mu_0.astype(np.float64)
        else:
            self._mu_0 = np.zeros(self.d, dtype=np.float64)

        if Lambda_0 is not None:
            self._Lambda_0 = Lambda_0.astype(np.float64)
        else:
            self._Lambda_0 = lambda_prior * np.eye(self.d, dtype=np.float64)

        self._Lambda: np.ndarray = np.eye(self.d)
        self._b: np.ndarray = np.zeros(self.d)
        self._mu: np.ndarray = np.zeros(self.d)
        self._selected: Set[int] = set()
        self._rng = np.random.default_rng(42)

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
            z = self._rng.standard_normal(self.d)
            x = np.linalg.solve(L.T, z)
            theta_sample = self._mu + x
        except np.linalg.LinAlgError:
            theta_sample = self._mu

        Phi_avail = self.phi[available]
        scores = Phi_avail @ theta_sample

        best_idx = int(np.argmax(scores))
        return available[best_idx]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)

        phi_a = self.phi[action]

        self._Lambda += self.noise_precision * np.outer(phi_a, phi_a)
        self._b += self.noise_precision * reward * phi_a
        self._mu = np.linalg.solve(self._Lambda, self._b)
