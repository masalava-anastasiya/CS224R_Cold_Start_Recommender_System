"""Greedy collaborative-filtering baseline policy."""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import numpy as np
from sklearn.decomposition import TruncatedSVD


class GreedyCFBaseline:
    """
    ratings_by_user: Full interaction dict from ratings_by_user.pt.
    warm_users: List of warm user indices.
    n_items: Total number of items.
    k: Number of latent factors for truncated SVD.
    reg: Ridge regularisation coefficient for the per-episode user-vector update.
    """

    def __init__(
        self,
        ratings_by_user: Dict,
        warm_users: List[int],
        n_items: int,
        k: int = 50,
        reg: float = 1.0,
    ) -> None:
        self.k = k
        self.reg = reg
        self.n_items = n_items
        self._fit(ratings_by_user, warm_users)
        self._selected: Set[int] = set()
        self._history_items: List[int] = []
        self._history_rewards: List[float] = []
        self._theta: np.ndarray = np.zeros(k, dtype=np.float64)

    def _fit(self, ratings_by_user: Dict, warm_users: List[int]) -> None:
        """Fit item factors Q (n_items x k) from warm-user ratings only."""
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
        self._global_prior: np.ndarray = U_sigma.mean(axis=0)

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._selected = set()
        self._history_items = []
        self._history_rewards = []
        self._theta = self._global_prior.copy()

    def select_action(self, state: Dict) -> int:
        """Return the candidate item index with the highest predicted rating.
        state: State dict from ColdStartEnv.
        Returns: item_idx:int
        """
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates 
        Q_sub = self.Q[available]
        preds = Q_sub @ self._theta + self.item_means[available]
        return available[int(np.argmax(preds))]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        """Observe the reward and update the per-episode user vector.
        """
        self._selected.add(action)
        self._history_items.append(action)
        self._history_rewards.append(reward)

        Q_hist = self.Q[self._history_items]          
        centered = (
            np.array(self._history_rewards, dtype=np.float64)
            - self.item_means[self._history_items]
        )
        A = Q_hist.T @ Q_hist + self.reg * np.eye(self.k)
        b = Q_hist.T @ centered + self.reg * self._global_prior
        self._theta = np.linalg.solve(A, b)
