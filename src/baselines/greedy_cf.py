"""Greedy collaborative-filtering baseline policy.

Item latent factors are learned from warm-user ratings only via truncated SVD.
For each cold-user episode the policy maintains a per-episode user vector
(theta) that is updated via MAP ridge regression after every revealed rating,
then selects the candidate with the highest predicted rating greedily.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.decomposition import TruncatedSVD


class GreedyCFBaseline:
    """Greedy collaborative-filtering baseline policy.

    Parameters
    ----------
    ratings_by_user:
        Full interaction dict from ratings_by_user.pt.
        Format: {user_idx: [(item_idx, rating, timestamp), ...]}.
    warm_users:
        List of warm user indices.  Only these users contribute to MF training.
    n_items:
        Total number of items (item_emb.shape[0]).
    k:
        Number of latent factors for truncated SVD.
    reg:
        Ridge regularisation coefficient for the per-episode user-vector update.
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

        # per-episode state (initialised by reset)
        self._selected: Set[int] = set()
        self._history_items: List[int] = []
        self._history_rewards: List[float] = []
        self._theta: np.ndarray = np.zeros(k, dtype=np.float64)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _fit(self, ratings_by_user: Dict, warm_users: List[int]) -> None:
        """Fit item factors Q (n_items x k) from warm-user ratings only."""
        n_warm = len(warm_users)
        user_to_row = {u: i for i, u in enumerate(warm_users)}

        # Dense warm-user rating matrix (n_warm x n_items).
        # 3936 x 3706 x 4 bytes ~= 58 MB -- fits comfortably in RAM.
        R = np.zeros((n_warm, self.n_items), dtype=np.float32)
        for u in warm_users:
            row = user_to_row[u]
            for item_idx, rating, _ in ratings_by_user[u]:
                R[row, item_idx] = float(rating)

        # Item means over warm users who actually rated the item.
        rated = R > 0                                     # (n_warm, n_items) bool
        counts = rated.sum(axis=0)                        # (n_items,)
        self.item_means: np.ndarray = np.where(
            counts > 0, R.sum(axis=0) / np.maximum(counts, 1), 0.0
        ).astype(np.float64)

        # Mean-centre only the rated entries so unrated zeros stay zero.
        R_centered = R.astype(np.float64)
        r_idx, c_idx = np.where(rated)
        R_centered[r_idx, c_idx] -= self.item_means[c_idx]

        # Truncated SVD: R_centered ~= U Sigma V^T
        # Uses randomised SVD internally -- fast even for dense matrices.
        svd = TruncatedSVD(n_components=self.k, random_state=42)
        U_sigma = svd.fit_transform(R_centered)  # (n_warm, k)
        self.Q: np.ndarray = svd.components_.T   # (n_items, k)  item factors

        # Global prior: mean warm-user vector in latent space.
        # At t=0 (no cold-user history) theta is set to this, so the policy
        # recommends items that appeal to the average warm user.
        self._global_prior: np.ndarray = U_sigma.mean(axis=0)  # (k,)

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, user_idx: Optional[int] = None) -> None:
        """Reset per-episode state.  user_idx accepted but not used
        (the policy has no side information about the cold user)."""
        self._selected = set()
        self._history_items = []
        self._history_rewards = []
        self._theta = self._global_prior.copy()

    def select_action(self, state: Dict) -> int:
        """Return the candidate item index with the highest predicted rating.

        Parameters
        ----------
        state:
            State dict from ColdStartEnv.  Only ``state["candidates"]``
            (LongTensor of item indices) is used.

        Returns
        -------
        item_idx : int
        """
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates  # fallback; should not occur in normal use

        Q_sub = self.Q[available]                          # (n_avail, k)
        preds = Q_sub @ self._theta + self.item_means[available]
        return available[int(np.argmax(preds))]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        """Observe the reward and update the per-episode user vector.

        MAP ridge regression regularised toward the global warm-user prior t0:
            theta* = argmin_theta  ||Q_hist @ theta - centered_rewards||^2
                                   + reg * ||theta - t0||^2
        Normal equations: (Q^T Q + reg*I) theta = Q^T y + reg * t0
        """
        self._selected.add(action)
        self._history_items.append(action)
        self._history_rewards.append(reward)

        Q_hist = self.Q[self._history_items]              # (t, k)
        centered = (
            np.array(self._history_rewards, dtype=np.float64)
            - self.item_means[self._history_items]
        )

        # Regularise toward global_prior so early steps stay near the
        # warm-user average rather than collapsing toward zero.
        A = Q_hist.T @ Q_hist + self.reg * np.eye(self.k)
        b = Q_hist.T @ centered + self.reg * self._global_prior
        self._theta = np.linalg.solve(A, b)
