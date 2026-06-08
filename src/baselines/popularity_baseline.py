"""Popularity-based baseline policy."""
from __future__ import annotations
from typing import Dict, List, Optional, Set
import numpy as np

class PopularityBaseline:
    """
    ratings_by_user: Full interaction dict from ratings_by_user.pt.
    warm_users: List of warm user indices.
    n_items: Total number of items.
    shrinkage: Bayesian shrinkage strength toward the global mean.
    """
    def __init__(
        self,
        ratings_by_user: Dict,
        warm_users: List[int],
        n_items: int,
        shrinkage: float = 10.0,
    ) -> None:
        self.n_items = n_items
        self._fit(ratings_by_user, warm_users, shrinkage)
        self._selected: Set[int] = set()

    def _fit(self, ratings_by_user: Dict, warm_users: List[int], shrinkage: float) -> None:
        """Compute shrinkage-adjusted item means from warm-user ratings only."""
        counts = np.zeros(self.n_items, dtype=np.float64)
        sums   = np.zeros(self.n_items, dtype=np.float64)
        for u in warm_users:
            for item_idx, rating, _ in ratings_by_user[u]:
                counts[item_idx] += 1
                sums[item_idx]   += float(rating)
        global_mean = float(sums.sum() / max(counts.sum(), 1.0))
        self.item_scores: np.ndarray = (
            (sums + shrinkage * global_mean) / (counts + shrinkage)
        )
        self._global_mean: float = global_mean

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._selected = set()

    def select_action(self, state: Dict) -> int:
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates
        return available[int(np.argmax(self.item_scores[available]))]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)

    def score_candidates(self, candidates: List[int]) -> np.ndarray:
        return self.item_scores[np.asarray(candidates, dtype=int)]

    def expected_score(self, item_idx: int) -> float:
        return float(self.item_scores[item_idx])
