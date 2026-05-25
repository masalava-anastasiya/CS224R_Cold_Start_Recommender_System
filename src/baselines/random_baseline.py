"""Random recommendation baseline.

Selects uniformly at random from the available (not yet recommended)
candidates.  No training.  Used as a lower-bound reference.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set

import numpy as np


class RandomBaseline:
    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._selected: Set[int] = set()

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._selected = set()

    def select_action(self, state: Dict) -> int:
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates
        return int(self._rng.choice(available))

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)
