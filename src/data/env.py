"""Cold-start recommendation environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class ColdStartEnv:
    """
    ratings_by_user: Dict[user_idx -> list of (item_idx, rating, timestamp)].
    item_emb: Cached item embedding matrix (n_items, emb_dim).
    config: DataConfig instance.
    user_pool: Users to sample episodes from.
    user_emb: Optional static user features (n_users, user_emb_dim).
    warm_users: Required when expose_user_features is "warm_only".
    rng: Optional numpy Generator.
    use_full_candidate_pool: If True, candidates are all rated items, not just the first T.
    reward_noise_std: Gaussian noise std added to rewards before clipping to [1, 5].
    """

    def __init__(
        self,
        ratings_by_user: Dict[int, List[Tuple[int, float, int]]],
        item_emb: torch.Tensor,
        config,
        user_pool: List[int],
        user_emb: Optional[torch.Tensor] = None,
        warm_users: Optional[Set[int]] = None,
        rng: Optional[np.random.Generator] = None,
        use_full_candidate_pool: bool = False,
        reward_noise_std: float = 0.0,
    ) -> None:
        self.ratings_by_user = ratings_by_user
        self.item_emb = item_emb
        self.user_emb = user_emb
        self.config = config
        self.user_pool = list(user_pool)
        self.warm_users = warm_users
        self.use_full_candidate_pool = use_full_candidate_pool
        self.reward_noise_std = reward_noise_std
        self.rng = rng if rng is not None else np.random.default_rng(config.random_seed)

        self.emb_dim: int = item_emb.shape[1]
        self.user_emb_dim: int = user_emb.shape[1] if user_emb is not None else 0
        self._validate_expose_user_features()

        self._current_user: Optional[int] = None
        self._candidates: List[int] = []
        self._candidate_rating: Dict[int, float] = {}
        self._revealed: List[Tuple[int, float]] = []
        self._t: int = 0

    def reset(self, user_idx: Optional[int] = None) -> Dict[str, Any]:
        """Start a new episode and return the initial state."""
        if user_idx is None:
            user_idx = int(self.rng.choice(self.user_pool))

        self._current_user = user_idx
        T = self.config.cold_start_horizon_T
        full_history = self.ratings_by_user[user_idx]

        if self.use_full_candidate_pool:
            candidate_window = full_history
        else:
            candidate_window = full_history[:T]

        self._candidates = [item_idx for item_idx, _, _ in candidate_window]
        self._candidate_rating = {
            item_idx: rating for item_idx, rating, _ in candidate_window
        }

        self._revealed = []
        self._t = 0

        return self._build_state()

    def step(
        self, action_item_idx: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Recommend one item and return (next_state, reward, done, info)."""
        if action_item_idx not in self._candidate_rating:
            raise ValueError(
                f"action_item_idx={action_item_idx} is not in the candidate set. "
                f"Candidates: {self._candidates[:10]}..."
            )

        raw_rating = self._candidate_rating[action_item_idx]
        reward = self._compute_reward(raw_rating)

        self._revealed.append((action_item_idx, reward))
        self._t += 1
        done = self._t >= self.config.cold_start_horizon_T

        info: Dict[str, Any] = {
            "user_idx": self._current_user,
            "item_idx": action_item_idx,
            "raw_rating": raw_rating,
            "reward": reward,
            "step": self._t,
            "remaining_candidates": len(self._candidates) - self._t,
        }

        return self._build_state(), reward, done, info

    @property
    def current_user(self) -> Optional[int]:
        return self._current_user

    @property
    def candidates(self) -> List[int]:
        return list(self._candidates)

    def _validate_expose_user_features(self) -> None:
        mode = self.config.expose_user_features
        valid_modes = {"never", "always", "warm_only"}
        if mode not in valid_modes:
            raise ValueError(
                f"expose_user_features must be one of {sorted(valid_modes)}, got {mode!r}"
            )
        if mode != "never" and self.user_emb is None:
            raise ValueError(
                "user_emb must be provided when expose_user_features is not 'never'"
            )
        if mode == "warm_only" and self.warm_users is None:
            raise ValueError(
                "warm_users must be provided when expose_user_features is 'warm_only'"
            )

    def _should_expose_user_features(self, user_idx: int) -> bool:
        mode = self.config.expose_user_features
        if mode == "never" or self.user_emb is None:
            return False
        if mode == "always":
            return True
        assert self.warm_users is not None
        return user_idx in self.warm_users

    def _get_user_emb(self, user_idx: Optional[int]) -> torch.Tensor:
        if user_idx is None or not self._should_expose_user_features(user_idx):
            return torch.zeros(self.user_emb_dim, dtype=torch.float32)
        assert self.user_emb is not None
        return self.user_emb[user_idx]

    def _compute_reward(self, raw_rating: float) -> float:
        if self.config.reward_mode == "binary":
            return 1.0 if raw_rating >= self.config.rating_threshold else 0.0
        reward = float(raw_rating)
        if self.reward_noise_std > 0.0:
            reward += float(self.rng.normal(0.0, self.reward_noise_std))
            reward = float(np.clip(reward, 1.0, 5.0))
        return reward

    def _build_state(self) -> Dict[str, Any]:
        if self._revealed:
            revealed_indices = [idx for idx, _ in self._revealed]
            revealed_emb = self.item_emb[revealed_indices]
        else:
            revealed_emb = torch.zeros(0, self.emb_dim)

        return {
            "history": list(self._revealed),
            "revealed_emb": revealed_emb,
            "user_emb": self._get_user_emb(self._current_user),
            "candidates": torch.tensor(self._candidates, dtype=torch.long),
            "t": self._t,
        }


class VectorizedColdStartEnv:
    """Run multiple ColdStartEnv episodes in parallel for batched training."""

    def __init__(self, envs: List[ColdStartEnv]) -> None:
        self.envs = envs
        self.B = len(envs)
        self.emb_dim = envs[0].emb_dim
        self.user_emb_dim = envs[0].user_emb_dim
        self._states: List[Dict[str, Any]] = [{}] * self.B

    def reset_all(self) -> Dict[str, Any]:
        self._states = [env.reset() for env in self.envs]
        return self._stack_states(self._states)

    def step_all(
        self, actions: List[int]
    ) -> Tuple[Dict[str, Any], List[float], List[bool], List[Dict[str, Any]]]:
        assert len(actions) == self.B
        rewards, dones, infos = [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, r, done, info = env.step(action)
            self._states[i] = state
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return self._stack_states(self._states), rewards, dones, infos

    def _stack_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_t = max(s["t"] for s in states)
        padded_embs = []
        masks = []
        for s in states:
            t = s["t"]
            emb = s["revealed_emb"]
            if t < max_t:
                pad = torch.zeros(max_t - t, self.emb_dim)
                emb = torch.cat([emb, pad], dim=0)
            padded_embs.append(emb)
            mask = torch.tensor([1] * t + [0] * (max_t - t), dtype=torch.bool)
            masks.append(mask)

        return {
            "revealed_emb": torch.stack(padded_embs, dim=0),
            "user_emb": torch.stack([s["user_emb"] for s in states], dim=0),
            "mask": torch.stack(masks, dim=0),
            "histories": [s["history"] for s in states],
            "candidates": [s["candidates"] for s in states],
            "t": max_t,
        }
