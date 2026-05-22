"""Cold-start bandit environment (the runtime "dataloader").

ColdStartEnv presents each user's interaction history one step at a time,
following a Gym-like (reset / step) interface so all three methods can use
the same loop:

    env = ColdStartEnv(ratings_by_user, item_emb, config, user_pool=cold_users)
    state = env.reset()
    for _ in range(config.cold_start_horizon_T):
        action = policy.act(state)
        state, reward, done, info = env.step(action)

Action space design choice (documented deliberately):
    The candidate set per episode is restricted to items the user ACTUALLY
    RATED within their first cold_start_horizon_T interactions. This is the
    tractable choice for logged-data evaluation: we always have a ground-truth
    reward for every candidate action. The trade-off is that the agent cannot
    recommend items outside this set, making the evaluation optimistic and
    limiting exploration realism. A principled alternative is the offline
    replay / rejection-sampling protocol (Li et al. 2010), which we plan to
    revisit in the ablations.
    # TODO: implement optional replay-protocol evaluation for unbiased offline
    #       evaluation (requires a broader candidate pool + rejection sampling).

Hooks for future model components:
    # TODO (baseline): load processed/baseline.pt here and expose
    #   self.mu_base (non-personalised score vector over items) and
    #   self.baseline_rewards (expected cumulative reward curve) so the
    #   Constrained Bandit's feasibility check can read them.
    # TODO (Neural Linear): the revealed_emb tensor in the state dict is the
    #   design matrix Φ_t; the reward vector is the label vector y_t.
    # TODO (Meta-RL / RL²): feed state['history'] as the token sequence into
    #   the recurrent policy. Use VectorizedColdStartEnv below for batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class ColdStartEnv:
    """Single-episode cold-start recommendation environment.

    Parameters
    ----------
    ratings_by_user:
        Dict mapping user_idx -> list of (item_idx, rating, timestamp) in
        chronological order (produced by preprocess.build_ratings_by_user).
    item_emb:
        Tensor of shape (n_items, emb_dim) — the cached φ(x) matrix.
    user_emb:
        Optional tensor of shape (n_users, user_emb_dim) — static demographic
        features ψ(u). Pass None if use_user_features is False.
    config:
        DataConfig instance.
    user_pool:
        List of user_idxs this env draws episodes from. Pass the cold set
        for evaluation, or the warm set for any warm-user diagnostics.
    warm_users:
        Set of warm user indices. Required when expose_user_features is
        "warm_only" so the env knows which users may receive side info.
    rng:
        Optional seeded numpy Generator. Created from config.random_seed if
        not provided.
    """

    def __init__(
        self,
        ratings_by_user: Dict[int, List[Tuple[int, float, int]]],
        item_emb: torch.Tensor,
        config,  # DataConfig — not annotated to avoid circular import
        user_pool: List[int],
        user_emb: Optional[torch.Tensor] = None,
        warm_users: Optional[Set[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.ratings_by_user = ratings_by_user
        self.item_emb = item_emb          # (n_items, emb_dim)
        self.user_emb = user_emb
        self.config = config
        self.user_pool = list(user_pool)
        self.warm_users = warm_users
        self.rng = rng if rng is not None else np.random.default_rng(config.random_seed)

        self.emb_dim: int = item_emb.shape[1]
        self.user_emb_dim: int = user_emb.shape[1] if user_emb is not None else 0
        self._validate_expose_user_features()

        # Populated by reset()
        self._current_user: Optional[int] = None
        self._episode_history: List[Tuple[int, float, int]] = []  # full chrono list
        self._candidates: List[int] = []   # item_idxs available this episode
        self._candidate_rating: Dict[int, float] = {}
        self._revealed: List[Tuple[int, float]] = []  # (item_idx, reward) so far
        self._t: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, user_idx: Optional[int] = None) -> Dict[str, Any]:
        """Start a new episode.

        Parameters
        ----------
        user_idx:
            If given, use this specific user. Otherwise sample uniformly
            from user_pool.

        Returns
        -------
        state dict at t=0 (no items revealed yet).
        """
        if user_idx is None:
            user_idx = int(self.rng.choice(self.user_pool))

        self._current_user = user_idx
        T = self.config.cold_start_horizon_T

        full_history = self.ratings_by_user[user_idx]  # chronological
        # Episode window: first T interactions
        episode_window = full_history[:T]

        self._episode_history = episode_window
        self._candidates = [item_idx for item_idx, _, _ in episode_window]
        self._candidate_rating = {
            item_idx: rating for item_idx, rating, _ in episode_window
        }

        # TODO: optional negative sampling — add unrated items to _candidates
        # and assign reward 0.0 (or use rejection-sampling protocol).

        self._revealed = []
        self._t = 0

        return self._build_state()

    def step(
        self, action_item_idx: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take one recommendation step.

        Parameters
        ----------
        action_item_idx:
            The item the policy recommends. Must be in the candidate set.

        Returns
        -------
        (next_state, reward, done, info)
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        return float(raw_rating)  # "raw": return the 1-5 rating directly

    def _build_state(self) -> Dict[str, Any]:
        """Construct the state dict exposed to the policy.

        Keys
        ----
        history:
            List of (item_idx, reward) tuples for the steps revealed so far.
        revealed_emb:
            Tensor (t, emb_dim) — stacked φ(x) for each revealed item.
            Shape is (0, emb_dim) at t=0 so downstream code can always
            concatenate without a special-case branch.
        user_emb:
            Tensor (user_emb_dim,) — static demographic features ψ(u), or
            zeros when expose_user_features is "never" or the user is cold
            under "warm_only".
        candidates:
            1-D LongTensor of candidate item_idxs for this episode.
        t:
            Current step index (0 before any step has been taken).
        """
        if self._revealed:
            revealed_indices = [idx for idx, _ in self._revealed]
            revealed_emb = self.item_emb[revealed_indices]  # (t, emb_dim)
        else:
            revealed_emb = torch.zeros(0, self.emb_dim)

        return {
            "history": list(self._revealed),
            "revealed_emb": revealed_emb,
            "user_emb": self._get_user_emb(self._current_user),
            "candidates": torch.tensor(self._candidates, dtype=torch.long),
            "t": self._t,
        }


# ---------------------------------------------------------------------------
# Batched wrapper for RL² / recurrent policies
# ---------------------------------------------------------------------------

class VectorizedColdStartEnv:
    """Run B independent ColdStartEnv episodes in parallel.

    States are stacked into (B, t, ...) tensors with a padding mask for
    variable-length histories, so a recurrent policy can train on mini-batches.
    The single-env API (reset/step) is preserved per-slot, so Neural Linear
    and Constrained Bandit code that uses ColdStartEnv directly is unchanged.

    Parameters
    ----------
    envs:
        List of B pre-constructed ColdStartEnv instances (can share
        ratings_by_user and item_emb by reference).
    """

    def __init__(self, envs: List[ColdStartEnv]) -> None:
        self.envs = envs
        self.B = len(envs)
        self.emb_dim = envs[0].emb_dim
        self.user_emb_dim = envs[0].user_emb_dim
        self._states: List[Dict[str, Any]] = [{}] * self.B

    def reset_all(self) -> Dict[str, Any]:
        """Reset all B environments and return a batched state dict."""
        self._states = [env.reset() for env in self.envs]
        return self._stack_states(self._states)

    def step_all(
        self, actions: List[int]
    ) -> Tuple[Dict[str, Any], List[float], List[bool], List[Dict[str, Any]]]:
        """Step all B environments with the given per-env actions.

        Parameters
        ----------
        actions:
            List of length B with one action per environment.

        Returns
        -------
        (batched_state, rewards, dones, infos)
        """
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
        """Stack per-env states; pad revealed_emb along the time axis."""
        max_t = max(s["t"] for s in states)
        padded_embs = []
        masks = []
        for s in states:
            t = s["t"]
            emb = s["revealed_emb"]  # (t, emb_dim)
            if t < max_t:
                pad = torch.zeros(max_t - t, self.emb_dim)
                emb = torch.cat([emb, pad], dim=0)
            padded_embs.append(emb)
            mask = torch.tensor([1] * t + [0] * (max_t - t), dtype=torch.bool)
            masks.append(mask)

        return {
            "revealed_emb": torch.stack(padded_embs, dim=0),  # (B, max_t, emb_dim)
            "user_emb": torch.stack([s["user_emb"] for s in states], dim=0),  # (B, user_emb_dim)
            "mask": torch.stack(masks, dim=0),                  # (B, max_t)
            "histories": [s["history"] for s in states],
            "candidates": [s["candidates"] for s in states],
            "t": max_t,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from src.config import DataConfig

    config = DataConfig()
    processed = Path(config.data_dir) / "processed"

    required = [
        "id_maps.pt",
        "ratings_by_user.pt",
        "user_split.pt",
        "item_emb.pt",
        "user_emb.pt",
    ]
    missing = [f for f in required if not (processed / f).exists()]
    if missing:
        print(
            f"Missing preprocessed artifacts: {missing}\n"
            "Run `python -m src.data.preprocess` first."
        )
        sys.exit(1)

    print("Loading cached artifacts...")
    ratings_by_user = torch.load(processed / "ratings_by_user.pt", weights_only=False)
    user_split = torch.load(processed / "user_split.pt", weights_only=False)
    item_emb = torch.load(processed / "item_emb.pt", weights_only=False)
    user_emb = torch.load(processed / "user_emb.pt", weights_only=False)
    cold_users = user_split["cold"]
    warm_users = set(user_split["warm"])

    print(f"Cold users available: {len(cold_users)}")
    print(f"Item embedding shape: {tuple(item_emb.shape)}")
    print(f"User embedding shape: {tuple(user_emb.shape)}")

    env = ColdStartEnv(
        ratings_by_user,
        item_emb,
        config,
        user_pool=cold_users,
        user_emb=user_emb,
        warm_users=warm_users,
    )

    print("\n--- Reset ---")
    state = env.reset()
    print(f"  user_idx:          {env.current_user}")
    print(f"  candidates:        {state['candidates'].shape}  (# items in episode)")
    print(f"  revealed_emb:      {state['revealed_emb'].shape}")
    print(f"  user_emb:          {state['user_emb'].shape}")
    print(f"  history:           {state['history']}")
    print(f"  t:                 {state['t']}")

    print("\n--- 5 random steps ---")
    for step_num in range(1, 6):
        candidate_list = state["candidates"].tolist()
        action = int(np.random.choice(candidate_list))
        state, reward, done, info = env.step(action)
        print(
            f"  step={step_num} | action={action} | reward={reward:.2f} "
            f"| raw_rating={info['raw_rating']} | done={done}"
        )
        print(f"    revealed_emb shape: {state['revealed_emb'].shape}")
        print(f"    history length:     {len(state['history'])}")

    print("\nSmoke test passed.")
