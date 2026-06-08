"""Constrained LinUCB bandit with popularity baseline fallback."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch


class ConstrainedLinearUCBBandit:

    def __init__(
        self,
        item_emb: torch.Tensor,
        baseline_policy,
        lambda_reg: float = 1.0,
        sigma2: float = 1.0,
        beta: float = 1.0,
        beta_safe: float = 2.0,
        alpha: float = 0.90,
        prior_mean: Optional[np.ndarray] = None,
        constraint_mode: str = "cumulative",
    ) -> None:
        self.phi = item_emb.detach().cpu().numpy().astype(np.float64)
        self.d = self.phi.shape[1]
        self.baseline = baseline_policy
        self.lambda_reg = lambda_reg
        self.sigma2 = sigma2
        self.noise_prec = 1.0 / sigma2
        self.beta = beta
        self.beta_safe = beta_safe
        self.alpha = alpha
        self.prior_mean = (
            np.zeros(self.d, dtype=np.float64)
            if prior_mean is None
            else np.asarray(prior_mean, dtype=np.float64)
        )
        if constraint_mode not in ("per_step", "cumulative"):
            raise ValueError(
                f"constraint_mode must be 'per_step' or 'cumulative', got {constraint_mode!r}"
            )
        self.constraint_mode = constraint_mode

        self._selected: Set[int] = set()
        self._A_inv: np.ndarray = np.eye(self.d) / lambda_reg
        self._b_vec: np.ndarray = np.zeros(self.d)
        self._theta: np.ndarray = np.zeros(self.d)
        self._actual_reward: float = 0.0
        self._baseline_expected: float = 0.0
        self._last_baseline_score: float = 0.0
        self.num_fallbacks: int = 0
        self.num_steps: int = 0

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._selected = set()
        self._A_inv = np.eye(self.d, dtype=np.float64) / self.lambda_reg
        self._b_vec = self.lambda_reg * self.prior_mean.copy()
        self._theta = self.prior_mean.copy()
        self._actual_reward = 0.0
        self._baseline_expected = 0.0
        self._last_baseline_score = 0.0
        self.num_fallbacks = 0
        self.num_steps = 0
        self.baseline.reset(user_idx)

    def select_action(self, state: Dict) -> int:
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates

        mean, std, ucb, lcb = self._ucb_scores(available)
        best_idx = int(np.argmax(ucb))
        proposed_action = available[best_idx]
        proposed_lcb = float(lcb[best_idx])

        fallback_action = self.baseline.select_action(state)
        baseline_score = self.baseline.expected_score(fallback_action)

        if self._feasible(proposed_lcb, baseline_score):
            chosen = proposed_action
        else:
            chosen = fallback_action
            self.num_fallbacks += 1

        self._last_baseline_score = baseline_score
        return chosen

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)

        phi_a = self.phi[action]
        A_inv_phi = self._A_inv @ phi_a
        phi_A_inv_phi = float(phi_a @ A_inv_phi)
        c = self.noise_prec
        self._A_inv -= (
            c * np.outer(A_inv_phi, A_inv_phi) / (1.0 + c * phi_A_inv_phi)
        )

        self._b_vec += self.noise_prec * reward * phi_a
        self._theta = self._A_inv @ self._b_vec
        self.baseline.update(action, reward, next_state, done)

        self._actual_reward += reward
        self._baseline_expected += self._last_baseline_score
        self.num_steps += 1

    @property
    def fallback_rate(self) -> float:
        return self.num_fallbacks / max(self.num_steps, 1)

    def step_diagnostics(self, state: Dict) -> Dict:
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates

        mean, std, ucb, lcb = self._ucb_scores(available)
        best_idx = int(np.argmax(ucb))
        proposed = available[best_idx]
        proposed_lcb = float(lcb[best_idx])

        fallback = self.baseline.select_action(state)
        bs = self.baseline.expected_score(fallback)
        feasible = self._feasible(proposed_lcb, bs)

        return {
            "proposed_action": proposed,
            "proposed_ucb": float(ucb[best_idx]),
            "proposed_lcb": proposed_lcb,
            "fallback_action": fallback,
            "baseline_score": bs,
            "feasible": feasible,
            "selected_action": proposed if feasible else fallback,
            "actual_so_far": self._actual_reward,
            "baseline_expected_so_far": self._baseline_expected,
        }

    def _ucb_scores(self, candidates: List[int]) -> Tuple:
        Phi = self.phi[candidates]
        mean = Phi @ self._theta
        Phi_A_inv = Phi @ self._A_inv
        var = (Phi_A_inv * Phi).sum(axis=1)
        std = np.sqrt(np.maximum(var, 0.0))
        ucb = mean + self.beta * std
        lcb = mean - self.beta_safe * std
        return mean, std, ucb, lcb

    def _feasible(self, proposed_lcb: float, baseline_score: float) -> bool:
        if self.constraint_mode == "per_step":
            return proposed_lcb >= self.alpha * baseline_score
        return (
            self._actual_reward + proposed_lcb
            >= self.alpha * (self._baseline_expected + baseline_score)
        )
