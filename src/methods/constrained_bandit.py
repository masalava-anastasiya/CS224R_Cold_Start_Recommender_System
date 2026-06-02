"""Constrained Linear UCB Bandit for cold-start recommendation.

Maintains a Bayesian linear model over each cold-user episode (same
structure as NeuralLinearTS) but selects actions using a LinUCB-style
optimistic score.  Exploration is only permitted when the lower
confidence bound of the proposed action clears a relevance floor
defined by a non-personalized baseline policy.

Two constraint modes
--------------------
"per_step"
    LCB(proposed) >= alpha * baseline_score(fallback)
    Checked independently at every step.

"cumulative"   (default)
    actual_reward_so_far + LCB(proposed)
        >= alpha * (baseline_expected_so_far + baseline_score(fallback))
    Closer to the conservative bandit guarantee in the proposal: the
    cumulative reward stays within a multiplicative slack of what the
    non-personalized baseline would have accumulated.

When the constraint is violated the policy falls back to the baseline's
recommended action, which is always safe by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class ConstrainedLinearUCBBandit:
    """LinUCB bandit with a conservative cumulative feasibility constraint.

    Parameters
    ----------
    item_emb:
        (n_items, d) item embedding tensor from item_emb.pt.
    baseline_policy:
        Trained non-personalized baseline.  Must expose:
        reset(), select_action(state), update(action, reward, ns, done),
        expected_score(item_idx).
    lambda_reg:
        Ridge regularisation for the prior Lambda = lambda_reg * I.
        Also controls the prior precision: higher = tighter prior.
    sigma2:
        Assumed observation noise variance.
    beta:
        UCB exploration coefficient (optimism for action selection).
    beta_safe:
        LCB coefficient for the safety lower bound.  Should be >= beta
        so the safety interval is at least as wide as the UCB interval.
    alpha:
        Constraint slack in [0, 1].  alpha=1 enforces the constraint
        strictly; alpha=0 always allows exploration.
    prior_mean:
        Optional (d,) prior mean for theta.  Pass mu_0 from
        compute_warm_prior() for a warm-user-initialised prior.
    constraint_mode:
        "per_step" or "cumulative".
    """

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
        self.phi = item_emb.detach().cpu().numpy().astype(np.float64)  # (n_items, d)
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

        # episode state (reset each call to reset())
        self._selected: Set[int] = set()
        self._A_inv: np.ndarray = np.eye(self.d) / lambda_reg
        self._b_vec: np.ndarray = np.zeros(self.d)
        self._theta: np.ndarray = np.zeros(self.d)
        self._actual_reward: float = 0.0
        self._baseline_expected: float = 0.0
        self._last_baseline_score: float = 0.0
        self.num_fallbacks: int = 0
        self.num_steps: int = 0

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def reset(self, user_idx: Optional[int] = None) -> None:
        """Reset the Bayesian model and episode counters for a new user."""
        self._selected = set()
        # Prior: Lambda = lambda_reg * I  =>  A_inv = (1/lambda_reg) * I
        self._A_inv = np.eye(self.d, dtype=np.float64) / self.lambda_reg
        # b = Lambda @ mu_0  so  theta = A_inv @ b = mu_0 initially
        self._b_vec = self.lambda_reg * self.prior_mean.copy()
        self._theta = self.prior_mean.copy()
        self._actual_reward = 0.0
        self._baseline_expected = 0.0
        self._last_baseline_score = 0.0
        self.num_fallbacks = 0
        self.num_steps = 0
        self.baseline.reset(user_idx)

    def select_action(self, state: Dict) -> int:
        """Return the item to recommend, subject to the safety constraint."""
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

        # store for the cumulative constraint update in update()
        self._last_baseline_score = baseline_score
        return chosen

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        """Bayesian rank-1 update using Sherman-Morrison identity.

        Maintains A_inv directly (O(d^2) per step) rather than solving
        an O(d^3) system at each step.
        """
        self._selected.add(action)

        phi_a = self.phi[action]                         # (d,)

        # Sherman-Morrison: A_new^{-1} for A_new = A + c * outer(phi, phi)
        #   A_new^{-1} = A_inv - c * (A_inv phi)(A_inv phi)^T
        #                          / (1 + c * phi^T A_inv phi)
        A_inv_phi = self._A_inv @ phi_a                  # (d,)
        phi_A_inv_phi = float(phi_a @ A_inv_phi)         # scalar
        c = self.noise_prec
        self._A_inv -= (
            c * np.outer(A_inv_phi, A_inv_phi) / (1.0 + c * phi_A_inv_phi)
        )

        self._b_vec += self.noise_prec * reward * phi_a
        self._theta = self._A_inv @ self._b_vec

        # keep the baseline's selected-set in sync so it never re-selects
        self.baseline.update(action, reward, next_state, done)

        self._actual_reward += reward
        self._baseline_expected += self._last_baseline_score
        self.num_steps += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fallback_rate(self) -> float:
        return self.num_fallbacks / max(self.num_steps, 1)

    def step_diagnostics(self, state: Dict) -> Dict:
        """Return per-step debug info without advancing any state.

        Call before select_action() to inspect the decision the policy
        is about to make.
        """
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
            "proposed_action":        proposed,
            "proposed_ucb":           float(ucb[best_idx]),
            "proposed_lcb":           proposed_lcb,
            "fallback_action":        fallback,
            "baseline_score":         bs,
            "feasible":               feasible,
            "selected_action":        proposed if feasible else fallback,
            "actual_so_far":          self._actual_reward,
            "baseline_expected_so_far": self._baseline_expected,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ucb_scores(self, candidates: List[int]) -> Tuple:
        Phi = self.phi[candidates]               # (n, d)
        mean = Phi @ self._theta                 # (n,)
        Phi_A_inv = Phi @ self._A_inv            # (n, d)   one BLAS call
        var = (Phi_A_inv * Phi).sum(axis=1)      # (n,)  diag(Phi A_inv Phi^T)
        std = np.sqrt(np.maximum(var, 0.0))      # (n,)
        ucb = mean + self.beta      * std
        lcb = mean - self.beta_safe * std
        return mean, std, ucb, lcb

    def _feasible(self, proposed_lcb: float, baseline_score: float) -> bool:
        if self.constraint_mode == "per_step":
            return proposed_lcb >= self.alpha * baseline_score
        # cumulative: at t=0, degrades to same as per_step
        return (
            self._actual_reward + proposed_lcb
            >= self.alpha * (self._baseline_expected + baseline_score)
        )
