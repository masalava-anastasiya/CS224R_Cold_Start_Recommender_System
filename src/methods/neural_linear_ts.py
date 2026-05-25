"""Neural Linear Thompson Sampling policy for cold-start recommendation.

Uses pretrained item embeddings phi(x) as features and maintains a
Bayesian linear regression model over the user preference vector theta.
At each step, we sample theta from the posterior and pick the candidate
with the highest predicted reward.

The "neural" part is the sentence-transformer embeddings (already computed
in preprocessing). The "linear" part is the reward model: r = phi(x)^T theta + noise.
Because the model is linear-Gaussian, Bayesian updates are exact (closed-form).

Reference: Riquelme et al., "Deep Bayesian Bandits Showdown" (ICLR 2018).
"""

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
    """Compute a prior mean mu_0 and precision Lambda_0 from warm-user data.

    Pools ALL warm-user (item, rating) pairs into a single ridge regression
    to get a robust "global preference direction" in embedding space. This
    works better than averaging per-user thetas, which is noisy when d >> n
    per user. Also returns the empirical precision matrix so the prior
    reflects where warm data has concentrated information.

    Args:
        ratings_by_user: dict mapping user_idx -> [(item_idx, rating, ts), ...]
        warm_users: list of warm user indices
        item_emb: (n_items, d) item embedding matrix
        reg: ridge regularization strength

    Returns:
        (mu_0, Lambda_0): prior mean (d,) and prior precision (d, d)
    """
    phi = item_emb.numpy().astype(np.float64)
    d = phi.shape[1]

    # pool all warm-user ratings into one big design matrix
    # (too large to materialize explicitly, so accumulate A = Phi^T Phi and b = Phi^T r)
    A = reg * np.eye(d, dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)

    for u in warm_users:
        interactions = ratings_by_user[u]
        item_idxs = [iid for iid, _, _ in interactions]
        rewards = np.array([r for _, r, _ in interactions], dtype=np.float64)

        Phi_u = phi[item_idxs]  # (n_interactions, d)
        A += Phi_u.T @ Phi_u
        b += Phi_u.T @ rewards

    mu_0 = np.linalg.solve(A, b)

    # return a scaled-down version of A as the prior precision. the idea:
    # A encodes where warm data concentrates info, but we don't want it
    # to be *so* tight that the agent can't explore at all. scale it by
    # a factor that makes the prior covariance reasonable.
    # we normalize so trace(Lambda_0) = reg * d (same total precision as reg * I)
    scale = (reg * d) / np.trace(A)
    Lambda_0 = A * scale

    return mu_0, Lambda_0


class NeuralLinearTS:
    """Neural Linear Thompson Sampling policy.

    Maintains a Bayesian linear regression posterior over the user
    preference vector theta. At each step:
      1. Sample theta_tilde ~ N(mu, Lambda^{-1})
      2. Score each candidate: score = phi(x)^T theta_tilde
      3. Pick the candidate with the highest score

    After observing the reward, update the posterior:
      Lambda_new = Lambda_old + (1/sigma^2) * phi @ phi^T
      b_new = b_old + (1/sigma^2) * reward * phi
      mu_new = Lambda_new^{-1} @ b_new

    Parameters
    ----------
    item_emb:
        (n_items, d) tensor of precomputed item embeddings.
    lambda_prior:
        Regularization for the prior precision matrix. If Lambda_0 is not
        provided, uses Lambda_0 = lambda_prior * I. Higher = tighter prior
        = less initial exploration.
    sigma_noise:
        Assumed std dev of observation noise. Controls how much each
        observation shifts the posterior. Smaller = each rating matters more.
    mu_0:
        Optional prior mean for theta. If None, defaults to zeros.
        Pass the output of compute_warm_prior() for a warm-user-informed start.
    Lambda_0:
        Optional prior precision matrix (d, d). If None, uses lambda_prior * I.
        Pass the second return value of compute_warm_prior() for an
        informed prior that reflects where warm data concentrates.
    """

    def __init__(
        self,
        item_emb: torch.Tensor,
        lambda_prior: float = 1.0,
        sigma_noise: float = 1.0,
        mu_0: Optional[np.ndarray] = None,
        Lambda_0: Optional[np.ndarray] = None,
    ) -> None:
        self.phi = item_emb.numpy().astype(np.float64)  # (n_items, d)
        self.d = self.phi.shape[1]
        self.lambda_prior = lambda_prior
        self.sigma_noise = sigma_noise
        self.noise_precision = 1.0 / (sigma_noise ** 2)  # precompute 1/sigma^2

        # prior mean: either provided (from warm users) or zeros
        if mu_0 is not None:
            self._mu_0 = mu_0.astype(np.float64)
        else:
            self._mu_0 = np.zeros(self.d, dtype=np.float64)

        # prior precision: either a full matrix or lambda * I
        if Lambda_0 is not None:
            self._Lambda_0 = Lambda_0.astype(np.float64)
        else:
            self._Lambda_0 = lambda_prior * np.eye(self.d, dtype=np.float64)

        # these get set properly in reset()
        self._Lambda: np.ndarray = np.eye(self.d)
        self._b: np.ndarray = np.zeros(self.d)
        self._mu: np.ndarray = np.zeros(self.d)
        self._selected: Set[int] = set()
        self._rng = np.random.default_rng(42)

    def reset(self, user_idx: Optional[int] = None) -> None:
        """Reset the posterior for a new cold-start episode.

        Starts fresh with the prior: Lambda = Lambda_0, mu = mu_0.
        """
        self._Lambda = self._Lambda_0.copy()

        # b = Lambda_0 @ mu_0 so that mu = Lambda^{-1} @ b = mu_0 initially
        self._b = self._Lambda @ self._mu_0
        self._mu = self._mu_0.copy()

        self._selected = set()

    def select_action(self, state: Dict) -> int:
        """Sample theta from the posterior and pick the best candidate.

        This is where the exploration happens: when Lambda is "loose"
        (early in the episode), samples of theta vary a lot, so the agent
        tries different types of movies. As Lambda tightens, samples
        converge and the agent exploits what it's learned.
        """
        candidates: List[int] = state["candidates"].tolist()
        available = [c for c in candidates if c not in self._selected]
        if not available:
            available = candidates  # fallback, shouldn't happen in practice

        # sample theta from the posterior N(mu, Lambda^{-1})
        # use Cholesky of Lambda to avoid explicitly inverting the matrix:
        #   Lambda = L @ L^T  =>  Lambda^{-1} = L^{-T} @ L^{-1}
        #   to sample: z ~ N(0,I), solve L^T @ x = z, then theta = mu + x
        try:
            L = np.linalg.cholesky(self._Lambda)
            z = self._rng.standard_normal(self.d)
            x = np.linalg.solve(L.T, z)  # x ~ N(0, Lambda^{-1})
            theta_sample = self._mu + x
        except np.linalg.LinAlgError:
            # if Cholesky fails (shouldn't happen with proper regularization),
            # fall back to the posterior mean (pure exploitation)
            theta_sample = self._mu

        # score every available candidate under the sampled theta
        Phi_avail = self.phi[available]  # (n_available, d)
        scores = Phi_avail @ theta_sample  # (n_available,)

        best_idx = int(np.argmax(scores))
        return available[best_idx]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        """Update the posterior after observing a (item, reward) pair.

        Bayesian linear regression update (conjugate, exact):
          Lambda_new = Lambda_old + (1/sigma^2) * phi @ phi^T
          b_new = b_old + (1/sigma^2) * reward * phi
          mu_new = Lambda_new^{-1} @ b_new
        """
        self._selected.add(action)

        phi_a = self.phi[action]  # (d,)

        # rank-1 update to the precision matrix
        self._Lambda += self.noise_precision * np.outer(phi_a, phi_a)

        # update the information vector
        self._b += self.noise_precision * reward * phi_a

        # recompute the posterior mean
        self._mu = np.linalg.solve(self._Lambda, self._b)
