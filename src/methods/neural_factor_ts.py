"""Method #2: "true" Neural-Linear Thompson Sampling with learned features.

Our NLTS runs Bayesian linear regression on FROZEN sentence-transformer
embeddings --- the representation is never shaped by the reward signal.
Riquelme et al. (2018) instead train a neural net on the reward, then do
Bayesian linear regression on its LAST layer. This module follows that
recipe.

We train a two-tower model on warm-user ratings:
    item tower   g(phi(x)) -> R^k     (content embedding -> learned factor)
    user factors theta_u  in R^k      (one per warm user)
    rating_hat = g(phi(x))^T theta_u + item_mean_x   (MSE on warm triples)

The item tower g is shared and trained on actual ratings, so the learned
factors encode "what about a movie drives ratings". Unlike SVD factors,
g generalises to ANY item via its content embedding. At test time we run
the SAME exact Bayesian linear TS as Hybrid, but over g(phi(x)) instead of
SVD factors.

Reference: Riquelme et al., "Deep Bayesian Bandits Showdown" (ICLR 2018).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


class ItemTower(nn.Module):
    """MLP mapping a content embedding phi(x) to a k-dim learned factor."""

    def __init__(self, in_dim: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_neural_factors(
    ratings_by_user: Dict,
    warm_users: List[int],
    item_emb: torch.Tensor,
    k: int = 50,
    hidden: int = 128,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 8192,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the two-tower model on warm ratings.

    Returns
    -------
    Q : (n_items, k) learned item factors g(phi(x)) for every item
    item_means : (n_items,) warm item-mean ratings (for centering)
    mu_0 : (k,) prior mean = average warm-user factor
    Lambda_0 : (k, k) prior precision from warm-user factor covariance
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    n_items = item_emb.shape[0]
    in_dim = item_emb.shape[1]
    item_emb_t = item_emb.float().to(dev)

    # warm item means (same convention as Greedy CF / Hybrid)
    sums = np.zeros(n_items, dtype=np.float64)
    counts = np.zeros(n_items, dtype=np.float64)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    user_to_row = {u: i for i, u in enumerate(warm_users)}
    for u in warm_users:
        for item_idx, rating, _ in ratings_by_user[u]:
            sums[item_idx] += rating
            counts[item_idx] += 1
            rows.append(user_to_row[u])
            cols.append(item_idx)
            vals.append(rating)
    item_means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    row_idx = torch.tensor(rows, dtype=torch.long, device=dev)
    col_idx = torch.tensor(cols, dtype=torch.long, device=dev)
    # center ratings by item mean -> model learns the residual structure
    centered = torch.tensor(
        np.array(vals, dtype=np.float32) - item_means[cols].astype(np.float32),
        device=dev,
    )
    n_triples = len(vals)

    tower = ItemTower(in_dim, k, hidden).to(dev)
    user_factors = nn.Embedding(len(warm_users), k).to(dev)
    nn.init.normal_(user_factors.weight, std=0.1)

    opt = torch.optim.Adam(
        list(tower.parameters()) + list(user_factors.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"  Training two-tower on {n_triples:,} warm triples "
              f"(k={k}, hidden={hidden}, epochs={epochs})...")

    for epoch in range(1, epochs + 1):
        perm = torch.tensor(rng.permutation(n_triples), device=dev)
        tower.train()
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_triples, batch_size):
            b = perm[start:start + batch_size]
            phi_b = item_emb_t[col_idx[b]]              # (B, in_dim)
            q_b = tower(phi_b)                          # (B, k)
            theta_b = user_factors(row_idx[b])          # (B, k)
            pred = (q_b * theta_b).sum(dim=1)           # (B,)
            loss = loss_fn(pred, centered[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    epoch {epoch:2d}/{epochs}  MSE={ep_loss / n_batches:.4f}")

    # precompute learned factors for all items
    tower.eval()
    with torch.no_grad():
        Q = tower(item_emb_t).cpu().numpy().astype(np.float64)   # (n_items, k)
    theta = user_factors.weight.detach().cpu().numpy().astype(np.float64)  # (n_warm, k)

    mu_0 = theta.mean(axis=0)
    cov = np.cov(theta.T) + 1e-6 * np.eye(k)
    Lambda_0 = np.linalg.inv(cov)

    return Q, item_means.astype(np.float64), mu_0, Lambda_0


class NeuralFactorTS:
    """Thompson Sampling over learned (reward-trained) item factors.

    Identical Bayesian-linear math to HybridNeuralLinearTS, but the factor
    matrix Q comes from the trained item tower rather than SVD.
    """

    def __init__(
        self,
        Q: np.ndarray,
        item_means: np.ndarray,
        mu_0: np.ndarray,
        Lambda_0: np.ndarray,
        sigma_noise: float = 1.0,
    ) -> None:
        self.Q = Q
        self.item_means = item_means
        self.k = Q.shape[1]
        self.noise_precision = 1.0 / (sigma_noise ** 2)

        self._mu_0 = mu_0.astype(np.float64)
        self._Lambda_0 = Lambda_0.astype(np.float64)

        self._Lambda = np.eye(self.k)
        self._b = np.zeros(self.k)
        self._mu = np.zeros(self.k)
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
            z = self._rng.standard_normal(self.k)
            x = np.linalg.solve(L.T, z)
            theta_sample = self._mu + x
        except np.linalg.LinAlgError:
            theta_sample = self._mu
        scores = self.Q[available] @ theta_sample + self.item_means[available]
        return available[int(np.argmax(scores))]

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        self._selected.add(action)
        q_a = self.Q[action]
        centered_reward = reward - self.item_means[action]
        self._Lambda += self.noise_precision * np.outer(q_a, q_a)
        self._b += self.noise_precision * centered_reward * q_a
        self._mu = np.linalg.solve(self._Lambda, self._b)
