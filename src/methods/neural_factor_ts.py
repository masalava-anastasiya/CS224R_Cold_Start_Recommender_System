"""Neural-factor Thompson Sampling (reward-trained item tower)."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn


class ItemTower(nn.Module):

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
    torch.manual_seed(seed)
    dev = torch.device(device)

    n_items = item_emb.shape[0]
    in_dim = item_emb.shape[1]
    item_emb_t = item_emb.float().to(dev)

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
        print(
            f"training two-tower on {n_triples:,} warm triples "
            f"(k={k}, hidden={hidden}, epochs={epochs})..."
        )

    for epoch in range(1, epochs + 1):
        perm = torch.tensor(rng.permutation(n_triples), device=dev)
        tower.train()
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_triples, batch_size):
            b = perm[start:start + batch_size]
            phi_b = item_emb_t[col_idx[b]]
            q_b = tower(phi_b)
            theta_b = user_factors(row_idx[b])
            pred = (q_b * theta_b).sum(dim=1)
            loss = loss_fn(pred, centered[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch}/{epochs}, MSE={ep_loss / n_batches:.4f}")

    tower.eval()
    with torch.no_grad():
        Q = tower(item_emb_t).cpu().numpy().astype(np.float64)
    theta = user_factors.weight.detach().cpu().numpy().astype(np.float64)

    mu_0 = theta.mean(axis=0)
    cov = np.cov(theta.T) + 1e-6 * np.eye(k)
    Lambda_0 = np.linalg.inv(cov)

    return Q, item_means.astype(np.float64), mu_0, Lambda_0


class NeuralFactorTS:

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
