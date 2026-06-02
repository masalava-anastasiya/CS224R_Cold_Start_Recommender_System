"""RL² (Meta-RL) recurrent policy for cold-start recommendation.

The LSTM hidden state serves as an in-context "task posterior": by observing
(prev_item_emb, prev_reward) at each step, it accumulates evidence about the
cold user's preferences without any parameter updates at test time.

Meta-training: run REINFORCE across warm users (each user is one "task").
Meta-testing:  reset hidden state per cold user, run greedy.

References
----------
Duan et al. (2016) "RL²: Fast Reinforcement Learning via Slow Reinforcement
Learning". arXiv:1611.02779.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class RL2Policy(nn.Module):
    """LSTM meta-policy for cold-start recommendation.

    Parameters
    ----------
    item_emb : Tensor
        (n_items, emb_dim) item feature matrix — shared, not trained.
    hidden_dim : int
        LSTM hidden state size.
    gamma : float
        Discount factor for return computation (1.0 = undiscounted).
    """

    def __init__(
        self,
        item_emb: torch.Tensor,
        hidden_dim: int = 256,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()

        n_items, emb_dim = item_emb.shape
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # item_emb is fixed — register as buffer so it moves with .to(device)
        self.register_buffer("item_emb", item_emb.float())

        # LSTM input: [prev_item_emb (emb_dim) | prev_reward (1)]
        self.lstm = nn.LSTM(
            input_size=emb_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Projects hidden state into item embedding space for bilinear scoring
        self.proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        self._init_weights()

        # Holds LSTM state between env.reset() and env.step() calls at eval time
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.proj.weight)

    # ------------------------------------------------------------------
    # Helpers shared by train and eval paths
    # ------------------------------------------------------------------

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a zeroed (h_0, c_0) on the correct device."""
        dev = self.proj.weight.device
        h = torch.zeros(1, 1, self.hidden_dim, device=dev)
        c = torch.zeros(1, 1, self.hidden_dim, device=dev)
        return (h, c)

    def _build_lstm_input(
        self,
        prev_item_idx: Optional[int],
        prev_reward: float,
    ) -> torch.Tensor:
        """Shape (1, 1, emb_dim+1) ready for nn.LSTM."""
        dev = self.proj.weight.device
        if prev_item_idx is None:
            emb = torch.zeros(self.emb_dim, device=dev)
        else:
            emb = self.item_emb[prev_item_idx]
        r = torch.tensor([prev_reward], dtype=torch.float32, device=dev)
        return torch.cat([emb, r]).unsqueeze(0).unsqueeze(0)  # (1,1,d+1)

    def _score_candidates(
        self,
        h: torch.Tensor,
        candidates: List[int],
        excluded: set,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Return (logits, available_indices), masking already-selected items.

        h : (1, hidden_dim)
        """
        available = [c for c in candidates if c not in excluded]
        if not available:               # all selected (shouldn't happen in practice)
            available = list(candidates)

        cand_embs = self.item_emb[available]    # (n, emb_dim)
        h_proj = self.proj(h.squeeze(0))        # (emb_dim,)
        scores = cand_embs @ h_proj             # (n,)
        return scores, available

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def forward_train(
        self,
        state: Dict,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One step of the training rollout (stochastic action, tracked grads).

        Returns
        -------
        action : int
        log_prob : scalar Tensor (gradient attached)
        entropy : scalar Tensor (gradient attached)
        new_hidden : updated LSTM state
        """
        history: List = state["history"]
        candidates: List[int] = state["candidates"].tolist()
        excluded = {idx for idx, _ in history}

        prev_item, prev_r = history[-1] if history else (None, 0.0)

        x = self._build_lstm_input(prev_item, prev_r)
        out, new_hidden = self.lstm(x, hidden)  # out: (1,1,hidden)
        h = out.squeeze(1)                       # (1, hidden)

        scores, available = self._score_candidates(h, candidates, excluded)

        dist = Categorical(logits=scores)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()
        action = available[idx.item()]

        return action, log_prob, entropy, new_hidden

    # ------------------------------------------------------------------
    # Evaluation API  (compatible with run_episodes in existing eval scripts)
    # ------------------------------------------------------------------

    def reset(self, user_idx: Optional[int] = None) -> None:
        """Reset hidden state at the start of each new episode."""
        self._hidden = self.init_hidden()

    @torch.no_grad()
    def select_action(self, state: Dict) -> int:
        """Greedy action selection — called by run_episodes() helpers."""
        history: List = state["history"]
        candidates: List[int] = state["candidates"].tolist()
        excluded = {idx for idx, _ in history}

        prev_item, prev_r = history[-1] if history else (None, 0.0)

        x = self._build_lstm_input(prev_item, prev_r)
        out, self._hidden = self.lstm(x, self._hidden)
        h = out.squeeze(1)

        scores, available = self._score_candidates(h, candidates, excluded)
        action = available[scores.argmax().item()]
        return action

    def update(self, action: int, reward: float, next_state: Dict, done: bool) -> None:
        """No-op: RL² does not update parameters at test time."""
        pass
