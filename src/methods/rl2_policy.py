"""RL2 LSTM meta-policy for cold-start recommendation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class RL2Policy(nn.Module):

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

        self.register_buffer("item_emb", item_emb.float())

        self.lstm = nn.LSTM(
            input_size=emb_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        self._init_weights()

        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.proj.weight)

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.proj.weight.device
        h = torch.zeros(1, 1, self.hidden_dim, device=dev)
        c = torch.zeros(1, 1, self.hidden_dim, device=dev)
        return (h, c)

    def _build_lstm_input(
        self,
        prev_item_idx: Optional[int],
        prev_reward: float,
    ) -> torch.Tensor:
        dev = self.proj.weight.device
        if prev_item_idx is None:
            emb = torch.zeros(self.emb_dim, device=dev)
        else:
            emb = self.item_emb[prev_item_idx]
        r = torch.tensor([prev_reward], dtype=torch.float32, device=dev)
        return torch.cat([emb, r]).unsqueeze(0).unsqueeze(0)

    def _score_candidates(
        self,
        h: torch.Tensor,
        candidates: List[int],
        excluded: set,
    ) -> Tuple[torch.Tensor, List[int]]:
        available = [c for c in candidates if c not in excluded]
        if not available:
            available = list(candidates)

        cand_embs = self.item_emb[available]
        h_proj = self.proj(h.squeeze(0))
        scores = cand_embs @ h_proj
        return scores, available

    def forward_train(
        self,
        state: Dict,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        history: List = state["history"]
        candidates: List[int] = state["candidates"].tolist()
        excluded = {idx for idx, _ in history}

        prev_item, prev_r = history[-1] if history else (None, 0.0)

        x = self._build_lstm_input(prev_item, prev_r)
        out, new_hidden = self.lstm(x, hidden)
        h = out.squeeze(1)

        scores, available = self._score_candidates(h, candidates, excluded)

        dist = Categorical(logits=scores)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()
        action = available[idx.item()]

        return action, log_prob, entropy, new_hidden

    def reset(self, user_idx: Optional[int] = None) -> None:
        self._hidden = self.init_hidden()

    @torch.no_grad()
    def select_action(self, state: Dict) -> int:
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
        pass
