from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    # Paths
    data_dir: str = "data"

    # User filtering
    min_ratings_per_user: int = 25

    # Episode / cold-start protocol
    cold_start_horizon_T: int = 20

    # User split
    warm_user_fraction: float = 0.7

    # Item encoding
    encoder_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384          # must match the sentence-transformer output
    use_genre_features: bool = True   # concatenate multi-hot genre vector to text emb

    # Reward
    rating_threshold: float = 4.0    # ratings >= this are "positive" in binary mode
    reward_mode: str = "raw"         # "raw" (1-5 rating) | "binary"

    # Reproducibility
    random_seed: int = 42

    def config_hash(self) -> str:
        """Stable hash of config used to detect stale cached artifacts."""
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
