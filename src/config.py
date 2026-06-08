from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass
class DataConfig:
    data_dir: str = "data"
    min_ratings_per_user: int = 25
    cold_start_horizon_T: int = 20
    warm_user_fraction: float = 0.7
    encoder_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_genre_features: bool = True
    use_user_features: bool = True
    expose_user_features: str = "never"
    rating_threshold: float = 4.0
    reward_mode: str = "raw"
    random_seed: int = 42

    def config_hash(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
