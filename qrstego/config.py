from dataclasses import dataclass
from typing import Optional

@dataclass
class RunConfig:
    audio_dir: str
    ffmpeg_path: str
    epochs: int = 10
    batch_size: int = 8
    weight_finder: float = 100.0
    noise_multiplier: float = 12.0
    audio_loss_weight: float = 0.5
    learning_rate: float = 0.0001