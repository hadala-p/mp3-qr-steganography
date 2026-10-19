import io

import numpy as np
import torch
from pydub import AudioSegment

from .ffmpeg_io import run_ffmpeg


def decode_audio_ffmpeg_mono(ffmpeg_path: str, path: str, sr: int) -> torch.Tensor:
    raw = run_ffmpeg(ffmpeg_path, [
        "-hide_banner", "-loglevel", "error",
        "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(int(sr)), "pipe:1"
    ])
    x = np.frombuffer(raw, dtype=np.float32).copy()
    if x.size == 0:
        raise RuntimeError(f"Pusty plik audio z FFmpeg: {path}")
    return torch.from_numpy(x).view(1, 1, -1).clamp(-1.0, 1.0)


def mp3_roundtrip_tensor(ffmpeg_path, audio_tensor, sample_rate, bitrate):
    # Konwersja na int16 do AudioSegment
    audio_np = audio_tensor.detach().cpu().numpy().squeeze()
    audio_int16 = (audio_np * 32767).astype('int16')

    seg = AudioSegment(audio_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

    # Eksport do RAM
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate=f"{bitrate}k")
    buf.seek(0)

    # Odczyt z bufora
    stego_seg = AudioSegment.from_file(buf, format="mp3")
    stego_np = np.array(stego_seg.get_array_of_samples()).astype('float32') / 32768.0

    # Przycinanie/Dopełnianie do oryginalnej długości
    t_len = audio_tensor.shape[-1]
    stego_np = stego_np[:t_len]
    if len(stego_np) < t_len:
        stego_np = np.pad(stego_np, (0, t_len - len(stego_np)))

    return torch.from_numpy(stego_np).unsqueeze(0)
