import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as aF

from qrstego.model.net import QRAudioEncoder
from qrstego.qr.qr_codec import load_qr_bits_and_images
from qrstego.constants import MSG_SIZE, EPS, SEGMENT_SAMPLES
from script.constants import DIR_MP3_INPUT, DIR_QR_INPUT, DIR_MP3_STEGO, ENCODER_WEIGHTS, FFMPEG_PATH


def embed(audio_filename: str, qr_filename: str, output_filename: str = None):
    os.makedirs(DIR_MP3_STEGO, exist_ok=True)

    audio_path = os.path.join(DIR_MP3_INPUT, audio_filename)
    qr_path = os.path.join(DIR_QR_INPUT, qr_filename)

    if output_filename is None:
        name, ext = os.path.splitext(audio_filename)
        output_filename = f"{name}_stego{ext}"
    output_path = os.path.join(DIR_MP3_STEGO, output_filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- UKRYWANIE ({device}) ---")
    print(f"Audio wejściowe: {audio_path}")
    print(f"QR wejściowy:   {qr_path}")

    encoder = QRAudioEncoder(msg_dim=MSG_SIZE).to(device)
    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location=device, weights_only=True))
    encoder.eval()

    msg_bits_original, _ = load_qr_bits_and_images(qr_path)
    msg_bits = msg_bits_original.view(1, -1).to(device)

    full_audio_stereo, sr = torchaudio.load(audio_path)
    full_audio_stereo = full_audio_stereo.to(device)
    mono_audio = full_audio_stereo.mean(dim=0, keepdim=True)

    window_size = sr // 2
    min_distance = SEGMENT_SAMPLES

    start_positions = [None, None, None]

    if mono_audio.shape[-1] > SEGMENT_SAMPLES:
        window_conv = torch.ones(1, 1, window_size, device=device) / window_size
        full_energy = F.conv1d(mono_audio.abs(), window_conv, stride=window_size).squeeze()
        sorted_indices = torch.argsort(full_energy, descending=True)

        for idx in sorted_indices:
            sample_pos = idx.item() * window_size
            if sample_pos + SEGMENT_SAMPLES > mono_audio.shape[-1]:
                continue
            if start_positions[0] is None:
                start_positions[0] = sample_pos
            elif start_positions[1] is None:
                if abs(sample_pos - start_positions[0]) >= min_distance:
                    start_positions[1] = sample_pos
            elif start_positions[2] is None:
                if (abs(sample_pos - start_positions[0]) >= min_distance and
                    abs(sample_pos - start_positions[1]) >= min_distance):
                    start_positions[2] = sample_pos
                    break
    else:
        start_positions[0] = 0

    def get_delta_for_start(start_idx):
        seg = mono_audio[..., start_idx: start_idx + SEGMENT_SAMPLES]
        if seg.shape[-1] < SEGMENT_SAMPLES:
            seg = F.pad(seg, (0, SEGMENT_SAMPLES - seg.shape[-1]))
        seg_batched = seg.unsqueeze(0)
        window_calc = torch.ones(1, 1, 1024, device=device) / 1024
        base_e = F.conv1d(seg_batched.abs(), window_calc, padding=512)[..., :seg_batched.shape[-1]]
        loc_e = (base_e ** 2) * 1.5
        with torch.no_grad():
            raw_d = encoder(seg_batched, msg_bits)
            d = raw_d * (EPS * loc_e * 15.0)
            d = aF.highpass_biquad(d, sr, cutoff_freq=300.0)
        return d.squeeze(0)

    for st_sample in start_positions:
        if st_sample is not None:
            delta = get_delta_for_start(st_sample)
            if full_audio_stereo.shape[-1] > SEGMENT_SAMPLES:
                full_audio_stereo[..., st_sample: st_sample + SEGMENT_SAMPLES] += delta[..., :full_audio_stereo.shape[-1] - st_sample]
            else:
                full_audio_stereo += delta[..., :full_audio_stereo.shape[-1]]

    full_audio_stereo = full_audio_stereo.clamp(-1.0, 1.0)

    temp_wav = os.path.join(DIR_MP3_STEGO, "temp_embed.wav")
    torchaudio.save(temp_wav, full_audio_stereo.cpu(), sr)

    cmd = [
        FFMPEG_PATH, "-y", "-loglevel", "error",
        "-i", temp_wav, "-i", audio_path,
        "-map", "0:a:0", "-map", "1:v?", "-map_metadata", "1",
        "-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "320k",
        output_path
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    print(f"Zapisano stego audio do: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ukrywanie kodu QR w pliku audio.")
    parser.add_argument(
        "audio",
        nargs="?",
        default="muzyka.mp3",
        help="Nazwa pliku MP3 z katalogu DIR_MP3_INPUT (domyślnie: muzyka.mp3)",
    )
    parser.add_argument(
        "qr",
        nargs="?",
        default="kod.png",
        help="Nazwa pliku PNG z katalogu DIR_QR_INPUT (domyślnie: kod.png)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Opcjonalna nazwa pliku wyjściowego (domyślnie: <audio>_stego.mp3)",
    )

    args = parser.parse_args()
    embed(audio_filename=args.audio, qr_filename=args.qr, output_filename=args.output)