import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import torch.nn.functional as F
from pyzbar.pyzbar import decode

from qrstego.audio.mp3_tools import decode_audio_ffmpeg_mono
from qrstego.constants import MSG_SIZE, SAMPLE_RATE, SEGMENT_SAMPLES
from qrstego.model.net import QRAudioStego
from qrstego.qr.qr_codec import bits_to_image
from script.constants import (
    DECODER_WEIGHTS,
    DIR_MP3_STEGO,
    DIR_QR_EXTRACTED,
    FFMPEG_PATH,
)


def read_qr_text(image_path: str) -> str:
    """Odczytuje zakodowany tekst z pliku PNG."""
    img = cv2.imread(image_path)
    if img is None:
        return "BŁĄD_WCZYTANIA_OBRAZU"
    decoded_objects = decode(img)
    if not decoded_objects:
        return "BRAK_ODCZYTU"
    return decoded_objects[0].data.decode("utf-8")


def get_qr_corners_mask_and_values():
    """Generuje maskę stałych pikseli dla kodu QR w wersji 1 (21x21)."""
    size = 21
    mask = torch.zeros((size, size), dtype=torch.bool)
    values = torch.zeros((size, size), dtype=torch.float32)

    def draw_finder(r, c):
        mask[r : r + 7, c : c + 7] = True
        values[r : r + 7, c : c + 7] = 1.0
        values[r + 1 : r + 6, c + 1 : c + 6] = 0.0
        values[r + 2 : r + 5, c + 2 : c + 5] = 1.0

    draw_finder(0, 0)
    draw_finder(0, size - 7)
    draw_finder(size - 7, 0)

    mask[7, 0:8] = True; values[7, 0:8] = 0.0
    mask[0:8, 7] = True; values[0:8, 7] = 0.0
    mask[7, 13:21] = True; values[7, 13:21] = 0.0
    mask[0:8, 13] = True; values[0:8, 13] = 0.0
    mask[13, 0:8] = True; values[13, 0:8] = 0.0
    mask[13:21, 7] = True; values[13:21, 7] = 0.0

    for i in range(8, 13):
        mask[6, i] = True; values[6, i] = 1.0 if i % 2 == 0 else 0.0
        mask[i, 6] = True; values[i, 6] = 1.0 if i % 2 == 0 else 0.0

    mask[13, 8] = True; values[13, 8] = 1.0

    return mask.view(-1), values.view(-1)


def extract(audio_filename: str, qr_output_filename: str = None):
    os.makedirs(DIR_QR_EXTRACTED, exist_ok=True)

    audio_path = (
        audio_filename
        if os.path.isabs(audio_filename) or os.path.dirname(audio_filename)
        else os.path.join(DIR_MP3_STEGO, audio_filename)
    )

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Nie znaleziono pliku stego audio: {audio_path}")

    if qr_output_filename is None:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        qr_output_filename = f"extracted_{base_name}.png"

    qr_output_path = (
        qr_output_filename
        if os.path.isabs(qr_output_filename) or os.path.dirname(qr_output_filename)
        else os.path.join(DIR_QR_EXTRACTED, qr_output_filename)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- EKSTRAKCJA ({device}) ---")
    print(f"Audio wejściowe: {audio_path}")
    print(f"Zapis QR do:     {qr_output_path}")

    decoder = QRAudioStego(msg_dim=MSG_SIZE).to(device)
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS, map_location=device, weights_only=True))
    decoder.eval()

    mask, ideal_values = get_qr_corners_mask_and_values()
    mask, ideal_values = mask.to(device), ideal_values.to(device)

    audio_stego = decode_audio_ffmpeg_mono(FFMPEG_PATH, audio_path, SAMPLE_RATE).to(device)
    if audio_stego.ndim == 1:
        audio_stego = audio_stego.unsqueeze(0)

    window_size = SAMPLE_RATE // 2
    min_distance = SEGMENT_SAMPLES

    base_starts = [None, None, None]

    if audio_stego.shape[-1] > SEGMENT_SAMPLES:
        window_conv = torch.ones(1, 1, window_size, device=device) / window_size
        full_energy = F.conv1d(audio_stego.abs(), window_conv, stride=window_size).squeeze()
        sorted_indices = torch.argsort(full_energy, descending=True)

        for idx in sorted_indices:
            sample_pos = idx.item() * window_size
            if sample_pos + SEGMENT_SAMPLES > audio_stego.shape[-1]:
                continue
            if base_starts[0] is None:
                base_starts[0] = sample_pos
            elif base_starts[1] is None:
                if abs(sample_pos - base_starts[0]) >= min_distance:
                    base_starts[1] = sample_pos
            elif base_starts[2] is None:
                if (
                    abs(sample_pos - base_starts[0]) >= min_distance
                    and abs(sample_pos - base_starts[1]) >= min_distance
                ):
                    base_starts[2] = sample_pos
                    break
    else:
        base_starts[0] = 0

    def run_radar(base_start):
        if base_start is None:
            return None
        b_acc = 0.0
        b_logits = None
        for offset in range(-5000, 5000, 10):
            test_start = base_start + offset
            if test_start < 0 or test_start + SEGMENT_SAMPLES > audio_stego.shape[-1]:
                continue
            segment = audio_stego[..., test_start : test_start + SEGMENT_SAMPLES]
            if segment.shape[-1] < SEGMENT_SAMPLES:
                segment = F.pad(segment, (0, SEGMENT_SAMPLES - segment.shape[-1]))

            with torch.no_grad():
                logits = decoder.decode_logits(segment).squeeze()
                predicted_bits = (logits > 0).float()

            corner_bits = predicted_bits[mask]
            matches = (corner_bits == ideal_values[mask]).sum().item()
            acc = matches / mask.sum().item()

            if acc > b_acc:
                b_acc = acc
                b_logits = logits
            if b_acc == 1.0:
                break
        return b_logits

    valid_logits = [run_radar(pos) for pos in base_starts if pos is not None]
    valid_logits = [l for l in valid_logits if l is not None]

    if not valid_logits:
        raise RuntimeError("Nie udało się odnaleźć synchronizacji w żadnym z segmentów.")

    combined_logits = sum(valid_logits)
    best_bits = (combined_logits > 0).float()

    # Korekcja pikseli wzorca
    corner_bits_extracted = best_bits[mask]
    pixels_corrected = (corner_bits_extracted != ideal_values[mask]).sum().item()
    best_bits[mask] = ideal_values[mask]

    reconstructed_img = bits_to_image(best_bits.cpu())
    reconstructed_img.save(qr_output_path)

    extracted_text = read_qr_text(qr_output_path)
    print(f"Naprawione piksele stałe: {pixels_corrected}")
    print(f"Odczytana treść: {extracted_text}")
    print(f"Zapisano odtworzony kod do: {qr_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrakcja ukrytego kodu QR z pliku stego audio.")
    parser.add_argument(
        "audio",
        nargs="?",
        default="muzyka_stego.mp3",
        help=f"Nazwa/ścieżka do pliku audio MP3 (domyślnie: muzyka_stego.mp3)",
    )
    parser.add_argument(
        "qr",
        nargs="?",
        default=None,
        help="Opcjonalna nazwa pliku wyjściowego PNG (domyślnie: extracted_<audio>.png)",
    )
    args = parser.parse_args()

    extract(audio_filename=args.audio, qr_output_filename=args.qr)