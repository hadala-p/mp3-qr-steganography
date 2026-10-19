import os
import shutil
import platform
from pathlib import Path

def setup_ffmpeg():
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin

    if platform.system() == "Windows":
        fallback = Path(
            r"C:\Users\USER\AppData\Local\Microsoft\WinGet\Packages"
            r"\SCIEZKA\DO\FFMPEG\DO\WPISANIA"
        )
    else:
        fallback = Path("/usr/bin/ffmpeg")

    if fallback.is_file():
        os.environ["PATH"] = str(fallback.parent) + os.pathsep + os.environ.get("PATH", "")
        return str(fallback)

    raise FileNotFoundError(f"Nie znaleziono ffmpeg ani w PATH, ani pod adresem: {fallback}")

FFMPEG_PATH = setup_ffmpeg()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DIR_MP3_INPUT = os.path.join(PROJECT_ROOT, "mp3_do_ukrycia")
DIR_QR_INPUT = os.path.join(PROJECT_ROOT, "qr_do_ukrycia")
DIR_MP3_STEGO = os.path.join(PROJECT_ROOT, "mp3_stego")
DIR_QR_EXTRACTED = os.path.join(PROJECT_ROOT, "odkodowane_qr")

# Katalog i wagi modeli
DIR_MODELS = os.path.join(PROJECT_ROOT, "script", "modele")
ENCODER_WEIGHTS = os.path.join(DIR_MODELS, "encoder.pt")
DECODER_WEIGHTS = os.path.join(DIR_MODELS, "decoder.pt")