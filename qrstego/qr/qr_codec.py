import cv2
import numpy as np
import torch
from qrstego.constants import QR_CORE
from PIL import Image

def decode_qr_payload(img_u8: np.ndarray) -> str | None:
    det = cv2.QRCodeDetector()
    data, _, _ = det.detectAndDecode(img_u8)
    return data if data else None


def extract_core_modules_optimized(img_u8: np.ndarray, qr_core: int = 21) -> np.ndarray:
    # progowanie, bo obraz jest czarno-biały
    ys, xs = np.where(img_u8 < 128)
    if ys.size == 0:
        return np.zeros((qr_core, qr_core), dtype=np.float32)

    # Wycinamy sam rdzeń kodu
    crop = img_u8[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # Skalujemy bezpośrednio do docelowej siatki
    rs = cv2.resize(crop, (qr_core, qr_core), interpolation=cv2.INTER_AREA)

    # Progowanie i zwrot wyniku
    return (rs < 128).astype(np.float32)


def load_qr_bits_and_images(qr_path: str):
    img = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Nie mogę wczytać QR: {qr_path}")

    gt_payload = decode_qr_payload(img) or "UNKNOWN"

    mod = extract_core_modules_optimized(img)
    msg_bits = torch.from_numpy(mod.reshape(-1)).float()

    return msg_bits, gt_payload


def bits_to_image(bits: torch.Tensor, version_size: int = 21) -> Image.Image:
    """
    Przekształca 441 bitów z powrotem w widoczny obrazek kodu QR.
    """
    bits = bits.squeeze().cpu().numpy()
    grid = bits.reshape((version_size, version_size))

    # 1 = czarny (0), 0 = biały (255)
    pixels = ((1.0 - grid) * 255).astype(np.uint8)

    # DODANIE QUIET ZONE czyli Margines 4 białych pikseli ze wszystkich stron
    pixels_with_margin = np.pad(pixels, pad_width=4, mode='constant', constant_values=255)

    img = Image.fromarray(pixels_with_margin, mode='L')

    # Skalujemy obrazek mocniej x15, żeby łatwiej było go zeskanować telefonem z ekranu
    final_size = (version_size + 8) * 15
    img = img.resize((final_size, final_size), Image.NEAREST)

    return img