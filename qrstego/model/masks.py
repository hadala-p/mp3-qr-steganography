import torch
from qrstego.constants import QR_CORE

def build_weight_mask(
        qr_core: int = QR_CORE,
        weight_finder: float = 10.0,    # Kary za zepsucie narożników
        weight_timing: float = 5.0,     # Kary za paski synchronizacyjne
        weight_separator: float = 8.0,  # Kary za białe separatory
        weight_format: float = 15.0     # kara za zepsucie bitów formatu
) -> torch.Tensor:
    # Inicjalizacja macierzy jedynek czyli tło i waga 1 dla zwykłych danych
    w = torch.ones((qr_core, qr_core), dtype=torch.float32)

    # Paski synchronizacyjne
    w[6, :] = weight_timing
    w[:, 6] = weight_timing

    # białe ramki wokół wzorców pozycyjnych
    # Lewy-góra
    w[7, 0:8] = weight_separator
    w[0:8, 7] = weight_separator

    # Prawy-góra
    w[7, qr_core - 8:qr_core] = weight_separator
    w[0:8, qr_core - 8] = weight_separator

    # Lewy-dół
    w[qr_core - 8, 0:8] = weight_separator
    w[qr_core - 8:qr_core, 7] = weight_separator

    # Bity Formatu
    w[8, 0:9] = weight_format                 # Poziomy pasek lewy
    w[0:9, 8] = weight_format                 # Pionowy pasek górny
    w[8, qr_core - 8:qr_core] = weight_format # Poziomy pasek prawy
    w[qr_core - 8:qr_core, 8] = weight_format # Pionowy pasek dolny

    # Wzorce pozycyjne
    w[0:7, 0:7] = weight_finder
    w[0:7, qr_core - 7:qr_core] = weight_finder
    w[qr_core - 7:qr_core, 0:7] = weight_finder

    # 6. Spłaszczenie do jednego wymiaru
    return w.view(-1)