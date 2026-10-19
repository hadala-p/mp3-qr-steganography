import os
import random
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
import torchaudio.functional as aF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from qrstego.audio.mp3_tools import mp3_roundtrip_tensor
from qrstego.constants import SAMPLE_RATE, EPS, THR, MSG_SIZE
from qrstego.model.masks import build_weight_mask
from qrstego.model.net import QRAudioStego, QRAudioEncoder
from qrstego.utils.metrics import ber_from_logits


class FastBinaryDataset(Dataset):
    def __init__(self, cache_dir, msg_dim, qr_size=21):
        self.chunks = []
        files = sorted([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')])

        print(f"Ładuję {len(files)} paczek z RAM-dysku...")
        for f in files:
            self.chunks.append(torch.load(f, weights_only=True))

        self.data = torch.cat(self.chunks, dim=0)
        print(f"Dane gotowe! Kształt głównego tensora: {self.data.shape}")

        self.msg_dim = msg_dim
        self.qr_size = qr_size

        # W fazie inicjalizacji budujemy "pieczątkę" struktury QR
        self.fixed_mask, self.fixed_values = self._build_qr_template()

    def _build_qr_template(self):
        """Generuje stałe elementy kodu QR"""
        size = self.qr_size
        # mask - True oznacza, że ten piksel jest stałym elementem budowy"
        mask = torch.zeros((size, size), dtype=torch.bool)
        values = torch.zeros((size, size), dtype=torch.float32)

        def draw_finder(r, c):
            mask[r:r + 7, c:c + 7] = True
            values[r:r + 7, c:c + 7] = 1.0  # Zewn. czarna ramka
            values[r + 1:r + 6, c + 1:c + 6] = 0.0  # Biała przerwa (separator wewn.)
            values[r + 2:r + 5, c + 2:c + 5] = 1.0  # Czarne jądro 3x3

        # 1. Separatory = Białe obwódki wokół wzorców
        mask[7, 0:8] = True;
        values[7, 0:8] = 0.0
        mask[0:8, 7] = True;
        values[0:8, 7] = 0.0

        mask[7, size - 8:size] = True;
        values[7, size - 8:size] = 0.0
        mask[0:8, size - 8] = True;
        values[0:8, size - 8] = 0.0

        mask[size - 8, 0:8] = True;
        values[size - 8, 0:8] = 0.0
        mask[size - 8:size, 7] = True;
        values[size - 8:size, 7] = 0.0

        # Wzorce Pozycyjne = Nadpisują ewentualne styki z separatorami
        draw_finder(0, 0)
        draw_finder(0, size - 7)
        draw_finder(size - 7, 0)

        # Paski synchronizacyjne = Timing patterns na zmianę 1 i 0
        for i in range(size):
            if not mask[6, i]:
                mask[6, i] = True
                values[6, i] = 1.0 if i % 2 == 0 else 0.0
            if not mask[i, 6]:
                mask[i, 6] = True
                values[i, 6] = 1.0 if i % 2 == 0 else 0.0

        return mask.view(-1), values.view(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio = self.data[index]

        # Generujemy losowe bity = dane wewnątrz QR
        msg_bits = torch.randint(0, 2, (self.msg_dim,)).float()

        # 2. NAKŁADAMY STRUKTURĘ:
        # Zastępujemy losowe piksele naszymi stałymi wzorcami
        msg_bits[self.fixed_mask] = self.fixed_values[self.fixed_mask]

        return audio, msg_bits

def calculate_local_energy(audio: torch.Tensor, energy_multiplier: float = 1.5, device: str = "cpu") -> torch.Tensor:
    window = torch.ones(1, 1, 1024, device=device) / 1024 # 23ms
    base_energy = F.conv1d(audio.abs(), window, padding=512)[..., :audio.shape[-1]]
    local_energy = (base_energy ** 2) * energy_multiplier # narazie podniesienie do kwadratu aby podkreślić różnice w głośności
    return local_energy


def batch_mp3_roundtrip(stego_batch: torch.Tensor, ffmpeg_path: str, bitrates: list = [320]) -> torch.Tensor:
    bitrate = random.choice(bitrates)
    device = stego_batch.device

    # Funkcja pomocnicza dla jednego wątku
    def process_single(i):
        single_audio = stego_batch[i].detach()
        if single_audio.dim() > 1:
            single_audio = single_audio.squeeze()

        s_mp3 = mp3_roundtrip_tensor(
            ffmpeg_path,
            single_audio.unsqueeze(0),
            SAMPLE_RATE,
            bitrate
        )

        if s_mp3.dim() == 3:
            s_mp3 = s_mp3.squeeze(1)
        return s_mp3

    with ThreadPoolExecutor(max_workers=12) as executor:
        stego_mp3_list = list(executor.map(process_single, range(stego_batch.size(0))))

    return torch.stack(stego_mp3_list, dim=0).to(device)


def save_checkpoints(encoder, decoder, epoch: int, enc_path: str, dec_path: str):
    """Zapisuje bieżące wagi oraz archiwizuje kopię z danej epoki"""
    # Zapis standardowy do łatwego wznawiania
    torch.save(encoder.state_dict(), enc_path)
    torch.save(decoder.state_dict(), dec_path)

    # Zapis do logów
    os.makedirs("checkpoints", exist_ok=True)
    hist_enc_path = os.path.join("checkpoints", f"encoder_epoch_{epoch}.pt")
    hist_dec_path = os.path.join("checkpoints", f"decoder_epoch_{epoch}.pt")

    torch.save(encoder.state_dict(), hist_enc_path)
    torch.save(decoder.state_dict(), hist_dec_path)
    print(f"-> Zabezpieczono kopię z epoki {epoch} w folderze checkpoints/")


def train(cfg, device: str):
    # dataset z ramu
    dataset = FastBinaryDataset(cache_dir=cfg.audio_dir, msg_dim=MSG_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        pin_memory=True
    )

    encoder = QRAudioEncoder(msg_dim=MSG_SIZE).to(device)
    decoder = QRAudioStego(msg_dim=MSG_SIZE).to(device)

    enc_path = "encoder_checkpoint.pt"
    dec_path = "decoder_checkpoint.pt"

    # Wznawianie z checkpointów
    if os.path.exists(enc_path) and os.path.exists(dec_path):
        print(f"--- Znaleziono checkpointy! Wznawiam uczenie z plików... ---")
        encoder.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(dec_path, map_location=device, weights_only=True))

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.learning_rate)
    wmask = build_weight_mask().to(device).view(1, -1)

    # NOISE_MULTIPLIER = 7.0
    # AUDIO_LOSS_WEIGHT = 2.0
    NOISE_MULTIPLIER = cfg.noise_multiplier
    AUDIO_LOSS_WEIGHT = cfg.audio_loss_weight

    print("Rozpoczynam trenowanie...")

    for epoch in range(cfg.epochs):
        encoder.train()
        decoder.train()

        for batch_idx, (audio, msg) in enumerate(dataloader):
            audio = audio.to(device)
            msg = msg.to(device)

            # ENKODOWANIE
            local_energy = calculate_local_energy(audio, 1.5, device)
            raw_delta = encoder(audio, msg)

            delta = raw_delta * (EPS * local_energy * NOISE_MULTIPLIER)

            # OCHRONA BASU
            delta = aF.highpass_biquad(delta, SAMPLE_RATE, cutoff_freq=300.0)

            # ŁĄCZENIE
            stego = (audio + delta).clamp(-1.0, 1.0)

            # KOMPRESJA MP3
            stego_mp3 = batch_mp3_roundtrip(stego, cfg.ffmpeg_path)

            # STE
            stego_differentiable = stego + (stego_mp3 - stego).detach()

            # DEKODOWANIE
            logits = decoder.decode_logits(stego_differentiable)
            msg_bce = (msg > 0).float()
            bce = F.binary_cross_entropy_with_logits(logits, msg_bce, reduction="none")

            loss_decode = (bce * wmask).sum() / bce.size(0)
            loss_audio = torch.mean(delta ** 2) * AUDIO_LOSS_WEIGHT
            loss = loss_decode + loss_audio

            # WSTECZNA PROPAGACJA
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()

            # LOGOWANIE
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    ber = ber_from_logits(logits, msg_bce, THR)
                    print(
                        f"Epoka {epoch} | B {batch_idx} | L_Dec: {loss_decode.item():.4f} | L_Aud: {loss_audio.item():.6f} | BER: {ber:.4f}")

        # Zapis po każdej epoce
        save_checkpoints(encoder, decoder, epoch, enc_path, dec_path)

    return encoder, decoder