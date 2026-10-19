import torch
import torch.nn as nn


class QRAudioEncoder(nn.Module):
    def __init__(self, msg_dim: int):
        super().__init__()

        self.msg_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.GELU()
        )

        # Łączenie: audio ma 128 kanałów, a wiadomość ma 32 kanały
        self.joint_conv = nn.Sequential(
            nn.Conv1d(128 + 32, 128, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=9, padding=4),
            nn.Tanh()  # Ogranicza amplitudę szumu do [-1, 1]
        )

    def forward(self, audio: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:

        # Ekstrakcja cech audio
        a = self.audio_conv(audio)

        # Przetworzenie wiadomości bez utraty kolejności
        m = msg.unsqueeze(1)  # Dodajemy wymiar kanału: [batch, 1, 441]
        m = self.msg_conv(m)  # [batch, 32, 441]

        # rozciągamy 441 bitów w czasie, aby dopasować je do długości skróconego audio.
        # tryb nearest, aby bity się ze sobą nie zlewały.
        m = torch.nn.functional.interpolate(m, size=a.shape[2], mode='nearest')

        # Połączenie audio i ułożonej w czasie wiadomości
        x = torch.cat([a, m], dim=1)

        # Generowanie szumu i przywrócenie pierwotnej długości
        delta_compressed = self.joint_conv(x)
        delta = torch.nn.functional.interpolate(delta_compressed, size=audio.shape[2], mode='linear',
                                                align_corners=False)

        return delta


class QRAudioStego(nn.Module):
    def __init__(self, msg_dim: int):
        super().__init__()

        self.features = nn.Sequential(
            # ZMIANA: stride=2 zamiast stride=4 w całej sieci
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(msg_dim)
        )
        self.classifier = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)

    def decode_logits(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.features(audio)
        x = self.classifier(x)
        return x.view(audio.size(0), -1)
