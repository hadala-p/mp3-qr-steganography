# MP3-QR-Steganography

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep learning-based cross-modal audio steganography system designed to embed structured QR codes into MP3 audio tracks with high acoustic transparency and resilience against lossy compression.**

---

## 📌 Overview

Traditional audio steganography methods (such as Least Significant Bit – LSB) fail catastrophically when subjected to lossy perceptual compression like MP3. Non-linear psychoacoustic quantization and encoder delays completely wipe out uncompressed time-domain bit modifications.

**MP3-QR-Steganography** addresses this limitation by training an end-to-end convolutional encoder-decoder pipeline directly against real MP3 compression. By utilizing the **Straight-Through Estimator (STE)**, non-differentiable MP3 encoding/decoding is integrated into the backpropagation training loop, forcing the model to embed perturbations into the most resilient, high-energy acoustic components.

Trained on a curated large-scale dataset of nearly **40,000 music tracks** from the [Free Music Archive (FMA)](https://github.com/mdeff/fma), the model acts as a universal steganographic key, generalising to previously unseen audio files.

---

## 🚀 Key Features

* **STE Gradient Approximation:** Uses Straight-Through Estimation (`x_diff = x_stego + (x_mp3 - x_stego).detach()`) to allow gradient propagation through non-differentiable MP3 codecs during training.
* **Psychoacoustic Energy Masking:** Dynamic amplitude adaptation scales embedding strength relative to the local signal energy envelope ($E_{local}$), keeping perturbations acoustically imperceptible.
* **Temporal Synchronization ("Radar"):** Compensates for MP3 encoder delay and padding via sliding-window cross-correlation against the structural finder patterns of the QR code.
* **Triple Embedding & Soft Voting:** Embeds identical QR payload replicas into the three loudest audio segments; extracted continuous logits are averaged prior to thresholding to mitigate localized artifacts.
* **QR Template & Geometric Correction:** Leverages standardized structural features (finder patterns, timing patterns, format info) combined with Reed-Solomon error correction for deterministic error recovery.
* **Zero Payload Size Footprint:** Stego audio introduces 0 additional bytes to the compressed audio stream compared to standard re-encoded baseline MP3 files.

---

## 📊 Experimental Results

Tested on a benchmark dataset of music tracks evaluated under realistic transmission conditions:

| Metric / Scenario | Proposed Deep Stego System | Classical LSB Method |
| :--- | :---: | :---: |
| **MP3 Compression (320 kbps)** | **100.0% Success** (0.00 BER) | **0.0% Success** (190.9 avg bit errors) |
| **Streaming Bitrate (128 kbps)** | **94.0% Success** (1.29 avg bit errors) | 0.0% Success |
| **Aggressive Bitrate (64 kbps)** | **55.0% Success** (8.84 avg bit errors) | 0.0% Success |
| **Audio Cropping (6 sec removed)** | **70.0% – 83.0% Success** | 0.0% (Desynchronization) |
| **Additive Noise (SNR 20 dB)** | **89.0% Success** | 0.0% |
| **Acoustic Transparency (SNR)** | **27.01 dB average** (min > 20 dB) | N/A |
| **Acoustic Transparency (PSNR)**| **41.34 dB average** (peak 60.97 dB) | N/A |

---

## 🛠️ System Architecture

```text
       [ Cover MP3 ] ──> FFmpeg PCM (44.1kHz) ──┐
                                                │
[ QR Code (21x21) ] ─> Flatten Tensor [441] ─> [ CNN Encoder ] ──> Psychoacoustic Masking
                                                                           │
                                                                  [ Stego Audio PCM ]
                                                                           │
                                                            ┌──────────────┴──────────────┐
                                                       Forward Pass                  Backward Pass
                                                    (Real MP3 Codec)              (Identity Bypass - STE)
                                                            │                                     ▲
                                                            ▼                                     │
                                                  [ MP3 Re-encoded Signal ] ──────────────────────┘
                                                            │
                                                     [ CNN Decoder ]
                                                            │
                                            Temporal Radar Synchronization
                                                            │
                                            Soft Voting (Triple Embedding)
                                                            │
                                             Geometric Template Correction
                                                            │
                                                            ▼
                                                 [ Recovered QR Code ]
```

---

## 💻 Quickstart

### Prerequisites
* Python 3.10+ (tested on Python 3.12)
* [FFmpeg](https://ffmpeg.org/) installed and added to system `PATH`
* NVIDIA GPU with CUDA support (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/hadala-p/mp3-qr-steganography.git
cd mp3-qr-steganography

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Embedding & Extracting
```bash
# 1. Embed QR code into an audio file
python -m src.embed --audio "data/samples/input.mp3" \
                    --qr "data/samples/target_qr.png" \
                    --output "output/stego.mp3"

# 2. Extract QR code from the stego audio
python -m src.extract --audio "output/stego.mp3" \
                      --output "output/recovered_qr.png"
```

---

## 📁 Repository Structure

```text
mp3-qr-steganography/
├── instrukcja.md
├── README.md
├── requirements.txt
├── mp3_do_ukrycia/
├── mp3_stego/
├── odkodowane_qr/
├── qr_do_ukrycia/
│   ├── STEGANOGRAFIA.png
│   ├── TAJNYTEXT.png
│   └── TRUDNYTEXT.png
├── qrstego/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── ffmpeg_io.py
│   │   └── mp3_tools.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── masks.py
│   │   └── net.py
│   ├── qr/
│   │   ├── __init__.py
│   │   └── qr_codec.py
│   ├── stego/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
└── script/
    ├── __init__.py
    ├── constants.py
    ├── embed.py
    ├── extract.py
    ├── train.py
    └── modele/
        ├── decoder.pt
        └── encoder.pt
```

---

## 🔬 Citation

If you use this work, codebase, or results in your research, please cite:

```bibtex
@mastersthesis{hadala2026steganography,
  author       = {Piotr Hadała},
  title        = {Badanie algorytmów sztucznej inteligencji do steganograficznej ochrony danych w plikach audio (Investigation of Artificial Intelligence Algorithms for Steganographic Data Protection in Audio Files)},
  school       = {Uniwersytet Komisji Edukacji Narodowej w Krakowie (UKEN), Instytut Bezpieczeństwa i Informatyki},
  year         = {2026},
  address      = {Kraków, Poland},
  note         = {Supervised by dr hab. prof. UKEN Serhii Semenov}
}
```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
