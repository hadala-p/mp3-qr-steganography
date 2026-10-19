# Projekt Magisterski

Instrukcja konfiguracji środowiska oraz uruchomienia projektu na systemie Windows.

---

## 1. Wymagania wstępne

- **Python:** Wersja `3.12.x` (zalecana `3.12.2`).
- **FFmpeg:** Zainstalowany w systemie i dostępny pod określoną ścieżką lub w zmiennej środowiskowej `PATH`.
- **Kompilator/Biblioteki systemowe:** Visual C++ Redistributable (wymagane m.in. przez bibliotekę `pyzbar`).

---

## 2. Przygotowanie środowiska wirtualnego

Otwórz terminal (PowerShell lub CMD) w głównym katalogu projektu:

```powershell
# 1. Utworzenie środowiska wirtualnego opartego na Pythonie 3.12
py -3.12 -m venv .venv

# 2. Aktywacja środowiska (opcjonalnie w PowerShell)
.venv\Scripts\Activate.ps1

3. Instalacja zależności

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

```

## 3. Specyfikacja danych wejściowych
Aby proces osadzania i odczytu przebiegał poprawnie, pliki wejściowe muszą spełniać następujące parametry:

* Pliki audio (MP3):

    Zalecany stały bitrate: 320 kbps.

    Częstotliwość próbkowania: 44.1 kHz (stereo lub mono).


* Obrazy kodów QR:

    Wymagana wersja: Wersja 1 (matryca 21×21 pikseli).

    Generowanie kompatybilnych kodów: https://pewscorner.github.io/barcodes/qrcode_generator.html

    Format zapisu: PNG

## 4. Struktura katalogów

Przed pierwszym uruchomieniem upewnij się, że pliki wejściowe oraz wytrenowane wagi modeli znajdują się w odpowiednich katalogach:
```
PROJEKT/
│
├── mp3_do_ukrycia/         # [Wejście] Pliki źródłowe audio (.mp3)
├── qr_do_ukrycia/          # [Wejście] Obrazy kodów QR w formacie PNG (21x21 px)
├── mp3_stego/              # [Wyjście] Wygenerowane pliki MP3 z osadzoną informacją
├── odkodowane_qr/          # [Wyjście] Zrekonstruowane obrazy PNG z kodami QR
│
├── script/
│   ├── constants.py        # Ścieżki bazowe, detekcja FFmpeg i stałe projektu
│   ├── embed.py            # Skrypt osadzania (ukrywania) QR w wybranym utworze
│   ├── extract.py          # Skrypt ekstrakcji i weryfikacji QR z pliku stego
│   ├── train.py            # Skrypt do trenowania sieci neuronowej (opcjonalny, wymaga GPU)
│   └── modele/
│       ├── encoder.pt      # Wagi sieci neuronowej enkodera
│       └── decoder.pt      # Wagi sieci neuronowej dekodera
│
└── qrstego/                # Moduły biblioteczne (architektura sieci, obsługa audio, kodek QR)
```

## 5. Konfiguracja FFmpeg (script/constants.py)

Skrypty korzystają ze scentralizowanej metody setup_ffmpeg() zdefiniowanej w constants.py:
1. Sprawdza, czy polecenie ffmpeg jest bezpośrednio dostępne w zmiennej środowiskowej PATH.

2. Jeśli nie, na systemie Windows przeszukuje domyślną ścieżkę instalacyjną WinGet:
`C:\Users\<Użytkownik>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg...`
   (* WAŻNE: W przypadku niestandardowej instalacji należy ręcznie zaktualizować zmienną FFmpeg_PATH w constants.py).

3. Na maszynach z systemem Linux oraz w środowisku Google Colab automatycznie wskazuje /usr/bin/ffmpeg.

## 6. Przeznaczenie skryptów i sposób użycia

* Ukrywanie danych (embed.py)

Przykłady uruchomienia:
```
# 1. Domyślne uruchomienie (szuka muzyka.mp3 i kod.png w dedykowanych folderach):
python script/embed.py

# 2. Przekazanie nazw plików jako argumenty pozycyjne:
python script/embed.py utwor.mp3 kod_qr.png

# 3. Zdefiniowanie własnej nazwy pliku wyjściowego:
python script/embed.py utwor.mp3 kod_qr.png -o utwor_stego_custom.mp3
```

* Ekstrakcja danych (extract.py)

Przykłady uruchomienia:
```
# 1. Odczyt domyślnego pliku określonego w OUTPUT_STEGO_PATH:
python script/extract.py

# 2. Odczyt wskazanego pliku stego z folderu mp3_stego/:
python script/extract.py utwor_stego.mp3

# 3. Zapis pod konkretną nazwą wyjściową pliku PNG:
python script/extract.py utwor_stego.mp3 zdekodowany_kod.png
```