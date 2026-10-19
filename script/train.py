from qrstego.config import RunConfig
from qrstego.stego.trainer import train

if __name__ == "__main__":
    cfg = RunConfig(
        audio_dir=r"D:\sciezka\do\audio_cache_bin",
        ffmpeg_path=r"C:\sciezka\do\ffmpeg.exe",
        epochs=1,
        batch_size=1,
        weight_finder=10.0,
        noise_multiplier=15.0,
        audio_loss_weight=2.0,
        learning_rate=0.0001

    )
    encoder, decoder = train(cfg, device="cuda")
