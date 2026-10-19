import subprocess

def run_ffmpeg(ffmpeg_path: str, cmd: list[str], input_bytes: bytes | None = None) -> bytes:
    p = subprocess.run([ffmpeg_path] + cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"Błąd FFmpeg:\n{p.stderr.decode('utf-8', errors='ignore')}")
    return p.stdout