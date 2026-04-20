"""
download_audio.py
-----------------
Downloads a 10-minute segment of a public NPTEL Hinglish lecture using yt-dlp
and trims it to exactly 600 seconds using torchaudio.

Usage:
    python download_audio.py --url <YouTube_URL> --start 00:05:00 --output data/original_segment.wav
"""

import argparse
import os
import subprocess
import torchaudio
import torch

# ── Default NPTEL URL (Speech Understanding course, IIT Madras) ──────────────
DEFAULT_URL = "https://www.youtube.com/watch?v=X2iLjQ4Y7_M"  # Replace with valid NPTEL URL
SEGMENT_DURATION = 600   # 10 minutes in seconds
TARGET_SR = 22050        # 22.05 kHz as required by assignment


def download_youtube_audio(url: str, out_path: str = "data/raw_lecture.wav") -> str:
    """Download audio from YouTube using yt-dlp and convert to WAV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path.replace(".wav", ".%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", tmp_path,
        url
    ]
    print(f"[download_audio] Downloading: {url}")
    subprocess.run(cmd, check=True)

    # yt-dlp may name the file differently — locate it
    base = out_path.replace(".wav", "")
    for ext in [".wav", ".webm.wav", ".m4a.wav"]:
        candidate = base + ext
        if os.path.exists(candidate):
            if candidate != out_path:
                os.rename(candidate, out_path)
            break

    print(f"[download_audio] Saved raw audio → {out_path}")
    return out_path


def trim_and_resample(
    in_path: str,
    out_path: str,
    start_sec: float = 300.0,     # skip first 5 min (intro)
    duration_sec: float = SEGMENT_DURATION,
    target_sr: int = TARGET_SR,
) -> str:
    """Trim to [start, start+duration] and resample to target_sr."""
    waveform, sr = torchaudio.load(in_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim
    start_frame = int(start_sec * sr)
    end_frame = int((start_sec + duration_sec) * sr)
    waveform = waveform[:, start_frame:end_frame]

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, waveform, target_sr)
    print(f"[download_audio] Trimmed & resampled → {out_path}  "
          f"({waveform.shape[1]/target_sr:.1f}s @ {target_sr}Hz)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download NPTEL lecture audio")
    parser.add_argument("--url", default=DEFAULT_URL, help="YouTube URL")
    parser.add_argument("--start", type=float, default=300.0,
                        help="Start offset in seconds (default: 300 = skip 5 min intro)")
    parser.add_argument("--output", default="data/original_segment.wav")
    args = parser.parse_args()

    raw = download_youtube_audio(args.url, out_path="data/raw_lecture.wav")
    trim_and_resample(raw, args.output, start_sec=args.start)


if __name__ == "__main__":
    main()
