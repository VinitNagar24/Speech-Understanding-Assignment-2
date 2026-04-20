"""
part1_stt/task1_3_denoising.py
-------------------------------
Denoises audio using Spectral Subtraction (no external DeepFilterNet needed).
Estimates noise PSD from the first 0.5s (assumed noise-only) then subtracts
a over-subtraction-scaled estimate from every frame.

Ref: Boll (1979) "Suppression of acoustic noise in speech using spectral subtraction"
"""

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import os


# ── Spectral Subtraction ──────────────────────────────────────────────────────

class SpectralSubtractor:
    """
    Single-channel spectral subtraction denoiser.

    Parameters
    ----------
    sr            : sample rate
    n_fft         : FFT size
    hop_length    : hop between frames
    noise_frames  : number of leading frames assumed to be noise-only
    alpha         : over-subtraction factor  (1.0 – 2.0; higher → more aggressive)
    beta          : spectral floor  (prevents musical noise, 0.01 – 0.1)
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_frames: int = 10,
        alpha: float = 1.5,
        beta: float = 0.02,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_frames = noise_frames
        self.alpha = alpha
        self.beta = beta

    def _stft(self, waveform: np.ndarray):
        """Returns (complex spectrogram, magnitude, phase)."""
        window = np.hanning(self.n_fft)
        frames = []
        for i in range(0, len(waveform) - self.n_fft, self.hop_length):
            frame = waveform[i : i + self.n_fft] * window
            frames.append(np.fft.rfft(frame))
        S = np.array(frames).T            # shape (freq, time)
        mag = np.abs(S)
        phase = np.angle(S)
        return S, mag, phase

    def _istft(self, mag: np.ndarray, phase: np.ndarray, orig_len: int) -> np.ndarray:
        """Reconstruct waveform from magnitude + phase."""
        S = mag * np.exp(1j * phase)
        window = np.hanning(self.n_fft)
        out = np.zeros(orig_len)
        norm = np.zeros(orig_len)
        for t in range(S.shape[1]):
            frame = np.fft.irfft(S[:, t])
            start = t * self.hop_length
            end = start + self.n_fft
            if end > orig_len:
                frame = frame[: orig_len - start]
                out[start:] += frame * window[: len(frame)]
                norm[start:] += window[: len(frame)] ** 2
            else:
                out[start:end] += frame * window
                norm[start:end] += window ** 2
        norm = np.maximum(norm, 1e-8)
        return out / norm

    def denoise(self, waveform: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        waveform : 1-D float32 numpy array, range [-1, 1]

        Returns
        -------
        denoised  : 1-D float32 numpy array
        """
        _, mag, phase = self._stft(waveform)

        # Estimate noise PSD from leading frames
        noise_est = mag[:, : self.noise_frames].mean(axis=1, keepdims=True)  # (freq, 1)

        # Spectral subtraction: max(|X| - α·|N|, β·|X|)
        mag_clean = np.maximum(mag - self.alpha * noise_est, self.beta * mag)

        return self._istft(mag_clean, phase, len(waveform)).astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def denoise_audio(
    input_path: str,
    output_path: str,
    alpha: float = 1.5,
    beta: float = 0.02,
) -> str:
    """
    Load WAV → denoise via spectral subtraction → save WAV.

    Parameters
    ----------
    input_path  : path to noisy WAV
    output_path : path to write cleaned WAV
    alpha       : over-subtraction factor
    beta        : spectral floor

    Returns
    -------
    output_path
    """
    waveform, sr = torchaudio.load(input_path)

    # Convert to mono numpy
    waveform_np = waveform.mean(dim=0).numpy()

    denoiser = SpectralSubtractor(sr=sr, alpha=alpha, beta=beta)
    clean_np = denoiser.denoise(waveform_np)

    # Back to torch tensor
    clean_tensor = torch.from_numpy(clean_np).unsqueeze(0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(output_path, clean_tensor, sr)
    print(f"[Denoising] Saved denoised audio → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha",  type=float, default=1.5)
    p.add_argument("--beta",   type=float, default=0.02)
    args = p.parse_args()
    denoise_audio(args.input, args.output, args.alpha, args.beta)
