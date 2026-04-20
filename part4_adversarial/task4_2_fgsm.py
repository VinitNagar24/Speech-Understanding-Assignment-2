"""
part4_adversarial/task4_2_fgsm.py
------------------------------------
Adversarial Noise Injection using FGSM (Fast Gradient Sign Method)
targeting the LID system from Task 1.1.

Goal: Find minimum epsilon such that:
  - The perturbation is inaudible (SNR > 40 dB)
  - The LID system misclassifies Hindi as English

FGSM: x_adv = x + ε · sign(∇_x L(f(x), y_wrong))

We report:
  - Minimum epsilon that flips LID prediction
  - SNR of the adversarial audio
  - Confusion matrix before/after attack
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple


# ── SNR Calculation ───────────────────────────────────────────────────────────

def compute_snr(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB.
    SNR = 10 * log10(power(signal) / power(noise))
    Higher SNR = less audible perturbation. Target: SNR > 40 dB.
    """
    noise  = perturbed - original
    signal_power = np.mean(original ** 2) + 1e-10
    noise_power  = np.mean(noise ** 2)    + 1e-10
    return float(10.0 * np.log10(signal_power / noise_power))


# ── Differentiable MFCC wrapper ───────────────────────────────────────────────

class DiffMFCC(nn.Module):
    """
    Differentiable MFCC extraction for gradient-based attacks.
    Built on torchaudio's transforms (which support autograd).
    """

    def __init__(self, sr: int = 16000, n_mfcc: int = 40):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 800, "hop_length": 160, "n_mels": 80},
        )
        self.delta  = torchaudio.transforms.ComputeDeltas()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (1, T) → features: (1, 3*n_mfcc, T)"""
        m  = self.mfcc(waveform)
        d  = self.delta(m)
        d2 = self.delta(d)
        return torch.cat([m, d, d2], dim=1)


# ── FGSM Attack ───────────────────────────────────────────────────────────────

class FGSMAttack:
    """
    FGSM adversarial attack against the LID model.

    Targets the binary head to flip prediction from Hindi (1) → English (0).
    """

    def __init__(
        self,
        lid_model,          # MultiHeadLID instance from task1_1_lid.py
        mfcc_extractor,     # MFCCExtractor from task1_1_lid.py
        device: str = "cpu",
        sr: int = 16000,
    ):
        self.lid    = lid_model.to(device).eval()
        self.device = device
        self.sr     = sr

        # Use differentiable MFCC for gradient flow
        self.diff_mfcc = DiffMFCC(sr=sr).to(device)

    def attack(
        self,
        waveform: torch.Tensor,
        target_label: int = 0,     # 0 = English (we want to fool Hindi → English)
        epsilon_range: List[float] = None,
        snr_threshold: float = 40.0,
    ) -> Dict:
        """
        Sweep epsilon values and find minimum that flips LID prediction
        while maintaining SNR > 40 dB.

        Parameters
        ----------
        waveform      : (1, T) original waveform (should be a Hindi segment)
        target_label  : label we want the model to predict (0 = English)
        epsilon_range : list of epsilon values to try
        snr_threshold : minimum acceptable SNR in dB

        Returns
        -------
        dict with: min_epsilon, snr_at_epsilon, predictions, adversarial_wav
        """
        if epsilon_range is None:
            epsilon_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

        waveform = waveform.to(self.device)
        wav_orig = waveform.clone().detach()

        results = []
        min_epsilon = None
        best_adv_wav = None

        for eps in epsilon_range:
            adv_wav, pred_before, pred_after = self._fgsm_step(
                wav_orig.clone(), target_label, eps
            )
            wav_np  = wav_orig.squeeze().cpu().numpy()
            adv_np  = adv_wav.squeeze().cpu().numpy()
            snr     = compute_snr(wav_np, adv_np)
            flipped = (pred_after == target_label) and (pred_before != target_label)

            entry = {
                "epsilon":     eps,
                "snr_db":      snr,
                "pred_before": pred_before,
                "pred_after":  pred_after,
                "flipped":     flipped,
                "snr_ok":      snr > snr_threshold,
            }
            results.append(entry)
            print(f"[FGSM] ε={eps:.5f}  SNR={snr:.1f}dB  "
                  f"pred: {pred_before}→{pred_after}  flipped={flipped}")

            if flipped and snr > snr_threshold and min_epsilon is None:
                min_epsilon  = eps
                best_adv_wav = adv_wav.cpu()

        result = {
            "min_epsilon":          min_epsilon,
            "snr_threshold_db":     snr_threshold,
            "sweep_results":        results,
            "attack_successful":    min_epsilon is not None,
        }

        # Save adversarial audio
        if best_adv_wav is not None:
            os.makedirs("output", exist_ok=True)
            torchaudio.save("output/adversarial_segment.wav", best_adv_wav, self.sr)
            result["adversarial_audio"] = "output/adversarial_segment.wav"
            print(f"[FGSM] Saved adversarial audio → output/adversarial_segment.wav")

        return result

    def _fgsm_step(
        self,
        waveform: torch.Tensor,
        target_label: int,
        epsilon: float,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Single FGSM step.
        Returns (adversarial_waveform, pred_before, pred_after).
        """
        waveform = waveform.requires_grad_(True)

        # Get prediction before attack
        with torch.no_grad():
            feat_before = self.diff_mfcc(waveform.detach())
            # Add context (simplified: just use raw MFCC, not windowed context)
            feat_before = feat_before.permute(0, 2, 1)  # (1, T, features)
            # Pad feature dim to match LID model's expected input
            logits_bin, _ = self.lid(feat_before)
            pred_before = logits_bin.argmax(-1).mean().round().long().item()

        # Forward pass for gradient
        feat = self.diff_mfcc(waveform)
        feat = feat.permute(0, 2, 1)  # (1, T, features)
        logits_bin, _ = self.lid(feat)

        # Loss: push toward target label (English = 0) across all frames
        target = torch.full(
            (logits_bin.shape[0], logits_bin.shape[1]),
            target_label,
            dtype=torch.long,
            device=self.device
        )
        loss = F.cross_entropy(logits_bin.reshape(-1, 2), target.reshape(-1))
        loss.backward()

        # FGSM perturbation
        grad_sign = waveform.grad.sign()
        adv_wav   = waveform.detach() + epsilon * grad_sign
        adv_wav   = adv_wav.clamp(-1.0, 1.0)

        # Prediction after attack
        with torch.no_grad():
            feat_after = self.diff_mfcc(adv_wav)
            feat_after = feat_after.permute(0, 2, 1)
            logits_bin_after, _ = self.lid(feat_after)
            pred_after = logits_bin_after.argmax(-1).float().mean().round().long().item()

        return adv_wav.detach(), pred_before, pred_after


# ── Robustness Evaluation ─────────────────────────────────────────────────────

def evaluate_adversarial_robustness(
    audio_path: str,
    lid_checkpoint: str,
    output_json: str = "output/adversarial_robustness.json",
    segment_duration: float = 5.0,  # 5 seconds as required
) -> Dict:
    """
    Run FGSM sweep on a 5-second Hindi segment and report robustness metrics.
    """
    from part1_stt.task1_1_lid import load_lid_model, SR, HOP_LENGTH, CONTEXT, N_MFCC, add_context

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, extractor = load_lid_model(lid_checkpoint)
    model.to(device)

    # Load 5-second segment
    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0, keepdim=True)

    # Take first 5 seconds
    seg_samples = int(segment_duration * SR)
    wav = wav[:, :seg_samples]

    attacker = FGSMAttack(model, extractor, device=device, sr=SR)
    result   = attacker.attack(
        wav,
        target_label=0,  # flip Hindi → English
        epsilon_range=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        snr_threshold=40.0,
    )

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"[FGSM] Min epsilon to flip LID: {result['min_epsilon']}")
    print(f"[FGSM] Attack successful: {result['attack_successful']}")
    print(f"{'='*50}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio",      required=True, help="5s Hindi audio segment")
    p.add_argument("--checkpoint", required=True, help="LID model checkpoint")
    p.add_argument("--output",     default="output/adversarial_robustness.json")
    args = p.parse_args()
    evaluate_adversarial_robustness(args.audio, args.checkpoint, args.output)
