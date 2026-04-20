"""
part4_adversarial/task4_1_antispoofing.py
------------------------------------------
Anti-Spoofing Countermeasure (CM) System.

Architecture:
  LFCC features → Lightweight LCNN (Light CNN) classifier
  → Binary: Bona Fide (0) vs Spoof (1)
  → Evaluated by Equal Error Rate (EER)

LFCC: Linear Frequency Cepstral Coefficients
  Unlike MFCC (mel-scale), LFCC uses a linear filter bank, which preserves
  fine-grained spectral details important for spoofing detection.

EER: point where FAR (False Accept Rate) = FRR (False Reject Rate).
Target: EER < 10%.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from typing import List, Tuple, Dict, Optional
import scipy.signal


SR      = 16000
N_LFCC  = 60       # number of LFCC coefficients
HOP     = 160      # 10ms hop at 16kHz
N_FFT   = 512


# ── LFCC Feature Extraction ───────────────────────────────────────────────────

class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients.

    Steps:
      1. STFT → power spectrum
      2. Linear filter bank (uniform triangular filters)
      3. Log → DCT → LFCC
    """

    def __init__(self, sr: int = SR, n_filters: int = 70, n_lfcc: int = N_LFCC,
                 n_fft: int = N_FFT, hop: int = HOP):
        self.sr       = sr
        self.n_filters = n_filters
        self.n_lfcc   = n_lfcc
        self.n_fft    = n_fft
        self.hop      = hop
        self.filter_bank = self._build_linear_filter_bank()

    def _build_linear_filter_bank(self) -> np.ndarray:
        """Triangular filters evenly spaced on linear frequency axis."""
        freqs   = np.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        centers = np.linspace(0, self.sr / 2, self.n_filters + 2)
        fb = np.zeros((self.n_filters, len(freqs)))
        for m in range(1, self.n_filters + 1):
            f_m1 = centers[m - 1]
            f_m  = centers[m]
            f_m2 = centers[m + 1]
            for k, f in enumerate(freqs):
                if f_m1 <= f < f_m:
                    fb[m - 1, k] = (f - f_m1) / (f_m - f_m1 + 1e-8)
                elif f_m <= f <= f_m2:
                    fb[m - 1, k] = (f_m2 - f) / (f_m2 - f_m + 1e-8)
        return fb   # (n_filters, n_fft//2+1)

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        waveform : 1-D float32 numpy array
        returns  : (n_lfcc, T) numpy array
        """
        # STFT
        _, _, S = scipy.signal.stft(
            waveform, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop
        )
        power = np.abs(S) ** 2  # (n_fft//2+1, T)

        # Linear filter bank
        fb_out = np.dot(self.filter_bank, power)  # (n_filters, T)
        log_fb = np.log(fb_out + 1e-8)

        # DCT
        from scipy.fftpack import dct
        lfcc = dct(log_fb, axis=0, norm="ortho")[: self.n_lfcc]  # (n_lfcc, T)

        # Add Δ and ΔΔ
        delta  = np.diff(lfcc,  prepend=lfcc[:,  :1], axis=1)
        delta2 = np.diff(delta, prepend=delta[:, :1], axis=1)
        return np.concatenate([lfcc, delta, delta2], axis=0).astype(np.float32)  # (3*n_lfcc, T)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpoofDataset(Dataset):
    """
    labels_json: [{"audio": path, "label": 0|1}, ...]
    0 = bona fide, 1 = spoof
    """

    def __init__(self, labels_json: str, max_frames: int = 300):
        with open(labels_json) as f:
            self.items = json.load(f)
        self.extractor = LFCCExtractor()
        self.max_frames = max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav, sr = torchaudio.load(item["audio"])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav_np = wav.mean(0).numpy()
        feat = self.extractor.extract(wav_np)  # (3*n_lfcc, T)

        # Pad or truncate to max_frames
        T = feat.shape[1]
        if T < self.max_frames:
            pad = np.zeros((feat.shape[0], self.max_frames - T), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)
        else:
            feat = feat[:, : self.max_frames]

        return torch.from_numpy(feat), torch.tensor(item["label"], dtype=torch.long)


def spoof_collate(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.stack(labels)


# ── Model: Lightweight LCNN ───────────────────────────────────────────────────

class LCNN(nn.Module):
    """
    Light CNN for anti-spoofing.
    Input: (B, 3*N_LFCC, T) → Binary logits (B, 2)
    """

    def __init__(self, in_channels: int = 3 * N_LFCC):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ── EER Calculation ───────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate.
    y_scores: higher = more likely spoof.
    Returns (EER, threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


# ── Training ──────────────────────────────────────────────────────────────────

def train_antispoofing(
    labels_json: str,
    save_path: str = "checkpoints/antispoofing.pt",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpoofDataset(labels_json)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=spoof_collate)

    model = LCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[AntiSpoof] Epoch {epoch}/{epochs}  loss={total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[AntiSpoof] Saved → {save_path}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_antispoofing(
    labels_json: str,
    checkpoint: str,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dataset = SpoofDataset(labels_json)
    loader  = DataLoader(dataset, batch_size=8, collate_fn=spoof_collate)
    extractor = LFCCExtractor()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            logits = model(feats.to(device))
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())

    eer, thresh = compute_eer(np.array(all_labels), np.array(all_scores))
    print(f"[AntiSpoof] EER = {eer*100:.2f}%  (target < 10%)  threshold={thresh:.4f}")
    result = {"EER_percent": eer * 100, "threshold": thresh, "pass": eer < 0.10}

    with open("output/antispoofing_eer.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def predict_bonafide_or_spoof(audio_path: str, checkpoint: str) -> Dict:
    """Predict single audio file: Bona Fide or Spoof."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav_np = wav.mean(0).numpy()
    extractor = LFCCExtractor()
    feat = extractor.extract(wav_np)

    max_frames = 300
    T = feat.shape[1]
    if T < max_frames:
        feat = np.pad(feat, ((0, 0), (0, max_frames - T)))
    else:
        feat = feat[:, :max_frames]

    feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(feat_t)
        probs  = torch.softmax(logits, dim=-1).squeeze()

    label = "Spoof" if probs[1] > probs[0] else "Bona Fide"
    return {
        "audio": audio_path,
        "prediction": label,
        "confidence_bonafide": float(probs[0]),
        "confidence_spoof": float(probs[1])
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("--labels", required=True)
    tr.add_argument("--save",   default="checkpoints/antispoofing.pt")
    tr.add_argument("--epochs", type=int, default=30)

    ev = sub.add_parser("eval")
    ev.add_argument("--labels",     required=True)
    ev.add_argument("--checkpoint", required=True)

    pr = sub.add_parser("predict")
    pr.add_argument("--audio",      required=True)
    pr.add_argument("--checkpoint", required=True)

    args = p.parse_args()
    if args.cmd == "train":
        train_antispoofing(args.labels, args.save, args.epochs)
    elif args.cmd == "eval":
        evaluate_antispoofing(args.labels, args.checkpoint)
    elif args.cmd == "predict":
        result = predict_bonafide_or_spoof(args.audio, args.checkpoint)
        print(json.dumps(result, indent=2))
