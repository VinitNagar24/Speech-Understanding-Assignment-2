"""
part1_stt/task1_1_lid.py
------------------------
Multi-Head Frame-Level Language Identification (LID)
Distinguishes English (L2) vs Hindi at the frame level.

Architecture
------------
  Shared MFCC encoder  →  [Head-A: binary EN/HI]
                        →  [Head-B: 3-class EN / HI / code-switch]

Training uses weak labels from langdetect on forced-aligned word segments,
then propagates labels at the frame level.

Target: F1 ≥ 0.85 (binary EN vs HI)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import f1_score, classification_report
from typing import List, Tuple, Dict, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
SR          = 16000
N_MFCC      = 40
FRAME_LEN   = 0.025      # 25 ms
FRAME_SHIFT = 0.010      # 10 ms
HOP_LENGTH  = int(SR * FRAME_SHIFT)   # 160
N_FFT       = int(SR * FRAME_LEN * 2) # 800
CONTEXT     = 15         # ± 15 frames context window
LABELS      = {"en": 0, "hi": 1, "cs": 2}   # cs = code-switch
BINARY_MAP  = {0: 0, 1: 1, 2: 1}            # cs → Hindi for binary head


# ── Feature Extraction ────────────────────────────────────────────────────────

class MFCCExtractor(nn.Module):
    """Extracts Δ + ΔΔ MFCC features → (time, 3*N_MFCC)."""

    def __init__(self, sr: int = SR, n_mfcc: int = N_MFCC):
        super().__init__()
        self.mfcc = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": 80},
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform : (1, T)  float32 waveform
        returns  : (time_steps, 3*N_MFCC)
        """
        mfcc = self.mfcc(waveform)          # (1, n_mfcc, time)
        delta  = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        feat = torch.cat([mfcc, delta, delta2], dim=1)  # (1, 3*n_mfcc, time)
        return feat.squeeze(0).T            # (time, 3*n_mfcc)


def add_context(feat: torch.Tensor, context: int = CONTEXT) -> torch.Tensor:
    """
    Stacks ± context frames around each frame.
    feat    : (T, D)
    returns : (T, D * (2*context+1))
    """
    T, D = feat.shape
    padded = torch.nn.functional.pad(feat.unsqueeze(0).permute(0, 2, 1),
                                     (context, context), mode="replicate"
                                     ).permute(0, 2, 1).squeeze(0)
    windows = [padded[i: i + T] for i in range(2 * context + 1)]
    return torch.cat(windows, dim=-1)       # (T, D*(2C+1))


# ── Model ─────────────────────────────────────────────────────────────────────

class MultiHeadLID(nn.Module):
    """
    Shared LSTM encoder + two classification heads.
      Head A : binary  (EN=0 / HI=1)
      Head B : 3-class (EN=0 / HI=1 / CS=2)
    """

    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        enc_out = hidden * 2  # bidirectional

        self.head_binary = nn.Sequential(
            nn.Linear(enc_out, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        self.head_3class = nn.Sequential(
            nn.Linear(enc_out, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, x: torch.Tensor):
        """
        x       : (B, T, input_dim)
        returns : logits_binary (B,T,2), logits_3class (B,T,3)
        """
        enc, _ = self.encoder(x)          # (B, T, 2*hidden)
        return self.head_binary(enc), self.head_3class(enc)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LIDFrameDataset(Dataset):
    """
    Each sample = (feature_tensor, label_tensor) for one utterance.

    labels_json format:
    [
      {
        "audio": "data/utt1.wav",
        "frame_labels": [0, 0, 1, 1, 2, ...]   # per-frame 0=EN,1=HI,2=CS
      },
      ...
    ]
    """

    def __init__(self, labels_json: str, extractor: MFCCExtractor, context: int = CONTEXT):
        with open(labels_json) as f:
            self.items = json.load(f)
        self.extractor = extractor
        self.context = context

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav, sr = torchaudio.load(item["audio"])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(0, keepdim=True)

        with torch.no_grad():
            feat = self.extractor(wav)      # (T, D)
        feat = add_context(feat, self.context)

        labels = torch.tensor(item["frame_labels"], dtype=torch.long)
        # Align lengths
        min_len = min(feat.shape[0], labels.shape[0])
        return feat[:min_len], labels[:min_len]


def collate_fn(batch):
    feats, labels = zip(*batch)
    feats  = nn.utils.rnn.pad_sequence(feats,  batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    return feats, labels


# ── Weak Label Generator (no manual annotation needed) ───────────────────────

def generate_weak_labels(
    transcript_segments: List[Dict],  # [{start, end, text}, ...]
    total_frames: int,
    sr: int = SR,
) -> List[int]:
    """
    Propagate word-level language tags (from langdetect) to frames.
    Falls back to EN for ambiguous segments.
    """
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        print("[LID] langdetect not installed; all frames labelled EN")
        return [0] * total_frames

    frame_labels = [0] * total_frames   # default: English

    for seg in transcript_segments:
        try:
            lang = detect(seg["text"])
        except Exception:
            lang = "en"

        if lang in ("hi", "mr", "ur"):
            label = 1
        else:
            label = 0

        start_f = int(seg["start"] * sr / HOP_LENGTH)
        end_f   = int(seg["end"]   * sr / HOP_LENGTH)
        for f in range(start_f, min(end_f, total_frames)):
            frame_labels[f] = label

    return frame_labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_lid(
    labels_json: str,
    save_path: str = "checkpoints/lid_model.pt",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 8,
):
    extractor = MFCCExtractor()
    input_dim = N_MFCC * 3 * (2 * CONTEXT + 1)
    model = MultiHeadLID(input_dim=input_dim)

    dataset = LIDFrameDataset(labels_json, extractor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, labels in loader:
            feats  = feats.to(device)
            labels = labels.to(device)

            logits_bin, logits_3cls = model(feats)

            # Binary labels
            bin_labels = labels.clone()
            bin_labels[bin_labels == 2] = 1   # CS → HI

            loss_bin  = criterion(logits_bin.reshape(-1, 2),  bin_labels.reshape(-1))
            loss_3cls = criterion(logits_3cls.reshape(-1, 3), labels.reshape(-1))
            loss = loss_bin + 0.5 * loss_3cls

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"[LID] Epoch {epoch}/{epochs}  loss={total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[LID] Model saved → {save_path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def load_lid_model(checkpoint: str) -> Tuple[MultiHeadLID, MFCCExtractor]:
    extractor = MFCCExtractor()
    input_dim = N_MFCC * 3 * (2 * CONTEXT + 1)
    model = MultiHeadLID(input_dim=input_dim)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, extractor


def predict_language_frames(
    audio_path: str,
    model: MultiHeadLID,
    extractor: MFCCExtractor,
    device: str = "cpu",
) -> Dict:
    """
    Returns:
      {
        "frame_labels_binary": [...],      # 0=EN, 1=HI per frame
        "frame_labels_3class": [...],      # 0=EN, 1=HI, 2=CS per frame
        "switch_timestamps_sec": [...],    # seconds where language flips
        "frame_duration_sec": 0.010
      }
    """
    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0, keepdim=True)

    with torch.no_grad():
        feat = extractor(wav)
        feat = add_context(feat)
        feat = feat.unsqueeze(0).to(device)
        logits_bin, logits_3cls = model(feat)

    preds_bin  = logits_bin.squeeze(0).argmax(-1).cpu().tolist()
    preds_3cls = logits_3cls.squeeze(0).argmax(-1).cpu().tolist()

    # Detect switch points
    switches = []
    for i in range(1, len(preds_bin)):
        if preds_bin[i] != preds_bin[i - 1]:
            switches.append(round(i * FRAME_SHIFT, 3))

    return {
        "frame_labels_binary": preds_bin,
        "frame_labels_3class": preds_3cls,
        "switch_timestamps_sec": switches,
        "frame_duration_sec": FRAME_SHIFT,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_lid(labels_json: str, model: MultiHeadLID, extractor: MFCCExtractor):
    """Compute F1 score on held-out utterances."""
    with open(labels_json) as f:
        items = json.load(f)

    all_true, all_pred = [], []
    for item in items:
        result = predict_language_frames(item["audio"], model, extractor)
        preds  = result["frame_labels_binary"]
        gt     = item["frame_labels"]
        min_l  = min(len(preds), len(gt))
        all_pred.extend(preds[:min_l])
        all_true.extend(gt[:min_l])

    f1 = f1_score(all_true, all_pred, average="macro")
    print(f"[LID] Macro F1 = {f1:.4f}  (target ≥ 0.85)")
    print(classification_report(all_true, all_pred, target_names=["EN", "HI"]))
    return f1


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("--labels", required=True, help="Path to labels JSON")
    tr.add_argument("--save",   default="checkpoints/lid_model.pt")
    tr.add_argument("--epochs", type=int, default=20)

    inf = sub.add_parser("infer")
    inf.add_argument("--audio",      required=True)
    inf.add_argument("--checkpoint", required=True)

    args = p.parse_args()

    if args.cmd == "train":
        train_lid(args.labels, args.save, args.epochs)

    elif args.cmd == "infer":
        model, ext = load_lid_model(args.checkpoint)
        result = predict_language_frames(args.audio, model, ext)
        print(json.dumps(result, indent=2))
