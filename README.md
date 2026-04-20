# Speech Understanding — Programming Assignment 2
## Code-Switched Hinglish → Rajasthani TTS Pipeline

---

## Overview

This repository implements a full end-to-end pipeline for:
1. **Transcribing** Hinglish (Code-Switched English+Hindi) lecture audio
2. **Converting** the transcript to IPA and translating to Rajasthani
3. **Synthesizing** the lecture in Rajasthani using zero-shot voice cloning
4. **Evaluating** robustness via anti-spoofing and adversarial attacks

**Target Low-Resource Language:** Rajasthani (Marwari dialect)  
**Source Audio:** NPTEL Hinglish lecture (10 minutes)

---

## Repository Structure

```
pipeline/
├── pipeline.py                          ← Master runner
├── download_audio.py                    ← NPTEL audio downloader (yt-dlp)
├── requirements.txt
├── README.md
│
├── part1_stt/
│   ├── task1_1_lid.py                   ← Multi-head frame-level LID (BiLSTM)
│   ├── task1_2_constrained_decoding.py  ← Whisper + N-gram logit bias
│   └── task1_3_denoising.py             ← Spectral subtraction denoising
│
├── part2_phonetic/
│   ├── task2_1_ipa_mapping.py           ← Hinglish → IPA (custom G2P)
│   └── task2_2_translation.py          ← IPA/Text → Rajasthani (500-word corpus)
│
├── part3_tts/
│   ├── task3_1_voice_embedding.py       ← x-vector (ECAPA-TDNN / d-vector)
│   ├── task3_2_prosody_warping.py       ← F0 + Energy DTW warping
│   └── task3_3_synthesis.py             ← YourTTS / MMS synthesis + MCD eval
│
├── part4_adversarial/
│   ├── task4_1_antispoofing.py          ← LFCC + LCNN classifier, EER
│   └── task4_2_fgsm.py                  ← FGSM attack on LID, SNR analysis
│
└── data/                                ← (created at runtime)
    ├── original_segment.wav
    ├── student_voice_ref.wav            ← YOU must record this (60s)
    └── denoised_segment.wav
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/SpeechUnderstanding-PA2.git
cd SpeechUnderstanding-PA2
```

### 2. Create environment
```bash
conda create -n su_pa2 python=3.10 -y
conda activate su_pa2
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# Install ffmpeg (required by yt-dlp and whisper)
sudo apt install ffmpeg -y   # Linux
brew install ffmpeg           # macOS
```


---

## Usage

### Run the full pipeline
```bash
python pipeline.py --step all
```

### Run individual steps
```bash
# Download NPTEL lecture
python pipeline.py --step download

# Denoise audio
python pipeline.py --step denoise

# Train LID model
python pipeline.py --step lid

# Transcribe with Whisper
python pipeline.py --step transcribe

# Convert to IPA
python pipeline.py --step ipa

# Translate to Rajasthani
python pipeline.py --step translate

# Extract speaker embedding (needs student_voice_ref.wav!)
python pipeline.py --step embedding

# Synthesize TTS
python pipeline.py --step synthesize

# Apply prosody warping
python pipeline.py --step prosody

# Train & eval anti-spoofing
python pipeline.py --step antispoofing

# FGSM adversarial attack
python pipeline.py --step fgsm
```

### Use a custom config
```bash
python pipeline.py --config config.json --step all
```

Example `config.json`:
```json
{
  "nptel_url": "https://www.youtube.com/watch?v=YOUR_LECTURE_ID",
  "whisper_model": "large-v3",
  "lid_epochs": 20,
  "spoof_epochs": 30
}
```

---

## Important: Record Your Voice

Before running synthesis, you **must** record 60 seconds of your own voice:
```bash
# Using sox
rec -r 22050 -c 1 data/student_voice_ref.wav trim 0 60

# Or using ffmpeg
ffmpeg -f alsa -i default -t 60 -ar 22050 -ac 1 data/student_voice_ref.wav
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/original_segment.wav` | 10-min NPTEL lecture segment |
| `data/student_voice_ref.wav` | Your 60s reference voice |
| `output/output_LRL_cloned.wav` | Final Rajasthani TTS (22.05kHz) |
| `output/transcript.json` | Whisper transcript with word timestamps |
| `output/ipa_transcript.json` | IPA-tagged transcript |
| `output/rajasthani_transcript.json` | Rajasthani translation |
| `output/speaker_embedding.npy` | x-vector (192-D) |
| `output/mcd_score.json` | MCD evaluation result |
| `output/antispoofing_eer.json` | EER result |
| `output/adversarial_robustness.json` | FGSM epsilon sweep results |
| `output/rajasthani_corpus.csv` | 500-word parallel corpus |

---

## Evaluation Metrics

| Metric | Target | Script |
|--------|--------|--------|
| WER (English) | < 15% | `task1_2_constrained_decoding.py` |
| WER (Hindi) | < 25% | `task1_2_constrained_decoding.py` |
| LID F1 | ≥ 0.85 | `task1_1_lid.py` |
| MCD | < 8.0 dB | `task3_3_synthesis.py` |
| EER | < 10% | `task4_1_antispoofing.py` |
| LID Switch Precision | ≤ 200ms | `task1_1_lid.py` |
| Min FGSM ε (SNR > 40dB) | Reported | `task4_2_fgsm.py` |

---

## Architecture Decisions (Implementation Notes)

### Task 1.1 — Multi-Head BiLSTM LID
- **Why BiLSTM?** Bidirectional context improves code-switching detection at boundaries. A CNN-only approach misses temporal dependencies across ~200ms windows.
- Two heads: binary (EN/HI) + 3-class (EN/HI/CS) share the encoder, reducing parameters while improving boundary detection via multitask learning.

### Task 1.2 — N-gram Logit Bias
- Logit bias formula: `logit_final(t) = logit_whisper(t) + λ · log P_ngram(t | history)`
- λ = 0.3 prevents the N-gram from dominating; acts as soft constraint not hard beam pruning.

### Task 2.1 — Custom Hinglish G2P
- Standard G2P tools (CMUdict, espeak) fail on romanized Hindi (e.g., "theek", "wala") because they're trained on standard English orthography. Our rule table covers 300+ Hinglish phonological patterns.

### Task 3.2 — DTW Prosody Transfer
- DTW aligns voiced segments only (F0 > 0) to avoid warping silence regions.
- Energy transfer is handled separately via gain normalization after pitch shifting.

### Task 4.1 — LFCC over MFCC for Spoofing
- LFCC uses a linear (not mel) filter bank, capturing high-frequency artifacts introduced by neural vocoders (e.g., VITS GAN artifacts above 8kHz). MFCC's mel compression loses these discriminative cues.

---

## References

1. Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE TASLP*.
2. Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*. (Whisper)
3. Kim, J. et al. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS. (VITS)
4. Desplanques, B. et al. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN. *Interspeech*.
5. Goodfellow, I. et al. (2014). Explaining and Harnessing Adversarial Examples. *ICLR*. (FGSM)
6. Todisco, M. et al. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Speech Detection. *Interspeech*.
7. Pratap, V. et al. (2023). Scaling Speech Technology to 1,000+ Languages. (Meta MMS)
8. Salvador, S. & Chan, P. (2007). Toward Accurate Dynamic Time Warping. *KDD*.
