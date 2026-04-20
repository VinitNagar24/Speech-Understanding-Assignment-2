"""
pipeline.py
-----------
Master runner for the full Speech Understanding Assignment 2 pipeline.

Steps:
  0. Download NPTEL lecture audio
  1. Denoise
  2. Frame-level LID (train or load)
  3. Constrained transcription (Whisper + N-gram logit bias)
  4. IPA conversion
  5. Rajasthani translation
  6. Speaker embedding extraction
  7. TTS synthesis (voice cloning)
  8. Prosody warping (DTW)
  9. Anti-spoofing training & evaluation
  10. Adversarial FGSM attack

Usage:
    python pipeline.py --config config.json
    python pipeline.py --step all         # run everything
    python pipeline.py --step transcribe  # run only transcription
"""

import os
import sys
import json
import argparse
import subprocess

# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "nptel_url":        "https://www.youtube.com/watch?v=X2iLjQ4Y7_M",
    "raw_audio":        "data/original_segment.wav",
    "denoised_audio":   "data/denoised_segment.wav",
    "student_voice":    "data/student_voice_ref.wav",
    "lid_labels":       "data/lid_labels.json",
    "lid_checkpoint":   "checkpoints/lid_model.pt",
    "spoof_labels":     "data/spoof_labels.json",
    "spoof_checkpoint": "checkpoints/antispoofing.pt",
    "transcript":       "output/transcript.json",
    "ipa_transcript":   "output/ipa_transcript.json",
    "raj_transcript":   "output/rajasthani_transcript.json",
    "speaker_embed":    "output/speaker_embedding.npy",
    "raw_tts":          "output/raw_tts.wav",
    "final_output":     "output/output_LRL_cloned.wav",
    "whisper_model":    "large-v3",
    "lid_epochs":       20,
    "spoof_epochs":     30,
}


# ── Step implementations ───────────────────────────────────────────────────────

def step_download(cfg):
    print("\n" + "="*60)
    print("STEP 0: Downloading NPTEL lecture audio")
    print("="*60)
    from download_audio import download_youtube_audio, trim_and_resample
    raw = download_youtube_audio(cfg["nptel_url"])
    trim_and_resample(raw, cfg["raw_audio"])


def step_denoise(cfg):
    print("\n" + "="*60)
    print("STEP 1: Denoising audio (Spectral Subtraction)")
    print("="*60)
    from part1_stt.task1_3_denoising import denoise_audio
    denoise_audio(cfg["raw_audio"], cfg["denoised_audio"])


def step_lid_train(cfg):
    print("\n" + "="*60)
    print("STEP 2a: Training LID model")
    print("="*60)
    if not os.path.exists(cfg["lid_labels"]):
        print(f"[Pipeline] LID labels not found: {cfg['lid_labels']}")
        print("[Pipeline] Generating synthetic labels from transcript ...")
        _generate_synthetic_lid_labels(cfg)
    from part1_stt.task1_1_lid import train_lid
    train_lid(cfg["lid_labels"], cfg["lid_checkpoint"], epochs=cfg["lid_epochs"])


def step_transcribe(cfg):
    print("\n" + "="*60)
    print("STEP 3: Constrained transcription (Whisper + N-gram logit bias)")
    print("="*60)
    from part1_stt.task1_2_constrained_decoding import ConstrainedWhisper
    cw = ConstrainedWhisper(model_name=cfg["whisper_model"])
    result = cw.transcribe(cfg["denoised_audio"])
    os.makedirs("output", exist_ok=True)
    with open(cfg["transcript"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] Transcript saved → {cfg['transcript']}")


def step_ipa(cfg):
    print("\n" + "="*60)
    print("STEP 4: IPA conversion (Hinglish → Unified IPA)")
    print("="*60)
    from part2_phonetic.task2_1_ipa_mapping import transcript_to_ipa_file
    transcript_to_ipa_file(cfg["transcript"], cfg["ipa_transcript"])


def step_translate(cfg):
    print("\n" + "="*60)
    print("STEP 5: Rajasthani translation")
    print("="*60)
    from part2_phonetic.task2_2_translation import translate_transcript, export_corpus_csv
    translate_transcript(cfg["ipa_transcript"], cfg["raj_transcript"])
    export_corpus_csv("output/rajasthani_corpus.csv")


def step_embedding(cfg):
    print("\n" + "="*60)
    print("STEP 6: Speaker embedding extraction")
    print("="*60)
    if not os.path.exists(cfg["student_voice"]):
        print(f"[Pipeline] ERROR: student_voice_ref.wav not found at {cfg['student_voice']}")
        print("[Pipeline] Please record 60 seconds of your voice and save to that path.")
        return
    from part3_tts.task3_1_voice_embedding import extract_speaker_embedding
    extract_speaker_embedding(cfg["student_voice"], cfg["speaker_embed"])


def step_synthesize(cfg):
    print("\n" + "="*60)
    print("STEP 7: TTS synthesis (zero-shot voice cloning)")
    print("="*60)
    from part3_tts.task3_3_synthesis import synthesize_lecture
    synthesize_lecture(
        cfg["raj_transcript"],
        cfg["student_voice"],
        cfg["raw_tts"],
        reference_wav=cfg["student_voice"],
    )


def step_prosody(cfg):
    print("\n" + "="*60)
    print("STEP 8: Prosody warping (DTW F0 + Energy)")
    print("="*60)
    from part3_tts.task3_2_prosody_warping import warp_prosody
    warp_prosody(cfg["raw_audio"], cfg["raw_tts"], cfg["final_output"])


def step_antispoofing(cfg):
    print("\n" + "="*60)
    print("STEP 9: Anti-spoofing training & evaluation")
    print("="*60)
    if not os.path.exists(cfg["spoof_labels"]):
        print("[Pipeline] Generating spoof labels ...")
        _generate_spoof_labels(cfg)
    from part4_adversarial.task4_1_antispoofing import train_antispoofing, evaluate_antispoofing
    train_antispoofing(cfg["spoof_labels"], cfg["spoof_checkpoint"], epochs=cfg["spoof_epochs"])
    evaluate_antispoofing(cfg["spoof_labels"], cfg["spoof_checkpoint"])


def step_fgsm(cfg):
    print("\n" + "="*60)
    print("STEP 10: Adversarial FGSM attack")
    print("="*60)
    from part4_adversarial.task4_2_fgsm import evaluate_adversarial_robustness
    evaluate_adversarial_robustness(
        cfg["denoised_audio"],
        cfg["lid_checkpoint"],
    )


# ── Label Generators (for when manual labels are unavailable) ─────────────────

def _generate_synthetic_lid_labels(cfg):
    """
    Generate weak LID labels using langdetect on Whisper word-level timestamps.
    Saves lid_labels.json.
    """
    if not os.path.exists(cfg["transcript"]):
        print("[Pipeline] No transcript found; skipping label generation.")
        return

    from part1_stt.task1_1_lid import generate_weak_labels, SR, HOP_LENGTH
    import torchaudio

    with open(cfg["transcript"], encoding="utf-8") as f:
        data = json.load(f)

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in data.get("segments", [])
    ]

    wav, sr = torchaudio.load(cfg["denoised_audio"])
    total_frames = int(wav.shape[-1] / sr * SR / HOP_LENGTH)
    labels = generate_weak_labels(segments, total_frames)

    items = [{"audio": cfg["denoised_audio"], "frame_labels": labels}]
    os.makedirs(os.path.dirname(cfg["lid_labels"]) or ".", exist_ok=True)
    with open(cfg["lid_labels"], "w") as f:
        json.dump(items, f)
    print(f"[Pipeline] LID labels saved → {cfg['lid_labels']}")


def _generate_spoof_labels(cfg):
    """
    Generate spoof dataset labels:
      - Bona fide: student_voice_ref.wav (label=0)
      - Spoof:     output_LRL_cloned.wav (label=1)
    """
    items = []
    if os.path.exists(cfg["student_voice"]):
        items.append({"audio": cfg["student_voice"], "label": 0})
    if os.path.exists(cfg["final_output"]):
        items.append({"audio": cfg["final_output"], "label": 1})
    elif os.path.exists(cfg["raw_tts"]):
        items.append({"audio": cfg["raw_tts"], "label": 1})

    os.makedirs(os.path.dirname(cfg["spoof_labels"]) or ".", exist_ok=True)
    with open(cfg["spoof_labels"], "w") as f:
        json.dump(items, f)
    print(f"[Pipeline] Spoof labels → {cfg['spoof_labels']} ({len(items)} items)")


# ── Main ──────────────────────────────────────────────────────────────────────

STEP_MAP = {
    "download":     step_download,
    "denoise":      step_denoise,
    "lid":          step_lid_train,
    "transcribe":   step_transcribe,
    "ipa":          step_ipa,
    "translate":    step_translate,
    "embedding":    step_embedding,
    "synthesize":   step_synthesize,
    "prosody":      step_prosody,
    "antispoofing": step_antispoofing,
    "fgsm":         step_fgsm,
}

ALL_STEPS = [
    "download", "denoise", "lid", "transcribe",
    "ipa", "translate", "embedding", "synthesize",
    "prosody", "antispoofing", "fgsm"
]


def main():
    parser = argparse.ArgumentParser(description="Speech Understanding PA-2 Pipeline")
    parser.add_argument("--config", default=None,  help="JSON config file path")
    parser.add_argument("--step",   default="all", help=f"Step to run: all | {' | '.join(STEP_MAP)}")
    args = parser.parse_args()

    # Load config
    cfg = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))

    # Create output dirs
    for d in ["data", "output", "checkpoints", "output/chunks"]:
        os.makedirs(d, exist_ok=True)

    # Run steps
    if args.step == "all":
        steps = ALL_STEPS
    elif args.step in STEP_MAP:
        steps = [args.step]
    else:
        print(f"Unknown step: {args.step}")
        print(f"Available: all, {', '.join(STEP_MAP)}")
        sys.exit(1)

    for step in steps:
        try:
            STEP_MAP[step](cfg)
        except Exception as e:
            print(f"\n[Pipeline] ERROR in step '{step}': {e}")
            import traceback
            traceback.print_exc()
            print("[Pipeline] Continuing to next step ...")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"  Original audio   : {cfg['raw_audio']}")
    print(f"  Transcript       : {cfg['transcript']}")
    print(f"  Rajasthani text  : {cfg['raj_transcript']}")
    print(f"  Final TTS output : {cfg['final_output']}")
    print(f"  Speaker embed    : {cfg['speaker_embed']}")


if __name__ == "__main__":
    main()
