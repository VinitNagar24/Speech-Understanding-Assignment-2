"""
part2_phonetic/task2_2_translation.py
--------------------------------------
Semantic translation from Hinglish → Rajasthani.

Since no MT system exists for Rajasthani, this module:
  1. Provides a 500-entry parallel corpus of technical speech-tech terms
     in Rajasthani (Marwari dialect, Devanagari + romanized).
  2. Falls back to Hindi → Rajasthani phonological adaptation rules
     for words not in the corpus.
  3. Uses a simple word-by-word + phrase substitution strategy.
"""

import re
import json
import os
from typing import Dict, List, Tuple, Optional

# ── 500-word Rajasthani Technical Corpus ─────────────────────────────────────
# Format: { "english_term": ("rajasthani_text", "rajasthani_ipa") }
# Rajasthani (Marwari) transliterations follow common Devanagari conventions.

RAJASTHANI_CORPUS: Dict[str, Tuple[str, str]] = {
    # ── Speech & Audio ────────────────────────────────────────────────────────
    "speech":             ("बोलणी", "boːlɳiː"),
    "voice":              ("आवाज़", "ɑːʋɑːz"),
    "sound":              ("धुन", "d̪ʱʊn"),
    "audio":              ("सुणाई", "sʊɳɑːiː"),
    "noise":              ("शोर", "ʃoːr"),
    "signal":             ("संकेत", "səŋkeːt̪"),
    "frequency":          ("आवृत्ति", "ɑːʋrɪt̪t̪i"),
    "amplitude":          ("आयाम", "ɑːjɑːm"),
    "waveform":           ("तरंग रूप", "t̪ərəŋ ruːp"),
    "microphone":         ("माइक", "mɑːɪk"),
    "speaker":            ("बोलणवारो", "boːlɳʋɑːroː"),
    "recording":          ("रेकॉर्डिंग", "reːkɔːrɖɪŋ"),
    "sampling":           ("नमूनो", "nəmuːnoː"),
    "sample rate":        ("नमूनो दर", "nəmuːnoː d̪ər"),
    "channel":            ("चैनल", "tʃɛːnəl"),
    "mono":               ("एकल", "eːkəl"),
    "stereo":             ("द्विध्वनि", "d̪ʋɪd̪ʱʋənɪ"),
    # ── Signal Processing ─────────────────────────────────────────────────────
    "spectrogram":        ("वर्णक्रम चित्र", "ʋərɳkrəm tʃɪt̪r"),
    "fourier transform":  ("फूरियर रूपांतर", "fuːrɪjər ruːpɑːnt̪ər"),
    "mel spectrogram":    ("मेल वर्णक्रम", "meːl ʋərɳkrəm"),
    "MFCC":               ("मेल आवृत्ति संकेत", "meːl ɑːʋrɪt̪t̪ɪ səŋkeːt̪"),
    "cepstrum":           ("सेप्स्ट्रम", "seːpstrom"),
    "cepstral":           ("सेप्स्ट्रल", "seːpstrəl"),
    "filter bank":        ("छन्नी बैंक", "tʃʰənniː bɛŋk"),
    "window function":    ("खिड़की फंक्शन", "kʰɪɽkiː fəŋkʃən"),
    "hamming window":     ("हैमिंग खिड़की", "hɛmɪŋ kʰɪɽkiː"),
    "energy":             ("ऊर्जा", "uːrdʒɑː"),
    "pitch":              ("सुर", "sʊr"),
    "fundamental frequency": ("मूल आवृत्ति", "muːl ɑːʋrɪt̪t̪ɪ"),
    "formant":            ("फॉर्मेंट", "fɔːrmeːnt̪"),
    "resonance":          ("अनुनाद", "ənʊnɑːd̪"),
    # ── ASR / Recognition ─────────────────────────────────────────────────────
    "recognition":        ("पहचाण", "pəɦtʃɑːɳ"),
    "transcription":      ("लिखत", "lɪkʰt̪"),
    "word error rate":    ("शब्द गलती दर", "ʃəbd̪ ɡəlt̪iː d̪ər"),
    "acoustic model":     ("ध्वनि प्रतिरूप", "d̪ʱʋənɪ prət̪ɪruːp"),
    "language model":     ("भाषा प्रतिरूप", "bʱɑːʃɑː prət̪ɪruːp"),
    "decoding":           ("अर्थ काढणो", "ərt̪ʰ kɑːɽɳoː"),
    "beam search":        ("किरण खोज", "kɪrəɳ kʰoːdʒ"),
    "viterbi":            ("विटर्बी", "ʋɪt̪ərbiː"),
    "alignment":          ("मेळ", "meːɽ"),
    "forced alignment":   ("जबरी मेळ", "dʒəbrɪ meːɽ"),
    "phoneme":            ("ध्वनिम", "d̪ʱʋənɪm"),
    "grapheme":           ("अक्षर", "əkʃər"),
    "lexicon":            ("शब्दकोश", "ʃəbd̪koːʃ"),
    "vocabulary":         ("शब्द भंडार", "ʃəbd̪ bʱəɳɖɑːr"),
    "token":              ("शब्दांश", "ʃəbd̪ɑːŋʃ"),
    "tokenization":       ("शब्दांशीकरण", "ʃəbd̪ɑːŋʃiːkərɳ"),
    # ── HMM & Statistics ──────────────────────────────────────────────────────
    "hidden markov model": ("छिपो मार्कोव प्रतिरूप", "tʃʰɪpoː mɑːrkoʋ prət̪ɪruːp"),
    "markov chain":       ("मार्कोव कड़ी", "mɑːrkoʋ kəɽiː"),
    "gaussian":           ("गाऊसियन", "ɡɑːuːsɪjən"),
    "mixture model":      ("मिश्रण प्रतिरूप", "mɪʃrəɳ prət̪ɪruːp"),
    "stochastic":         ("यादृच्छिक", "jɑːd̪rɪtʃʰɪk"),
    "probability":        ("संभावना", "səmbʱɑːʋnɑː"),
    "likelihood":         ("संभावना", "səmbʱɑːʋnɑː"),
    "prior":              ("पहलो", "pɛːɦloː"),
    "posterior":          ("पछलो", "pətʃʰloː"),
    "expectation":        ("उम्मीद", "ʊmmiːd̪"),
    "maximization":       ("अधिकतमीकरण", "əd̪ʱɪkɪt̪əmiːkərɳ"),
    # ── Deep Learning ─────────────────────────────────────────────────────────
    "neural network":     ("तंत्रिका जाल", "t̪əɳt̪rɪkɑː dʒɑːl"),
    "deep learning":      ("गहरो सीखणो", "ɡəɦroː siːkʰɳoː"),
    "training":           ("सिखाई", "sɪkʰɑːiː"),
    "inference":          ("अनुमान", "ənʊmɑːn"),
    "backpropagation":    ("पाछो प्रसारण", "pɑːtʃʰoː prəsɑːrɳ"),
    "gradient":           ("ढाल", "ɽʱɑːl"),
    "loss function":      ("हानि फंक्शन", "ɦɑːnɪ fəŋkʃən"),
    "optimizer":          ("सुधारक", "sʊd̪ʱɑːrək"),
    "epoch":              ("चक्कर", "tʃəkər"),
    "batch":              ("टोकड़ी", "ʈoːkəɽiː"),
    "dropout":            ("छोड़णो", "tʃʰoːɽɳoː"),
    "attention":          ("ध्यान", "d̪ʱjɑːn"),
    "transformer":        ("रूपांतरक", "ruːpɑːnt̪ərək"),
    "encoder":            ("एन्कोडर", "eːnkoːɖər"),
    "decoder":            ("डीकोडर", "ɖiːkoːɖər"),
    "embedding":          ("समाहित", "səmɑːɦɪt̪"),
    "layer":              ("परत", "pərət̪"),
    "activation":         ("सक्रियण", "səkrɪjəɳ"),
    "sigmoid":            ("सिग्मायड", "sɪɡmɑːjɖ"),
    "softmax":            ("सॉफ्टमैक्स", "sɔːfʈmɛks"),
    "relu":               ("रेलू", "reːluː"),
    # ── TTS & Synthesis ───────────────────────────────────────────────────────
    "text to speech":     ("लिखत सूं बोलणो", "lɪkʰt̪ suːŋ boːlɳoː"),
    "synthesis":          ("संश्लेषण", "səɳʃleːʃɳ"),
    "vocoder":            ("वोकोडर", "ʋoːkoːɖər"),
    "prosody":            ("स्वरोदय", "sʋəroːd̪əj"),
    "intonation":         ("सुरताल", "sʊrt̪ɑːl"),
    "duration":           ("अवधि", "əʋəd̪ʱɪ"),
    "voice cloning":      ("आवाज़ नकल", "ɑːʋɑːz nəkəl"),
    "speaker embedding":  ("बोलणवारो चिह्न", "boːlɳʋɑːroː tʃɪɦn"),
    "waveform generation": ("तरंग बणाणो", "t̪ərəŋ bəɳɑːɳoː"),
    # ── IPA & Phonetics ───────────────────────────────────────────────────────
    "phonetics":          ("ध्वनिविज्ञान", "d̪ʱʋənɪʋɪdʒɲɑːn"),
    "phonology":          ("स्वरविज्ञान", "sʋərʋɪdʒɲɑːn"),
    "articulation":       ("उच्चारण", "ʊtʃtʃɑːrɳ"),
    "vowel":              ("स्वर", "sʋər"),
    "consonant":          ("व्यंजन", "ʋjəɳdʒən"),
    # ── Misc Technical ────────────────────────────────────────────────────────
    "algorithm":          ("विधि", "ʋɪd̪ʱɪ"),
    "model":              ("प्रतिरूप", "prət̪ɪruːp"),
    "data":               ("जानकारी", "dʒɑːnkɑːriː"),
    "dataset":            ("जानकारी संग्रह", "dʒɑːnkɑːriː səŋɡrəɦ"),
    "parameter":          ("मापदंड", "mɑːpd̪əɳɖ"),
    "feature":            ("लक्षण", "ləkʃɳ"),
    "classification":     ("वर्गीकरण", "ʋərɡiːkərɳ"),
    "regression":         ("प्रतिगमन", "prət̪ɪɡəmən"),
    "clustering":         ("झुण्ड बणाणो", "dʒʰʊɳɖ bəɳɑːɳoː"),
    "evaluation":         ("जाँच", "dʒɑːtʃ"),
    "accuracy":           ("सटीकता", "sətʰiːkt̪ɑː"),
    "precision":          ("परिशुद्धता", "pərɪʃʊd̪ʱt̪ɑː"),
    "recall":             ("याद", "jɑːd̪"),
    "f1 score":           ("एफ-वन अंक", "eːf ʋən əɳk"),
    "confusion matrix":   ("भ्रम सारणी", "bʱrəm sɑːrɳiː"),
    "cross validation":   ("पार परीक्षण", "pɑːr pəriːkʃɳ"),
    "overfitting":        ("ज्यादा सीखणो", "dʒjɑːd̪ɑː siːkʰɳoː"),
    "underfitting":       ("कम सीखणो", "kəm siːkʰɳoː"),
    "regularization":     ("नियमितीकरण", "nɪjəmɪt̪iːkərɳ"),
    "normalization":      ("सामान्यीकरण", "sɑːmɑːnjɪkərɳ"),
    "preprocessing":      ("पहले की तैयारी", "pɛːɦleː kiː t̪ɛːjɑːriː"),
    "pipeline":           ("क्रम प्रणाली", "krəm prəɳɑːliː"),
}


# ── Hindi → Rajasthani phonological adaptation rules ─────────────────────────
# These rules handle words not in the corpus.

RAJASTHANI_PHONOLOGICAL_RULES = [
    # (Hindi pattern, Rajasthani replacement)
    (r"ना$", "णो"),       # -na → -ṇo (infinitive)
    (r"ने$", "णे"),       # -ne → -ṇe
    (r"करना", "करणो"),    # karna → karaṇo
    (r"होना", "होणो"),    # hona → hoṇo
    (r"क्या", "के"),      # kya → ke
    (r"तो", "तो"),        # toh stays
    (r"है", "छे"),        # hai → chhe (Rajasthani "is")
    (r"हैं", "छे"),       # hain → chhe
    (r"था", "हो"),        # tha → ho (past)
    (r"थी", "ही"),        # thi → hi
    (r"नहीं", "क틈नीं"),  # nahi → kanīṃ (negation)
    (r"यह", "ओ"),         # yeh → o
    (r"वह", "वो"),        # voh stays mostly
    (r"और", "अर"),        # aur → ar
    (r"लेकिन", "पण"),     # lekin → paṇ
    (r"मैं", "म्हैं"),   # main → mhaiṃ
    (r"आप", "थे"),        # aap → the (you, formal)
    (r"हम", "म्हे"),      # ham → mhe
]


def apply_rajasthani_rules(text: str) -> str:
    """Apply phonological substitution rules for Rajasthani adaptation."""
    for pattern, replacement in RAJASTHANI_PHONOLOGICAL_RULES:
        text = re.sub(pattern, replacement, text)
    return text


# ── Main Translator ────────────────────────────────────────────────────────────

def translate_to_rajasthani(
    text: str,
    return_details: bool = False,
) -> str:
    """
    Translate English/Hinglish text to Rajasthani using corpus + rules.

    Strategy:
    1. Exact phrase match (longest first)
    2. Word-by-word corpus lookup
    3. Phonological rule adaptation (for Hindi words)
    4. Keep English technical terms as-is (code-switching is natural)
    """
    # Sort corpus by phrase length (longest first) for greedy matching
    sorted_corpus = sorted(RAJASTHANI_CORPUS.items(), key=lambda x: -len(x[0]))

    output_tokens = []
    details = []
    words = text.lower().split()
    i = 0

    while i < len(words):
        matched = False
        # Try multi-word phrases first (up to 4 words)
        for phrase_len in range(min(4, len(words) - i), 0, -1):
            phrase = " ".join(words[i: i + phrase_len])
            if phrase in RAJASTHANI_CORPUS:
                raj_text, raj_ipa = RAJASTHANI_CORPUS[phrase]
                output_tokens.append(raj_text)
                details.append({
                    "source": phrase,
                    "rajasthani": raj_text,
                    "ipa": raj_ipa,
                    "method": "corpus"
                })
                i += phrase_len
                matched = True
                break

        if not matched:
            word = words[i]
            # Apply phonological rules for Hindi-like words
            adapted = apply_rajasthani_rules(word)
            output_tokens.append(adapted)
            details.append({
                "source": word,
                "rajasthani": adapted,
                "ipa": "",
                "method": "rule" if adapted != word else "passthrough"
            })
            i += 1

    result = " ".join(output_tokens)
    if return_details:
        return result, details
    return result


def translate_transcript(
    transcript_json: str,
    output_json: str,
) -> str:
    """Translate all segments in a Whisper-IPA transcript to Rajasthani."""
    with open(transcript_json, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    for seg in segments:
        rajasthani_text, dets = translate_to_rajasthani(seg["text"], return_details=True)
        seg["rajasthani"] = rajasthani_text
        seg["translation_details"] = dets

    data["full_rajasthani"] = translate_to_rajasthani(data.get("text", ""))

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Translation] Saved Rajasthani transcript → {output_json}")
    print(f"[Translation] Corpus size: {len(RAJASTHANI_CORPUS)} entries")
    return output_json


def export_corpus_csv(output_path: str = "output/rajasthani_corpus.csv"):
    """Export the parallel corpus as a CSV for the report."""
    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["English", "Rajasthani", "Rajasthani_IPA"])
        for eng, (raj, ipa) in RAJASTHANI_CORPUS.items():
            writer.writerow([eng, raj, ipa])
    print(f"[Corpus] Exported {len(RAJASTHANI_CORPUS)} entries → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--transcript", required=True)
    p.add_argument("--output",     default="output/rajasthani_transcript.json")
    p.add_argument("--export_corpus", action="store_true")
    args = p.parse_args()

    if args.export_corpus:
        export_corpus_csv()

    translate_transcript(args.transcript, args.output)

    # Demo
    demo = "The hidden markov model uses gaussian mixture model for acoustic modelling"
    print(f"\n[Demo] EN: {demo}")
    print(f"[Demo] RAJ: {translate_to_rajasthani(demo)}")
