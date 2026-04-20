"""
part2_phonetic/task2_1_ipa_mapping.py
--------------------------------------
Converts a code-switched (Hinglish) transcript into a unified IPA string.

Strategy
--------
1. Split transcript into tokens; detect each token's language (EN / HI).
2. Apply English G2P (via epitran or gruut) for English tokens.
3. Apply custom Hindi G2P mapping table for Hindi/transliterated tokens.
4. Output unified IPA string with language tags for downstream use.

Note: Standard G2P tools fail on Hinglish because they don't handle
      transliterated Hindi (Devanagari romanized) or mixed-script inputs.
      This module adds a manual Hinglish → IPA mapping layer.
"""

import re
import json
from typing import List, Tuple, Dict, Optional

# ── Hindi/Hinglish → IPA mapping table ────────────────────────────────────────
# Covers common Devanagari romanizations (ITRANS-style) + common Hinglish words.
# Extend this dict as needed for your lecture domain.

HINDI_IPA_MAP: Dict[str, str] = {
    # Vowels
    "a": "ə", "aa": "aː", "i": "ɪ", "ii": "iː", "u": "ʊ", "uu": "uː",
    "e": "eː", "ai": "æ", "o": "oː", "au": "ɔː",
    # Consonants (ITRANS-style)
    "k": "k", "kh": "kʰ", "g": "ɡ", "gh": "ɡʱ", "ng": "ŋ",
    "ch": "tʃ", "chh": "tʃʰ", "j": "dʒ", "jh": "dʒʱ",
    "t": "t̪", "th": "t̪ʰ", "d": "d̪", "dh": "d̪ʱ", "n": "n",
    "T": "ʈ", "Th": "ʈʰ", "D": "ɖ", "Dh": "ɖʱ", "N": "ɳ",
    "p": "p", "ph": "pʰ", "b": "b", "bh": "bʱ", "m": "m",
    "y": "j", "r": "r", "l": "l", "v": "ʋ", "w": "ʋ",
    "sh": "ʃ", "Sh": "ʂ", "s": "s", "h": "ɦ",
    "f": "f", "z": "z", "x": "x",
    # Common Hinglish words (direct IPA)
    "hai": "ɦæː", "hain": "ɦæːn", "kya": "kjɑː", "aur": "ɔːr",
    "matlab": "mətləb", "lekin": "ləkɪn", "toh": "toː", "bhi": "bʰiː",
    "wala": "ʋɑːlɑː", "mera": "meːrɑː", "tera": "teːrɑː", "yeh": "jeː",
    "vo": "ʋoː", "unka": "ʊnkɑː", "isliye": "ɪslɪjeː",
    "samajh": "səmədʒ", "pehle": "pɛːɦleː", "phir": "pʰɪr",
    "theek": "ʈʰiːk", "sahi": "sɑːɦɪ", "nahi": "nəɦɪː",
    "matlab": "mətləb", "basically": "beɪsɪkli",  # mixed
    "actually": "æktʃuəli", "okay": "oːkeː",
    # Speech tech terms in Hindi context
    "shabda": "ʃəbd̪ɑː", "dhwani": "d̪ʱʋɑːniː", "bhaasha": "bʱɑːʃɑː",
    "awaaz": "əʋɑːz", "paehchaan": "pɛːtʃɦɑːn",
}

# Common English → IPA (fallback table for offline use)
ENGLISH_IPA_FALLBACK: Dict[str, str] = {
    "speech": "spiːtʃ", "recognition": "ˌrɛkəɡˈnɪʃən",
    "hidden": "ˈhɪdən", "markov": "ˈmɑːrkɒv", "model": "ˈmɒdəl",
    "acoustic": "əˈkuːstɪk", "language": "ˈlæŋɡwɪdʒ",
    "neural": "ˈnjʊərəl", "network": "ˈnɛtwɜːrk",
    "cepstrum": "ˈsɛpstrəm", "stochastic": "stəˈkæstɪk",
    "phoneme": "ˈfoʊniːm", "spectrogram": "ˈspɛktrəɡræm",
    "frequency": "ˈfriːkwənsi", "fundamental": "ˌfʌndəˈmɛntəl",
    "gaussian": "ˈɡaʊsiən", "mixture": "ˈmɪkstʃər",
    "connectionist": "kəˈnɛkʃənɪst", "temporal": "ˈtɛmpərəl",
    "classification": "ˌklæsɪfɪˈkeɪʃən", "attention": "əˈtɛnʃən",
    "transformer": "trænsˈfɔːrmər", "encoder": "ɛnˈkoʊdər",
    "decoder": "diːˈkoʊdər", "algorithm": "ˈælɡərɪðəm",
    "prosody": "ˈprɒsədi", "intonation": "ˌɪntəˈneɪʃən",
    "synthesis": "ˈsɪnθəsɪs", "vocoder": "ˈvoʊkoʊdər",
}


# ── Language detection at token level ─────────────────────────────────────────

def detect_token_lang(token: str) -> str:
    """
    Classify a single token as 'en', 'hi', or 'cs' (code-switch/ambiguous).
    Uses Unicode range check (Devanagari) + known Hinglish word list.
    """
    # Devanagari range: U+0900 – U+097F
    if any('\u0900' <= c <= '\u097F' for c in token):
        return "hi"
    token_lower = token.lower().strip(".,!?")
    if token_lower in HINDI_IPA_MAP:
        return "hi"
    # Simple heuristic: common Hindi suffixes in romanized form
    hindi_suffixes = ("wala", "wali", "waale", "iya", "oon", "een",
                      "hai", "hain", "tha", "thi", "kar", "ke", "ki", "ka")
    if any(token_lower.endswith(suf) for suf in hindi_suffixes):
        return "hi"
    return "en"


# ── English G2P ───────────────────────────────────────────────────────────────

def english_to_ipa(token: str) -> str:
    """Convert English token to IPA. Uses epitran if available, else fallback table."""
    token_lower = token.lower().strip(".,!?\"'")
    # Try fallback table first (fast, no dependency)
    if token_lower in ENGLISH_IPA_FALLBACK:
        return ENGLISH_IPA_FALLBACK[token_lower]
    # Try epitran
    try:
        import epitran
        epi = epitran.Epitran("eng-Latn")
        return epi.transliterate(token_lower)
    except ImportError:
        pass
    # Try gruut
    try:
        from gruut import sentences
        for sent in sentences(token_lower, lang="en-us"):
            for word in sent:
                if word.phonemes:
                    return " ".join(word.phonemes)
    except Exception:
        pass
    # Last resort: return token as-is in angle brackets
    return f"⟨{token_lower}⟩"


# ── Hindi/Hinglish G2P ────────────────────────────────────────────────────────

def hindi_to_ipa(token: str) -> str:
    """Convert Hindi/Hinglish token to IPA using the manual map + syllable rules."""
    token_lower = token.lower().strip(".,!?\"'")

    # Direct lookup
    if token_lower in HINDI_IPA_MAP:
        return HINDI_IPA_MAP[token_lower]

    # Devanagari: use epitran if available
    if any('\u0900' <= c <= '\u097F' for c in token):
        try:
            import epitran
            epi = epitran.Epitran("hin-Deva")
            return epi.transliterate(token)
        except Exception:
            pass

    # Romanized Hindi: greedy phoneme parsing
    ipa_out = []
    i = 0
    s = token_lower
    while i < len(s):
        matched = False
        # Try longest match (up to 3 chars)
        for length in [3, 2, 1]:
            chunk = s[i: i + length]
            if chunk in HINDI_IPA_MAP:
                ipa_out.append(HINDI_IPA_MAP[chunk])
                i += length
                matched = True
                break
        if not matched:
            ipa_out.append(s[i])
            i += 1

    return "".join(ipa_out)


# ── Main Converter ────────────────────────────────────────────────────────────

def hinglish_to_ipa(
    text: str,
    return_tagged: bool = False,
) -> str:
    """
    Convert a full Hinglish transcript to a unified IPA string.

    Parameters
    ----------
    text         : raw transcript (may contain Hindi and English words)
    return_tagged: if True, returns list of (token, lang, ipa) tuples

    Returns
    -------
    Unified IPA string (space-separated phones) or tagged list.
    """
    tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z']+|[0-9]+|[^\w\s]", text)
    result = []

    for token in tokens:
        lang = detect_token_lang(token)
        if lang == "hi":
            ipa = hindi_to_ipa(token)
        else:
            ipa = english_to_ipa(token)
        result.append((token, lang, ipa))

    if return_tagged:
        return result

    return " ".join(ipa for _, _, ipa in result)


def transcript_to_ipa_file(
    transcript_json: str,
    output_json: str,
) -> str:
    """
    Load a Whisper transcript JSON and add IPA fields to each segment.
    """
    with open(transcript_json, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    for seg in segments:
        tagged = hinglish_to_ipa(seg["text"], return_tagged=True)
        seg["ipa"] = " ".join(ipa for _, _, ipa in tagged)
        seg["token_langs"] = [{"token": t, "lang": l, "ipa": i} for t, l, i in tagged]

    data["full_ipa"] = hinglish_to_ipa(data.get("text", ""))

    import os
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[IPA] Saved IPA-tagged transcript → {output_json}")
    return output_json


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--transcript", required=True, help="Whisper transcript JSON")
    p.add_argument("--output",     default="output/ipa_transcript.json")
    args = p.parse_args()
    transcript_to_ipa_file(args.transcript, args.output)

    # Quick demo
    sample = "Toh basically, the Hidden Markov Model ka use karein ge speech recognition mein"
    print("\n[IPA Demo]")
    print(f"Input : {sample}")
    print(f"IPA   : {hinglish_to_ipa(sample)}")
