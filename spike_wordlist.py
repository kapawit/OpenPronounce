"""Throwaway spike: phoneme-aware wordlist expansion.

For each word in the sample:
  1. Phonemize it to get the actual phone sequence
  2. Check which error patterns actually apply to THIS word's phones
  3. Generate misspelled variant only when the phone difference is real

This avoids the spelling-based false positives from the first spike.
"""

import sys
sys.path.insert(0, "/mnt/data/Project/OpenPronounce")

from phonemizer import phonemize
from phonemizer.separator import Separator
import re

# Test sample
SAMPLE_WORDS = [
    # Voicing candidates (p/t/k initial)
    "pagi", "tahu", "kota", "sapu", "paku", "tari", "kaca", "sate", "tiga", "pita",
    "jalan", "pasar", "tangan", "tengah", "pinggir", "tangga",
    "kelurahan", "kecil", "terima", "peti", "pelajar", "besar", "belajar", "desa", "depan",
    "pulau", "pantai", "pisau", "sungai", "kerbau", "pakai", "laut", "daun", "harimau", "santai",
    "enak", "merah", "dekat", "gelas", "petani", "sebelah",
    "paling", "asing", "bintang", "gantung",
    "kepala", "pekerja", "bekerja", "sekolah", "mekarna", "tekun",
    "kecap", "ketiga", "pentol", "bendera", "tengki", "singa",
    # Schwa candidates (prefixes ke-, pe-, be-, se-)
    "keluar", "kepal", "peti", "pelajar", "besar", "belajar", "desa", "depan",
    "seorang", "pekerja", "belakang", "kemarin", "pepat", "semangat",
    "ketawa", "betawi", "sekolah", "berita", "pedoman", "beasiswa",
    # Diphthong words
    "raja", "baja", "gula", "bui", "selai", "bilik",
    "bau", "sai", "koi", "selui", "bikini",
    # NG words
    "nganga", "nginang", "bangun", "senang", "tangan", "bunga",
    "asing", "paling", "bintang", "gantung", "enak", "benang",
    # Final consonant words
    "makan", "rumah", "hitam", "besar", "kecil", "market",
]


def phonemize_word(word):
    """Return (phones list, raw string) for a word."""
    try:
        out = phonemize(
            word,
            language="id",
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=Separator(phone=" ", word="", syllable=""),
        )
        phones = out.strip().split()
        return phones, out.strip()
    except Exception:
        return [], ""


def phoneme_aware_errors(word, correct_phones):
    """Return dict of error_type -> list of (misspelled_word, correct_phones, error_phones)."""
    results = {}
    lower = word.lower()
    phones_str = " ".join(correct_phones)

    # ---- VOICING: initial p/t/k → b/d/g ----
    if correct_phones and correct_phones[0] in ("p", "t", "k"):
        voicing_map = {"p": "b", "t": "d", "k": "g"}
        orig = correct_phones[0]
        misspelled = lower.replace(orig, voicing_map[orig], 1)
        # Only if the misspelled word is actually a different string
        if misspelled != lower:
            err_phones, _ = phonemize_word(misspelled)
            if err_phones and err_phones != correct_phones:
                results.setdefault("voicing", []).append((misspelled, correct_phones, err_phones))

    # ---- FINAL_DROPPED: final consonant removed ----
    if correct_phones and not correct_phones[-1] in ("a", "e", "ɛ", "i", "o", "u", "ə", "ʊ"):
        misspelled = lower[:-1]
        err_phones, _ = phonemize_word(misspelled)
        if err_phones and err_phones != correct_phones:
            results.setdefault("final_dropped", []).append((misspelled, correct_phones, err_phones))

    # ---- NG_TO_N: ng → n anywhere in word ----
    if "ŋ" in correct_phones:
        misspelled = re.sub(r"n(?=[aeiouɛəoɔuʊ]|$)", "ŋ", lower.replace("ŋ", "n"), count=1)
        # simpler: replace first occurrence of "n" that is part of "ng" context
        ng_idx = lower.find("ng")
        if ng_idx >= 0:
            misspelled = lower[:ng_idx] + "n" + lower[ng_idx+2:]
            if misspelled != lower:
                err_phones, _ = phonemize_word(misspelled)
                if err_phones and err_phones != correct_phones:
                    results.setdefault("ng_to_n", []).append((misspelled, correct_phones, err_phones))

    # ---- SCHWA_TO_A: /ə/ in phones → learner says /a/ instead ----
    # The error: learner says /a/ instead of /ə/. Misspelling uses 'a' spelling.
    # We generate misspelling by replacing 'e'→'a' in prefix spelling (ke→ka, pe→pa etc.)
    if "ə" in correct_phones:
        for prefix_spell, prefix_rep in [("ke", "ka"), ("pe", "pa"), ("be", "ba"), ("se", "sa")]:
            if lower.startswith(prefix_spell):
                misspelled = prefix_rep + lower[len(prefix_spell):]
                if misspelled != lower:
                    err_phones, _ = phonemize_word(misspelled)
                    if err_phones and err_phones != correct_phones:
                        results.setdefault("schwa_to_a", []).append((misspelled, correct_phones, err_phones))
                break

    # ---- DIPHTHONG_REDUCED: /aʊ/, /aɪ/, /oɪ/ → single vowel ----
    diphthong_info = [
        ("aʊ", "au", ["a", "u"]),
        ("aɪ", "ai", ["a", "i"]),
        ("oɪ", "oi", ["o", "i"]),
    ]
    for diph, orth, replacements in diphthong_info:
        if diph in phones_str and orth in lower:
            for rep in replacements:
                misspelled = lower.replace(orth, rep, 1)
                if misspelled != lower:
                    err_phones, _ = phonemize_word(misspelled)
                    if err_phones and err_phones != correct_phones:
                        results.setdefault("diphthong_reduced", []).append((misspelled, correct_phones, err_phones))

    # ---- E_TO_I: /ɛ/ spelled 'e' → 'i' spelling ----
    # Only for words where 'e' spelling maps to /ɛ/ (open-mid e, not close-mid e)
    # In Indonesian, 'e' in closed syllables = /ɛ/
    if "ɛ" in correct_phones:
        # Find position of ɛ in phones, map back to 'e' in spelling
        # Heuristic: replace first 'e' that results in different phones
        for match in re.finditer(r"e", lower):
            misspelled = lower[:match.start()] + "i" + lower[match.start()+1:]
            if misspelled != lower:
                err_phones, _ = phonemize_word(misspelled)
                if err_phones and err_phones != correct_phones:
                    results.setdefault("e_to_i", []).append((misspelled, correct_phones, err_phones))

    return results


def main():
    total_pairs = 0
    all_results = {}

    for word in SAMPLE_WORDS:
        correct_phones, phones_str = phonemize_word(word)
        if not correct_phones:
            continue
        errors = phoneme_aware_errors(word, correct_phones)
        for error_type, pairs in errors.items():
            for misspelled, cp, ep in pairs:
                key = (word, misspelled)
                if key not in all_results:
                    all_results[key] = (error_type, cp, ep)
                    print(f"  + ({word!r}, {misspelled!r}) [{error_type}]")
                    print(f"      correct: {cp}")
                    print(f"      error:   {ep}")
                total_pairs += 1

    # Group by error type and count
    by_type = {}
    for (word, misspelled), (etype, cp, ep) in all_results.items():
        by_type.setdefault(etype, []).append((word, misspelled, cp, ep))

    print(f"\n\nSUMMARY BY TYPE:")
    for etype, pairs in by_type.items():
        print(f"  {etype}: {len(pairs)} unique pairs")
    print(f"\n  Total unique pairs: {len(all_results)}")


if __name__ == "__main__":
    main()
