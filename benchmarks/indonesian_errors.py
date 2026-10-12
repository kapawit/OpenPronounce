"""Indonesian learner-error detection benchmark: measure localization and false alarms.

    python benchmarks/indonesian_errors.py --run            # recognize + score + cache rows
    python benchmarks/indonesian_errors.py --tune            # threshold grid on the tuning half
    python benchmarks/indonesian_errors.py --report          # per-type table, tune vs held-out
    python benchmarks/indonesian_errors.py --assets          # tiny fast subset, live
    python benchmarks/indonesian_errors.py --real DIR        # score real wav+txt pairs
"""

import argparse
import hashlib
import itertools
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Levenshtein  # noqa: E402

from openpronounce import phones  # noqa: E402

ID_LANG = "id"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROW_CACHE = os.path.join(RESULTS_DIR, "indonesian-errors-rows.json")
NOT_TARGETED = ("voicing", "ng_to_n", "schwa_to_a", "diphthong_reduced")

logger = logging.getLogger("indonesian_errors")


def expected_phones(word, lang=ID_LANG):
    """espeak phones of ``word`` as a flat list."""
    return [p for group in phones.get_expected_phones(word, lang)[1] for p in group]


def changed_expected_phones(correct, wrong, lang=ID_LANG):
    """Phones of ``correct`` that the ``wrong`` spelling replaces or deletes (nearest phone for an insertion)."""
    reference, hypothesis = expected_phones(correct, lang), expected_phones(wrong, lang)
    changed = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(reference, hypothesis):
        if tag in ("replace", "delete"):
            changed.extend(reference[i1:i2])
        elif tag == "insert" and reference:
            changed.append(reference[i1] if i1 < len(reference) else reference[-1])
    return changed


def localized(report, changed):
    """True when a phone flagged in ``report`` is one the misspelling changed."""
    return any(p["confidence"] > 0 and p["expected"] in changed for p in report["phones"])


def is_flagged(report, threshold, min_edits):
    """A word is flagged when its weighted edits reach ``threshold`` of its phones or ``min_edits``."""
    edits, length = report["weighted_edits"], len(report["expected"])
    return report["distance"] > 0 and (edits / length >= threshold or edits >= min_edits)


def is_tune_word(word):
    """Stable hash parity, so the tuning/held-out split is identical in every process."""
    return int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16) % 2 == 0


GRID = list(itertools.product([t / 20 for t in range(4, 25)], [1, 1.5, 2, 2.5, 3, 4, 99]))


def metrics(rows, flag):
    """Localization recall (primary), word recall, precision and false-alarm rate over ``rows``.

    Positives (``bad``) are induced-error cells: a hit needs the flag AND ``localized``.
    Negatives are correct audio: any flag is a false alarm. ``f1`` is the harmonic mean of
    ``precision`` and ``word_recall`` (word level); localization is reported separately.
    """
    positive = [r for r in rows if r["bad"]]
    negative = [r for r in rows if not r["bad"]]
    hit = sum(1 for r in positive if flag(r) and r["localized"])
    mislocalized = sum(1 for r in positive if flag(r) and not r["localized"])
    flagged_pos = hit + mislocalized
    fp = sum(1 for r in negative if flag(r))
    n_pos, n_neg = len(positive), len(negative)
    precision = flagged_pos / (flagged_pos + fp) if flagged_pos + fp else 0.0
    word_recall = flagged_pos / n_pos if n_pos else 0.0
    f1 = 2 * precision * word_recall / (precision + word_recall) if precision + word_recall else 0.0
    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "localized": hit, "mislocalized": mislocalized,
        "flagged_pos": flagged_pos, "unflagged_pos": n_pos - flagged_pos, "fp": fp,
        "localization_recall": hit / n_pos if n_pos else 0.0,
        "word_recall": word_recall,
        "precision": precision,
        "far": fp / n_neg if n_neg else 0.0,
        "f1": f1,
    }
