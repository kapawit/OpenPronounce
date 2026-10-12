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
