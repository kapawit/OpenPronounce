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
from indonesian_errors_corpus import CORPUS, ERROR_TYPES  # noqa: E402

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


def _row(report, error_type, correct, wrong, changed, bad):
    length = len(report["expected"])
    return {
        "error_type": error_type, "correct": correct, "wrong": wrong, "word_len": length,
        "tune": is_tune_word(correct), "bad": bad,
        "localized": localized(report, changed) if bad else False,
        "changed": changed, "distance": report["distance"], "expected": report["expected"],
        "actual": report["actual"], "weighted_edits": report["weighted_edits"],
        "phones": report["phones"],
    }


def score_cell(error_type, correct, wrong, heard, lang=ID_LANG):
    """Score one induced error: ``heard`` is what the recognizer produced for the misspelling."""
    changed = changed_expected_phones(correct, wrong, lang)
    if not changed:
        return None
    reports = phones._word_reports(heard, correct, lang)
    if not reports or not reports[0]["expected"]:
        return None
    return _row(reports[0], error_type, correct, wrong, changed, bad=True)


def negative_row(word, heard, lang=ID_LANG):
    """Score correct audio against its own text; any later flag is a false alarm."""
    reports = phones._word_reports(heard, word, lang)
    if not reports or not reports[0]["expected"]:
        return None
    return _row(reports[0], "none", word, word, [], bad=False)


def corpus_cells(lang=ID_LANG):
    """Every scoreable positive cell plus skip statistics (identical phones, empty expected)."""
    cells, stats = [], {"skipped": 0, "empty": 0}
    for error_type in ERROR_TYPES:
        for correct, wrong in CORPUS[error_type]:
            changed = changed_expected_phones(correct, wrong, lang)
            if not changed:
                stats["skipped"] += 1
                continue
            if not expected_phones(correct, lang):
                stats["empty"] += 1
                continue
            cells.append({"error_type": error_type, "correct": correct, "wrong": wrong,
                          "word_len": len(expected_phones(correct, lang)),
                          "tune": is_tune_word(correct), "changed": changed})
    return cells, stats


def build_rows(cells, recognize, lang=ID_LANG):
    """Score every cell, add one negative per distinct correct word, count failures."""
    rows, stats = [], {"failed": 0, "skipped": 0, "empty": 0}
    seen = set()
    for cell in cells:
        try:
            heard = recognize(cell["wrong"])
        except Exception:  # network/JIT failure: skip the cell, never abort the run
            logger.warning("recognize failed for %r", cell["wrong"], exc_info=True)
            stats["failed"] += 1
            continue
        row = score_cell(cell["error_type"], cell["correct"], cell["wrong"], heard, lang)
        if row is None:
            stats["skipped"] += 1
            continue
        rows.append(row)
        if cell["correct"] not in seen:
            seen.add(cell["correct"])
            try:
                negative_heard = recognize(cell["correct"])
            except Exception:
                logger.warning("recognize failed for %r", cell["correct"], exc_info=True)
                stats["failed"] += 1
                continue
            negative = negative_row(cell["correct"], negative_heard, lang)
            if negative is not None:
                rows.append(negative)
    return rows, stats


def save_rows(rows, path=ROW_CACHE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f)


def load_rows(path=ROW_CACHE):
    with open(path) as f:
        return json.load(f)


def _recognizer(lang, threads):
    import torch

    torch.set_num_threads(threads)
    from openpronounce import audio

    def recognize(word):
        return phones.recognize_phones(audio.load(audio.text2speech(word, lang=lang)), lang=lang)

    return recognize


def run(args):
    cells, stats = corpus_cells(args.lang)
    rows, build_stats = build_rows(cells, _recognizer(args.lang, args.threads), args.lang)
    for key in ("failed", "skipped", "empty"):
        build_stats[key] += stats[key]
    save_rows(rows, args.out)
    print(f"{len(cells)} cells, {sum(1 for r in rows if r['bad'])} scored, "
          f"{sum(1 for r in rows if not r['bad'])} negatives -> {args.out}")
    print(f"skipped (identical phones) {build_stats['skipped']}, empty {build_stats['empty']}, "
          f"failed {build_stats['failed']}")


def assets(args):
    """Fast live check on the first two cells of a few error types (regression smoke)."""
    recognize = _recognizer(args.lang, args.threads)
    sample = [c for t in ("voicing", "e_to_i") for c in CORPUS[t][:2]]
    cells, _ = corpus_cells(args.lang)
    lookup = {(c["correct"], c["wrong"]): c for c in cells}
    for correct, wrong in sample:
        cell = lookup.get((correct, wrong))
        if cell is None:
            print(f"{correct:8s} -> {wrong:8s}  SKIP (same phones)")
            continue
        row = score_cell(cell["error_type"], correct, wrong, recognize(wrong), args.lang)
        print(f"{correct:8s} -> {wrong:8s}  localized={row['localized']} bad={row['bad']}")


def _flag(threshold, min_edits):
    def flag(row):
        return is_flagged(row, threshold, min_edits)
    return flag


def _bucket(word_len):
    if word_len <= 2:
        return "len2"
    if word_len == 3:
        return "len3"
    return "len4+"


def tune_table(rows, top=15, budgets=(0.05, 0.1, 0.2, 0.3)):
    """Recall-first: best localization per false-alarm budget, then the top points by F1."""
    lines = []
    for budget in budgets:
        points = [(metrics(rows, _flag(t, m)), t, m) for t, m in GRID]
        eligible = [p for p in points if p[0]["far"] <= budget]
        if not eligible:
            lines.append(f"far<={budget:.2f}: no point in the grid")
            continue
        m, t, me = max(eligible, key=lambda p: (p[0]["localization_recall"], -p[0]["far"]))
        lines.append(f"far<={budget:.2f}: threshold={t:.2f} min_edits={me} "
                     f"localization_recall={m['localization_recall']:.3f} far={m['far']:.3f} "
                     f"precision={m['precision']:.3f}")
    lines.append("")
    lines.append(f"{'threshold':>9} {'min_edits':>9} {'loc_recall':>10} {'word_recall':>11} "
                 f"{'P':>6} {'far':>6} {'F1':>6}")
    ranked = sorted(((metrics(rows, _flag(t, m)), t, m) for t, m in GRID),
                    key=lambda p: (-p[0]["f1"], p[0]["far"]))
    for m, t, me in ranked[:top]:
        lines.append(f"{t:9.2f} {me:9} {m['localization_recall']:10.3f} {m['word_recall']:11.3f} "
                     f"{m['precision']:6.3f} {m['far']:6.3f} {m['f1']:6.3f}")
    return "\n".join(lines)


def report_table(rows):
    """Localization per error type and word-length bucket, then overall metrics per split."""
    types = [r["error_type"] for r in rows if r["bad"]]
    ordered = list(dict.fromkeys(types))
    lines = [f"{'error_type':20} {'len2':>7} {'len3':>7} {'len4+':>7} {'total':>7}  target"]
    for error_type in ordered:
        positives = [r for r in rows if r["bad"] and r["error_type"] == error_type]
        cells, hit = [], 0
        for bucket in ("len2", "len3", "len4+"):
            group = [r for r in positives if _bucket(r["word_len"]) == bucket]
            hits = sum(1 for r in group if r["localized"])
            cells.append(f"{hits}/{len(group)}" if group else "-")
            hit += hits
        mark = "not-targeted" if error_type in NOT_TARGETED else "targeted"
        lines.append(f"{error_type:20} {cells[0]:>7} {cells[1]:>7} {cells[2]:>7} "
                     f"{f'{hit}/{len(positives)}':>7}  {mark}")
    lines.append("")
    current = _flag(phones.PHONE_ERROR_THRESHOLD, phones.PHONE_ERROR_MIN_EDITS)
    for split, keep in (("tune", True), ("held-out", False)):
        subset = [r for r in rows if r["tune"] == keep]
        m = metrics(subset, current)
        lines.append(f"overall {split:9} n_pos={m['n_pos']} n_neg={m['n_neg']} "
                     f"localization_recall={m['localization_recall']:.3f} word_recall={m['word_recall']:.3f} "
                     f"precision={m['precision']:.3f} far={m['far']:.3f}")
    for name, want in (("hit", True), ("miss", False)):
        confidences = [max((p["confidence"] for p in r["phones"]), default=0.0)
                       for r in rows if r["bad"] and r["localized"] == want]
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        lines.append(f"mean phone confidence, {name}: {mean:.3f} (n={len(confidences)})")
    return "\n".join(lines)


def tune(args):
    print(tune_table(load_rows(args.out)))


def report(args):
    print(report_table(load_rows(args.out)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="recognize, score and cache the rows")
    mode.add_argument("--tune", action="store_true", help="threshold grid on the tuning half")
    mode.add_argument("--report", action="store_true", help="per-type table, tune vs held-out")
    mode.add_argument("--assets", action="store_true", help="fast live subset")
    mode.add_argument("--real", metavar="DIR", help="score real wav+txt pairs under DIR")
    parser.add_argument("--lang", default=ID_LANG)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", default=ROW_CACHE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.run:
        run(args)
    elif args.tune:
        tune(args)
    elif args.report:
        report(args)
    elif args.assets:
        assets(args)
    else:
        print("mode not implemented yet")  # replaced in Task 6 and Task 7


if __name__ == "__main__":
    main()
