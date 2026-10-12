# Indonesian Learner-Error Detection Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark that measures how well OpenPronounce localizes a foreign learner's induced Indonesian mistakes while counting false alarms on correct audio, and tunes the flag thresholds for it.

**Architecture:** A corpus of `(correct, misspelled)` word pairs per error type. gTTS reads the misspelling, `phones.recognize_phones` hears it, and `phones._word_reports` scores it against the correct word. Pure functions (corpus validation, localization ground truth, metrics, threshold grid) are unit-tested with plain phone lists and no model; a thin runner wires the neural recognizer and caches one JSON of scored rows, mirroring `benchmarks/word_detection.py` (`--extract`/`--tune`/`--report`/`--assets`).

**Tech Stack:** Python >= 3.10, `Levenshtein` (phone alignment, already a dependency), `phonemizer`/espeak (ground truth), gTTS (audio), the existing Wav2Vec2 phone model, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-12-indonesian-learner-error-benchmark-design.md`

## Global Constraints

- Python `>= 3.10`; dependencies already declared (`phonemizer>=3.2`, `Levenshtein>=0.20`, `gTTS>=2.3`).
- Indonesian language code is `id`; gTTS code `id`; espeak voice `id`. The corpus is Indonesian only.
- The benchmark MUST report detection **and** false alarms from the same run; a report with only one is a bug.
- Tuning is **recall-first**; the primary measure is **phone localization** (`localized`), not "word flagged".
- The tune/held-out split MUST be deterministic across processes (no builtin `hash()`).
- No production scoring constant changes without held-out evidence, and per-language changes MUST NOT alter English/French behavior.
- Run tests with `.venv/bin/pytest -q`; run the benchmark with `.venv/bin/python benchmarks/indonesian_errors.py ...`.
- The CLI script follows the `benchmarks/word_detection.py` import pattern (`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`).
- Scoring uses a plain phone list when the caller passes one (no posteriors, full confidence); a `phones.PhoneRecognition` adds plausibility scaling. Tests use plain lists.

## Review Focus

Inputs and failure modes the spec implies but this plan's happy-path tests do not fully exercise; each has a test in the owning task below.

1. A `(correct, wrong)` pair whose espeak phones are identical (e.g. two spellings that phonemize the same) MUST be skipped and counted, never scored as a miss. (Task 4)
2. A word the phonemizer returns no phones for (empty expected) MUST not crash the run or divide by zero. (Task 4)
3. gTTS or the recognizer failing on one cell (network/JIT) MUST not abort the whole run; the cell is skipped and reported. (Task 5)
4. A correct word that the recognizer mishears (a negative) MUST count as a false alarm, not be silently dropped from the metrics. (Task 5, Task 3)
5. Tune/held-out leakage: a base word reused across several error types MUST land entirely on one side of the split. (Task 3, Task 4)
6. Rerunning from the cached rows MUST reproduce the same tables (the cache is the reproducibility contract). (Task 5, Task 6)

---

### Task 1: Corpus data module

**Files:**
- Create: `benchmarks/indonesian_errors_corpus.py`
- Create: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CORPUS: dict[str, list[tuple[str, str]]]` (error-type name -> list of `(correct, wrong)` spellings) and `ERROR_TYPES: tuple[str, ...]` (insertion order of `CORPUS`).

- [ ] **Step 1: Write the failing test**

```python
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import indonesian_errors_corpus as corpus  # noqa: E402


class TestCorpus:
    def test_has_the_expected_error_types(self):
        for name in ("voicing", "final_dropped", "ng_to_n", "schwa_to_a",
                     "diphthong_reduced", "e_to_i"):
            assert name in corpus.CORPUS

    def test_each_type_has_enough_distinct_pairs(self):
        for name, pairs in corpus.CORPUS.items():
            assert len(pairs) >= 10, name
            assert len(set(pairs)) == len(pairs), name
            for correct, wrong in pairs:
                assert correct != wrong, (name, correct)
                assert correct.isascii() and wrong.isascii(), (name, correct, wrong)
                assert correct == correct.lower() and wrong == wrong.lower(), (name, correct, wrong)

    def test_error_types_matches_corpus_order(self):
        assert corpus.ERROR_TYPES == tuple(corpus.CORPUS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'indonesian_errors_corpus'`.

- [ ] **Step 3: Implement the corpus module**

Create `benchmarks/indonesian_errors_corpus.py`. Seed each type with the spike's pairs and extend to at least 10 per type with words of 2, 3 and 4+ phones (use real Indonesian words). Error types and starter pairs:

```python
"""Indonesian learner-error corpus: (correct, misspelled) word pairs per error type.

gTTS reads Indonesian orthography literally, so the misspelling below is what the
synthesized audio says; espeak phones of the misspelling are the ground truth for the
error. A pair whose espeak phones equal the correct word's is skipped by the runner.
"""

CORPUS = {
    "voicing": [
        ("pagi", "bagi"), ("tahu", "dahu"), ("kota", "gota"), ("sapu", "sabu"),
        ("paku", "baku"), ("tari", "dari"), ("kaca", "gaca"), ("sate", "sade"),
        ("tiga", "diga"), ("pita", "bida"),
    ],
    "final_dropped": [
        ("makan", "maka"), ("jalan", "jala"), ("rumah", "ruma"), ("tidak", "tida"),
        ("besar", "besa"), ("kapan", "kapa"), ("malam", "mala"), ("pasar", "pasa"),
        ("jangan", "janga"), ("hitam", "hita"),
    ],
    "ng_to_n": [
        ("tangan", "tanan"), ("bunga", "buna"), ("datang", "datan"), ("senang", "senan"),
        ("bangku", "banku"), ("tengah", "tenah"), ("pinggir", "pinir"), ("tangga", "tanga"),
        ("sengaja", "senaja"), ("bangun", "banun"),
    ],
    "schwa_to_a": [
        ("besar", "basar"), ("terima", "tarima"), ("selamat", "salamat"), ("kecil", "kacil"),
        ("empat", "ampat"), ("berat", "barat"), ("medan", "madan"), ("telur", "talur"),
        ("semua", "samua"), ("kertas", "kartas"),
    ],
    "diphthong_reduced": [
        ("pulau", "pula"), ("pantai", "panta"), ("pisau", "pisa"), ("sungai", "sunga"),
        ("kerbau", "kerba"), ("pakai", "paka"), ("laut", "lat"), ("daun", "dan"),
        ("harimau", "harima"), ("santai", "santa"),
    ],
    "e_to_i": [
        ("enak", "inak"), ("empat", "impat"), ("merah", "mirah"), ("kecil", "kicil"),
        ("senang", "sinang"), ("besar", "bisar"), ("dekat", "dikat"), ("gelas", "gilas"),
        ("petani", "pitani"), ("sebelah", "sibilah"),
    ],
}

ERROR_TYPES = tuple(CORPUS)
```

All ten pairs of each type must be distinct (the uniqueness test enforces this). A base word reused across types (`kecil` in `schwa_to_a` and `e_to_i`, `senang` in `ng_to_n` and `e_to_i`) is fine and intended: the runner keys tune/held-out on the base word, so it lands on one side under every type (see Task 3/4). Extend any type that ends up with fewer than ten *scoring* pairs after the identical-phones skip in Task 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors_corpus.py tests/test_indonesian_errors.py
git commit -m "feat(bench): add Indonesian learner-error corpus"
```

---

### Task 2: Localization ground truth

**Files:**
- Create: `benchmarks/indonesian_errors.py`
- Modify: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: `openpronounce.phones.get_expected_phones`, `Levenshtein.opcodes`.
- Produces:
  - `ID_LANG = "id"`
  - `expected_phones(word, lang=ID_LANG) -> list[str]` — flattened espeak phones.
  - `changed_expected_phones(correct, wrong, lang=ID_LANG) -> list[str]` — phones of `correct` at positions replaced or deleted by `wrong`, plus the nearest correct phone for a pure insertion.
  - `localized(report, changed) -> bool` — `report` is a `phones._word_reports` entry; true when any of its phones has `confidence > 0` and its `expected` is in `changed`.

- [ ] **Step 1: Write the failing test**

```python
import indonesian_errors as ie  # noqa: E402  (added to sys.path in Task 1)


class TestGroundTruth:
    def test_expected_phones_flattens(self):
        assert ie.expected_phones("makan") == ["m", "a", "k", "a", "n"]

    def test_changed_phones_for_substitution(self):
        assert ie.changed_expected_phones("pagi", "bagi") == ["p"]

    def test_changed_phones_for_final_deletion(self):
        assert ie.changed_expected_phones("makan", "maka") == ["n"]

    def test_localized_when_the_changed_phone_is_flagged(self):
        from openpronounce import phones
        # espeak's own phones of "maka", so this holds whatever symbol espeak uses.
        report = phones._word_reports(ie.expected_phones("maka"), "makan", lang="id")[0]
        assert ie.localized(report, ie.changed_expected_phones("makan", "maka")) is True

    def test_not_localized_when_another_phone_is_flagged(self):
        from openpronounce import phones
        # /b/ for /m/ is flagged, but the ground-truth change (makan -> maka) is the final /n/.
        heard = ["b"] + ie.expected_phones("maka")[1:]
        report = phones._word_reports(heard, "makan", lang="id")[0]
        assert ie.localized(report, ie.changed_expected_phones("makan", "maka")) is False

    def test_no_change_gives_empty(self):
        assert ie.changed_expected_phones("makan", "makan") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'indonesian_errors'`.

- [ ] **Step 3: Implement the module and the ground-truth functions**

Create `benchmarks/indonesian_errors.py` with the word_detection import header:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors.py tests/test_indonesian_errors.py
git commit -m "feat(bench): add localization ground truth for Indonesian errors"
```

---

### Task 3: Flag rule, metrics and deterministic split

**Files:**
- Modify: `benchmarks/indonesian_errors.py`
- Modify: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `is_flagged(report, threshold, min_edits) -> bool` — `report["distance"] > 0 and (weighted_edits / len(expected) >= threshold or weighted_edits >= min_edits)`.
  - `is_tune_word(word) -> bool` — stable SHA-1 parity, same across processes.
  - `GRID: list[tuple[float, float]]` — `itertools.product([t/20 for t in range(4, 25)], [1, 1.5, 2, 2.5, 3, 4, 99])`.
  - `metrics(rows, flag) -> dict` with keys `n_pos`, `n_neg`, `localized`, `mislocalized`, `flagged_pos`, `unflagged_pos`, `fp`, `localization_recall`, `word_recall`, `precision`, `far`, `f1`.

- [ ] **Step 1: Write the failing test**

```python
class TestMetrics:
    def rows(self):
        def r(bad, localized, flag):
            return {"bad": bad, "localized": localized, "_flag": flag}
        return [
            r(True, True, True), r(True, True, True), r(True, True, True),   # localized hits
            r(True, False, True),                                            # flagged, wrong phone
            r(True, True, False), r(True, False, False),                     # missed
            r(False, False, True),                                           # false alarm
            r(False, False, False), r(False, False, False),
            r(False, False, False), r(False, False, False),
        ]

    def test_metrics_exact(self):
        m = ie.metrics(self.rows(), lambda row: row["_flag"])
        assert m["n_pos"] == 6 and m["n_neg"] == 5
        assert m["localized"] == 3 and m["mislocalized"] == 1
        assert m["flagged_pos"] == 4 and m["fp"] == 1
        assert m["localization_recall"] == 3 / 6
        assert m["word_recall"] == 4 / 6
        assert m["precision"] == 4 / 5
        assert m["far"] == 1 / 5
        assert m["f1"] == 2 * (4 / 5) * (4 / 6) / ((4 / 5) + (4 / 6))

    def test_metrics_never_divides_by_zero(self):
        m = ie.metrics([], lambda row: row["_flag"])
        assert m["localization_recall"] == 0.0 and m["far"] == 0.0 and m["precision"] == 0.0


class TestSplit:
    def test_tune_word_is_deterministic_and_stable(self):
        assert ie.is_tune_word("makan") == ie.is_tune_word("makan")
        # A literal value pinned so a change of hash breaks the test, not silently the split.
        assert ie.is_tune_word("makan") == (int(hashlib.sha1("makan".encode()).hexdigest(), 16) % 2 == 0)


class TestFlagRule:
    def test_flag_rule_threshold_or_min_edits(self):
        long = {"distance": 1, "expected": ["m", "a", "k", "a", "n"], "weighted_edits": 2.0}
        assert ie.is_flagged(long, 0.4, 2) is True       # 2.0 >= min_edits
        short = {"distance": 1, "expected": ["m", "a", "k", "a", "n"], "weighted_edits": 0.5}
        assert ie.is_flagged(short, 0.4, 2) is False     # 0.5 / 5 = 0.1 < 0.4 and 0.5 < 2
        ratio = {"distance": 1, "expected": ["m", "a"], "weighted_edits": 1.0}
        assert ie.is_flagged(ratio, 0.4, 2) is True      # 1.0 / 2 = 0.5 >= 0.4
        clean = {"distance": 0, "expected": ["m", "a"], "weighted_edits": 0.0}
        assert ie.is_flagged(clean, 0.4, 2) is False     # a perfect word is never flagged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `AttributeError: module 'indonesian_errors' has no attribute 'metrics'`.

- [ ] **Step 3: Implement the rule, split, grid and metrics**

Add to `benchmarks/indonesian_errors.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors.py tests/test_indonesian_errors.py
git commit -m "feat(bench): add flag rule, localization metrics and stable split"
```

---

### Task 4: Score one cell and build the corpus rows

**Files:**
- Modify: `benchmarks/indonesian_errors.py`
- Modify: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: `CORPUS`, `ERROR_TYPES` (Task 1); `changed_expected_phones`, `localized` (Task 2); `is_tune_word`, `is_flagged` (Task 3); `phones._word_reports`.
- Produces:
  - `score_cell(error_type, correct, wrong, heard, lang=ID_LANG) -> dict | None` — one positive row; `None` when the misspelling does not change the phones or no word report is produced.
  - `negative_row(word, heard, lang=ID_LANG) -> dict | None` — one correct-audio row (`bad=False`).
  - `corpus_cells(lang=ID_LANG) -> tuple[list[dict], dict]` — `(cells, stats)`; each cell has `error_type`, `correct`, `wrong`, `word_len`, `tune`, `changed`; `stats` counts `skipped` (identical phones) and `empty`.
  - Row keys: `error_type`, `correct`, `wrong`, `word_len`, `tune`, `bad`, `localized`, `changed`, `distance`, `expected`, `actual`, `weighted_edits`, `phones`.

- [ ] **Step 1: Write the failing test**

```python
class TestCells:
    def test_score_cell_localizes_a_substitution(self):
        # pagi -> bagi: we hear espeak's phones of "bagi" scored against the correct "pagi".
        row = ie.score_cell("voicing", "pagi", "bagi", ie.expected_phones("bagi"))
        assert row["bad"] is True
        assert row["error_type"] == "voicing"
        assert row["changed"] == ["p"]
        assert row["localized"] is True

    def test_score_cell_returns_none_when_phones_do_not_change(self):
        assert ie.score_cell("schwa_to_a", "enak", "enak", ie.expected_phones("enak")) is None

    def test_corpus_cells_skips_identical_pairs_and_counts_them(self):
        cells, stats = ie.corpus_cells()
        assert stats["skipped"] >= 0
        assert all(c["correct"] != "" for c in cells)
        # every scored cell carries a stable tune flag and a non-empty changed list
        for c in cells:
            assert c["tune"] == ie.is_tune_word(c["correct"])
            assert c["changed"]

    def test_empty_phonemization_returns_none(self):
        # A word espeak gives no phones for must not crash the run or divide by zero.
        assert ie.negative_row("", ["a"]) is None

    def test_reused_base_word_is_on_one_side(self):
        # "enak" appears in schwa_to_a and e_to_i; both cells must share the tune flag.
        cells, _ = ie.corpus_cells()
        flags = {c["tune"] for c in cells if c["correct"] == "enak"}
        assert len(flags) <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `AttributeError: module 'indonesian_errors' has no attribute 'score_cell'`.

- [ ] **Step 3: Implement `score_cell`, `negative_row` and `corpus_cells`**

Add the corpus import at the top (`from indonesian_errors_corpus import CORPUS, ERROR_TYPES`) and:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors.py tests/test_indonesian_errors.py
git commit -m "feat(bench): score Indonesian error cells and build corpus rows"
```

---

### Task 5: Runner, cache and `--run` / `--assets`

**Files:**
- Modify: `benchmarks/indonesian_errors.py`
- Modify: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: `corpus_cells`, `score_cell`, `negative_row` (Task 4); `audio.text2speech`, `audio.load`, `phones.recognize_phones`.
- Produces:
  - `build_rows(cells, recognize, lang=ID_LANG) -> tuple[list[dict], dict]` — scored positives plus one negative per distinct correct word; `recognize(word) -> heard` is injected (default wraps gTTS+recognizer). `stats` adds `"failed"` (recognizer raised, cell skipped) and `"skipped"`/`"empty"` from `corpus_cells`.
  - `save_rows(rows, path=ROW_CACHE) -> None` and `load_rows(path=ROW_CACHE) -> list[dict]`.
  - `run(args)`, `assets(args)`, and `main(argv=None)` wiring `--run/--tune/--report/--assets/--real` (mutually exclusive), `--lang`, `--threads`, `--out`.

- [ ] **Step 1: Write the failing test**

```python
class TestBuildRows:
    def test_build_rows_adds_one_negative_per_correct_word(self):
        cells = [
            {"error_type": "voicing", "correct": "pagi", "wrong": "bagi",
             "word_len": 4, "tune": ie.is_tune_word("pagi"), "changed": ["p"]},
            {"error_type": "e_to_i", "correct": "enak", "wrong": "inak",
             "word_len": 4, "tune": ie.is_tune_word("enak"), "changed": ["e"]},
        ]

        def fake_recognize(word):
            return {"bagi": ie.expected_phones("bagi"), "inak": ie.expected_phones("inak"),
                    "pagi": ie.expected_phones("pagi"), "enak": ie.expected_phones("enak")}[word]

        rows, stats = ie.build_rows(cells, fake_recognize)
        positives = [r for r in rows if r["bad"]]
        negatives = [r for r in rows if not r["bad"]]
        assert len(positives) == 2
        assert sorted(r["correct"] for r in negatives) == ["enak", "pagi"]
        assert stats["failed"] == 0

    def test_a_failing_recognizer_does_not_abort_the_run(self):
        cells = [{"error_type": "voicing", "correct": "pagi", "wrong": "bagi",
                  "word_len": 4, "tune": True, "changed": ["p"]}]

        def broken(word):
            raise RuntimeError("network down")

        rows, stats = ie.build_rows(cells, broken)
        assert rows == [] and stats["failed"] >= 1

    def test_rows_round_trip_through_json(self, tmp_path):
        cells = [{"error_type": "voicing", "correct": "pagi", "wrong": "bagi",
                  "word_len": 4, "tune": True, "changed": ["p"]}]
        rows, _ = ie.build_rows(cells, lambda w: ["b", "a", "ɡ", "i"])
        path = str(tmp_path / "rows.json")
        ie.save_rows(rows, path)
        assert ie.load_rows(path) == rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `AttributeError: module 'indonesian_errors' has no attribute 'build_rows'`.

- [ ] **Step 3: Implement the runner, cache and CLI**

Add `from indonesian_errors_corpus import CORPUS, ERROR_TYPES` (if not already), and:

```python
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
    elif args.assets:
        assets(args)
    else:
        print("mode not implemented yet")  # replaced in Task 6 and Task 7


if __name__ == "__main__":
    main()
```

Note: `--tune` and `--report` are added as CLI flags here but raise/print until Task 6; that keeps the argparse surface in one place. Adjust the `else` branch to dispatch to `tune(args)` / `report(args)` in Task 6 and `real(args)` in Task 7.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors.py tests/test_indonesian_errors.py
git commit -m "feat(bench): add Indonesian error runner, cache and CLI skeleton"
```

---

### Task 6: `--tune` and `--report`

**Files:**
- Modify: `benchmarks/indonesian_errors.py`
- Modify: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: `load_rows`, `GRID`, `is_flagged`, `metrics` (Tasks 3, 5); `NOT_TARGETED`.
- Produces:
  - `tune_table(rows, top=15, budgets=(0.05, 0.1, 0.2, 0.3)) -> str` — recall-first: for each false-alarm budget, the grid point with the highest localization recall, plus the global top-`top` by `f1`.
  - `report_table(rows) -> str` — per-error-type rows with one `localized/total` cell per word-length bucket (`len2` = <=2 phones, `len3`, `len4+`), the total, and whether the type is targeted; then the overall `metrics` for the tune and held-out halves separately at the current globals `phones.PHONE_ERROR_THRESHOLD` / `phones.PHONE_ERROR_MIN_EDITS`; then the mean confidence of localized vs missed phones.
  - `tune(args)`, `report(args)` printing the tables and loading `args.out`.

- [ ] **Step 1: Write the failing test**

```python
class TestTables:
    def rows(self):
        # One localized positive, one missed positive, one false alarm.
        return [
            {"error_type": "voicing", "bad": True, "localized": True, "tune": True,
             "distance": 1, "word_len": 4, "expected": ["p", "a", "ɡ", "i"],
             "weighted_edits": 1.0, "phones": [{"expected": "p", "heard": "b", "confidence": 0.5}]},
            {"error_type": "e_to_i", "bad": True, "localized": False, "tune": True,
             "distance": 1, "word_len": 4, "expected": ["e", "n", "a", "k"],
             "weighted_edits": 0.5, "phones": [{"expected": "e", "heard": "i", "confidence": 0.5}]},
            {"error_type": "none", "bad": False, "localized": False, "tune": True,
             "distance": 0, "word_len": 4, "expected": ["p", "a", "ɡ", "i"],
             "weighted_edits": 0.0, "phones": []},
        ]

    def test_report_table_contains_every_error_type_and_buckets(self):
        out = ie.report_table(self.rows())
        assert "voicing" in out and "e_to_i" in out and "overall" in out
        assert "len4+" in out and "1/1" in out  # voicing localized in the >=4 bucket

    def test_report_table_marks_non_targeted_types(self):
        out = ie.report_table(self.rows())
        assert "voicing" in out and "not-targeted" in out

    def test_tune_table_reports_a_best_point_per_budget(self):
        out = ie.tune_table(self.rows(), budgets=(0.0, 1.0))
        assert "far<=" in out and "localization_recall" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `AttributeError: module 'indonesian_errors' has no attribute 'tune_table'`.

- [ ] **Step 3: Implement the tables and dispatch**

Add to `benchmarks/indonesian_errors.py`:

```python
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
```

Dispatch in `main`: `elif args.tune: tune(args)` / `elif args.report: report(args)` / `elif args.real: real(args)` (the last added in Task 7).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors.py tests/test_indonesian_errors.py
git commit -m "feat(bench): add tune and report tables for Indonesian errors"
```

---

### Task 7: `--real` recordings mode

**Files:**
- Modify: `benchmarks/indonesian_errors.py`
- Modify: `tests/test_indonesian_errors.py`

**Interfaces:**
- Consumes: `phones.recognize_phones`, `phones.compare_phones`, `audio.load`.
- Produces: `score_directory(directory, lang=ID_LANG) -> list[dict]` where each dict is `{wav, text, errors}` (the flagged words of `compare_phones`), reading every `*.wav` with a sibling `*.txt` of the same stem; `real(args)` prints them.

- [ ] **Step 1: Write the failing test**

```python
class TestRealMode:
    def test_score_directory_pairs_wav_and_text(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"")
        (tmp_path / "a.txt").write_text("makan")
        (tmp_path / "b.wav").write_bytes(b"")
        (tmp_path / "b.txt").write_text("pagi")

        def fake_recognize(path):
            return ["m", "a", "k", "a"]

        results = ie.score_directory(str(tmp_path), recognize=fake_recognize)
        assert [r["text"] for r in results] == ["makan", "pagi"]
        # "makan" heard as "maka" is flagged; "pagi" heard as "maka" is also flagged
        assert all("errors" in r for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: FAIL with `AttributeError: module 'indonesian_errors' has no attribute 'score_directory'`.

- [ ] **Step 3: Implement `score_directory` and `real`**

Add to `benchmarks/indonesian_errors.py`:

```python
def score_directory(directory, lang=ID_LANG, recognize=None):
    """Score every ``*.wav`` under ``directory`` against its sibling ``*.txt`` (real readings)."""
    if recognize is None:
        from openpronounce import audio
        recognize = lambda path: phones.recognize_phones(audio.load(path), lang=lang)  # noqa: E731
    results = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".wav"):
            continue
        stem = name[:-4]
        text_path = os.path.join(directory, stem + ".txt")
        if not os.path.exists(text_path):
            logger.warning("no text for %s, skipped", name)
            continue
        with open(text_path) as f:
            text = f.read().strip()
        result = phones.compare_phones(recognize(os.path.join(directory, name)), text, lang=lang)
        results.append({"wav": name, "text": text,
                        "errors": [e["word"] for e in result["errors"]]})
    return results


def real(args):
    for row in score_directory(args.real, args.lang):
        flagged = ", ".join(row["errors"]) or "-"
        print(f"{row['wav']:30s} {row['text']:30s} flagged: {flagged}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_indonesian_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/indonesian_errors.py tests/test_indonesian_errors.py
git commit -m "feat(bench): add --real mode for real Indonesian readings"
```

---

### Task 8: Per-language threshold hook in `phones.py`

**Files:**
- Modify: `openpronounce/phones.py` (constants block near line 32; `compare_phones` near line 530)
- Modify: `tests/test_languages.py`

**Interfaces:**
- Consumes: `PHONE_ERROR_THRESHOLD`, `PHONE_ERROR_MIN_EDITS`.
- Produces: `PHONE_THRESHOLDS: dict[str, tuple[float, float]]` (empty by default) and `phone_thresholds(lang) -> tuple[float, float]` returning the per-language override or the globals; `compare_phones` uses it for the current `lang`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_languages.py` in `TestIndonesianPhones`:

```python
    def test_per_language_threshold_override(self):
        from openpronounce import phones
        default = (phones.PHONE_ERROR_THRESHOLD, phones.PHONE_ERROR_MIN_EDITS)
        self.assertEqual(phones.phone_thresholds("id"), default)
        self.assertEqual(phones.phone_thresholds("en"), default)
        phones.PHONE_THRESHOLDS["zz"] = (0.1, 1)
        try:
            self.assertEqual(phones.phone_thresholds("zz"), (0.1, 1))
            self.assertEqual(phones.phone_thresholds("en"), default)
        finally:
            del phones.PHONE_THRESHOLDS["zz"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_languages.py -q`
Expected: FAIL with `AttributeError: module 'openpronounce.phones' has no attribute 'PHONE_THRESHOLDS'`.

- [ ] **Step 3: Implement the hook**

Add after the constants (near line 37) in `openpronounce/phones.py`:

```python
#: Per-language overrides of (threshold, min_edits); empty means the globals above apply.
PHONE_THRESHOLDS = {}


def phone_thresholds(lang=DEFAULT_LANGUAGE):
    """``(PHONE_ERROR_THRESHOLD, PHONE_ERROR_MIN_EDITS)`` for ``lang``, or its override."""
    return PHONE_THRESHOLDS.get(lang, (PHONE_ERROR_THRESHOLD, PHONE_ERROR_MIN_EDITS))
```

In `compare_phones`, replace the direct use of the globals in the flag test and the confidence:

```python
    threshold, min_edits = phone_thresholds(lang)
    ...
        if report["distance"] and (edits / length >= threshold or edits >= min_edits):
            ...
                "confidence": round(min(1.0, max(edits / length, edits / min_edits)), 3),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_languages.py tests/test_phones.py -q`
Expected: PASS (no regression in the existing language tests).

- [ ] **Step 5: Commit**

```bash
git add openpronounce/phones.py tests/test_languages.py
git commit -m "feat(phones): allow per-language error thresholds"
```

---

### Task 9: Run the benchmark, record results, set the Indonesian point

**Files:**
- Modify: `benchmarks/README.md` (new `## Indonesian learner errors` section)
- Modify: `openpronounce/phones.py` (set `PHONE_THRESHOLDS["id"]` only if the grid justifies it, with a comment)
- Modify: `progress.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: every earlier task.
- Produces: the results section and, if adopted, the Indonesian operating point.

- [ ] **Step 1: Run the full benchmark**

Run: `.venv/bin/python benchmarks/indonesian_errors.py --run` (network + model; cached after the first call), then
`.venv/bin/python benchmarks/indonesian_errors.py --report` and `--tune`.
Expected: a per-type table with localization rates and a false-alarm rate, plus the trade-off lines.

- [ ] **Step 2: Decide the Indonesian operating point**

If a grid point within `far <= 0.10` reaches localization recall clearly above the current globals' held-out recall, adopt it: set `PHONE_THRESHOLDS["id"] = (threshold, min_edits)` in `openpronounce/phones.py` with a comment citing the held-out number. Otherwise leave `PHONE_THRESHOLDS` empty and say so in the README.

- [ ] **Step 3: Write the README section**

Add `## Indonesian learner errors` to `benchmarks/README.md`: the corpus size, how to run, the per-type table, the tune/held-out overall numbers, the chosen operating point (or why the globals were kept), the close-pair rows marked not-targeted, and the synthetic-corpus caveat plus the real-recording check (sample size and result).

- [ ] **Step 4: Run the real check and fill it in**

Run: `.venv/bin/python benchmarks/indonesian_errors.py --real <dir>` on the recordings the human partner supplied and add the outcome to the README. If no recordings are available yet, write that the section is pending and leave the command documented.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

```bash
git add benchmarks/README.md benchmarks/indonesian_errors.py openpronounce/phones.py progress.md CHANGELOG.md
git commit -m "docs(bench): Indonesian learner-error results and operating point"
```
