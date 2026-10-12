# Indonesian learner-error detection benchmark — design

Date: 2026-10-12
Status: draft for review

## Intent and why

OpenPronounce scores a learner's pronunciation of a target text: it aligns the
phones the recognizer heard with the phones espeak expects, flags the words whose
phones differ, and reports a score. For English there is a human-rated corpus
(speechocean762) and a benchmark (`benchmarks/word_detection.py`) that measures how
well the word-flagging rule catches words the raters marked mispronounced.

Indonesian (`id`) was added recently, but every test so far used gTTS reading
*correct* Indonesian and checked that nothing was flagged (currently 0 of 41 words
and 0 of 56 sentences). That only proves half the capability. The product goal is
the opposite half: a foreign learner says an Indonesian word wrong, and the tool
points at that sound.

A throwaway spike (24 induced errors, 6 types, gTTS reading a misspelled word so the
error is known) found the tool catches only about 1 in 5: voicing p/b, t/d, k/g 0/4;
final consonant dropped 0/4; ŋ→n 1-2/4; schwa→a 1/4; diphthong reduced 0-1/4;
e→i 1-2/4. The recognizer heard almost every error correctly — the misses are the
*scorer* deciding those differences do not matter. This benchmark exists to measure
that reliably, on a corpus with known ground truth, and to tune the decision.

## Goals

- One command that, for a fixed Indonesian corpus of induced errors, reports
  per-error-type **localization rate** (the wrong sound pointed at) and the
  **false-alarm rate on correct audio** in the same run, so neither can be hidden
  behind the other.
- A tuning sweep (same shape as `word_detection.py`: grid over the flag thresholds)
  that produces the localization/false-alarm trade-off and recommends an operating
  point, tuned on one half of the corpus and reported on the other. Tuning is
  recall-first.
- A checked-in corpus definition (words + error transformations) so the numbers are
  reproducible and a regression is caught.
- A fast subset for quick checks and a full run for the real number.
- A small set of real recordings, scored through the same path, as the final sanity
  check that the synthetic result transfers.

## Non-goals

- Building a human-rated learner corpus. Real recordings are a small add-on, not the
  bulk of the benchmark.
- A general scoring redesign for all languages. Tuning here is Indonesian-specific
  unless the held-out evidence shows a safe global change.
- Acoustics, fluency, or prosody. This is word/phone mispronunciation detection only.

## Anti-goals (what would make this a failure)

- A benchmark that looks good by letting correct audio false-alarm. Every report
  must show both detection and false alarms.
- Overfitting the thresholds to the synthetic corpus. A tune/held-out split of the
  corpus is mandatory, mirroring `word_detection.py` (even/odd).
- Changing production scoring constants without held-out evidence, or without saying
  whether the change is per-language or global.

## Constraints

- Environment: no `espeak-ng` CLI (phonemizer works through `libespeak-ng`); gTTS is
  the TTS backend (network, cached by `audio.text2speech`); the phone and ASR models
  are cached. CPU only.
- Build on what exists: `benchmarks/` layout and README, `phones.compare_phones`,
  `phones.recognize_phones`, `phones._word_reports`, and the constants
  `PHONE_ERROR_THRESHOLD`, `PHONE_ERROR_MIN_EDITS`, `NEAR_PHONE_COST`,
  `FINAL_DELETION_COST`, `FINAL_EXTRA_COST`, `PHONE_PLAUSIBLE_POSTERIOR`.
- Errors are induced by misspelling the correct word, because gTTS reads Indonesian
  orthography literally (e.g. `pagi`→`bagi`, `makan`→`maka`, `pulau`→`pula`). The
  ground-truth error is the espeak phone difference between the correct and the
  misspelled spelling; a pair whose phones are identical is skipped and reported.

## The how (decided)

- **Corpus.** A grid of error type × word. Each cell is `(correct, misspelled,
  error_type)`. Starter error types (the spike's set): voicing p→b, t→d, k→g; final
  consonant dropped; ŋ→n; ny ɲ→n; schwa→a; a→schwa; e→i; o→u; diphthong reduced
  (aʊ/aɪ/oɪ → one vowel); r dropped or →ɹ; schwa inserted between consonants;
  syllable dropped. Each error type gets 12-20 words, spread over word lengths 2, 3,
  4+ phones. (Builder call: N that large because a 50 % held-out split leaves a
  per-type rate moving in ~5-10 point steps; the report prints counts next to every
  rate so a noisy cell is visible.)
- **Running a cell.** Synthesize the *misspelled* word with gTTS (`lang="id"`),
  recognize it (`recognize_phones`), score it against the *correct* word
  (`compare_phones`, `lang="id"`). `caught` = the correct word appears in
  `words_with_errors`; `localized` = the flagged phone's expected phone is the one
  the misspelling changed. **`localized` is the primary success measure** (the
  product goal is to point at the wrong sound); `caught` is reported alongside it as
  the weaker measure.
- **Negatives.** The correct audio of every word in the corpus, scored against its
  own text, in the same run. A flag there is a false alarm.
- **Metrics.** Per error type: localization rate (primary) and word-caught rate
  (secondary). Overall: localization recall, word recall, precision, F1, false-alarm
  rate (flagged correct words / correct words), and the confidence distribution of
  hits vs misses. Output a table: rows = error type, columns = word length
  (2 / 3 / 4+), cells = localized/total. **Tuning is recall-first**: among the points
  in the grid, prefer the ones that maximize localization, then minimize false
  alarms.
- **Tuning.** Sweep `PHONE_ERROR_THRESHOLD` and `PHONE_ERROR_MIN_EDITS` over a grid
  (as `word_detection.py` does), on a tune half of the corpus; report on the
  held-out half; print the points that reach a target localization with the fewest
  false alarms, and vice versa. Optional second sweep over `NEAR_PHONE_COST` to see
  what the "close pair" leniency costs in detection. (Builder call: the split is by
  base word, not by cell, so the same word never appears in both halves under
  different error types; a per-type held-out rate is printed only when that type has
  held-out cells.)
- **Per-language operating point.** An Indonesian-specific tuning is allowed: the
  benchmark may recommend adding per-language thresholds to `phones.py`
  (`PHONE_ERROR_THRESHOLD`, `PHONE_ERROR_MIN_EDITS`, and optionally the cost
  constants), so English/French behaviour is untouched. The default path stays
  global when the Indonesian grid does not justify a separate value.
- **Close pairs are reported, not targeted (yet).** Voicing p/b, t/d, k/g and
  reduced diphthongs stay at the half-cost rule for this iteration. They appear as
  explicit "not targeted by design" rows in the table. Whether to raise
  `NEAR_PHONE_COST` toward 1 is a follow-up decided from the measured trade-off.
- **Deliverables.**
  - `benchmarks/indonesian_errors.py`: `--run` (full corpus + negatives, prints the
    tables), `--tune`, `--report` (tune vs held-out + trade-off curve), `--assets`
    (a tiny fast subset for regression), `--real <dir>` (score a folder of
    `*.wav` + sidecar text, no ground-truth errors required — just flags; used for
    the real recordings below).
  - Corpus data: a data module or JSON next to the script listing the cells.
  - A `## Indonesian learner errors` section in `benchmarks/README.md` with the
    results table and the recommended operating point.
  - Unit tests for the corpus loader (every non-skipped pair really changes phones)
    and for the metric functions, so the benchmark itself is trustworthy.
  - If the grid justifies it, a small change in `phones.py` adding per-language
    thresholds (with tests) plus an update to the README; if not, a note saying the
    global constants were kept and why.
- **Real recordings.** The human partner will record a small set (some correct, some
  deliberately wrong) as wav + text pairs. These are run through `--real` as a
  sanity check that the synthetic numbers transfer; the outcome and the sample size
  go in the README. This is validation, not a second benchmark.

## Success

The product call is **recall-first** and the primary signal is **localization** (the
right phone pointed at), not merely "the word was flagged":

- On the held-out corpus, localization of the error types that matter for learners
  **≥ 0.7**, while the false-alarm rate on correct audio stays **≤ 10 %**.
- The per-error-type table makes the trade-off explicit; close-pair errors (voicing,
  reduced diphthongs) are a visible "not targeted by design" row, not hidden.
- The real-recording sanity check does not contradict the synthetic result (roughly
  the same flags on the same kind of error).

## Resolved decisions (from the human partner)

1. **Operating point.** Recall-first: missing a learner's mistake is worse than a
   false alarm. Tuning maximizes localization within a false-alarm budget.
2. **Caught word vs localized phone.** The **phone** is the target. `localized` is
   the primary success measure and the tuning objective.
3. **Close-pair errors.** Report them, do not target them yet. Keep the half-cost
   rule this iteration; show them as "not targeted by design" rows; revisit
   `NEAR_PHONE_COST` from the measured trade-off.
4. **Blast radius.** Per-language tuning is allowed: an Indonesian operating point
   may be added as per-language thresholds in `phones.py`, leaving English/French
   untouched. Global stays the default when the grid does not justify a split.
5. **Real recordings.** The human partner will record a small set (some correct,
   some deliberately wrong) for the `--real` sanity check.

## Left to the builder

- The exact word list and misspellings per error type (the spike has a starter set).
- Grid ranges for the sweep, and whether to add a supervised classifier later.
- File format of the corpus (Python data vs JSON).
- How to wire `--assets` into the existing test/CI style.
