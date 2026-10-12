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
  per-error-type **detection rate** and the **false-alarm rate on correct audio** in
  the same run, so neither can be hidden behind the other.
- A tuning sweep (same shape as `word_detection.py`: grid over the flag thresholds)
  that produces the detection/false-alarm trade-off and recommends an operating
  point, tuned on one half of the corpus and reported on the other.
- A checked-in corpus definition (words + error transformations) so the numbers are
  reproducible and a regression is caught.
- A fast subset for quick checks and a full run for the real number.
- A place for a small set of real recordings to be dropped in and scored, as the
  final sanity check.

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
  the misspelling changed.
- **Negatives.** The correct audio of every word in the corpus, scored against its
  own text, in the same run. A flag there is a false alarm.
- **Metrics.** Per error type: detection rate (recall). Overall: recall, precision,
  F1, false-alarm rate (flagged correct words / correct words), and the confidence
  distribution of hits vs misses. Output a table: rows = error type, columns = word
  length (2 / 3 / 4+), cells = caught/total.
- **Tuning.** Sweep `PHONE_ERROR_THRESHOLD` and `PHONE_ERROR_MIN_EDITS` over a grid
  (as `word_detection.py` does), on a tune half of the corpus; report on the
  held-out half; print the points that reach a target recall with the fewest false
  alarms, and vice versa. Optional second sweep over `NEAR_PHONE_COST` to see what
  the "close pair" leniency costs in detection. (Builder call: the split is by base
  word, not by cell, so the same word never appears in both halves under different
  error types; a per-type held-out rate is printed only when that type has held-out
  cells.)
- **Deliverables.**
  - `benchmarks/indonesian_errors.py`: `--run` (full corpus + negatives, prints the
    tables), `--tune`, `--report` (tune vs held-out + trade-off curve), `--assets`
    (a tiny fast subset for regression), `--real <dir>` (score a folder of
    `*.wav` + sidecar text, no ground-truth errors required — just flags).
  - Corpus data: a data module or JSON next to the script listing the cells.
  - A `## Indonesian learner errors` section in `benchmarks/README.md` with the
    results table and the recommended operating point.
  - Unit tests for the corpus loader (every non-skipped pair really changes phones)
    and for the metric functions, so the benchmark itself is trustworthy.

## Success

Proposed target to react to (this is the product call to confirm, not a fact):

- On the held-out corpus, detection (recall) of the error types that matter for
  learners **≥ 0.7**, while the false-alarm rate on correct audio stays **≤ 10 %**.
- The per-error-type table makes the trade-off explicit, so "not detectable by
  design" (e.g. a cost-0.5 voicing pair) is visible as a row, not hidden.

## Open questions (for the human partner)

Ordered by how much they change what gets built.

1. **Operating point.** For a teaching tool, is missing a mistake worse than a false
   alarm? This sets whether tuning is recall-first or precision-first and whether the
   success targets above are the right ones. The proposed ≥ 0.7 recall / ≤ 10 %
   false-alarm pair is a product call, not derived.
2. **Caught word vs localized phone.** The product goal is "point at the wrong
   sound", but on single-word utterances "the word was flagged" is automatic once
   anything is flagged, so recall 0.7 is reachable while pointing at the wrong phone.
   Is the real target the *word* or the *phone*? If the phone, the tuning objective
   and the success gate change.
3. **Which error types are inside the target.** The spike's worst rows — voicing
   p/b, t/d, k/g and reduced diphthongs — are close pairs the scorer deliberately
   treats as half-cost, i.e. nearly invisible by design. Are those expected to be
   caught (which means changing the scoring rule, e.g. `NEAR_PHONE_COST` toward 1,
   at a false-alarm cost), or are they exempt and reported as a visible "not
   detectable by design" row?
4. **Blast radius.** Is an Indonesian-specific tuning acceptable (e.g. per-language
   thresholds), or must the production constants stay global? `PHONE_ERROR_THRESHOLD`
   etc. are module-level globals, so a per-language operating point is a real code
   change; a global-only benchmark can only recommend and may never ship the number.
5. **Real recordings.** Can a handful of real readings (the human partner, or a few
   people — some correct, some deliberately wrong) be recorded to sanity-check the
   synthetic result? If not, the benchmark is entirely synthetic and that limitation
   becomes a headline caveat in the README, not a footnote.

## Left to the builder

- The exact word list and misspellings per error type (the spike has a starter set).
- Grid ranges for the sweep, and whether to add a supervised classifier later.
- File format of the corpus (Python data vs JSON).
- How to wire `--assets` into the existing test/CI style.
