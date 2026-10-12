# Progress: add Bahasa Indonesia

Goal: add Indonesian (`id`) to OpenPronounce as a supported, selectable language
for the pronunciation scorer. POC first, then calibrate.

## Status: POC verified end-to-end; phone normalization (step 1) and diphthongs (step 2) calibrated

## What changed

### POC (registry + UI)

- `openpronounce/languages.py` — registered `"id"`:
  - espeak voice `id` (phonemization)
  - ASR model `indonesian-nlp/wav2vec2-large-xlsr-indonesian` (word transcription)
  - `acoustic_good=11.0` — placeholder, still not measured
- `static/ui.js` — `id: 'id-ID'` speech tag and an Indonesian example set
  (`EXAMPLES.id`); the language dropdown auto-populates from `GET /languages`.
- `templates/index.html` — header copy "6 other languages" -> "7".
- `tests/test_languages.py` — `TestIndonesian.test_registry_entry`.
- `README.md`, `CHANGELOG.md` — list Indonesian as experimental.

### Calibration step 1 (phone normalization)

- `openpronounce/phones.py`:
  - `_PHONE_MAP["id"]`: `ɑ/ʌ→a`, `ɛ→e`, `ɔ→o`, `x/χ/ħ→h`. The phone model
    (`facebook/wav2vec2-lv-60-espeak-cv-ft`) was fine-tuned on English espeak
    labels, so it spells Indonesian /a/ as ɑ/ʌ, /e/ as ɛ, /o/ as ɔ and /h/ as
    x/χ/ħ.
  - `_NEAR_PHONES["id"]`: close pairs `{p,b} {t,d} {k,ɡ} {s,z} {f,v} {ʃ,ʒ}
    {tʃ,dʒ} {n,ŋ}`. The recognizer systematically voices Indonesian's
    unaspirated stops and maps final /ŋ/ to n. `/ə/` vs `/a/` was kept
    contrastive here, then relaxed to a close pair in step 2 (see below).
- `tests/test_languages.py` — `TestIndonesianPhones` (merges, language-awareness,
  close-phone cost, and a `kata` behavioral check that flags in German but not
  in Indonesian).

### Calibration step 2 (diphthongs, rhotics, glottal stops, schwa)

- `openpronounce/phones.py`:
  - `_NEAR_PHONES["id"]` gained diphthong/vowel close pairs `{aʊ,a} {aʊ,u}
    {aɪ,a} {aɪ,i} {oɪ,o} {oɪ,i}`. The English model often hears the second half
    of an Indonesian diphthong weakly (`mau` -> `/ma/`); on a 2-phone word that
    otherwise counted as a full substitution. The pure vowels `a`/`u` are
    deliberately *not* a close pair (each diphthong is split into two separate
    groups so `a` and `u` never share one).
  - `_DIPHTHONG_MERGES["id"]`: adjacent `a+u / a+ʊ -> aʊ`, `a+i / a+ɪ -> aɪ`,
    `o+i / o+ɪ -> oɪ`. The recognizer sometimes splits a diphthong into two
    phones (`mau` -> `m a u`) while espeak writes one (`m aʊ`).
  - `_merge_diphthongs` / `_merge_diphthongs_indexed` apply that merge **per
    word** inside `_word_reports`, to both the expected candidate and the heard
    slice. This is deliberate: a global merge on the heard stream also joins
    `a|u` across a word boundary (`Dua ular` -> `...aʊ l...`) and corrupts the
    alignment. Per-word merging keeps `Dua ular` intact.
  - `_PHONE_MAP["id"]`: `ɹ -> r`, `ɾ -> r`. The recognizer spells Indonesian
    `/r/` as the English approximant (or a tap), which otherwise split a word
    (`ular` -> `ulaɹʔ`) and caused a false alarm.
  - `_PHONE_MAP["id"]`: `ʔ -> ""` (dropped). espeak never writes a glottal stop
    for Indonesian, but the recognizer inserts one between two vowels
    (`pisau itu` -> `pisaʔ itu`), which caused false alarms on `pisau`, `pulau`,
    `dua`.
  - `_NEAR_PHONES["id"]`: `{ə,a}` and `{ə,e}` as close pairs. The recognizer
    cannot tell pepet `/ə/` from `/a/` or `/e/` (`televisi` -> `dalafisi`), so a
    confusion is a minor error. Two separate groups keep `a` and `e` distinct
    (`enak` "tasty" vs `anak` "child"). This supersedes the earlier
    "keep `/ə/` contrastive" decision: the flag threshold already tolerates a
    single substitution, so `perang`/`parang` was never caught anyway, while
    schwa-heavy words false-alarmed.
- `tests/test_languages.py` — new `TestIndonesianPhones` cases:
  `test_rhotic_variants_map_to_r`, `test_glottal_stop_insertions_are_dropped`,
  `test_diphthong_glides_are_close_to_their_vowels`,
  `test_dropped_diphthong_glide_is_not_flagged`, `test_split_diphthong_is_reunited`
  (includes the `Dua ular` word-boundary control), `test_schwa_is_close_to_a_and_e`,
  and `test_genuine_error_is_still_flagged` (guards against over-tolerance).

## Verification

Environment: `.venv` (created with `virtualenv`; the system `python3-venv`
module and `espeak-ng` CLI are absent, but `libespeak-ng` + `espeak-ng-data`
are installed, so `phonemizer` works). Torch from the CPU wheel index.

- Full suite: `112 passed` (`.venv/bin/pytest`).
- `TestIndonesian::test_registry_entry` and `TestIndonesianPhones` pass.
- Phonemization: `"Saya mau pesan kopi, tolong."` -> `/saja maʊ pɛsan kopi toloŋ/`.
- `GET /languages` returns `{"code": "id", "name": "Indonesian"}`.
- Word-list sample (gTTS audio, espeak `id` vs recognizer): exact-match diffs
  fell 25 -> 20; the survivors are alignment-level and no longer flag words.
- Flagged-word sample (same 41 gTTS words, `compare_phones` after steps 1+2):
  **0/41 words flagged** (single words and the tricky diphthong words `mau`,
  `pisau`, `kerbau`, `pantai`, `sampai`, `pulau`).
- Sentence samples (56 gTTS sentences across three sets: greetings, questions,
  numbers, reduplication, loanwords, `ng`/`ny`, final glottal, diphthongs,
  daily conversation): **0/56 flagged**. Word-boundary controls
  `"Dua ular itu."` (PER 0.00) and `"Kota udara itu."` (PER 0.08) are clean,
  confirming the per-word diphthong merge and the glottal-drop do not bleed
  across words.
- Full pipeline on a gTTS Indonesian clip (`compare_audio_with_text`, lang=`id`):

  ```
  before:      SCORE 88.0  PER 0.30  WER 0.0   (apa/empat false alarms)
  step 1:      SCORE 90.0  PER 0.25  WER 0.0   (voicing/vowel false alarms gone; `mau` still flagged)
  steps 1+2:   SCORE 90.0  PER 0.25  WER 0.0   errors []
  ```

  Steps 1+2 leave no flagged words. `phone_error_rate` is still 0.25 because it
  is the raw recognizer-vs-espeak edit rate on the unmerged stream (the
  recognizer drops the final `ŋ`, splits `aʊ`, unvoices stops); it is *not* the
  learner-facing error signal, which is `errors`/`words_with_errors` (empty).
- Negative control: a genuine error still flags (`mau` heard as `b a u`:
  `/b/` for `/m/` is nasal-vs-stop, a full error, reported).
  Models are cached, so re-runs are offline.

## Known gaps (still to calibrate)

- `acoustic_good` is a guess; not measured against two native voices.
- No Indonesian reference voice besides gTTS (no Piper or Kokoro `id` voice), so
  the acoustic baseline cannot be measured the way the other languages were.
- No human-rated calibration data for Indonesian (English only).
- The close-pair cost (`NEAR_PHONE_COST = 0.5`) and the flag thresholds
  (`PHONE_ERROR_THRESHOLD`, `PHONE_ERROR_MIN_EDITS`) mean a *single* wrong phone
  is never flagged on a word of 3+ phones: a full vowel swap (`makan` ->
  `mokan`) or a dropped non-final syllable (`makan` -> `mak`) passes. This is the
  existing model, shared with English and French, and changing it globally would
  trade these misses for more false alarms, so it is documented, not changed.
  It is the main remaining accuracy gap for a pronunciation trainer and needs
  human-rated data to tune.
- End-to-end run used the reference clip as the "learner" audio, so
  `acoustic_distance` is 0.0; a real non-native recording has not been tested.
- `phone_error_rate` is a raw recognizer-vs-espeak rate and stays ~0.25 on
  correct audio (recognizer drops final `ŋ`, splits diphthongs, unvoices stops);
  the per-word `errors` are the signal users see. Not changed here.

## Next steps

1. (done) Sample espeak `id` vs the recognizer; add `_PHONE_MAP` / `_NEAR_PHONES`.
2. (done) Reunite diphthongs (`aɪ`, `aʊ`, `oɪ`) the recognizer splits into
   vowel+glide, map the rhotic variants, drop inserted glottal stops, and treat
   pepet `/ə/` as close to `/a/`/`/e/`, to clear the `mau`/`ular`/`pisau`/
   `televisi`-type false alarms.
3. Measure the acoustic baseline and set `acoustic_good` from data — blocked on a
   second Indonesian voice (Piper has no `id` voice today).
4. Collect native readings for human calibration, then tune the flag thresholds
   (the single-substitution miss above) with that data.
5. Optional: a Piper voice for `id` if one appears upstream.

## Reproduce

```bash
source .venv/bin/activate
pytest -q
openpronounce recording.wav "Saya mau pesan kopi" --lang id
```
