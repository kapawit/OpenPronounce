# Progress: add Bahasa Indonesia

Goal: add Indonesian (`id`) to OpenPronounce as a supported, selectable language
for the pronunciation scorer. POC first, then calibrate.

## Status: POC verified end-to-end; phone normalization calibrated (step 1)

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
    {tʃ,dʒ} {n,ŋ} {ə,a}`. The recognizer systematically voices Indonesian's
    unaspirated stops and maps final /ŋ/ to n.
- `tests/test_languages.py` — `TestIndonesianPhones` (merges, language-awareness,
  close-phone cost, and a `kata` behavioral check that flags in German but not
  in Indonesian).

## Verification

Environment: `.venv` (created with `virtualenv`; the system `python3-venv`
module and `espeak-ng` CLI are absent, but `libespeak-ng` + `espeak-ng-data`
are installed, so `phonemizer` works). Torch from the CPU wheel index.

- Full suite: `105 passed` (`.venv/bin/pytest`).
- `TestIndonesian::test_registry_entry` and `TestIndonesianPhones` pass.
- Phonemization: `"Saya mau pesan kopi, tolong."` -> `/saja maʊ pɛsan kopi toloŋ/`.
- `GET /languages` returns `{"code": "id", "name": "Indonesian"}`.
- Word-list sample (gTTS audio, espeak `id` vs recognizer): exact-match diffs
  fell 25 -> 20; the survivors are alignment-level and no longer flag words.
- Full pipeline on a gTTS Indonesian clip (`compare_audio_with_text`, lang=`id`):

  ```
  before:  SCORE 88.0  PER 0.30  WER 0.0
  after:   SCORE 90.0  PER 0.25  WER 0.0
  ```

  The `apa`/`empat`-style voicing and vowel-variant false alarms are gone.
  Models are cached, so re-runs are offline.

## Known gaps (still to calibrate)

- `acoustic_good` is a guess; not measured against two native voices.
- No Indonesian reference voice besides gTTS (no Piper or Kokoro `id` voice), so
  the acoustic baseline cannot be measured the way the other languages were.
- Residual false alarm: the diphthong /aʊ/ in `mau` is flagged (expected `maʊ`,
  heard `mau`/`u`). Beyond the conservative merge set; needs a diphthong rule.
- No human-rated calibration data for Indonesian (English only).
- End-to-end run used the reference clip as the "learner" audio, so
  `acoustic_distance` is 0.0; a real non-native recording has not been tested.

## Next steps

1. (done) Sample espeak `id` vs the recognizer; add `_PHONE_MAP` / `_NEAR_PHONES`.
2. Handle diphthongs (`aɪ`, `aʊ`, `oɪ`) recognized as vowel+glide, to clear the
   remaining `mau`-type flags.
3. Measure the acoustic baseline and set `acoustic_good` from data — blocked on a
   second Indonesian voice (Piper has no `id` voice today).
4. Collect native readings for human calibration.
5. Optional: a Piper voice for `id` if one appears upstream.

## Reproduce

```bash
source .venv/bin/activate
pytest -q
openpronounce recording.wav "Saya mau pesan kopi" --lang id
```
