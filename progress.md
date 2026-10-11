# Progress: add Bahasa Indonesia (POC)

Goal: add Indonesian (`id`) to OpenPronounce as a supported, selectable language
for the pronunciation scorer. Calibration deferred; this is a proof of concept
first.

## Status: POC verified end-to-end

## What changed

- `openpronounce/languages.py` — registered `"id"`:
  - espeak voice `id` (phonemization)
  - ASR model `indonesian-nlp/wav2vec2-large-xlsr-indonesian` (word transcription)
  - `acoustic_good=11.0` — placeholder, not measured
- `static/ui.js` — added `id: 'id-ID'` speech tag and an Indonesian example set
  (`EXAMPLES.id`); the language dropdown auto-populates from `GET /languages`.
- `templates/index.html` — header copy "6 other languages" -> "7".
- `tests/test_languages.py` — `TestIndonesian.test_registry_entry`.
- `README.md`, `CHANGELOG.md` — list Indonesian as experimental.

## Verification

Environment: `.venv` (created with `virtualenv`; the system `python3-venv`
module and `espeak-ng` CLI are absent, but `libespeak-ng` + `espeak-ng-data`
are installed, so `phonemizer` works). Torch installed from the CPU wheel
index.

- Full suite: `100 passed` (`.venv/bin/pytest`).
- `TestIndonesian::test_registry_entry` passes.
- Phonemization: `"Saya mau pesan kopi, tolong."` -> `/saja maʊ pɛsan kopi toloŋ/`.
- `GET /languages` returns `{"code": "id", "name": "Indonesian"}`.
- Full pipeline on a gTTS Indonesian clip (`compare_audio_with_text`, lang=`id`):

  ```
  SCORE      88.0
  TRANSCRIPT saya mau pesan kopi tolong
  PER 0.3    WER 0.0
  ```

  Models now cached in the Hugging Face cache, so re-runs are offline.

## Known gaps (calibration, deferred on purpose)

- `acoustic_good` is a guess; not measured against native voices.
- No Indonesian-specific phone normalization in `openpronounce/phones.py`
  (`_PHONE_MAP`, `_NEAR_PHONES`, `ALTERNATE_PRONUNCIATIONS`). Observed false
  alarm: `mau` expected `/maʊ/`, heard `/mɑu/` (the /aʊ/-vs-/ɑu/ merge is the
  likely first fix, alongside /e-ɛ/, /o-ɔ/).
- No human-rated calibration data for Indonesian (English is the only
  calibrated language).
- End-to-end run used the reference clip as the "learner" audio, so
  `acoustic_distance` was 0.0; a real non-native recording has not been tested.

## Next steps

1. Sample espeak `id` output vs the recognizer on a word list; add the needed
   `_PHONE_MAP` / `_NEAR_PHONES` entries.
2. Measure the acoustic baseline (gTTS vs Piper/reference) and set
   `acoustic_good` from data.
3. Collect native readings for human calibration.
4. Optional: add a Piper voice for `id` if one exists (default TTS is gTTS,
   which supports Indonesian; Kokoro does not).

## Reproduce

```bash
source .venv/bin/activate
pytest -q
openpronounce recording.wav "Saya mau pesan kopi" --lang id
```
