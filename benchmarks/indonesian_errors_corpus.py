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
