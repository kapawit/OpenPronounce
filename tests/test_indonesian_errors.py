import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import indonesian_errors_corpus as corpus  # noqa: E402
import indonesian_errors as ie  # noqa: E402


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
        heard = ["b"] + ie.expected_phones("makan")[1:]
        report = phones._word_reports(heard, "makan", lang="id")[0]
        assert ie.localized(report, ie.changed_expected_phones("makan", "maka")) is False

    def test_no_change_gives_empty(self):
        assert ie.changed_expected_phones("makan", "makan") == []
