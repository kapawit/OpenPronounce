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
