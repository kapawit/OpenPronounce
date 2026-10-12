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
