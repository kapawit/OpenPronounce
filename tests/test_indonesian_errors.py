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

    def test_every_type_has_at_least_ten_cells(self):
        cells, _ = ie.corpus_cells()
        counts = {}
        for c in cells:
            counts[c["error_type"]] = counts.get(c["error_type"], 0) + 1
        for name in corpus.ERROR_TYPES:
            assert counts.get(name, 0) >= 10, (name, counts.get(name, 0))

    def test_empty_phonemization_returns_none(self):
        # A word espeak gives no phones for must not crash the run or divide by zero.
        assert ie.negative_row("", ["a"]) is None

    def test_reused_base_word_is_on_one_side(self):
        # "kecil" appears in schwa_to_a and e_to_i; both cells must share the tune flag.
        cells, _ = ie.corpus_cells()
        flags = {c["tune"] for c in cells if c["correct"] == "kecil"}
        assert len(flags) <= 1


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


class TestTables:
    def rows(self):
        # Two tune positives (one localized, one missed), one tune negative,
        # one held-out localized positive, one held-out negative.
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
            {"error_type": "voicing", "bad": True, "localized": True, "tune": False,
             "distance": 1, "word_len": 4, "expected": ["p", "a", "ɡ", "i"],
             "weighted_edits": 1.0, "phones": [{"expected": "p", "heard": "b", "confidence": 0.5}]},
            {"error_type": "none", "bad": False, "localized": False, "tune": False,
             "distance": 0, "word_len": 4, "expected": ["p", "a", "ɡ", "i"],
             "weighted_edits": 0.0, "phones": []},
        ]

    def test_report_table_reports_held_out_types_and_buckets(self):
        out = ie.report_table(self.rows())
        assert "voicing" in out and "e_to_i" in out and "overall" in out
        assert "len4+" in out and "1/1" in out  # held-out voicing localized in the >=4 bucket

    def test_report_table_marks_non_targeted_types(self):
        out = ie.report_table(self.rows())
        assert "voicing" in out and "not-targeted" in out

    def test_report_table_surfaces_the_success_target(self):
        out = ie.report_table(self.rows())
        # At the global constants this fixture's held-out positives are unflagged,
        # so the target is not met and the line must say so.
        assert "success target" in out and "NOT MET" in out

    def test_tune_table_selects_from_the_tune_half_only(self):
        def tune_part(rows):
            out = ie.tune_table(rows, budgets=(0.1,))
            return out.splitlines()[0].split("|")[0]
        base = self.rows()
        poisoned = [dict(r) for r in base]
        for r in poisoned:
            if not r["tune"]:
                r["bad"] = False
                r["localized"] = False
                r["weighted_edits"] = 0.0
                r["distance"] = 0
        assert tune_part(base) == tune_part(poisoned)

    def test_tune_table_reports_a_best_point_per_budget(self):
        out = ie.tune_table(self.rows(), budgets=(0.0, 1.0))
        assert "far<=" in out and "localization_recall" in out
