"""P1 eval scoring (evals/scoring.py). Seam A: scoring functions on hand-built lists; no I/O, nothing faked.
Imports stay inside each test so one missing name never hides the other reds."""
import json

import pytest

REFUSAL = "I cannot find enough information in the indexed code to answer this."  # literal [code prompts.py:47]


def span(file_path, start, end, **extra):
    return {"file_path": file_path, "start_line": start, "end_line": end, **extra}


def misses(n):
    return [span("other.py", 1, 2) for _ in range(n)]


def test_result_sharing_lines_with_gold_counts_as_found_at_its_rank():  # P1-M1
    # Break: overlap written as "result inside gold" — 370-385 starts before 380, so rank 2 is lost.
    from evals.scoring import score_question
    s = score_question([span("other.py", 1, 9), span("sessions.py", 370, 385)], [span("sessions.py", 380, 390)])
    assert (s["hit@5"], s["rank"]) == (True, 2)


def test_recall_at_5_counts_each_gold_found_of_three():  # P1-M2
    # Break: recall as "any gold found" gives 1.0; all-golds@5 true with c.py missing.
    from evals.scoring import score_question
    golds = [span("a.py", 1, 5), span("b.py", 10, 20), span("c.py", 30, 40)]
    s = score_question([span("a.py", 3, 4), span("b.py", 15, 25)] + misses(3), golds)
    assert s["recall@5"] == pytest.approx(0.6667, abs=5e-5)  # 2/3 [brief]
    assert s["all_golds@5"] is False


def test_reciprocal_rank_uses_first_found_place():  # P1-M3
    # Break: ranks counted from 0 gives 1/2 = 0.5; using the later hit at rank 4 gives 0.25.
    from evals.scoring import score_question
    s = score_question(misses(2) + [span("sessions.py", 385, 386), span("sessions.py", 380, 381)], [span("sessions.py", 380, 390)])
    assert (s["rank"], s["rr"]) == (3, pytest.approx(0.3333, abs=5e-5))  # 1/3 [brief]


def test_flip_table_counts_b_and_c_and_keys_rows_by_question_id():  # P1-M4
    # Break: b/c swapped gives (2, 1); rows paired by position (after run in another order) gives q2 after = 4.
    from evals.scoring import flip_table
    before = {"q1": 1, "q2": 2, "q3": None, "q4": None, "q5": None}
    after = {"q2": None, "q3": 4, "q4": None, "q5": 1, "q1": 1}  # same questions, rotated order
    t = flip_table(before, after)
    assert (t["b"], t["c"]) == (1, 2)
    assert (t["rows"]["q2"]["rank_before"], t["rows"]["q2"]["rank_after"]) == (2, None)
    assert (t["rows"]["q3"]["rank_before"], t["rows"]["q3"]["rank_after"]) == (None, 4)


SOURCE = "\n".join(f"x_{n} = {n}" for n in range(1, 31))  # f.py: line n reads "x_n = n", every line unique


def chunk(start, end, text_end=None):
    lines = SOURCE.split("\n")[start - 1:(text_end or end)]
    return span("f.py", start, end, text="\n".join(lines))


def test_function_stored_as_two_windows_has_no_missing_lines():  # P1-M5
    # Break: a chunk counts only when its range equals the unit, so 10-15 + 16-20 report all 11 lines missing.
    from evals.scoring import missing_lines
    unit = [span("f.py", 10, 20)]
    assert missing_lines(unit, [chunk(10, 15), chunk(16, 20)], {"f.py": SOURCE}) == []
    # Able to fail [doc PROOF-blueprint.md:148]: last window dropped -> lines 16-20 missing.
    assert missing_lines(unit, [chunk(10, 15)], {"f.py": SOURCE}) == [("f.py", 16), ("f.py", 17), ("f.py", 18), ("f.py", 19), ("f.py", 20)]


GOLD = [span("sessions.py", 380, 390)]


def test_citation_precision_is_against_gold_lines_not_shown_lines():  # P1-M6
    # Break: precision against shown 380-400 gives 1.0; strict-inside against shown marks 395 true.
    from evals.scoring import score_answer
    a = score_answer("See `sessions.py:384` and `sessions.py:395`.", GOLD, [span("sessions.py", 380, 400)])
    assert (a["refused"], a["precision"]) == (False, 0.5)
    assert [(c["cite"], c["strict_inside"]) for c in a["citations"]] == [(("sessions.py", 384, 384), True), (("sessions.py", 395, 395), False)]


def test_exactly_one_shared_line_is_a_hit_and_zero_is_a_miss():  # P1-E1
    # Break: `<` where `<=` belongs drops A (shares only line 390); the reverse makes B (starts at 391) a hit.
    from evals.scoring import is_hit
    assert (is_hit(span("sessions.py", 390, 400), GOLD[0]), is_hit(span("sessions.py", 391, 400), GOLD[0])) == (True, False)


def test_gold_found_only_at_rank_6_misses_top_5_but_counts_for_hit20_and_rr():  # P1-E2
    # Break: top-5 cut taken as top-6 makes hit@5 true; rr ignoring ranks past 5 gives 0.0.
    from evals.scoring import score_question
    s = score_question(misses(5) + [span("sessions.py", 385, 386)], GOLD)
    assert (s["hit@5"], s["hit@20"], s["rank"]) == (False, True, 6)
    assert s["rr"] == pytest.approx(0.1667, abs=5e-5)  # 1/6 [brief]


@pytest.mark.parametrize("c, b, p, gain", [
    (6, 0, 0.03125, True),   # 2 * 0.5**6 [doc PROOF-blueprint.md:139]
    (5, 0, 0.0625, False),   # one-sided would read 0.03125, a gain
    (6, 1, 0.125, False),    # 2 * (1 + 7) / 2**7 [brief]
    (0, 0, 1.0, False),      # n = 0: no division by zero
])
def test_sign_test_is_exact_two_sided(c, b, p, gain):  # P1-E3
    # Break: one-sided p halves every value, so (5, 0) reads as a gain; n = 0 divides by zero.
    from evals.scoring import sign_test
    got_p, got_gain = sign_test(b=b, c=c)
    assert (got_p, got_gain) == (pytest.approx(p, abs=1e-12), gain)


def test_bare_line_number_binds_to_last_file_named_before_it():  # P1-E4
    # Break: bare numbers dropped gives []; binding to the first file named gives api.py:384.
    from evals.scoring import parse_citations
    answer = "`api.py` calls into the helper. In `sessions.py` the merge happens; see line 384."
    assert parse_citations(answer) == [("sessions.py", 384, 384)]


SHOWN = [span("sessions.py", 380, 400)]


def test_exact_refusal_sentence_counts_as_refusal_not_precision_0():  # P1-E5
    # Break: refusal scored as precision 0 gives mean 0.5 and refusals 0.
    from evals.scoring import score_answers
    s = score_answers([(REFUSAL, GOLD, SHOWN), ("See `sessions.py:384`.", GOLD, SHOWN)])
    assert (s["mean_precision"], s["refusals"]) == (1.0, 1)


def test_no_results_is_a_miss_with_rr_0():  # P1-E6
    # Break: empty list read as rank 1 gives hit@5 true; max() of [] raises. Control: one hit at rank 1.
    from evals.scoring import score_question
    empty, one = score_question([], GOLD), score_question([span("sessions.py", 380, 390)], GOLD)
    assert (empty["hit@5"], empty["rank"], empty["rr"]) == (False, None, 0.0)
    assert (one["hit@5"], one["rank"], one["rr"]) == (True, 1, 1.0)


def test_longer_answer_containing_refusal_sentence_is_scored_not_refused():  # P1-E7 (ruling 5)
    # Break: refusal found by substring gives refusals 2 and no scored answer.
    from evals.scoring import score_answers
    s = score_answers([(REFUSAL + " See `sessions.py:395`.", GOLD, SHOWN), (REFUSAL, GOLD, SHOWN)])
    assert (s["refusals"], s["mean_precision"]) == (1, 0.0)


def test_same_function_name_without_shared_line_is_a_miss():  # P1-B1
    # Break: name match counted as a hit (claim B2) puts rank at 1; the real overlap sits at rank 2.
    from evals.scoring import score_question
    gold = [span("sessions.py", 60, 75, name="merge_setting")]
    s = score_question([span("sessions.py", 1, 20, name="merge_setting"), span("sessions.py", 70, 80, name="other")], gold)
    assert s["rank"] == 2


def test_right_lines_in_wrong_file_with_same_base_name_is_a_miss():  # P1-B2
    # Break: file compared by base name (or not compared) puts tests/a.py at rank 1.
    from evals.scoring import score_question
    s = score_question([span("tests/a.py", 1, 5), span("src/a.py", 1, 5)], [span("src/a.py", 1, 5)])
    assert s["rank"] == 2


def gold_file(tmp_path, rows):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(rows))
    return path


ROW = {"id": "rq01", "question": "Where are settings merged?", "golds": [span("sessions.py", 60, 75)]}


def test_gold_row_without_line_range_is_refused_naming_its_id(tmp_path):  # P1-B3
    # Break: missing range defaulted to the whole file loads without error.
    from evals.scoring import load_gold
    bad = {"id": "rq07", "question": "Where is the adapter mounted?", "golds": [{"file_path": "sessions.py"}]}
    with pytest.raises(ValueError, match="rq07"):
        load_gold(gold_file(tmp_path, [ROW, bad]))
    assert [r["id"] for r in load_gold(gold_file(tmp_path, [ROW]))] == ["rq01"]  # control: a raise-always loader fails


def test_chunk_range_claiming_a_line_its_text_lacks_reports_that_line():  # P1-B4
    # Break: coverage trusts the stored range 10-20 and never reads the text, so line 20 counts as covered.
    from evals.scoring import missing_lines
    assert missing_lines([span("f.py", 10, 20)], [chunk(10, 20, text_end=19)], {"f.py": SOURCE}) == [("f.py", 20)]


def test_citation_to_a_line_never_shown_is_not_shown_only():  # P1-B5
    # Break: citations checked only against gold — shown_only missing, or equal to in-gold (382: shown, not gold).
    from evals.scoring import score_answer
    gold, shown = [span("sessions.py", 385, 390)], [span("sessions.py", 380, 390)]
    unseen, seen = score_answer("See `sessions.py:999`.", gold, shown), score_answer("See `sessions.py:382`.", gold, shown)
    assert (unseen["citations"][0]["shown_only"], unseen["precision"]) == (False, 0.0)
    assert (seen["citations"][0]["shown_only"], seen["precision"]) == (True, 0.0)


def test_two_gold_rows_with_one_id_are_refused_naming_the_id(tmp_path):  # P1-B6
    # Break: duplicate id accepted, so a flip table keyed by id merges two questions and hides one.
    from evals.scoring import load_gold
    first, second = {**ROW, "id": "rq03"}, {**ROW, "id": "rq03", "question": "Where are hooks dispatched?"}
    with pytest.raises(ValueError, match="rq03"):
        load_gold(gold_file(tmp_path, [first, second]))
    assert [r["id"] for r in load_gold(gold_file(tmp_path, [first, {**second, "id": "rq04"}]))] == ["rq03", "rq04"]
