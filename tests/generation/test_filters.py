from __future__ import annotations

from embbench.generation.sts.filters import (
    has_pipe_only_table,
    pair_reject_reason,
    question_reject_reason,
    reject_reason,
    strip_headers,
)


def test_strip_headers_drops_page_and_tables() -> None:
    raw = "\n".join(
        [
            "<!-- page: 1/3 -->",
            "Page 1 of 3",
            "SOP-Number: 12",
            "| --- | --- |",
            "|     |     |",
            "Investigator obtains a proforma.",
        ]
    )
    cleaned = strip_headers(raw)
    assert "Page 1" not in cleaned
    assert "SOP-Number" not in cleaned
    assert "proforma" in cleaned


def test_reject_page_marker_and_sop_number() -> None:
    assert reject_reason("See Page 2 of 10 for the rest of the rule.") == "page_marker"
    assert reject_reason("SOP-Number is printed on every leaf.") == "sop_number"


def test_reject_pipe_only_table() -> None:
    table = "| --- | --- |\n|     |     |\n|     |     |"
    assert has_pipe_only_table(table) is True
    assert reject_reason(table) == "pipe_only_table"


def test_pair_reject_either_side() -> None:
    reason = pair_reject_reason(
        "Investigator obtains a USD proforma and checks the count.",
        "Page 4 of 12 continues the procurement SOP.",
    )
    assert reason == "sentence2:page_marker"


def test_accepts_clean_gist() -> None:
    s1 = "Investigator obtains a USD or TZS proforma and checks type and count."
    s2 = "The investigator gets a proforma invoice and verifies quantity."
    assert pair_reject_reason(s1, s2) is None


def test_reject_identical_sides() -> None:
    text = "Investigator obtains a USD or TZS proforma and checks type and count."
    assert pair_reject_reason(text, text) == "identical"


def test_reject_generic_question() -> None:
    assert (
        question_reject_reason("What is this document about?") == "generic"
    )
    assert (
        question_reject_reason("What is the main research in this paper?")
        == "generic"
    )
    assert (
        question_reject_reason(
            "How does an investigator obtain a USD proforma?"
        )
        is None
    )

