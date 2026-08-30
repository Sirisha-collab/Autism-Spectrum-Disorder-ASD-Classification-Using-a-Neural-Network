"""Scoring rules are the one thing that must never silently change."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from questionnaire import QCHAT10, score_answers, score_item


def test_ten_items_present():
    assert len(QCHAT10) == 10
    assert [f[0] for f in QCHAT10] == [f"A{i}" for i in range(1, 11)]


def test_every_item_has_five_options():
    for feat, _q, options, _s in QCHAT10:
        assert len(options) == 5, feat


@pytest.mark.parametrize("feat", [f"A{i}" for i in range(1, 10)])
def test_items_one_to_nine_score_the_last_three_options(feat):
    """Items 1-9: the three least typical responses score 1."""
    assert score_item(feat, 0) == 0
    assert score_item(feat, 1) == 0
    assert score_item(feat, 2) == 1
    assert score_item(feat, 3) == 1
    assert score_item(feat, 4) == 1


def test_item_ten_is_reversed():
    """Item 10 asks about an atypical behaviour, so the scoring flips."""
    assert score_item("A10", 0) == 1
    assert score_item("A10", 2) == 1
    assert score_item("A10", 3) == 0
    assert score_item("A10", 4) == 0


def test_total_is_the_sum_of_items():
    answers = {f"A{i}": 4 for i in range(1, 10)} | {"A10": 0}
    scores = score_answers(answers)
    assert scores["total"] == 10
    assert scores["total"] == sum(scores[f"A{i}"] for i in range(1, 11))


def test_minimum_score():
    answers = {f"A{i}": 0 for i in range(1, 10)} | {"A10": 4}
    assert score_answers(answers)["total"] == 0


def test_out_of_range_option_rejected():
    with pytest.raises(ValueError):
        score_item("A1", 9)


def test_unknown_item_rejected():
    with pytest.raises(KeyError):
        score_item("A99", 0)
