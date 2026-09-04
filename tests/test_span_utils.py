from unittest.mock import Mock

import pytest

from spanerr.core import Span
from spanerr.span_utils import (
    ScoreLabelPair,
    ScoreSpanPair,
    composite_match_score,
    exact_match,
    min_overlap_factor,
    min_overlap_length,
    partial_overlap,
)


@pytest.mark.parametrize(
    "span_a,span_b,expected",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), True),
        (Span(2, 4, "a"), Span(2, 4, "b"), False),
        (Span(1, 4), Span(2, 3), False),
        (Span(1, 4), Span(2, 3, "o"), False),
        (Span(1, 3, "i"), Span(3, 5, "i"), False),
        (Span(1, 3, "i"), Span(3, 5, "j"), False),
    ],
)
def test_exact_match(span_a, span_b, expected):
    assert exact_match(span_a, span_b) == expected


@pytest.mark.parametrize(
    "span_a,span_b,expected",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), True),
        (Span(2, 4, "a"), Span(2, 4, "b"), False),
        (Span(1, 4), Span(2, 3), True),
        (Span(1, 4), Span(2, 3, "o"), False),
        (Span(1, 3, "i"), Span(3, 5, "i"), False),
        (Span(1, 3, "i"), Span(3, 5, "j"), False),
    ],
)
def test_partial_overlap(span_a, span_b, expected):
    assert partial_overlap(span_a, span_b) == expected


@pytest.mark.parametrize(
    "span_a,span_b,min_val,expected",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), 2, True),
        (Span(2, 4, "a"), Span(2, 4, "b"), 2, False),
        (Span(2, 4, "a"), Span(2, 4, "a"), 4, False),
        (Span(2, 4, "a"), Span(2, 4, "b"), 4, False),
        (Span(1, 4), Span(3, 5), 1, True),
        (Span(1, 4), Span(3, 5, "o"), 1, False),
        (Span(1, 4), Span(3, 5), 10, False),
        (Span(1, 4), Span(3, 5, "o"), 10, False),
        (Span(0, 2), Span(2, 4), 0, True),
        (Span(0, 2), Span(2, 4, "o"), 0, False),
    ],
)
def test_min_overlap_length(span_a, span_b, min_val, expected):
    assert min_overlap_length(span_a, span_b, min_val) == expected


@pytest.mark.parametrize(
    "span_a,span_b,min_val,expected",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), 1, True),
        (Span(2, 4, "a"), Span(2, 4, "b"), 1, False),
        (Span(1, 4), Span(3, 5), 0.25, True),
        (Span(1, 4), Span(3, 5, "o"), 0.25, False),
        (Span(1, 4), Span(3, 5), 0.5, False),
        (Span(1, 4), Span(3, 5, "o"), 0.5, False),
        (Span(0, 2), Span(2, 4), 0, True),
        (Span(0, 2), Span(2, 4, "o"), 0, False),
    ],
)
def test_min_overlap_factor(span_a, span_b, min_val, expected):
    # Label sensitive
    assert min_overlap_factor(span_a, span_b, min_val) == expected


@pytest.mark.parametrize(
    "span_a,span_b,expected_default, expected_custom",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), 1, 1),
        (Span(2, 4, "a"), Span(2, 4, "b"), 0, 0),
        (Span(2, 4, "a"), Span(2, 4, "A"), 0, 1),
        (Span(1, 4), Span(3, 5), 0.25, 0.25),
        (Span(1, 4, "i"), Span(3, 5, "i"), 0.25, 0.25),
        (Span(1, 4, "I"), Span(3, 5, "i"), 0, 0.25),
        (Span(1, 4), Span(3, 5, "o"), 0, 0),
        (Span(0, 2), Span(2, 4), 0, 0),
        (Span(0, 2, "o"), Span(2, 4, "O"), 0, 0),
    ],
)
def test_composite_match_score(span_a, span_b, expected_default, expected_custom):
    mock_score_bounds = Mock(spec=ScoreSpanPair, side_effect=Span.jaccard)
    # Default label scoring
    assert composite_match_score(span_a, span_b, mock_score_bounds) == expected_default
    mock_score_bounds.assert_called_once_with(span_a, span_b)
    # Custom label scoring
    mock_score_bounds.reset_mock()
    mock_score_label = Mock(
        spec=ScoreLabelPair, side_effect=lambda a, b: a.lower() == b.lower()
    )
    result = composite_match_score(
        span_a, span_b, mock_score_bounds, score_label=mock_score_label
    )
    assert result == expected_custom
    mock_score_label.assert_called_once_with(span_a.label, span_b.label)
    mock_score_bounds.assert_called_once_with(span_a, span_b)
