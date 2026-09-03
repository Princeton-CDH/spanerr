import pytest

from spanerr.core import Span
from spanerr.span_utils import (
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
