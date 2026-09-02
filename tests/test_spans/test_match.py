import pytest

from spanerr.core import Span
from spanerr.spans.match import (
    exact_match,
    min_overlap_factor,
    min_overlap_length,
    partial_overlap,
)


@pytest.mark.parametrize(
    "span_a,span_b,expected_default,expected_flag",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), True, True),
        (Span(2, 4, "a"), Span(2, 4, "b"), False, True),
        (Span(1, 4), Span(2, 3), False, False),
        (Span(1, 4), Span(2, 3, "o"), False, False),
        (Span(1, 3, "i"), Span(3, 5, "i"), False, False),
        (Span(1, 3, "i"), Span(3, 5, "j"), False, False),
    ],
)
def test_exact_match(span_a, span_b, expected_default, expected_flag):
    # Default: Label sensitive
    assert exact_match(span_a, span_b) == expected_default
    # Label insensitive
    assert exact_match(span_a, span_b, ignore_label=True) == expected_flag


@pytest.mark.parametrize(
    "span_a,span_b,expected_default,expected_flag",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), True, True),
        (Span(2, 4, "a"), Span(2, 4, "b"), False, True),
        (Span(1, 4), Span(2, 3), True, True),
        (Span(1, 4), Span(2, 3, "o"), False, True),
        (Span(1, 3, "i"), Span(3, 5, "i"), False, False),
        (Span(1, 3, "i"), Span(3, 5, "j"), False, False),
    ],
)
def test_partial_overlap(span_a, span_b, expected_default, expected_flag):
    # Default: Label sensitive
    assert partial_overlap(span_a, span_b) == expected_default
    # Label insensitive
    assert partial_overlap(span_a, span_b, ignore_label=True) == expected_flag


@pytest.mark.parametrize(
    "span_a,span_b,min_val,expected_default,expected_flag",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), 2, True, True),
        (Span(2, 4, "a"), Span(2, 4, "b"), 2, False, True),
        (Span(2, 4, "a"), Span(2, 4, "a"), 4, False, False),
        (Span(2, 4, "a"), Span(2, 4, "b"), 4, False, False),
        (Span(1, 4), Span(3, 5), 1, True, True),
        (Span(1, 4), Span(3, 5, "o"), 1, False, True),
        (Span(1, 4), Span(3, 5), 10, False, False),
        (Span(1, 4), Span(3, 5, "o"), 10, False, False),
        (Span(0, 2), Span(2, 4), 0, True, True),
        (Span(0, 2), Span(2, 4, "o"), 0, False, True),
    ],
)
def test_min_overlap_length(span_a, span_b, min_val, expected_default, expected_flag):
    # Label sensitive
    assert min_overlap_length(span_a, span_b, min_val) == expected_default
    # Label insensitive
    assert (
        min_overlap_length(span_a, span_b, min_val, ignore_label=True) == expected_flag
    )


@pytest.mark.parametrize(
    "span_a,span_b,min_val,expected_default,expected_flag",
    [
        (Span(2, 4, "a"), Span(2, 4, "a"), 1, True, True),
        (Span(2, 4, "a"), Span(2, 4, "b"), 1, False, True),
        (Span(1, 4), Span(3, 5), 0.25, True, True),
        (Span(1, 4), Span(3, 5, "o"), 0.25, False, True),
        (Span(1, 4), Span(3, 5), 0.5, False, False),
        (Span(1, 4), Span(3, 5, "o"), 0.5, False, False),
        (Span(0, 2), Span(2, 4), 0, True, True),
        (Span(0, 2), Span(2, 4, "o"), 0, False, True),
    ],
)
def test_min_overlap_factor(span_a, span_b, min_val, expected_default, expected_flag):
    # Label sensitive
    assert min_overlap_factor(span_a, span_b, min_val) == expected_default
    # Label insensitive
    assert (
        min_overlap_factor(span_a, span_b, min_val, ignore_label=True) == expected_flag
    )
