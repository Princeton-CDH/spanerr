from spanerr.core import Span
from spanerr.spans.match import (
    exact_match,
    min_overlap_factor,
    min_overlap_length,
    partial_overlap,
)


def test_exact_match():
    # Default: Label sensitive
    assert exact_match(Span(2, 4, "a"), Span(2, 4, "a"))
    assert not exact_match(Span(2, 4, "a"), Span(2, 4, "b"))
    assert not exact_match(Span(1, 4), Span(2, 3))
    assert not exact_match(Span(1, 3, "i"), Span(3, 5, "i"))
    # Label insensitive
    assert exact_match(Span(2, 4, "a"), Span(2, 4, "b"), ignore_label=True)
    assert not exact_match(Span(1, 4), Span(2, 3, "o"), ignore_label=True)
    assert not exact_match(Span(1, 3, "i"), Span(3, 5, "j"), ignore_label=True)


def test_partial_overlap():
    # Default: Label sensitive
    assert partial_overlap(Span(2, 4, "a"), Span(2, 4, "a"))
    assert not partial_overlap(Span(1, 3, "i"), Span(3, 5, "i"))
    assert partial_overlap(Span(1, 4), Span(2, 3))
    assert not partial_overlap(Span(0, 5, "a"), Span(2, 10, "b"))
    # Label insensitive
    assert partial_overlap(Span(2, 4, "a"), Span(2, 4, "b"), ignore_label=True)
    assert not partial_overlap(Span(1, 3, "i"), Span(3, 5, "j"), ignore_label=True)
    assert partial_overlap(Span(0, 5, "a"), Span(2, 10, "b"), ignore_label=True)


def test_min_overlap_length():
    # Label sensitive
    assert min_overlap_length(Span(2, 4, "a"), Span(2, 4, "a"), 2)
    assert not min_overlap_length(Span(2, 4, "a"), Span(2, 4, "b"), 2)
    assert not min_overlap_length(Span(2, 4, "a"), Span(2, 4, "a"), 4)
    assert min_overlap_length(Span(1, 4), Span(3, 5), 1)
    assert not min_overlap_length(Span(1, 4), Span(3, 5), 10)
    assert min_overlap_length(Span(0, 2), Span(2, 4), 0)
    assert not min_overlap_length(Span(0, 2), Span(2, 4, "o"), 0)
    # Label insensitive
    assert min_overlap_length(Span(2, 4, "a"), Span(2, 4, "b"), 2, ignore_label=True)
    assert not min_overlap_length(
        Span(2, 4, "a"), Span(2, 4, "b"), 4, ignore_label=True
    )
    assert min_overlap_length(Span(1, 4), Span(3, 5, "o"), 1, ignore_label=True)
    assert not min_overlap_length(Span(1, 4), Span(3, 5, "o"), 10, ignore_label=True)
    assert min_overlap_length(Span(0, 2), Span(2, 4, "o"), 0, ignore_label=True)


def test_min_overlap_factor():
    # Label sensitive
    assert min_overlap_factor(Span(2, 4, "a"), Span(2, 4, "a"), 1)
    assert min_overlap_factor(Span(1, 4), Span(3, 5), 0.25)
    assert not min_overlap_factor(Span(1, 4), Span(3, 5), 0.5)
    assert min_overlap_factor(Span(0, 2), Span(2, 4), 0)
    assert not min_overlap_factor(Span(0, 2), Span(2, 4, "o"), 0)
    # Label insensitive
    assert min_overlap_factor(Span(2, 4, "a"), Span(2, 4, "b"), 1, ignore_label=True)
    assert min_overlap_factor(Span(1, 4), Span(3, 5, "o"), 0.25, ignore_label=True)
    assert not min_overlap_factor(Span(1, 4), Span(3, 5, "o"), 0.5, ignore_label=True)
    assert min_overlap_factor(Span(0, 2), Span(2, 4, "o"), 0, ignore_label=True)
