"""
Library of methods for comparing spans
"""

from collections.abc import Callable

from spanerr.core import Span

# Custom function types
CheckSpanPair = Callable[[Span, Span], bool]
ScoreSpanPair = Callable[[Span, Span], float]
ScoreLabelPair = Callable[[str, str], float]


def exact_match(span_a: Span, span_b: Span) -> bool:
    """
    Test for an exact match where there there is an exact boundary match and
    label match. This corresponds to a strict match in SemEval'13.
    """
    return span_a == span_b


def partial_overlap(span_a: Span, span_b: Span) -> bool:
    """
    Test for an overlapping boundary match and label match. This corresponds to
    to a type match in SemEval'13.
    """
    return span_a.label == span_b.label and span_a.overlap_length(span_b) > 0


def min_overlap_length(span_a: Span, span_b: Span, min_len: int) -> bool:
    """
    Test for a label match and an overlap of at least `min_len` length.
    """
    return span_a.label == span_b.label and span_a.overlap_length(span_b) >= min_len


def min_overlap_factor(span_a: Span, span_b: Span, min_val: float) -> bool:
    """
    Tests for a label match and an overlap factor of at least  `min_val`.
    """
    return span_a.label == span_b.label and span_a.overlap_factor(span_b) >= min_val


def composite_match_score(
    span_a: Span,
    span_b: Span,
    score_bounds: ScoreSpanPair,
    score_label: ScoreLabelPair = str.__eq__,
) -> float:
    """
    Computes a composite match score that is the product of the two spans' boundary
    and label similarities. Boundary similarity is calculated using `score_bounds`.
    By default, label similarity is simple string equality but can be customized
    using `score_label`.
    """
    return score_label(span_a.label, span_b.label) * score_bounds(span_a, span_b)
