"""
Library of methods for comparing spans
"""

from spanerr.core import Span


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
