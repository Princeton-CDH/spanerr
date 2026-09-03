"""
Library of methods for checking for span-level matches
"""

from spanerr.core import Span


def exact_match(span_a: Span, span_b: Span, ignore_label: bool = False) -> bool:
    """
    Test for exact boundary match. By default, must have same label.
    Corresponds to strict and exact match in SemEval'13.
    """
    if ignore_label:
        return span_a.binarize() == span_b.binarize()
    else:
        return span_a == span_b


def partial_overlap(span_a: Span, span_b: Span, ignore_label: bool = False) -> bool:
    """
    Test for overlapping boundary match. By default, must have same label.
    Corresponds to type and partial match in SemEval'13.
    """
    if ignore_label or span_a.label == span_b.label:
        return span_a.overlap_length(span_b) > 0
    else:
        return False


def min_overlap_length(
    span_a: Span, span_b: Span, min_len: int, ignore_label: bool = False
) -> bool:
    """
    Test if spans overlap by at least `min_len` length.
    By default, must have same label.
    """
    if ignore_label or span_a.label == span_b.label:
        return span_a.overlap_length(span_b) >= min_len
    else:
        return False


def min_overlap_factor(
    span_a: Span, span_b: Span, min_val: float, ignore_label: bool = False
) -> bool:
    """
    Test if spans have an overlap factor of at least  `min_val`.
    By default, must have same label.
    """
    if ignore_label or span_a.label == span_b.label:
        return span_a.overlap_factor(span_b) >= min_val
    else:
        return False
