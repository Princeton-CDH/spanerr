"""
Library of methods for evaluating alignments of span annotations
"""

from spanerr.core import Span, SpanAlignment
from spanerr.span_utils import CheckSpanPair, ScoreSpanPair


def is_one_to_one(alignment: SpanAlignment) -> bool:
    """
    Tests if the alignment has a one-to-one mapping where each reference span (key)
    maps to (at most) a single distinct system span.
    """
    return all(len(sys_spans) == 1 for sys_spans in alignment.mapping.values()) and all(
        len(ref_spans) == 1 for ref_spans in alignment.reverse_mapping.values()
    )


def relevance_score(
    alignment: SpanAlignment,
    partial_score: ScoreSpanPair,
    partial_weight: float = 1,
    is_exact_match: CheckSpanPair = Span.__eq__,
) -> float:
    """
    Calculate the alignment's relevance score. This corresponds to the
    effective number of relevant spans retrieved (TP).

    The relevance score is computed as follows:
        # exact matches + partial_weight * sum of partial match scores

    The match scores for partial matches are determined by `partial_score`.
    By default, exact matches correspond to span equality (i.e., exact boundary
    and label match) but this can be customized via `is_exact_match`.
    """
    # Verify the weight is between 0 and 1
    if partial_weight < 0 or partial_weight > 1:
        raise ValueError("Partial weight must be between 0 and 1 (inclusive)")

    if is_one_to_one(alignment):
        score = 0
        for ref_span in alignment.mapping:
            sys_span = alignment.mapping[ref_span][0]
            if is_exact_match(ref_span, sys_span):
                score += 1
            else:
                score += partial_weight * partial_score(ref_span, sys_span)
        return score
    else:
        raise ValueError(
            "Unsupported alignment. Currently, cannot score alignments "
            "without one-to-one mappings."
        )


def precision(n_sys_spans: int, relevance_score: float):
    """
    Calculate precision. Returns 1 if there are no system spans.
    """
    return 1 if not n_sys_spans else relevance_score / n_sys_spans


def recall(n_ref_spans: int, relevance_score: float):
    """
    Calculate recall. Returns 1 if there are no reference spans.
    """
    return 1 if not n_ref_spans else relevance_score / n_ref_spans


def f_beta(beta: float, precision: float, recall: float):
    """
    Calculate F beta score. Beta indicates how many more times important recall
    is over precision. Returns 0 if precision and recall are 0.
    """
    if precision == 0 or recall == 0:
        return 0  # Since FP > 0 or FN > 0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
