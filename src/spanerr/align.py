"""
Library of methods for aligning document-level span annotations
"""

from spanerr.core import (
    CheckSpanPair,
    DocSpans,
    ScoreSpanPair,
    Span,
    SpanAlignment,
)
from spanerr.spans.match import partial_overlap


def select_first_match(
    ref: DocSpans,
    sys: DocSpans,
    is_match: CheckSpanPair,
    exclusive: bool = True,
) -> SpanAlignment:
    """
    Builds a span alignment using a select first match strategy. Each reference
    span is aligned with the first matching system span as determined by the provided
    `is_match` method. By default, alignments are exclusive.
    """
    align_map = {}
    sys_span_pool = dict.fromkeys(sys.spans)
    for ref_span in ref.spans:
        found_match = False
        for sys_span in sys_span_pool:
            if is_match(ref_span, sys_span):
                found_match = True
                break
        if found_match:
            align_map[ref_span] = [sys_span]
            if exclusive:
                # Remove system span from candidate pool
                del sys_span_pool[sys_span]
    return SpanAlignment(ref, sys, align_map)


def select_best_match(
    ref: DocSpans,
    sys: DocSpans,
    is_match: CheckSpanPair,
    score_match: ScoreSpanPair,
    exclusive: bool = True,
) -> SpanAlignment:
    """
    Builds a span alignment using a greedy select best match strategy. Each reference
    span is aligned with its best matching system span as defined by the provided
    `is_match` and `score_match` methods. By default, alignments are exclusive.
    """
    mapping = {}
    sys_span_pool = dict.fromkeys(sys.spans)
    for ref_span in ref.spans:
        best_span = None
        best_score = 0
        for sys_span in sys_span_pool:
            if is_match(ref_span, sys_span):
                new_score = score_match(ref_span, sys_span)
                if new_score > best_score:
                    # Update best span and score
                    best_span = sys_span
                    best_score = new_score
        # Add alignment if there's a match
        if best_span is not None:
            mapping[ref_span] = [best_span]
            if exclusive:
                # Remove system span from candidate pool
                del sys_span_pool[best_span]
    return SpanAlignment(ref, sys, mapping)


def corppa_align(
    ref: DocSpans,
    sys: DocSpans,
    is_match: CheckSpanPair = partial_overlap,
    score_match: ScoreSpanPair = Span.overlap_factor,
) -> SpanAlignment:
    """
    Builds a span alignment using `corppa`'s alignment strategy. Each reference
    span is aligned with its best matching system span as defined by the provided
    `is_match` and `score_match` methods. When a system span matches k > 1
    reference spans r_i, its split into multiple subspans with the following ranges:
        (s start, r_1 start), (r_1 start, r_2 start), ..., (r_k start, s end)

    Note: This strategy assumes that any reference spans matching the same system
    span do not overlap.
    """
    # Determine best system span for each reference span
    init_align = select_best_match(
        ref, sys, is_match=is_match, score_match=score_match, exclusive=False
    )
    sys_to_refs = init_align.reverse_mapping
    # Build final system spans and mapping
    final_mapping = {}
    ## Add unmatched system spans
    final_sys_spans = set(init_align.sys.spans) - sys_to_refs.keys()
    for sys_span, ref_spans in sys_to_refs.items():
        if len(ref_spans) == 1:
            final_sys_spans.add(sys_span)
            final_mapping[ref_spans[0]] = [sys_span]
        else:
            # Effectively, split system into k pieces (one for each reference span)
            for i, ref_span in enumerate(ref_spans):
                start = ref_span.start if i else sys_span.start
                if i == len(ref_spans) - 1:
                    # Final reference span, set end to the system span's end
                    end = sys_span.end
                else:
                    # Otherwise, set end to the start of the next (matched) reference span
                    next_ref_span = ref_spans[i + 1]
                    end = next_ref_span.start
                sub_span = Span(start, end, sys_span.label)
                final_sys_spans.add(sub_span)
                final_mapping[ref_span] = [sub_span]
    final_sys = DocSpans(sys.doc_id, final_sys_spans)
    return SpanAlignment(init_align.ref, final_sys, final_mapping)
