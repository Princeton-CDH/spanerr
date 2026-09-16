"""
Script for computing aggregated span evaluation metrics for a given reference
and system span annotations as JSONL files.

Currently this script supports the following functionality:
  * Computing entity- or document-level aggregated metrics (macro level)
  * Using the following span alignment strategies:
    * select_first : Select first match where matching spans must have the same
        label and overlapping boundaries.
    * select_best : Select the best match where matching spans must have the
        same label and overlapping boundaries and the best match corresponds to
        the match with the largest jaccard similarity.
      * corppa : The alignment strategy used by corppa.
  * Using the following scoring strategies:
      * jaccard: spans' jaccard similarity if their labels match, 0 otherwise.
      * overlap_factor: spans' overlap factor if their labels match, 0 otherwise.


This script takes the following parameters:
  1. reference span annotations JSONL file
  2. system span annotations JSONL
  3. macro-level in which metrics will be aggregated
  4. span alignment strategy for computing metrics
  5. scoring strategy for scoring span alignments

Example usage:

    compute_metrics.py ref.jsonl sys.jsonl entity select_first jaccard

    compute_metrics.py ref.jsonl sys.jsonl document corppa overlap_factor --no-progress
"""

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path

import orjsonl
from tqdm import tqdm

from spanerr.align import AlignSpans, construct_aligner
from spanerr.core import DocSpans, Span, SpanAlignment
from spanerr.eval import (
    f_beta,
    precision,
    recall,
    relevance_score,
)
from spanerr.span_utils import composite_match_score, partial_overlap

# Custom function type
ScoreAlignment = Callable[[SpanAlignment], float]


def get_aligner(
    strategy: str,
) -> AlignSpans:
    """
    Returns alignment method for the specified alignment strategy.
    Currently, the supported strategies use the following defaults:
        - select_first:
            * span match: same label and overlapping boundaries (spanerr.span_utils.partial_overlap)
        - select_best:
            * span match: same label and overlapping boundaries (spanerr.span_utils.partial_overlap)
            * match score: jaccard similarity
        - corppa: uses strategy's defaults
    """
    match strategy:
        case "select_first":
            return construct_aligner("select_first", is_match=partial_overlap)
        case "select_best":
            return construct_aligner(
                "select_best", is_match=partial_overlap, score_match=Span.jaccard
            )
        case "corppa":
            # Uses corppa_align's defaults
            return construct_aligner("corppa")
        case _:
            raise ValueError(f"Unknown alignment strategy: {strategy}")


def get_scorer(
    strategy: str,
    partial_weight: float = 1,
) -> ScoreAlignment:
    """
    Returns a method for calculating an alignment's relevance score using
    the provided scoring strategy and optionally a partial weight.

    Currently supports the following scoring strategies:
        - overlap_factor: spans' overlap factor if their labels match, 0 otherwise
        - jaccard: spans' jaccard similarity if their labels match, 0 otherwise
    """
    match strategy:
        case "overlap_factor":
            score_bounds = Span.overlap_factor
        case "jaccard":
            score_bounds = Span.jaccard
        case _:
            raise ValueError(f"Unknown scoring strategy: {strategy}")
    match_score = lambda a, b: composite_match_score(a, b, score_bounds)
    return lambda x: relevance_score(x, match_score, partial_weight=partial_weight)


def get_span_alignments(
    ref_file: Path,
    sys_file: Path,
    aligner: AlignSpans,
    ignore_unmatched: bool = False,
) -> Iterator[SpanAlignment]:
    """
    Yields document-level span alignments given two sets of span annotations (JSONL).
    By default, any documents represented by only a reference or system annotation
    are included by assuming its missing pair contains no span annotations.
    """
    # Validate input files
    if not ref_file.is_file():
        raise ValueError("Reference annotations file does not exist")
    if not sys_file.is_file():
        raise ValueError("System annotations file does not exist")

    # Read in system annotations
    sys_annos = {}
    for sys_dict in orjsonl.stream(sys_file):
        anno = DocSpans.from_dict(sys_dict)  # ty: ignore[invalid-argument-type]
        doc_id = anno.doc_id
        # Validate system annotations by checking for duplicate doc ids
        if doc_id in sys_annos:
            # Raise an error if a doc id is encountered more than once
            raise ValueError(f"Multiple system annotations with document id '{doc_id}'")
        sys_annos[doc_id] = anno
    # Then for each reference annotations, build corresponding alignment
    ref_doc_ids = set()  # for tracking encountered reference doc ids
    for ref_json in orjsonl.stream(ref_file):
        ref_anno = DocSpans.from_dict(ref_json)  # ty: ignore[invalid-argument-type]
        doc_id = ref_anno.doc_id
        # Validate reference annotations by checking for duplicate doc ids
        if doc_id in ref_doc_ids:
            # Raise an error if a doc id is encountered more than once
            raise ValueError(
                f"Multiple reference annotations with document id '{doc_id}'"
            )
        if doc_id not in sys_annos:
            # Optionally add unmatched reference annotation
            if ignore_unmatched:
                continue
            yield SpanAlignment(ref_anno, DocSpans(doc_id, []), {})
        else:
            # Remove matched system annotations
            yield aligner(ref_anno, sys_annos.pop(doc_id))
        ref_doc_ids.add(doc_id)
    # Then optionally add unmatched system annotations
    if not ignore_unmatched:
        for doc_id, sys_anno in sys_annos.items():
            yield SpanAlignment(DocSpans(doc_id, []), sys_anno, {})


def compute_entity_metrics(
    alignments: Iterator[SpanAlignment],
    scorer: ScoreAlignment,
    beta: float = 1,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Computes entity-level precision, recall, and F-score for series of SpanAlignments.
    By default, computes F1 scores.

    Returns results as a dict with the following fields:
        - n_docs = number of documents
        - precision = entity-level precision
        - recall = entity-level recall
        - f-score = entity-level F-score
    """
    if show_progress:
        print("Computing entity-level metrics...")
    # Compute aggregated statistics needed to calculate precision and recall
    n_docs = 0
    total_relevance = 0
    total_sys_spans = 0
    total_ref_spans = 0
    progress = tqdm(alignments, desc="Scoring alignments", disable=not show_progress)
    for a in progress:
        n_docs += 1
        rel_score = scorer(a)
        n_sys_spans = len(a.sys.spans)
        n_ref_spans = len(a.ref.spans)
        if show_progress:
            tqdm.write(
                f"  * {a.ref.doc_id}: relevance = {rel_score:.4g} | "
                f"{n_ref_spans} ref spans | {n_sys_spans} sys spans"
            )
        total_relevance += rel_score
        total_sys_spans += n_sys_spans
        total_ref_spans += n_ref_spans
    # Raise error if iterator contains no alignments
    if n_docs == 0:
        raise ValueError("Found no alignments to score")
    # Compute entity-level precision and recall
    avg_precision = precision(total_sys_spans, total_relevance)
    avg_recall = recall(total_ref_spans, total_relevance)
    avg_fscore = f_beta(beta, avg_precision, avg_recall)
    return {
        "n_docs": n_docs,
        "precision": avg_precision,
        "recall": avg_recall,
        "f-score": avg_fscore,
    }


def compute_document_metrics(
    alignments: Iterator[SpanAlignment],
    scorer: ScoreAlignment,
    beta: float = 1,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Computes document-level precision, recall, and F-score. By default,
    computes F1 scores.

    Returns results as a dict with the following fields:
        - n_docs = number of documents
        - precision = document-level precision
        - recall = document-level recall
        - f-score = document-level F-score
    """
    if show_progress:
        print("Computing document-level metrics...")
    # Compute aggregated statistics needed to calculate precision and recall
    n_docs = 0
    cumulative_precision = 0
    cumulative_recall = 0
    cumulative_fscore = 0
    progress = tqdm(alignments, desc="Scoring alignments", disable=not show_progress)
    for a in progress:
        n_docs += 1
        relevance = scorer(a)
        # Compute and accumulate precision and recall
        doc_precision = precision(len(a.sys.spans), relevance)
        doc_recall = recall(len(a.ref.spans), relevance)
        doc_fscore = f_beta(beta, doc_precision, doc_recall)
        if show_progress:
            tqdm.write(
                f"  * {a.ref.doc_id}: precision = {doc_precision:.4g} | "
                f"recall = {doc_recall:.4g} | F-{beta} = {doc_fscore:.4g}"
            )
        # Add to running totals
        cumulative_precision += doc_precision
        cumulative_recall += doc_recall
        cumulative_fscore += doc_fscore
    # Raise error if iterator contains no alignments
    if n_docs == 0:
        raise ValueError("Found no alignments to score")
    # Compute entity-level precision and recall
    avg_precision = cumulative_precision / n_docs
    avg_recall = cumulative_recall / n_docs
    avg_fscore = cumulative_fscore / n_docs
    return {
        "n_docs": n_docs,
        "precision": avg_precision,
        "recall": avg_recall,
        "f-score": avg_fscore,
    }


def compute_macro_metrics(
    ref_jsonl: Path,
    sys_jsonl: Path,
    macro_level: str,
    aligner: AlignSpans,
    scorer: ScoreAlignment,
    beta: float = 1,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute aggregated precision, recall, and F-score at the specified macro level.
    By default, computes F1-scores.

    Returns the following tuple:
        - number of documents: int
        - macro precision: float
        - macro recall: float
        - macro F-score: float
    """
    # Step 1: Span Alignments
    alignments = get_span_alignments(ref_jsonl, sys_jsonl, aligner)
    # Step 2: Compute macro metrics
    match macro_level:
        case "entity":
            return compute_entity_metrics(
                alignments, scorer, beta=beta, show_progress=show_progress
            )
        case "document":
            return compute_document_metrics(
                alignments, scorer, beta=beta, show_progress=show_progress
            )
        case _:
            raise ValueError(f"Unsupported macro level: {macro_level}")


def main():
    """
    Command-line access to computing macro-level evaluation metrics for a given pair of
    reference and system span annotations.
    """
    parser = argparse.ArgumentParser(
        description="Calculate aggregated span evaluation metrics"
    )
    # Required arguments
    parser.add_argument(
        "ref_jsonl",
        help="Path to reference span annotations (JSONL file)",
        type=Path,
    )
    parser.add_argument(
        "sys_jsonl",
        help="Path to system span annotations (JSONL file)",
        type=Path,
    )
    parser.add_argument(
        "macro_level",
        choices=["entity", "document"],
        help="Level in which the evaluation metrics are aggregated",
    )
    parser.add_argument(
        "alignment_method",
        choices=["select_first", "select_best", "corppa"],
        help="Strategy for aligning span annotations",
    )
    parser.add_argument(
        "scoring_method",
        choices=["jaccard", "overlap_factor"],
        help="Strategy for scoring partial matches",
    )
    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    # Compute aggregated evaluation metrics
    results = compute_macro_metrics(
        args.ref_jsonl,
        args.sys_jsonl,
        args.macro_level,
        get_aligner(args.alignment_method),
        get_scorer(args.scoring_method),
        show_progress=args.progress,
    )
    if args.progress:
        print()
    print(
        f"Macro {args.macro_level}-level metrics for {results['n_docs']:d} documents:"
    )
    print(f"- Precision = {results['precision']:.4g}")
    print(f"- Recall = {results['recall']:.4g}")
    print(f"- F1 = {results['f-score']:.4g}")


if __name__ == "__main__":
    main()
