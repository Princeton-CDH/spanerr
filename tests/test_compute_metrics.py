import json
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from spanerr.align import AlignSpans
from spanerr.compute_metrics import (
    ScoreAlignment,
    compute_document_metrics,
    compute_entity_metrics,
    compute_macro_metrics,
    get_aligner,
    get_scorer,
    get_span_alignments,
    main,
)
from spanerr.core import DocSpans, Span, SpanAlignment
from spanerr.span_utils import partial_overlap


@patch("spanerr.compute_metrics.construct_aligner", autospec=True)
def test_get_aligner(mock_constructor):
    mock_constructor.return_value = "result_aligner"
    # Unknown strategy
    err_msg = "Unknown alignment strategy: other"
    with pytest.raises(ValueError, match=err_msg):
        get_aligner("other")
    mock_constructor.assert_not_called()
    # select_first
    assert get_aligner("select_first") == "result_aligner"
    mock_constructor.assert_called_once_with("select_first", is_match=partial_overlap)
    # select_best
    mock_constructor.reset_mock()
    assert get_aligner("select_best") == "result_aligner"
    mock_constructor.assert_called_once_with(
        "select_best", is_match=partial_overlap, score_match=Span.jaccard
    )
    # corppa
    mock_constructor.reset_mock()
    assert get_aligner("corppa") == "result_aligner"
    mock_constructor.assert_called_once_with("corppa")


@patch("spanerr.compute_metrics.composite_match_score", autospec=True)
@patch("spanerr.compute_metrics.relevance_score", autospec=True)
def test_get_scorer(mock_relevance, mock_composite):
    mock_relevance.return_value = "relevance score"
    mock_composite.return_value = "match score"
    # Unknown strategy
    err_msg = "Unknown scoring strategy: other"
    with pytest.raises(ValueError, match=err_msg):
        get_scorer("other")
    mock_relevance.assert_not_called()
    mock_composite.assert_not_called()
    # overlap factor (with defaults)
    scorer = get_scorer("overlap_factor")
    assert callable(scorer)
    _ = scorer("span alignment")
    mock_relevance.assert_called_once()
    ## Check basic call args
    assert mock_relevance.call_args.args[0] == "span alignment"
    assert mock_relevance.call_args.kwargs == {"partial_weight": 1}
    ### Check function call arg
    match_score = mock_relevance.call_args.args[1]
    assert callable(match_score)
    _ = match_score("span_a", "span_b")
    mock_composite.assert_called_once_with("span_a", "span_b", Span.overlap_factor)
    # jaccard
    mock_relevance.reset_mock()
    mock_composite.reset_mock()
    scorer = get_scorer("jaccard")
    _ = scorer("span alignment")
    mock_relevance.assert_called_once()
    assert callable(scorer)
    ## Check basic call args
    assert mock_relevance.call_args.args[0] == "span alignment"
    assert mock_relevance.call_args.kwargs == {"partial_weight": 1}
    ### Check function call arg
    match_score = mock_relevance.call_args.args[1]
    assert callable(match_score)
    _ = match_score("span_a", "span_b")
    mock_composite.assert_called_once_with("span_a", "span_b", Span.jaccard)
    # set partial weight
    for strategy in ["overlap_factor", "jaccard"]:
        mock_relevance.reset_mock()
        scorer = get_scorer(strategy, partial_weight=0.5)
        _ = scorer("span alignment")
        assert mock_relevance.call_args.kwargs == {"partial_weight": 0.5}


@pytest.fixture
def ref_jsonl(tmp_path):
    """
    Create temporary reference span annotations file copied from test_data/ref.jsonl
    """
    test_dir = Path(__file__).resolve().parent
    ref_jsonl = test_dir / "test_data" / "ref.jsonl"
    tmp_ref = tmp_path / "mock_ref.jsonl"
    tmp_ref.write_bytes(ref_jsonl.read_bytes())
    return tmp_ref


@pytest.fixture
def sys_jsonl(tmp_path):
    """
    Create temporary reference span annotations file copied from test_data/sys.jsonl
    """
    test_dir = Path(__file__).resolve().parent
    sys_jsonl = test_dir / "test_data" / "sys.jsonl"
    # Create tmp system spans files
    tmp_sys = tmp_path / "mock_sys.jsonl"
    tmp_sys.write_bytes(sys_jsonl.read_bytes())
    return tmp_sys


# Mocking SpanAlignment since the objects are not hashable
@patch("spanerr.compute_metrics.SpanAlignment", autospec=True)
def test_get_span_alignments(mock_alignment, tmp_path):
    # Note: get_span_alignments is a generator so need to make sure to consume result
    # JSONL test files
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.touch()
    # Invalid ref and sys JSONL files
    with pytest.raises(ValueError, match="Reference annotations file does not exist"):
        list(get_span_alignments(tmp_path / "missing", empty_jsonl, "aligner"))
    with pytest.raises(ValueError, match="System annotations file does not exist"):
        list(get_span_alignments(empty_jsonl, tmp_path, "aligner"))

    # Ref and sys are 1-1 (don't make assumptions about yield order)
    mock_aligner = Mock(spec=AlignSpans, side_effect=lambda a, _: f"{a.doc_id}")
    ref_jsonl = tmp_path / "ref.jsonl"
    sys_jsonl = tmp_path / "sys.jsonl"
    ## same order
    ref_lines = [
        '{"doc_id":"a","spans":[{"start":1,"end":2}]}',
        '{"doc_id":"b","spans":[{"start":3,"end":4}]}',
    ]
    sys_lines = [
        '{"doc_id":"a","spans":[{"start":4,"end":5}]}',
        '{"doc_id":"b","spans":[{"start":6,"end":7}]}',
    ]
    ref_jsonl.write_text("\n".join(ref_lines) + "\n")
    sys_jsonl.write_text("\n".join(sys_lines) + "\n")
    ref_docspans = [DocSpans.from_dict(json.loads(l)) for l in ref_lines]
    sys_docspans = [DocSpans.from_dict(json.loads(l)) for l in sys_lines]
    result = get_span_alignments(ref_jsonl, sys_jsonl, mock_aligner)
    assert set(result) == {"a", "b"}
    assert mock_aligner.call_count == 2
    expected_calls = [
        call(ref_docspans[0], sys_docspans[0]),
        call(ref_docspans[1], sys_docspans[1]),
    ]
    mock_aligner.assert_has_calls(expected_calls, any_order=True)
    mock_alignment.assert_not_called()
    ## different order
    mock_aligner.reset_mock()
    sys_jsonl.write_text("\n".join(sys_lines[::-1]) + "\n")
    result = get_span_alignments(ref_jsonl, sys_jsonl, mock_aligner)
    assert set(result) == {"a", "b"}
    assert mock_aligner.call_count == 2
    mock_aligner.assert_has_calls(expected_calls, any_order=True)
    mock_alignment.assert_not_called()

    # Ref and sys are not 1-1
    ## No system spans
    mock_aligner.reset_mock()
    mock_alignment.side_effect = lambda a, b, c: a.doc_id
    result = get_span_alignments(ref_jsonl, empty_jsonl, mock_aligner)
    assert set(result) == {"a", "b"}
    mock_aligner.assert_not_called()
    assert mock_alignment.call_count == 2
    expected_calls = [
        call(ref_docspans[0], DocSpans("a", []), {}),
        call(ref_docspans[1], DocSpans("b", []), {}),
    ]
    mock_alignment.assert_has_calls(expected_calls, any_order=True)
    ### ignore unmatched
    mock_aligner.reset_mock()
    mock_alignment.reset_mock()
    result = get_span_alignments(
        ref_jsonl, empty_jsonl, mock_aligner, ignore_unmatched=True
    )
    assert list(result) == []
    mock_aligner.assert_not_called()
    mock_alignment.assert_not_called()

    ## No reference spans
    mock_aligner.reset_mock()
    result = get_span_alignments(empty_jsonl, sys_jsonl, mock_aligner)
    mock_alignment.reset_mock()
    assert set(result) == {"a", "b"}
    mock_aligner.assert_not_called()
    assert mock_alignment.call_count == 2
    expected_calls = [
        call(DocSpans("a", []), sys_docspans[0], {}),
        call(DocSpans("b", []), sys_docspans[1], {}),
    ]
    mock_alignment.assert_has_calls(expected_calls, any_order=True)
    ### ignore unmatched
    mock_aligner.reset_mock()
    mock_alignment.reset_mock()
    result = get_span_alignments(
        empty_jsonl, sys_jsonl, mock_aligner, ignore_unmatched=True
    )
    assert list(result) == []
    mock_aligner.assert_not_called()
    mock_alignment.assert_not_called()

    ## Mismatched
    ref_first_jsonl = tmp_path / "ref_first.jsonl"
    ref_first_jsonl.write_text(ref_lines[0] + "\n")
    sys_last_jsonl = tmp_path / "sys_last.jsonl"
    sys_last_jsonl.write_text(sys_lines[-1] + "\n")

    ### ref: 1 docs, sys: 2 docs
    mock_aligner.reset_mock()
    mock_alignment.reset_mock()
    result = get_span_alignments(ref_first_jsonl, sys_jsonl, mock_aligner)
    assert set(result) == {"a", "b"}
    mock_aligner.assert_called_once_with(ref_docspans[0], sys_docspans[0])
    mock_alignment.assert_called_once_with(DocSpans("b", []), sys_docspans[1], {})
    #### ignore unmatched
    mock_aligner.reset_mock()
    mock_alignment.reset_mock()
    result = get_span_alignments(
        ref_first_jsonl, sys_jsonl, mock_aligner, ignore_unmatched=True
    )
    assert set(result) == {"a"}
    mock_aligner.assert_called_once_with(ref_docspans[0], sys_docspans[0])
    mock_alignment.assert_not_called()

    ### ref: 2 docs, sys: 1 docs
    mock_aligner.reset_mock()
    mock_alignment.reset_mock()
    result = get_span_alignments(ref_jsonl, sys_last_jsonl, mock_aligner)
    assert set(result) == {"a", "b"}
    mock_aligner.assert_called_once_with(ref_docspans[1], sys_docspans[1])
    mock_alignment.assert_called_once_with(ref_docspans[0], DocSpans("a", []), {})
    #### ignore unmatched
    mock_aligner.reset_mock()
    mock_alignment.reset_mock()
    result = get_span_alignments(
        ref_jsonl, sys_last_jsonl, mock_aligner, ignore_unmatched=True
    )
    assert set(result) == {"b"}
    mock_aligner.assert_called_once_with(ref_docspans[1], sys_docspans[1])
    mock_alignment.assert_not_called()


@pytest.mark.parametrize(
    "jsonl_text,dup_id",
    [
        ['{"doc_id":"a","spans":[]}\n{"doc_id":"a","spans":[]}\n', "a"],
        ['{"spans":[]}\n{"spans":[]}\n', ""],
        ['{"doc_id":"","spans":[]}\n{"spans":[]}\n', ""],
        [
            '{"doc_id":"a","spans":[]}\n{"doc_id":"b","spans":[]}\n{"doc_id":"a","spans":[]}\n',
            "a",
        ],
        ['{"spans":[]}\n{"doc_id":"a","spans":[]}\n{"spans":[]}\n', ""],
    ],
)
def test_get_span_alignments_dup_doc_ids(tmp_path, jsonl_text, dup_id):
    # Create input JSONL files
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.touch()
    dup_jsonl = tmp_path / "dup.jsonl"
    dup_jsonl.write_text(jsonl_text)
    ## System annotations with duplicate doc_ids
    err_msg = f"Multiple system annotations with document id '{dup_id}'"
    with pytest.raises(ValueError, match=err_msg):
        list(get_span_alignments(empty_jsonl, dup_jsonl, "aligner"))
    ## Reference annotations with duplicate doc_ids
    err_msg = f"Multiple reference annotations with document id '{dup_id}'"
    with pytest.raises(ValueError, match=err_msg):
        list(get_span_alignments(dup_jsonl, empty_jsonl, "aligner"))


@patch("spanerr.compute_metrics.f_beta", autospec=True, return_value="f")
@patch("spanerr.compute_metrics.recall", autospec=True, return_value="r")
@patch("spanerr.compute_metrics.precision", autospec=True, return_value="p")
def test_compute_entity_metrics(mock_precision, mock_recall, mock_fscore):
    mock_scorer = Mock(autospec=ScoreAlignment, return_value=2)
    # Error: No alignments
    with pytest.raises(ValueError, match="Found no alignments to score"):
        compute_entity_metrics([], mock_scorer, show_progress=False)
    mock_scorer.assert_not_called()
    mock_precision.assert_not_called()
    mock_recall.assert_not_called()
    mock_fscore.assert_not_called()

    # Typical case with defaults (so F-1 score)
    ## Overall: n_ref_spans = 2, n_sys_spans = 1
    alignments = [
        SpanAlignment(DocSpans("a", []), DocSpans("a", []), {}),
        SpanAlignment(
            DocSpans("b", [Span(1, 2), Span(3, 4)]), DocSpans("b", [Span(4, 5)]), {}
        ),
    ]
    expected = {"n_docs": 2, "precision": "p", "recall": "r", "f-score": "f"}
    result = compute_entity_metrics(alignments, mock_scorer, show_progress=False)
    assert result == expected
    assert mock_scorer.call_count == 2
    mock_scorer.assert_has_calls([call(a) for a in alignments])
    mock_precision.assert_called_once_with(1, 2 * 2)
    mock_recall.assert_called_once_with(2, 2 * 2)
    mock_fscore.assert_called_once_with(1, "p", "r")

    # Typical case but using F-0.5 score (beta=0.5)
    mock_scorer.reset_mock()
    mock_precision.reset_mock()
    mock_recall.reset_mock()
    mock_fscore.reset_mock()
    ## Overall: n_ref_spans = 4, n_sys_spans = 6
    alignments.append(
        SpanAlignment(
            DocSpans("c", [Span(0, 1), Span(1, 2)]),
            DocSpans("c", [Span(0, 1), Span(1, 2), Span(2, 3), Span(3, 4), Span(4, 5)]),
            {},
        )
    )
    expected["n_docs"] = 3
    result = compute_entity_metrics(
        alignments, mock_scorer, beta=0.5, show_progress=False
    )
    assert result == expected
    assert mock_scorer.call_count == 3
    mock_scorer.assert_has_calls([call(a) for a in alignments])
    mock_precision.assert_called_once_with(6, 2 * 3)
    mock_recall.assert_called_once_with(4, 2 * 3)
    mock_fscore.assert_called_once_with(0.5, "p", "r")


@patch("spanerr.compute_metrics.f_beta", autospec=True)
@patch("spanerr.compute_metrics.recall", autospec=True)
@patch("spanerr.compute_metrics.precision", autospec=True)
def test_document_entity_metrics(mock_precision, mock_recall, mock_fscore):
    mock_scorer = Mock(autospec=ScoreAlignment, return_value="score")
    # Error: no alignments
    with pytest.raises(ValueError, match="Found no alignments to score"):
        compute_document_metrics([], mock_scorer, show_progress=False)
    mock_scorer.assert_not_called()
    mock_precision.assert_not_called()
    mock_recall.assert_not_called()
    mock_fscore.assert_not_called()

    # Typical case with defaults (so F-1 score)
    mock_precision.side_effect = [0.0, 0.1]
    mock_recall.side_effect = [0.4, 0.5]
    mock_fscore.side_effect = [0, 1]
    alignments = [
        ## n_ref_spans = 0, n_sys_spans = 0
        SpanAlignment(DocSpans("a", []), DocSpans("a", []), {}),
        ## n_ref_spans = 2, n_sys_spans = 1
        SpanAlignment(
            DocSpans("b", [Span(1, 2), Span(3, 4)]), DocSpans("b", [Span(4, 5)]), {}
        ),
    ]
    expected = {"n_docs": 2, "precision": 0.05, "recall": 0.45, "f-score": 0.5}
    result = compute_document_metrics(alignments, mock_scorer, show_progress=False)
    assert result == expected
    assert mock_scorer.call_count == 2
    mock_scorer.assert_has_calls([call(a) for a in alignments])
    assert mock_precision.call_count == 2
    mock_precision.assert_has_calls([call(0, "score"), call(1, "score")])
    assert mock_recall.call_count == 2
    mock_recall.assert_has_calls([call(0, "score"), call(2, "score")])
    assert mock_fscore.call_count == 2
    mock_fscore.assert_has_calls([call(1, 0.0, 0.4), call(1, 0.1, 0.5)])

    # Typical case but using F-0.5 score (beta=0.5)
    mock_scorer.reset_mock()
    mock_precision.reset_mock()
    mock_recall.reset_mock()
    mock_fscore.reset_mock()
    mock_precision.side_effect = [0.0, 0.1, 0.2]
    mock_recall.side_effect = [0.4, 0.5, 0.45]
    mock_fscore.side_effect = [0, 1, 0.2]
    alignments.append(
        # n_ref_spans = 2, n_sys_spans = 5
        SpanAlignment(
            DocSpans("c", [Span(0, 1), Span(1, 2)]),
            DocSpans("c", [Span(0, 1), Span(1, 2), Span(2, 3), Span(3, 4), Span(4, 5)]),
            {},
        )
    )
    expected = {
        "n_docs": 3,
        "precision": pytest.approx(0.1),
        "recall": 0.45,
        "f-score": pytest.approx(0.4),
    }
    result = compute_document_metrics(
        alignments, mock_scorer, beta=0.5, show_progress=False
    )
    assert result == expected
    assert mock_scorer.call_count == 3
    mock_scorer.assert_has_calls([call(a) for a in alignments])
    assert mock_precision.call_count == 3
    mock_precision.assert_has_calls(
        [call(0, "score"), call(1, "score"), call(5, "score")]
    )
    assert mock_recall.call_count == 3
    mock_recall.assert_has_calls([call(0, "score"), call(2, "score"), call(2, "score")])
    assert mock_fscore.call_count == 3
    mock_fscore.assert_has_calls(
        [call(0.5, 0.0, 0.4), call(0.5, 0.1, 0.5), call(0.5, 0.2, 0.45)]
    )


@patch(
    "spanerr.compute_metrics.compute_document_metrics",
    autospec=True,
    return_value="doc metrics",
)
@patch(
    "spanerr.compute_metrics.compute_entity_metrics",
    autospec=True,
    return_value="entity metrics",
)
@patch(
    "spanerr.compute_metrics.get_span_alignments",
    autospec=True,
    return_value="alignments",
)
def test_compute_macro_metrics(mock_alignments, mock_entity_metrics, mock_doc_metrics):
    # Unknown macro level
    with pytest.raises(ValueError, match="Unsupported macro level: unknown"):
        compute_macro_metrics("ref", "sys", "unknown", "aligner", "scorer")
    mock_alignments.assert_called_once_with("ref", "sys", "aligner")
    mock_entity_metrics.assert_not_called()
    mock_doc_metrics.assert_not_called()

    # Entity-level
    mock_alignments.reset_mock()
    result = compute_macro_metrics("ref", "sys", "entity", "aligner", "scorer")
    assert result == "entity metrics"
    mock_alignments.assert_called_once_with("ref", "sys", "aligner")
    mock_entity_metrics.assert_called_once_with(
        "alignments", "scorer", beta=1, show_progress=True
    )
    mock_doc_metrics.assert_not_called()
    ## Setting optional parameters
    mock_alignments.reset_mock()
    mock_entity_metrics.reset_mock()
    assert compute_macro_metrics(
        "ref", "sys", "entity", "aligner", "scorer", beta="float", show_progress="bool"
    )
    mock_alignments.assert_called_once_with("ref", "sys", "aligner")
    mock_entity_metrics.assert_called_once_with(
        "alignments", "scorer", beta="float", show_progress="bool"
    )
    mock_doc_metrics.assert_not_called()

    # Document-level
    mock_alignments.reset_mock()
    mock_entity_metrics.reset_mock()
    result = compute_macro_metrics("ref", "sys", "document", "aligner", "scorer")
    assert result == "doc metrics"
    mock_alignments.assert_called_once_with("ref", "sys", "aligner")
    mock_doc_metrics.assert_called_once_with(
        "alignments", "scorer", beta=1, show_progress=True
    )
    mock_entity_metrics.assert_not_called()
    ## Setting optional parameters
    mock_alignments.reset_mock()
    mock_doc_metrics.reset_mock()
    assert compute_macro_metrics(
        "ref",
        "sys",
        "document",
        "aligner",
        "scorer",
        beta="float",
        show_progress="bool",
    )
    mock_alignments.assert_called_once_with("ref", "sys", "aligner")
    mock_doc_metrics.assert_called_once_with(
        "alignments", "scorer", beta="float", show_progress="bool"
    )
    mock_entity_metrics.assert_not_called()


@pytest.mark.parametrize(
    "cli_args,call_params",
    [
        # all required params, default progress behavior
        [
            [
                "compute_metrics.py",
                "ref.jsonl",
                "sys.jsonl",
                "entity",
                "select_first",
                "overlap_factor",
            ],
            (
                [
                    Path("ref.jsonl"),
                    Path("sys.jsonl"),
                    "entity",
                    "select_first",
                    "overlap_factor",
                ],
                {"show_progress": True},
            ),
        ],
        # disable progress
        [
            [
                "compute_metrics.py",
                "ref.jsonl",
                "sys.jsonl",
                "document",
                "select_best",
                "jaccard",
                "--no-progress",
            ],
            (
                [
                    Path("ref.jsonl"),
                    Path("sys.jsonl"),
                    "document",
                    "select_best",
                    "jaccard",
                ],
                {"show_progress": False},
            ),
        ],
    ],
)
@patch("spanerr.compute_metrics.get_scorer", return_value="scorer")
@patch("spanerr.compute_metrics.get_aligner", return_value="aligner")
@patch("spanerr.compute_metrics.compute_macro_metrics")
def test_main(mock_metrics, mock_aligner, mock_scorer, cli_args, call_params, capsys):
    mock_metrics.return_value = {
        "n_docs": 0,
        "precision": 0.1,
        "recall": 0.1,
        "f-score": 0.1,
    }
    # patch in test args for argpars to parse
    with patch("sys.argv", cli_args):
        main()
        args, kwargs = call_params
        mock_aligner.assert_called_once_with(args[3])
        mock_scorer.assert_called_once_with(args[4])
        # Swap final args for the expected return values (based on patching)
        args[3] = "aligner"
        args[4] = "scorer"
        mock_metrics.assert_called_once_with(*args, **kwargs)
        # Check stdout output
        captured = capsys.readouterr()
        expected_reporting = "\n".join(
            [
                f"Macro {args[2]}-level metrics for 0 documents:",
                "- Precision = 0.1",
                "- Recall = 0.1",
                "- F1 = 0.1",
            ]
        )
        progress_pfx = "\n" if kwargs["show_progress"] else ""
        assert captured.out == f"{progress_pfx}{expected_reporting}\n"
