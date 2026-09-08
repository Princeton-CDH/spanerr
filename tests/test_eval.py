import re
from unittest.mock import Mock, call, patch

import pytest

from spanerr.core import Span, SpanAlignment
from spanerr.eval import (
    f_beta,
    is_one_to_one,
    precision,
    recall,
    relevance_score,
)
from spanerr.span_utils import (
    CheckSpanPair,
    ScoreSpanPair,
    min_overlap_length,
    partial_overlap,
)


def test_is_one_to_one():
    mock_alignment = Mock(spec=SpanAlignment)
    # one-to-one
    mapping = {Span(1, 2): [Span(2, 3)], Span(3, 4): [Span(4, 5)]}
    rev_mapping = {Span(2, 3): [Span(1, 2)], Span(4, 5): [Span(3, 4)]}
    mock_alignment.mapping = mapping
    mock_alignment.reverse_mapping = rev_mapping
    assert is_one_to_one(mock_alignment)
    # a reference span maps to multiple system spans
    mapping[Span(1, 2)].append(Span(5, 6))
    rev_mapping[Span(5, 6)] = [Span(1, 2)]
    assert not is_one_to_one(mock_alignment)
    # a system span maps to multiple reference spans
    mock_alignment.mapping = rev_mapping
    mock_alignment.reverse_mapping = mapping
    assert not is_one_to_one(mock_alignment)


@patch("spanerr.eval.is_one_to_one", autospec=True)
def test_relevance_score(mock_121):
    # initialize mock objects
    mock_align = Mock(spec=SpanAlignment)
    mock_score = Mock(spec=ScoreSpanPair, side_effect=Span.jaccard)
    # Bad weight
    err_msg = "Partial weight must be between 0 and 1 (inclusive)"
    for w in [-0.5, 1.5]:
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            relevance_score(mock_align, mock_score, partial_weight=w)
        mock_score.assert_not_called()
    # Unsupported alignment (does not have a one-to-one mapping)
    mock_121.return_value = False
    with pytest.raises(ValueError, match="Unsupported alignment"):
        relevance_score(mock_align, mock_score)
    mock_121.assert_called_once_with(mock_align)
    mock_score.assert_not_called()
    # Default case with supported alignment
    ## No matches
    mock_121.reset_mock()
    mock_121.return_value = True
    mock_align.mapping = {}
    assert relevance_score(mock_align, mock_score) == 0
    mock_121.assert_called_once_with(mock_align)
    mock_score.assert_not_called()
    ## Only exact matches
    exact_map = {Span(1, 3, "a"): [Span(1, 3, "a")], Span(3, 5, "b"): [Span(3, 5, "b")]}
    mock_align.mapping = exact_map
    assert relevance_score(mock_align, mock_score) == 2
    mock_score.assert_not_called()
    ## With partial matches
    partial_map = {Span(6, 10): [Span(7, 10)], Span(15, 20, "c"): [Span(18, 23, "c")]}
    mock_align.mapping = partial_map
    assert relevance_score(mock_align, mock_score) == 0.75 + 0.25
    assert mock_score.call_count == 2
    mock_score.assert_has_calls([call(key, val[0]) for key, val in partial_map.items()])
    ## Both exact and partial matches
    mock_score.reset_mock()
    both_map = exact_map | partial_map
    mock_align.mapping = both_map
    assert relevance_score(mock_align, mock_score) == 3
    assert mock_score.call_count == 2
    mock_score.assert_has_calls([call(key, val[0]) for key, val in partial_map.items()])
    # Customize partial weight
    assert relevance_score(mock_align, mock_score, partial_weight=0.5) == 2.5
    # Customize exact match
    mock_is_exact = Mock(spec=CheckSpanPair)
    ## Treat all partial matches as exact matches
    mock_score.reset_mock()
    mock_is_exact.side_effect = partial_overlap
    assert relevance_score(mock_align, mock_score, is_exact_match=mock_is_exact) == 4
    mock_score.assert_not_called()
    assert mock_is_exact.call_count == 4
    mock_is_exact.assert_has_calls([call(key, val[0]) for key, val in both_map.items()])
    ## Treat all partial matches with overlap length >= 3 as exact matches
    mock_is_exact.reset_mock()
    mock_is_exact.side_effect = lambda a, b: min_overlap_length(a, b, 3)
    assert relevance_score(mock_align, mock_score, is_exact_match=mock_is_exact) == 3.25


@pytest.mark.parametrize(
    "n_spans,rel_score,expected",
    [(0, 0, 1), (10, 0, 0), (4, 2, 0.5), (4, 4, 1.0), (12, 9, 0.75)],
)
def test_precision_recall(n_spans, rel_score, expected):
    assert precision(n_spans, rel_score) == expected
    assert recall(n_spans, rel_score) == expected


@pytest.mark.parametrize(
    "beta,precision,recall,expected",
    [
        # F-1
        (1, 0, 0, 0),
        (1, 0, 0.5, 0),
        (1, 0.2, 0, 0),
        (1, 1, 1, 1),
        (1, 0.5, 0.5, 0.5),
        (1, 0.75, 0.5, 0.6),
        (1, 0.5, 0.75, 0.6),
        # F-0.5
        (0.5, 0, 0, 0),
        (0.5, 0, 0.5, 0),
        (0.5, 0.2, 0, 0),
        (0.5, 1, 1, 1),
        (0.5, 0.5, 0.5, 0.5),
        (0.5, 0.75, 0.5, 0.681818),
        (0.5, 0.5, 0.75, 0.535714),
        # F-2
        (2, 0, 0, 0),
        (2, 0, 0.5, 0),
        (2, 0.2, 0, 0),
        (2, 1, 1, 1),
        (2, 0.5, 0.5, 0.5),
        (2, 0.75, 0.5, 0.535714),
        (2, 0.5, 0.75, 0.681818),
    ],
)
def test_f_beta_recall(beta, precision, recall, expected):
    # round to 6th decimal place
    assert round(f_beta(beta, precision, recall), 6) == expected
