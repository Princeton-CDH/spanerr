from unittest.mock import Mock

from spanerr.align import (
    corppa_align,
    select_best_match,
    select_first_match,
)
from spanerr.core import (
    CheckSpanPair,
    DocSpans,
    ScoreSpanPair,
    Span,
    SpanAlignment,
)
from spanerr.spans.match import partial_overlap


def test_select_first_match():
    mock_is_match = Mock(spec=CheckSpanPair)
    ref = DocSpans("", [Span(0, 1, "a"), Span(2, 3, "b")])
    sys = DocSpans("", [Span(4, 5, "c"), Span(5, 6, "d"), Span(6, 7, "e")])
    # No matches
    mock_is_match.return_value = False
    assert select_first_match(ref, sys, mock_is_match) == SpanAlignment(ref, sys, {})
    assert mock_is_match.call_count == 6
    # All match
    ## Default, exclusive matching
    mock_is_match.reset_mock()
    mock_is_match.return_value = True
    expected_map = {
        Span(0, 1, "a"): [Span(4, 5, "c")],
        Span(2, 3, "b"): [Span(5, 6, "d")],
    }
    assert select_first_match(ref, sys, mock_is_match) == SpanAlignment(
        ref, sys, expected_map
    )
    assert mock_is_match.call_count == 2
    ## Non-exclusive matching
    mock_is_match.reset_mock()
    expected_map = {
        Span(0, 1, "a"): [Span(4, 5, "c")],
        Span(2, 3, "b"): [Span(4, 5, "c")],
    }
    result = select_first_match(ref, sys, mock_is_match, exclusive=False)
    assert result == SpanAlignment(ref, sys, expected_map)
    assert mock_is_match.call_count == 2
    # Mixture
    ## Default, exclusive matching
    mock_is_match.reset_mock()
    mock_is_match.side_effect = [False, True, False, True]
    expected_map = {
        Span(0, 1, "a"): [Span(5, 6, "d")],
        Span(2, 3, "b"): [Span(6, 7, "e")],
    }
    assert select_first_match(ref, sys, mock_is_match) == SpanAlignment(
        ref, sys, expected_map
    )
    assert mock_is_match.call_count == 4
    ## Non-exclusive matching
    mock_is_match.reset_mock()
    mock_is_match.side_effect = [False, True, False, True]
    expected_map[Span(2, 3, "b")] = [Span(5, 6, "d")]
    result = select_first_match(ref, sys, mock_is_match, exclusive=False)
    assert result == SpanAlignment(ref, sys, expected_map)
    assert mock_is_match.call_count == 4


def test_select_best_match():
    mock_is_match = Mock(spec=CheckSpanPair)
    mock_score_match = Mock(spec=ScoreSpanPair, side_effect=Span.overlap_length)
    # No matches
    ref = DocSpans("", [Span(0, 1, "a"), Span(2, 3, "b")])
    sys = DocSpans("", [Span(4, 5, "c"), Span(5, 6, "d"), Span(6, 7, "e")])
    mock_is_match.return_value = False
    result = select_best_match(ref, sys, mock_is_match, mock_score_match)
    assert result == SpanAlignment(ref, sys, {})
    assert mock_is_match.call_count == 6
    mock_score_match.assert_not_called()
    # Simple case: select best match
    ref = DocSpans("", [Span(3, 8, "a")])
    sys = DocSpans(
        "", [Span(1, 4, "a"), Span(4, 7, "a"), Span(7, 10, "a"), Span(3, 8, "c")]
    )
    ## Label sensitive
    mock_is_match.reset_mock(return_value=True)
    mock_is_match.side_effect = partial_overlap
    expected_map = {Span(3, 8, "a"): [Span(4, 7, "a")]}
    result = select_best_match(ref, sys, mock_is_match, mock_score_match)
    assert result == SpanAlignment(ref, sys, expected_map)
    assert mock_is_match.call_count == 4
    assert mock_score_match.call_count == 3
    ## Label insensitive
    mock_is_match.reset_mock()
    mock_is_match.side_effect = Span.has_overlap
    mock_score_match.reset_mock()
    expected_map[Span(3, 8, "a")] = [Span(3, 8, "c")]
    result = select_best_match(ref, sys, mock_is_match, mock_score_match)
    assert result == SpanAlignment(ref, sys, expected_map)
    assert mock_is_match.call_count == 4
    assert mock_score_match.call_count == 4
    # Case: best option depends on exclusivity
    mock_is_match.reset_mock()
    mock_score_match.reset_mock()
    ref = DocSpans("", [Span(1, 3, "a"), Span(2, 5, "a"), Span(8, 10, "b")])
    sys = DocSpans("", [Span(2, 4, "a")])
    ## Default, exclusive matching
    expected_map = {Span(1, 3, "a"): [Span(2, 4, "a")]}
    result = select_best_match(ref, sys, mock_is_match, mock_score_match)
    assert result == SpanAlignment(ref, sys, expected_map)
    assert mock_is_match.call_count == 1
    assert mock_score_match.call_count == 1
    ## Non-exclusive matching
    mock_is_match.reset_mock()
    mock_score_match.reset_mock()
    expected_map[Span(2, 5, "a")] = [Span(2, 4, "a")]
    result = select_best_match(
        ref, sys, mock_is_match, mock_score_match, exclusive=False
    )
    assert result == SpanAlignment(ref, sys, expected_map)
    assert mock_is_match.call_count == 3
    assert mock_score_match.call_count == 2


def test_corppa_align():
    # Simple 1-1 cases
    ref = DocSpans("", [Span(2, 5, "a"), Span(10, 15, "b")])
    sys = DocSpans("", [Span(1, 6, "a"), Span(11, 13, "c")])
    ## With defaults
    expected_map = {Span(2, 5, "a"): [Span(1, 6, "a")]}
    result = corppa_align(ref, sys)
    assert result == SpanAlignment(ref, sys, expected_map)
    ## Label insensitive matching
    expected_map[Span(10, 15, "b")] = [Span(11, 13, "c")]
    result = corppa_align(ref, sys, is_match=Span.has_overlap)
    assert result == SpanAlignment(ref, sys, expected_map)

    # Best overlap
    ref = DocSpans("", [Span(3, 8, "a")])
    sys_a = DocSpans(
        "", [Span(1, 4, "a"), Span(4, 7, "a"), Span(7, 10, "a"), Span(3, 8, "c")]
    )
    sys_b = DocSpans("", [Span(2, 5, "a"), Span(5, 25, "a")])
    ## With defaults
    expected_map_a = {Span(3, 8, "a"): [Span(4, 7, "a")]}
    result = corppa_align(ref, sys_a)
    assert result == SpanAlignment(ref, sys_a, expected_map_a)
    expected_map_b = {Span(3, 8, "a"): [Span(2, 5, "a")]}
    result = corppa_align(ref, sys_b)
    assert result == SpanAlignment(ref, sys_b, expected_map_b)
    ## Label insensitive matching using overlap length
    expected_map_a = {Span(3, 8, "a"): [Span(3, 8, "c")]}
    result = corppa_align(
        ref, sys_a, is_match=Span.has_overlap, score_match=Span.overlap_length
    )
    assert result == SpanAlignment(ref, sys_a, expected_map_a)
    expected_map_b = {Span(3, 8, "a"): [Span(5, 25, "a")]}
    result = corppa_align(
        ref, sys_b, is_match=Span.has_overlap, score_match=Span.overlap_length
    )
    assert result == SpanAlignment(ref, sys_b, expected_map_b)

    # System span mapping with multiple reference spans
    ref = DocSpans("", [Span(2, 5, "a"), Span(7, 11, "b"), Span(18, 20, "a")])
    sys = DocSpans("", [Span(0, 25, "a")])
    ## With defaults
    expected_sys = DocSpans("", [Span(0, 18, "a"), Span(18, 25, "a")])
    expected_map = {
        Span(2, 5, "a"): [Span(0, 18, "a")],
        Span(18, 20, "a"): [Span(18, 25, "a")],
    }
    result = corppa_align(ref, sys)
    assert result == SpanAlignment(ref, expected_sys, expected_map)
    ## Label insensitive matching
    expected_sys = DocSpans("", [Span(0, 7, "a"), Span(7, 18, "a"), Span(18, 25, "a")])
    expected_map = {
        Span(2, 5, "a"): [Span(0, 7, "a")],
        Span(7, 11, "b"): [Span(7, 18, "a")],
        Span(18, 20, "a"): [Span(18, 25, "a")],
    }
    result = corppa_align(ref, sys, is_match=Span.has_overlap)
    assert result == SpanAlignment(ref, expected_sys, expected_map)
