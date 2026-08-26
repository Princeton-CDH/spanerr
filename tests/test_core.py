from unittest.mock import call, patch

import pytest

from spanerr.core import DocSpans, Span


class TestSpan:
    def test_init(self):
        # Test default label
        s = Span(3, 6)
        assert s.start == 3 and s.end == 6 and s.label == ""
        # Invalid range: end index < start index
        error_message = "Start index must be less than end index"
        with pytest.raises(ValueError, match=error_message):
            Span(9, 2, "label")
        # Invalid range: end index = < start index
        with pytest.raises(ValueError, match=error_message):
            Span(2, 2, "label")

    def test_len(self):
        assert len(Span(2, 5, "label")) == 3
        assert len(Span(0, 42, "label")) == 42

    def test_from_dict(self):
        # Missing required fields
        err_pfx = "Missing required fields: "
        with pytest.raises(ValueError, match=f"{err_pfx}start, end"):
            Span.from_dict({"a": 0, "b": 1, "c": "l"})
        with pytest.raises(ValueError, match=f"{err_pfx}start, end"):
            Span.from_dict({"s": 0, "e": 1, "label": "l"})
        with pytest.raises(ValueError, match=f"{err_pfx}start"):
            Span.from_dict({"end": 0})
        with pytest.raises(ValueError, match=f"{err_pfx}end"):
            Span.from_dict({"start": 0})

        # Typical input
        d = {"start": 0, "end": 3}
        assert Span.from_dict(d) == Span(0, 3)
        d["label"] = "l"
        assert Span.from_dict(d) == Span(0, 3, "l")
        # Ignores additional fields
        more_fields = d | {"poem_start": 52, "poem_end": 75}
        assert Span.from_dict(more_fields) == Span.from_dict(d)
        bad_label = {"start": 2, "end": 4, "poem_id": "i"}
        assert Span.from_dict(bad_label) == Span(2, 4)

    def test_has_overlap(self):
        s = Span(3, 6, "l")
        # exact overlap
        assert s.has_overlap(Span(3, 6, "l"))
        assert s.has_overlap(Span(3, 6))
        # partial overlap: subsets
        for start, end in [(4, 5), (3, 5), (4, 6)]:
            assert s.has_overlap(Span(start, end, "l"))
            assert s.has_overlap(Span(start, end))
        # partial overlap: not subsets
        for start, end in [(1, 5), (4, 8)]:
            assert s.has_overlap(Span(start, end, "l"))
            assert s.has_overlap(Span(start, end))
        # no overlap
        for start, end in [(0, 1), (0, 3)]:
            assert not s.has_overlap(Span(start, end, "l"))
            assert not s.has_overlap(Span(start, end))

    def test_is_adjacent(self):
        s = Span(3, 6, "l")
        assert s.is_adjacent(Span(0, 3))
        assert s.is_adjacent(Span(6, 8))
        assert not s.is_adjacent(Span(0, 2))
        assert not s.is_adjacent(Span(4, 5))
        assert not s.is_adjacent(Span(5, 10))

    @patch.object(Span, "has_overlap", autospec=True)
    def test_overlap_length(self, mock_has_overlap):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_has_overlap.return_value = False
        assert span_a.overlap_length("other span") == 0
        mock_has_overlap.assert_called_once_with(span_a, "other span")

        # has overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        ## exact overlap
        span_b = Span(3, 6, "label")
        assert span_a.overlap_length(span_b) == 3
        mock_has_overlap.assert_called_once_with(span_a, span_b)
        mock_has_overlap.reset_mock()
        ## partial overlap
        span_b = Span(3, 5, "label")
        assert span_a.overlap_length(span_b) == 2
        span_b = Span(2, 8, "label")
        assert span_a.overlap_length(span_b) == 3
        ## label is ignored
        span_b = Span(2, 8)
        assert span_a.overlap_length(span_b) == 3

    @patch.object(Span, "overlap_length", autospec=True)
    @patch.object(Span, "has_overlap", autospec=True)
    def test_jaccard(self, mock_has_overlap, mock_overlap_length):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_has_overlap.return_value = False
        assert span_a.jaccard("other span") == 0
        mock_has_overlap.assert_called_once_with(span_a, "other span")
        mock_overlap_length.assert_not_called()

        # has overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        ## exact overlap
        mock_overlap_length.return_value = 3
        span_b = Span(3, 6)
        assert span_a.jaccard(span_b) == 1
        mock_has_overlap.assert_called_once_with(span_a, span_b)
        mock_overlap_length.assert_called_once_with(span_a, span_b)
        ## partial overlap
        mock_overlap_length.return_value = 2
        assert span_a.jaccard(Span(3, 5)) == 2 / 3
        mock_overlap_length.return_value = 3
        assert span_a.jaccard(Span(2, 8)) == 3 / 6

    @patch.object(Span, "overlap_length", autospec=True)
    def test_overlap_factor(self, mock_overlap_length):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_overlap_length.return_value = 0
        assert span_a.overlap_factor("other span") == 0
        mock_overlap_length.assert_called_once_with(span_a, "other span")

        # has overlap
        ## exact overlap
        mock_overlap_length.reset_mock()
        mock_overlap_length.return_value = 3
        span_b = Span(3, 6, "label")
        assert span_a.overlap_factor(span_b) == 1
        mock_overlap_length.assert_called_once_with(span_a, span_b)
        ## partial overlap
        mock_overlap_length.reset_mock()
        mock_overlap_length.return_value = 2
        span_b = Span(3, 5, "label")
        assert span_a.overlap_factor(span_b) == 2 / 3
        span_b = Span(2, 8, "label")
        mock_overlap_length.return_value = 3
        assert span_a.overlap_factor(span_b) == 3 / 6
        ## label is ignored
        span_b = Span(2, 8)
        assert span_a.overlap_factor(span_b) == 3 / 6

    def test_binarize(self):
        assert Span(1, 4, "label").binarize() == Span(1, 4, "")
        assert Span(3, 5).binarize() == Span(3, 5)

    def test_merge(self):
        span_a = Span(3, 6, "label")

        # Different label
        with pytest.raises(
            ValueError, match="Cannot merge spans with different labels"
        ):
            span_a.merge(Span(3, 6, "other"))
        # No overlap, not adjacent
        with pytest.raises(
            ValueError, match="Cannot merge spans without overlap or adjacency"
        ):
            span_a.merge(Span(9, 12, "label"))
        # Adjacent
        assert span_a.merge(Span(1, 3, "label")) == Span(1, 6, "label")
        assert span_a.merge(Span(6, 10, "label")) == Span(3, 10, "label")
        # Exact overlap
        assert span_a.merge(span_a) == span_a
        # Partial overlap
        assert span_a.merge(Span(4, 5, "label")) == span_a
        assert span_a.merge(Span(1, 5, "label")) == Span(1, 6, "label")
        assert span_a.merge(Span(5, 10, "label")) == Span(3, 10, "label")
        assert span_a.merge(Span(1, 10, "label")) == Span(1, 10, "label")


class TestDocSpans:
    def test_init(self):
        a = Span(3, 6)
        b = Span(1, 2)
        c = Span(3, 5)
        d = Span(3, 6, "l")
        # Test post-initialization span sorting
        assert DocSpans("i", [a, b]).spans == [b, a]
        assert DocSpans("i", [a, b, c]).spans == [b, c, a]
        assert DocSpans("i", [d, a]).spans == [a, d]

    def test_from_dict(self):
        # Missing required field
        with pytest.raises(ValueError, match="Missing required field: spans"):
            assert DocSpans.from_dict({"doc_id": "i"})
        # Empty spans
        assert DocSpans.from_dict({"doc_id": "i", "spans": []}) == DocSpans("i", [])
        assert DocSpans.from_dict({"spans": []}) == DocSpans("", [])
        # With spans
        span_dicts = [{"start": 1, "end": 4, "label": "i"}, {"start": 2, "end": 5}]
        spans = [Span(1, 4, "i"), Span(2, 5)]
        assert DocSpans.from_dict({"spans": span_dicts}) == DocSpans("", spans)
        assert DocSpans.from_dict({"doc_id": "d", "spans": span_dicts}) == DocSpans(
            "d", spans
        )

    @patch.object(Span, "from_dict", autospec=True)
    def test_from_dict_calls(self, mock_span_from_dict):
        # No spans
        assert DocSpans.from_dict({"spans": []}) == DocSpans("", [])
        mock_span_from_dict.assert_not_called()
        # With spans
        span_dicts = [{"start": 1, "end": 4, "label": "i"}, {"start": 2, "end": 5}]
        spans = [Span(1, 4, "i"), Span(2, 5)]
        mock_span_from_dict.side_effect = spans
        assert DocSpans.from_dict({"spans": span_dicts}) == DocSpans("", spans)
        assert mock_span_from_dict.call_count == 2
        mock_span_from_dict.assert_has_calls([call(s) for s in span_dicts])

    def test_aggregate(self):
        a = Span(3, 6)
        b = Span(1, 2)  # non-adjacent
        c = Span(6, 8)  # adjacent
        d = Span(2, 4)  # overlapping
        e = Span(4, 5)  # nested
        f = Span(3, 6, "l1")  # nested but different label
        g = Span(3, 10, "l2")  # overlapping but different label

        # No spans
        assert DocSpans("", []).aggregate() == DocSpans("", [])
        # No overlap
        assert DocSpans("", [a, b]).aggregate() == DocSpans("", [a, b])
        assert DocSpans("", [a, c]).aggregate() == DocSpans("", [a, c])
        assert DocSpans("", [a, f]).aggregate() == DocSpans("", [a, f])
        ## With concat flag set
        assert DocSpans("", [a, b]).aggregate(concat=True) == DocSpans("", [a, b])
        assert DocSpans("", [a, c]).aggregate(concat=True) == DocSpans("", [Span(3, 8)])
        assert DocSpans("", [a, f]).aggregate(concat=True) == DocSpans("", [a, f])
        # With overlap
        assert DocSpans("", [a, d]).aggregate() == DocSpans("", [Span(2, 6)])
        assert DocSpans("", [a, e]).aggregate() == DocSpans("", [a])
        assert DocSpans("", [a, b, d]).aggregate() == DocSpans("", [b, Span(2, 6)])
        assert DocSpans("", [a, b, c, d, e, f, g]).aggregate() == DocSpans(
            "", [b, Span(2, 6), c, f, g]
        )
        ## With concat flag set
        assert DocSpans("", [a, b, d]).aggregate(concat=True) == DocSpans(
            "", [Span(1, 6)]
        )
        assert DocSpans("", [a, b, c, d, e, f, g]).aggregate(concat=True) == DocSpans(
            "", [Span(1, 8), f, g]
        )

    @patch.object(DocSpans, "aggregate", autospec=True, return_value="aggregated")
    def test_binarize(self, mock_aggregate):
        spans = [Span(3, 6, "a"), Span(1, 2, "b"), Span(6, 8, "b"), Span(2, 4, "d")]
        # Default
        x = DocSpans("i", spans)
        assert x.binarize(concat="concat") == "aggregated"
        mock_aggregate.assert_called_once_with(
            DocSpans("i", [s.binarize() for s in spans]), concat="concat"
        )
