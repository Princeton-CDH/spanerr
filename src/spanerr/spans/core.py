"""
Core span data types
"""

from dataclasses import dataclass
from typing import Self


@dataclass(order=True, frozen=True)
class Span:
    """
    Span object representing a Pythonic "closed open" interval.
    """

    #: start index
    start: int
    #: end index
    end: int
    #: label for the span
    label: str = ""

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("Start index must be less than end index")

    def __len__(self) -> int:
        return self.end - self.start

    @classmethod
    def from_dict(cls, span_dict: dict):
        """
        Create a Span object from a dict representing a span.

        The input dict is expected to have the following fields:
            - start (int): The starting index of the span
            - end (int): The ending index of the span
            - label (str, optional): The span's label
        """
        if "start" not in span_dict or "end" not in span_dict:
            missing = sorted({"start", "end"} - span_dict.keys(), reverse=True)
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        return Span(span_dict["start"], span_dict["end"], span_dict.get("label", ""))

    def has_overlap(self, other: Self) -> bool:
        """
        Returns whether this span overlaps with the other span.
        """
        return self.start < other.end and other.start < self.end

    def is_adjacent(self, other: Self) -> bool:
        """
        Returns whether this span is adjacent with the other span.
        """
        return self.end == other.start or self.start == other.end

    def overlap_length(self, other: Self) -> int:
        """
        Returns the length of overlap between this span and the other span.
        """
        if not self.has_overlap(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    def jaccard(self, other: Self) -> float:
        """
        Returns the jaccard index between this span and the other span.
        """
        if not self.has_overlap(other):
            return 0
        intersection = self.overlap_length(other)
        union = max(self.end, other.end) - min(self.start, other.start)
        return intersection / union

    def overlap_factor(self, other: Self) -> float:
        """
        Returns the overlap factor with the other span.

        The overlap factor is defined as follows:

            * If no overlap (overlap = 0), then overlap_factor = 0.

            * Otherwise, overlap_factor = overlap_length / longer_span_length

        So, the overlap factor has a range between 0 and 1 with higher values
        corresponding to a higher degree of overlap.
        """
        overlap = self.overlap_length(other)
        return overlap / max(len(self), len(other))

    def binarize(self) -> Self:
        """
        Returns the "binarized" version of this span (i.e., sets label to default)
        """
        default_label = self.__class__.label
        return self.__class__(self.start, self.end, default_label)

    def merge(self, other: Self) -> Self:
        """
        Returns the merged span of this span with the the other.
        Raises ValueError for spans with different labels, or without overlap or adjacency
        """
        if self.label != other.label:
            raise ValueError("Cannot merge spans with different labels")
        if not self.has_overlap(other) and not self.is_adjacent(other):
            raise ValueError("Cannot merge spans without overlap or adjacency")
        min_start = min(self.start, other.start)
        max_end = max(self.end, other.end)
        return self.__class__(min_start, max_end, self.label)


@dataclass(frozen=True)
class DocSpans:
    """
    Document-level span annotations object
    """

    doc_id: str  # Should this have a default
    spans: list[Span]

    def __post_init__(self):
        self.spans.sort()  # ensure spans are sorted

    @classmethod
    def from_dict(cls, doc_dict: dict) -> Self:
        """
        Load DocSpans from a dict representing an annotated document.

        The input dict is expected to have the following fields:
            - spans (list[dict]): Document's span annotations as a list of dicts
                                  compatible with `Span.from_dict`
            - doc_id (str, optional): Document ID (defaults to empty string)
        """
        if "spans" not in doc_dict:
            raise ValueError("Missing required field: spans")
        doc_id = doc_dict.get("doc_id", "")
        spans = [Span.from_dict(s) for s in doc_dict["spans"]]
        return cls(doc_id, spans)

    def aggregate(self, concat: bool = False) -> Self:
        """
        Aggregate spans by merging all overlapping spans with the same label.
        Optionally, can also merge adjacent spans.
        """
        new_spans = []
        last_label_index = {}  # track index of last added span with a given label
        for span in self.spans:
            # Check if current span's label has been seen previously
            if span.label in last_label_index:
                prev_index = last_label_index[span.label]
                prev = new_spans[prev_index]
                # Check if span can be merged with previous span with same label
                if prev.has_overlap(span) or (concat and prev.is_adjacent(span)):
                    # Replace previous span with its merged version
                    new_spans[prev_index] = prev.merge(span)
                else:
                    # Cannot merge so add span and update index
                    new_spans.append(span)
                    last_label_index[span.label] = len(new_spans) - 1
            else:
                # Unseen label, so update label index and append span
                new_spans.append(span)
                last_label_index[span.label] = len(new_spans) - 1
        return self.__class__(self.doc_id, new_spans)

    def binarize(self, concat: bool = False) -> Self:
        """
        Returns a "binarized" version of this DocSpans in which its spans are
        binarized with any overlapping spans merged. Optionally, adjacent spans
        can also be merged.
        """
        binary_spans = [s.binarize() for s in self.spans]
        return self.__class__(self.doc_id, binary_spans).aggregate(concat=concat)
