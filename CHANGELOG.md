# Change & Version Information

## 0.1.0

Initial alpha release.

### Span Logic

- Add `Span` object to represent a span annotation
- Add `DocSpans` object to represent a set of span annotations for a document
- Add library `span_utils` of methods for matching and scoring `Span` objects 

### Alignment Logic

- Add `SpanAlignment` object to represent an alignment between two sets of span annotations
- Add library `align` of methods for aligning two sets of span annotations

## Computing Evaluation Metrics

- Add library `eval` of methods for computing the evaluation metrics for a `SpanAlignment`
- Add script `compute_metrics` for computing document- and entity-level aggregated evaluation metrics
