# spanerr

`spanerr` is a Python library for evaluating span-level text annotations using customizable alignment and scoring strategies.
`spanerr` operates explicitly over span text boundaries (i.e., text indices) rather than over the annotated text itself.

Many span-level annotation tasks diverge significantly enough from named-entity recognition (e.g., long text spans, large label set) that the typical formulations for the evaluation metrics of precision, recall, and $F_1$ scores become insuitable.
`spanerr` addresses this issue by not only supporting customized scoring of (partial) span matches, but also customizing how the spans within document-level annotation sets are aligned for evaluation.
`spanerr` is designed for maximal flexibility generally leaving it to the user to determine what assumptions and restrictions are required in their use case.

[![unit tests](https://github.com/Princeton-CDH/spanerr/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/Princeton-CDH/spanerr/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/Princeton-CDH/spanerr/graph/badge.svg?token=Wd3vZ38Bxz)](https://codecov.io/gh/Princeton-CDH/spanerr)

## Basic Usage

### Installation

Use pip to install as a Python package directly from GitHub.
Use a branch or tag name, e.g. `@develop` or `@0.1.0` if you need to install a specific version

```sh
pip install git+https://github.com/Princeton-CDH/spanerr.git#egg=spanerr
```

### Core Data Types

`spanerr` has three first-class objects:

- `Span`: An individual span annotation.
- `DocSpans`: A set of span annotations for a document.
- `SpanAlignment`: A set of aligned span annotations (reference, system) within a single document.

All three of these data types are immutable, but `SpanAlignment` does not currently support hashing.

#### Binarization

`spanerr` provides functionality for "removing" span labels from `Span` and `DocSpans` by setting them to a default label (empty string).
For `DocSpans`, overlapping spans will be merged and optionally neighboring spans can be merged.

#### Loading from dictionaries

`Spans` can be loaded from dictionaries with the following fields:

- `start` (int): starting text index (inclusive)
- `end` (int): ending text index (exclusive)
- `label` (str): optional span label (defaults to empty string)

`DocSpans` can be loaded from dictionaries with the following fields:

- `doc_id` (str): optional document id (defaults to empty string)
- `spans` (list[dict]): list of spans (in dictionary form, see above)

### Core Functionality

In `spanerr` there are two core components to evaluating span annotations: (1) how spans are aligned and (2) how aligned spans are scored.
`spanerr` is intentionally designed so that these two components can be heavily customized.

#### Aligning span annotations

An alignment strategy is represented as a function that takes two sets of annotations (`DocSpans`) as input and returns an alignment (`SpanAlignment`) which will then be used for scoring.
There are little restrictions on the alignments themselves: the resulting `SpanAlignment` may contain transformed versions of the input `DocSpans` and no restrictions are made on its mapping between reference and system spans.
The idea is to allow for the creation of whatever alignment is useful for scoring.

The following alignment strategies are provided in `spanerr.align`:

- Select First : Select the first (sequential) matching system span for each reference span.
  By default, spans match if they overlap and have the same label.
- Select Best : Select the best matching system span for each reference span.
  By default, given spans that overlap and have the same label, the best match is the span pair with the highest jaccard similarity.
- Corppa : The alignment strategy used by [`corppa`](https://github.com/Princeton-CDH/corppa).
  See `corppa`'s [evaluation documentation](https://github.com/Princeton-CDH/corppa/tree/main/src/corppa/poetry_detection/evaluation) for more detail.

For additional flexibility, alignment strategies may take additional inputs to further customize their behavior (e.g., use different span matching and span scoring strategies), but these will generally need to be set to a specific value (e.g., via a lambda function) before they can be used within `spanerr`'s evaluation workflow.
See the `spanerr.align.construct_aligner` for an example.

### Scoring Aligmnents

Scoring an alignment (`SpanAlignment`) entails computing an alignment's *relevance score* (i.e., true positive, numerator for precision and recall).
So, a scoring strategy is represented as a function that takes an alignment (`SpanAlignment`) as input and returns its score (`float`).
This allows for both alignment-independent scoring strategies in which scoring depends solely on the individual scores of each reference-system span pair as well as those that don't.

Currently, `spanerr` provides the building blocks for constructing alignment-independent strategies using `spanerr.eval.relevance_score` and `spanerr.span_utils.composite_match_score`.
See `spanerr.compute_metrics.get_scorer` for an example of constructing scoring strategy functions.

### Span Annotations File Format

A set of span annotations can be loaded into `spanerr` by providing a JSONL file with each line corresponding to a different document's annotations (i.e. `DocSpans`).
For examples see the files in `tests/test_data`.

### Scripts

Installing `spanerr` currently provides access to the following command line script:

- `spanerr-metrics`: For calculating entity- or document-level aggregated precision, recall, and F-1 scores for given reference and system span annotation sets.
  (Corresponds to `spanerr.compute_metrics.py`)

## License

This project is licensed under the [Apache 2.0 License](LICENSE)

(c)2026 Trustees of Princeton University.
Permission granted for non-commercial distribution online under a standard Open Source license.
