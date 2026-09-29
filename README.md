# spanerr

Python library for evaluating span annotations using custom alignment and scoring strategies.

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

`spanerr` has three core data types (see `spanerr.core` for more details).

- `Span`: An individual span annotation.
- `DocSpans`: A set of span annotations for a document.
- `SpanAlignment`: An alignment between two sets of span annotations (reference, system) over a shared document.

#### Loading from dictionaries

`Spans` can be loaded from dictionaries with the following fields:

- `start` (int): starting text boundary (inclusive)
- `end` (int): ending text boundary (exclusive)
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

- Select First : Select the first matching system span for each reference span
- Select Best : Select the best matching system span for each reference span
- Corppa : The alignment strategy used by [`corppa`](https://github.com/Princeton-CDH/corppa) (see `corppa`'s [evaluation documentation](https://github.com/Princeton-CDH/corppa/tree/main/src/corppa/poetry_detection/evaluation) for more detail)

For additional flexibility, alignment strategies might take additional inputs, but these will generally need to be set to a specific value (e.g., via a lambda function) before they can be used within `spanerr`'s evaluation workflow.
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
