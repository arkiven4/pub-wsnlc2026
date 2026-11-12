# Program overview

The program starts with raw cough recordings and clinical TB labels. Its task is
to identify an acoustic mismatch within the TB-labelled clips, then compare
classification results before and after selecting clips. It does not change the
original clinical labels in the dataset.

Read this page first, then follow [the commands](workflow.md). For the Proposed
method itself, continue to [Proposed: from embeddings to decisions](proposed_method.md).

## The five steps

```text
datasets/raw/                   Original dataset downloads
      |
      v
01 Prepare data                 Select recordings, cut clips, harmonize metadata
      |                         datasets/preprocess/ + data/metadata/combine.csv
      v
02 Extract embeddings           Frozen HeAR or Qwen; no fine-tuning
      |                         Matrix + metadata in matching row order
      v
03 Clean labels                 Proposed, CL-PBNR, CL-CL; keep all non-TB rows
      |                         Per-clip decisions, scores and input hashes
      v
04 Evaluate                     Original / Random / PBNR / CL / Proposed
      |                         Grouped-CV classifier metrics + retained counts
      v
05 Make Table I                 Format results and compare with the paper
```

Each stage has one job. Stage 03 fits the cleaners, not evaluation classifiers.
Stage 04 reads the saved decisions and selects rows from the original frozen
embeddings. Stage 05 formats those measured results; it never replaces them with
the published reference numbers.

## What one row represents

One row is one cough clip. The metadata carries its filename, source database,
participant ID and original TB label. Embedding row i must describe metadata row
i. Every mask and score written by cleaning uses that same order.

A participant can contribute multiple clips. Outer classifier CV keeps a
participant's clips in one fold. Cleaner fitting follows the paper's original
pre-CV protocol; it is not an out-of-fold cleaning experiment.

## The two cleaning paths

| Path | Score source | Decision within TB-labelled clips |
|---|---|---|
| Proposed | Fisher coordinate -> class KDEs -> non-TB posterior | GMM membership, if the split passes its checks |
| CL-PBNR | Five-fold out-of-fold classifier probabilities | cleanlab prune_by_noise_rate flags |
| CL-CL | The same probability-estimation approach | cleanlab confident_learning flags |

Both paths retain every non-TB clip. Their scores are different quantities;
a CL probability and a Fisher/KDE posterior should not be treated as interchangeable.
Random Cleaning is a separate evaluation baseline with **15 seeds**. PBNR and CL
are not repeated 15 times.

## Where to read the code

1. `scripts/01_prepare_data.py` shows the preparation order. The dataset-specific
   rules live in `prep/`, with shared locations in `prep/paths.py`.
2. `lib/qwen_extractor.py` and `lib/hear_extractor.py` contain waveform-to-vector
   inference. `extract_embeddings.py` manages metadata and output alignment.
3. `lib/study_cleaning.py::compute_cleaning` shows which cleaners the paper runs.
   Read `lib/helper.py::run_fisher_and_split` for Proposed, then its three helpers.
4. `lib/study_evaluation.py::evaluate_study` builds the five experimental arms.
   `lib/utils.py::_prep_fold` and `_fit_clf_on_fold` define classifier fitting.
5. `lib/table_i.py::write_table` shows how means/stds become the final table.

Functions on this path have Doxygen-style docstrings with expected shapes and
return values. Run `doxygen Doxyfile` to build browsable pages in
`docs/_build/html/`.

The [architecture map](architecture.md) lists all modules. Differences between
the paper text and the code are listed in [reproducibility.md](reproducibility.md).
