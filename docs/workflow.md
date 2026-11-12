# Workflow: raw datasets to Table I

Run all commands from the repository root inside the Python 3.12 environment.
Each optional notebook has the same configuration block: keep encoders and
embedding/cleaning/result directories consistent across notebooks.
Use `--help` on each numbered script to see its arguments. The examples use
HeAR, which needs no Qwen weights or UK-COVID audio.

## 01 — Prepare the datasets

```bash
python scripts/01_prepare_data.py --workers 8
```

Inputs: original CODA, CIDRZ, TBScreen downloads under `datasets/raw/`; layout and
selection rules are in [prep/README.md](../prep/README.md). This script runs
`coda.py`, `cidrz.py`, `tbscreen.py`, then `build_db.py` in order.
Outputs: dataset WAVs under `datasets/preprocess/`, per-dataset CSVs and
`data/metadata/combine.csv`. The combined rows are the paper's 23,638 clip IDs:
CODA 9,279, CIDRZ 7,343, TBScreen 7,016. Preparation preserves manifest row order.

## 02 — Extract frozen embeddings

```bash
python scripts/02_extract_embeddings.py --encoder hear --check-only
python scripts/02_extract_embeddings.py --encoder hear
```

Inputs: prepared WAVs and `combine.csv`. `--check-only` validates metadata and
paths without loading a model. Outputs: `data/embeds/embeds_hear.npy` and
`metadata_hear.csv`; row i in the metadata describes row i in the matrix.
HeAR outputs 512 dimensions. It runs in batches of 64 in the paper order; the
UK-COVID slots of the original batches are filled with zeros, so no UK audio is needed.

For independent, three-dataset Qwen extraction (2048 dimensions; different
results from the publication):

```bash
python scripts/02_extract_embeddings.py --encoder qwen --qwen-mode independent --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```

Its separate default output directory is `data/embeds/independent/`. To use
stages 03–05 with it, pass matching directories as described below. The exact
paper Qwen setup is described in [reproducibility.md](reproducibility.md).

## 03 — Compute cleaning decisions

```bash
python scripts/03_clean_labels.py --encoders hear
```

This stage fits global PCA-256, then per-dataset CL/PBNR, CL/CL and Proposed
cleaners. It does **not** train evaluation classifiers. Outputs in `data/cleaning/`:

- `decisions_hear.csv`: original clip IDs/metadata, keep flags, acoustic scores
  and Fisher projections for each cleaner; non-TB rows are retained.
- `decisions_hear.npz`: decision masks used by stage 04, freshly computed from
  stage 02 outputs. This contains no reference embeddings.
- `decisions_hear.json`: encoder, row count, settings and SHA256 of input files
  and masks. It detects changed/reordered embeddings or metadata.

Inspect the CSV before evaluating. The two TB subgroups are Acoustically
Consistent TB and Acoustically Ambiguous TB. Stage 04 reuses these decisions;
it does not refit the cleaners.

## 04 — Evaluate classifiers and baselines

```bash
python scripts/04_evaluate.py --encoders hear --workers 16
```

Inputs: stage 02 matrices/metadata and matching stage 03 masks/provenance. The
script rejects missing/stale/mismatched decisions. Outputs in `result/table_i/`:
`overall_emb hear_perdb.csv`, `tbcount_emb hear_perdb.csv`, and supplementary
`lodo_emb hear_perdb.csv`. The first has 15 rows: 3 datasets × 5 methods.

The original study's folds, classifiers, random seeds, random baseline and
aggregation are preserved. Cleaning occurs before classifier CV, as in the
paper code. See the [reproduction notes](reproducibility.md) for interpretation.

## 05 — Format Table I

```bash
python scripts/05_make_table.py --encoders hear
```

Inputs: stage 04 aggregate CSVs. Outputs in the same result directory:
`table_i.csv`, `table_i.md`, and `table_i_comparison.csv`. The comparison reports
each published/local mean, standard deviation and mean difference in percentage
points. This stage only formats computed results; it does not run experiments
or replace results with reference numbers.

## 06 — Acoustic figures before/after cleaning (paper Figs. 1 and 4)

```bash
python scripts/06_acoustic_figures.py --encoder qwen --dataset coda --workers 8
```

Inputs: prepared WAVs (stage 01) and `data/cleaning/decisions_<encoder>.csv`
(stage 03; the Proposed keep flags define the subgroups). Every clip of the
chosen dataset is reloaded (mono, 16 kHz, peak-normalised) and 153 statistics
are computed: nine time summaries of 13 MFCCs plus four spectral streams
(FFT 2048, hop 64, window 512, 50–5000 Hz). The statistics are cached in
`data/acoustic/features_<dataset>.npz` (git-ignored).

Outputs in `result/acoustic_figures/` (tag `<dataset>_<encoder>`):

- `fig1_before_<tag>.{pdf,png}`: Non-TB vs TB, KDE (top) and box plots (bottom).
- `fig4_after_<tag>.{pdf,png}`: Non-TB vs Acoustically Ambiguous TB vs
  Acoustically Consistent TB.
- `feature_ranking_<tag>.csv`: all 153 statistics ranked by single-feature
  AUROC, max(AUC, 1−AUC), Consistent TB vs Non-TB (Ambiguous excluded).
- `distribution_summary_<tag>.csv`, `summary_<tag>.json`.

The default plotted features are the paper's MFCC-0 10th percentile, standard
deviation and 25th percentile; `--top 3` plots the top-ranked ones instead,
`--features` any others. With the paper Qwen decisions on CODA (6,506 Non-TB /
1,319 Ambiguous / 1,454 Consistent) the top three are exactly these, with
AUROC 0.745 / 0.735 / 0.727. The ranking is descriptive (groups were formed
from the same data), not held-out biomarker validation.

Figures use the paper's colours and layout. Fig. 1 depends only on audio and
labels, so it is identical for every run. Fig. 4 depends on the stage-03
grouping: HeAR and Qwen with `--include-uk-context` reproduce the paper groups
exactly (CODA Qwen 1,454 Consistent / 1,319 Ambiguous); Qwen without UK context
changes the grouping slightly (1,464 Consistent instead of 1,454), so the curves differ a little.

## Use a separate experiment directory

```bash
python scripts/02_extract_embeddings.py --encoder hear --output data/embeds/my_run/embeds_hear.npy
python scripts/03_clean_labels.py --encoders hear --embeddings-dir data/embeds/my_run --cleaning-dir data/cleaning/my_run
python scripts/04_evaluate.py --encoders hear --embeddings-dir data/embeds/my_run --cleaning-dir data/cleaning/my_run --output-dir result/my_run --workers 16
python scripts/05_make_table.py --encoders hear --results-dir result/my_run
```

For a complete independent Qwen + HeAR experiment, the driver selects separate
embedding/cleaning/result directories automatically:

```bash
python run_pipeline.py --qwen-mode independent --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```

To process both encoders with numbered stages 03–05, use `--encoders qwen hear`
and ensure both stage 02 matrix/metadata pairs exist in the chosen directory.
Compatibility command `paper_study.py` still runs stages 03–05 together.

## Dataset storage

Original downloads go in `datasets/raw/`, prepared audio in `datasets/preprocess/`.
Only the empty folders are versioned. Generated metadata, embeddings and
cleaning decisions are written under `data/`.
