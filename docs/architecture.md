# Implementation map

Start with [the overview](overview.md), then the [Proposed walkthrough](proposed_method.md).

| Responsibility | Implementation | Entry point |
|---|---|---|
| Per-dataset selection/audio processing | `prep/coda.py`, `cidrz.py`, `tbscreen.py` | Stage 01 |
| Combined metadata/order | `prep/build_db.py`, `prep/manifest/` | Stage 01 |
| Qwen frozen tower and packed inference | `lib/qwen_extractor.py` | Stage 02 |
| HeAR loading/framing/inference | `lib/hear_extractor.py` | Stage 02 |
| Model selection, metadata/output handling | `extract_embeddings.py` | Stage 02 |
| Cleaner orchestration/decision provenance | `lib/study_cleaning.py` | Stage 03 |
| PCA, Fisher, KDE, GMM | `lib/helper.py` | Stage 03 |
| Confident Learning baseline | `lib/confident_learning.py` | Stage 03 |
| Evaluation jobs/random baseline/aggregation | `lib/study_evaluation.py` | Stage 04 |
| Participant folds/classifiers/thresholds | `lib/utils.py` | Stage 04 |
| Table formatting/published references | `lib/table_i.py`, `table_i_reference.json` | Stage 05 |
| Whole workflow | `run_pipeline.py` | Stages 01–05 |

The numbered CLI files are small entry points. Numerical method implementations
live in `lib/`; dataset code stays in `prep/`. Stage 03/04 communicate through
explicit generated decisions and provenance, making each stage reviewable and
restartable. Stage 04 always evaluates original frozen features after selecting
rows; global cleaning PCA is not substituted for classifier inputs.

`paper_study.py` preserves the older combined CLI and `run_study` function.
`extract_embeddings.py` preserves its older CLI and `QwenReplay` import for
existing users. `extras/` is not part of the five-stage paper path.
