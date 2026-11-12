# Reproducibility notes

All embeddings are computed from raw audio. No stored paper features are
loaded by the pipeline.

| Extraction | Audio used | Table I |
|---|---|---|
| HeAR | study datasets only | all 60 means/stds match the paper |
| Qwen, `--include-uk-context` (paper setup) | study datasets + 42 UK-COVID clips during extraction | all 60 means/stds match the paper |
| Qwen, `--qwen-mode independent` | study datasets only | differs (Proposed AUROC 92/88/84 vs 86/88/89) |

The reproduced table is in [result/reproduced_table_i](../result/reproduced_table_i/table_i.md).

## HeAR

Exact numbers were obtained on an RTX 4090 with TensorFlow 2.19.0 and the
versions in `requirements.txt`. XLA autotuning can pick a different kernel for
the final projection, which changes the embeddings slightly.
`lib/hear_autotune_rtx4090_tf219.textproto` pins the kernels we used (it only
contains operation shapes and kernel choices, no data). Other GPUs and
TensorFlow versions were not tested.

## Qwen

The paper embeddings were extracted in packed batches of 128 clips with SDPA
attention. That path does not mask attention between clips, so clips in the
same batch slightly influence each other. One batch (184) contained 42 clips
from the UK COVID-19 Vocal Audio Dataset next to 86 TBScreen clips. UK-COVID is
not part of the study (it is never used for cleaning or evaluation), but its
audio still affects those 86 TBScreen embeddings.

To reproduce the paper exactly, prepare the UK clips (see `prep/README.md`) and run:

```bash
python run_pipeline.py --include-uk-context --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```

`--qwen-mode independent` embeds each clip on its own and needs only the three
study datasets. It is the cleaner setup, but the numbers differ from the paper.

## Differences between the paper text and the code

These follow the code that produced the published numbers:

- CIDRZ segments are cut from stored sample offsets
  (`prep/manifest/cidrz_segments.csv`), because the detector settings were
  changed after the paper. TBScreen uses the cough detector in `cad/`
  (June 2026 version, the one used for the paper).
- TBScreen is capped at a random 25–29 clips per patient and device (seed 42);
  the paper says 35.
- The Random baseline removes as many TB clips as Proposed keeps (Consistent
  count), while the paper describes removing the Ambiguous count.
- Cleaning and its global PCA are fitted on all data before classifier CV.
  Sensitivity on the cleaned subset is therefore an upper bound, not screening
  sensitivity.
- Classifier folds are grouped by participant. Scaling, PCA and calibration
  are fitted inside each fold. The threshold minimises |TPR − TNR| (the
  function is named after Youden's index).
- Means and standard deviations follow the original aggregation, including
  15 repeats for the Random baseline.
