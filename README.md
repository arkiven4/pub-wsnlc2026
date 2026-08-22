# pub-wsnlc2026

TB screening from cough audio, pooled across three corpora.

## Pipeline

```
01_make_data.ipynb    raw corpora -> one cough per wav + one unified table
                      data/{coda,cidrz,tbscreen}.csv -> data/combine.csv
                      (cidrz sessions are segmented with the CAD detector in cad/)

02_extract_embed.ipynb  wavs -> pretrained-encoder embeddings
                      result/metadata_{name}.csv + result/embeds_{name}.npy

03_proposed_method.ipynb  Fisher split: which TB+ clips actually sound TB-like
                      result/metadata_cleaned_{name}.csv + eval_{perdb,lodo}_*.csv
```

## Data

`data/combine.csv` - 27,455 clips, 1,487 subjects. Per-subject caps applied in
01 (cidrz 15 clips, tbscreen 35) keep one corpus from dominating the pool.

| corpus | db | clips | subjects | TB rate | `db_additional` holds |
|---|---|---|---|---|---|
| CODA | 0 | 9,580 | 1,082 | 0.30 | country (7) |
| CIDRZ | 1 | 7,699 | 361 | 0.17 | facility_phone (12) |
| TBscreen | 2 | 10,176 | 125 | 0.73 | device (3) |

All clips are 16 kHz mono, one cough each, 0.30-1.0 s. `db_additional` means a
different thing per corpus but is site/device nuisance in all three, so it is
the confounder column downstream.

## Embeddings

| name | model | dim |
|---|---|---|
| `qwen` | audio tower of Qwen3-Omni-30B-A3B-Thinking, mean-pooled | 2048 |
| `hear` | Google HeAR | 512 |

Each is cached as a `(metadata_{name}.csv, embeds_{name}.npy)` pair whose rows
line up one-to-one. Read them with `lib.utils.load_embeds(name)`.

## Layout

```
data/combine.csv   the unified table written by 01: one row per cough wav
lib/commons.py     constants: audio framing, paths, combine.csv semantics
lib/utils.py       metadata loading, wav loading, embedding cache I/O
lib/helper.py      the encoders, the extraction driver, the linear probe
cad/               standalone ONNX cough detector used by 01 (see cad/README.md)
```

## Proposed method (notebook 03)

Fisher LDA -> P(NTB | x) per clip -> GMM/BIC test for a natural split inside the
TB+ posteriors. TB+ clips divide into **biomarker** (acoustically TB-like) and
**label-only** (labelled TB, sitting with the NTB clips). Everything is inline in
the notebook; `lib/` covers only the embedding caches.

The notebook carries its own controls, and they matter more than the headline:
the in-sample comparison is circular (the split removes exactly the clips the
evaluation would score wrong), so §7.5's nested version - fit the split on
training folds, score untouched test folds - is the number that tests the claim.

## Notes

- The two extractors see byte-identical audio, so differences between the
  embedding sets are the model, not the framing.
- Qwen's audio tower consumes a *packed* batch and its `forward` never applies
  the block mask it builds, so on sdpa/eager the clips in a batch attend across
  each other. `lib/helper.py` runs it on flash-attention-2, whose varlen
  path honours `cu_seqlens`; `lib.helper.check_batch_invariance` is the
  regression test for this.
- `disease_status` in `combine.csv` is not binary: coda/cidrz store 0/1, but
  TBscreen stores the Xpert semi-quantitative grade (0 negative, 1 trace ... 5
  high). `lib.utils.load_meta` derives the binary `label` as `> 0`.
