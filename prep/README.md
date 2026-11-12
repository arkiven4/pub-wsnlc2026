# prep/ — from raw datasets to the paper clip list

Scripts that rebuild the 23,638 clips used in the paper (CODA 9,279 · CIDRZ
7,343 · TBScreen 7,016) and their metadata. Run from this folder:

```bash
python coda.py       # datasets/preprocess/coda/,     data/metadata/coda_solicited.csv
python cidrz.py      # datasets/preprocess/cidrz/,    data/metadata/cidrz_{cough,segments}.csv
python tbscreen.py   # datasets/preprocess/tbscreen/, data/metadata/tbscreen_coughs{_original,}.csv  (GPU for AST)
python build_db.py   # data/metadata/combine.csv  (paper order, 0.3–1.0 s filter)
```

## Raw data layout (`datasets/raw/`, git-ignored)

| DB | Source | Expected layout |
|---|---|---|
| CODA | CODA TB DREAM Challenge (Synapse), train split | `coda/meta_data/{Clinical,Cough Metadata}/*.csv`, `coda/raw_data/solicited_data/*.wav` |
| CIDRZ | CIDRZ / GHAI cough dataset | `cidrz/<Site> <Phone>/*.wav`, `cidrz/Metadata and Codebook/GHAI_Final_Data_2023.csv` |
| TBScreen | Sharma et al. | `tbscreen/Passive_coughs/{Passive_coughs.csv,Audio_files/}` |

The `datasets/raw/` and `datasets/preprocess/` folders are retained with `.gitkeep`;
all other contents are ignored by git. Download scripts will be added later.

## What each script reproduces

- **CODA** — official solicited metadata, `sound_prediction_score ≥ 0.90`,
  merged with the clinical table. One clip = one solicited recording.
  Output metadata is identical to the paper's.
- **CIDRZ** — the recordings were segmented with our cough detector (`cad/`). The segmentation parameters used for the
  paper were edited afterwards and could not be recovered, so the segments are
  cut from stored sample offsets (`manifest/cidrz_segments.csv`). This gives
  the paper's 7,343 segments bit-exact.
- **TBScreen** — passive coughs: trim (top_db 25) → June CAD posterior ≥ 0.75
  → AST cough−speech log-margin ≥ −4 → per patient-device cap (random 25–29,
  seed 42). Note: the paper text says "cap 35"; the code that produced the
  numbers caps at 25–29.
- **build_db** — combines the three DBs (paper-era `build_db` notebook) and
  keeps clips of 0.3–1.0 s.

`manifest/paper_clips.csv` lists the paper clips in the order they were
embedded (`paper_row`) with their 128-clip extraction batch (`qwen_batch`);
`build_db.py` sorts by it. 02 uses `qwen_batch` to replay the batched Qwen
extraction (padding inside a batch changes the embeddings slightly).


## UK-COVID clips for the exact Qwen extraction

UK-COVID is not a study dataset, but in the original Qwen extraction one batch
(184) held 42 UK clips next to 86 TBScreen clips, and they affect those TBScreen
embeddings (see `docs/reproducibility.md`). HeAR and independent Qwen do not
need them.

To reproduce the paper Qwen embeddings, put the 32 raw WAVs listed in
`manifest/qwen_context.csv` under `datasets/raw/ukcovid_context/` and run
`python ukcovid_context.py`. It cuts the 42 clips (RMS 0.1 detector, original
demeaned samples) and checks the raw and segment PCM16 hashes. Then pass
`--include-uk-context` to `extract_embeddings.py` or `run_pipeline.py`. The UK
clips never enter `combine.csv`, cleaning or evaluation. Downloads at a
different sample rate must pass the same hash checks.

Dataset: The UK COVID-19 Vocal Audio Dataset Open Access Edition, UK Health
Security Agency, 2023, Open Government Licence v3.0. Audio is not included here.

`manifest/hear_reference_rows.csv` maps clips to rows of the original HeAR
matrix used for the paper (its metadata file had been overwritten). It is only
used for checking; extraction always runs on the WAVs.
