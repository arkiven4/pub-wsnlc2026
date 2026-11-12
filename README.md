# Weakly Supervised Noisy Label Cleaning for Cough-Based TB Screening

Code for the APSIPA ASC 2026 paper by Aprianto Dwi Prasetyo, Dhany Arifianto,
Katherin Indriawati, Bagus Tris Atmaja and Sakriani Sakti.

TB labels come from sputum tests, so they say whether *M. tuberculosis* is
present, not whether the cough sounds like TB. We embed each cough with a frozen
audio encoder (Qwen3-Omni audio tower or HeAR), project the embeddings onto a
Fisher axis, turn that into a TB-likelihood with class KDEs, and split the TB
clips with a two-component GMM into **Acoustically Consistent TB** (kept) and
**Acoustically Ambiguous TB** (removed). No model is trained for the cleaning.

This README walks through the full reproduction: downloading the three datasets,
preparing the clips, extracting embeddings, cleaning, evaluating, and producing
Table I and Figs. 1 and 4.

- [Program overview](docs/overview.md): the five stages and how data flows between them
- [Proposed method walkthrough](docs/proposed_method.md): the method, following the code
- [Workflow details](docs/workflow.md): inputs and outputs of every stage
- [Reproducibility notes](docs/reproducibility.md): exact settings and where the code differs from the paper text

## What you need

- Python 3.12 and a CUDA GPU. The GPU is needed for the AST filter in TBScreen
  preparation, for HeAR (TensorFlow) and for Qwen (bfloat16).
- About 1.2 GB for the prepared clips, plus the raw downloads
  (CIDRZ raw is about 80 GB; the TBScreen archive is about 400 GB before you
  extract the part we use).
- A Hugging Face account with access to [google/hear](https://huggingface.co/google/hear).
- For Qwen: the [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
  checkpoint on local disk. Only its audio tower is loaded.

The paper numbers were produced on an RTX 4090 with the package versions in
`requirements.txt`.

## 1. Set up the environment

```bash
git clone https://github.com/arkiven4/pub-wsnlc2026.git
cd pub-wsnlc2026
python3.12 -m venv .venv
source .venv/bin/activate
# install the CUDA build of torch/torchaudio/torchvision for your machine first, then:
python -m pip install -r requirements.txt
huggingface-cli login          # needed for google/hear
```

Optional, for Qwen:

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Thinking --local-dir /path/to/Qwen3-Omni-30B-A3B-Thinking
```

## 2. Download the datasets

All raw data goes under `datasets/raw/`. When you are done it should look like this:

```text
datasets/raw/
├── coda/
│   ├── meta_data/
│   │   ├── Clinical/CODA_TB_Clinical_Meta_Info.csv, CODA_TB_additional_variables_train.csv
│   │   └── Cough Metadata/CODA_TB_Solicited_Meta_Info.csv
│   └── raw_data/solicited_data/*.wav
├── cidrz/
│   ├── Chawama Phone A/*.wav, Kanyama Phone B/*.wav, ...   (one folder per site/device)
│   └── Metadata and Codebook/GHAI_Final_Data_2023.csv
├── tbscreen/
│   └── Passive_coughs/
│       ├── Passive_coughs.csv
│       └── Audio_files/*.wav
└── ukcovid_context/*.wav        (only for the exact Qwen setup, see 2.4)
```

Nothing under `datasets/` is committed; only the empty folders are.

### 2.1 CODA (TB DREAM Challenge)

CODA is hosted on Synapse. Create a Synapse account, open the challenge page
[synapse.org/TBcough](https://www.synapse.org/TBcough) (project `syn31472953`)
and accept the data use conditions. Then download the training data with the
Synapse client:

```bash
pip install synapseclient
synapse login                  # or: synapse login -p <personal access token>
mkdir -p "datasets/raw/coda/meta_data/Clinical" "datasets/raw/coda/meta_data/Cough Metadata" \
         datasets/raw/coda/raw_data/solicited_data
synapse get -r syn53710097 --downloadLocation "datasets/raw/coda/meta_data/Clinical"
synapse get -r syn53710098 --downloadLocation "datasets/raw/coda/meta_data/Cough Metadata"
synapse get -r syn40358494 --downloadLocation datasets/raw/coda/raw_data/solicited_data
```

The three IDs are the Train folders `meta_data/Clinical`, `meta_data/Cough Metadata`
and `raw_data/solicited_data`. If an ID ever changes, download the same folders
from the Files tab of the project page. Only the solicited coughs are used.

### 2.2 CIDRZ

CIDRZ (Centre for Infectious Disease Research in Zambia, the TB cough dataset
used in the [HeAR paper](https://arxiv.org/abs/2403.02522)) is on Kaggle as
[Google + CIDRZ Health AI Evaluation Zambia](https://www.kaggle.com/datasets/googlehealthai/google-health-ai).
Sign in to Kaggle, accept the dataset terms on that page, create an API token
(Kaggle → Settings → API → Create New Token, saved as `~/.kaggle/kaggle.json`),
then:

```bash
pip install kaggle
kaggle datasets download -d googlehealthai/google-health-ai -p datasets/raw/cidrz --unzip
```

After unzipping, the recording folders (one per site and device, e.g.
`Chawama Phone A`, `Audio-Recorder-Kanyama`) and `Metadata and Codebook/` must
sit directly in `datasets/raw/cidrz/`. If the archive unpacks into an extra
top-level folder, move its contents up one level. The X-ray image folder is not
used.

CIDRZ recordings are not segmented. The paper's 7,343 cough segments are cut
from sample offsets stored in `prep/manifest/cidrz_segments.csv`, so you get
exactly the paper's clips.

### 2.3 TBScreen

TBScreen (Sharma et al., *Science Advances* 2024) is on
[Zenodo](https://zenodo.org/records/10431329). The archive is large, and only
the passive coughs are needed:

```bash
wget https://tbscreen.s3.amazonaws.com/TBscreen_Dataset.zip
unzip TBscreen_Dataset.zip "*Passive_coughs/Passive_coughs.csv" "*Passive_coughs/Audio_files/*" -d tbscreen_tmp
mkdir -p datasets/raw/tbscreen
mv "$(find tbscreen_tmp -type d -name Passive_coughs | head -1)" datasets/raw/tbscreen/
rm -r tbscreen_tmp
```

Check that `datasets/raw/tbscreen/Passive_coughs/Passive_coughs.csv` and
`datasets/raw/tbscreen/Passive_coughs/Audio_files/` exist.

### 2.4 UK-COVID clips (only for the exact Qwen numbers)

Skip this if you only run HeAR or independent Qwen. In the original Qwen
extraction, one batch contained 42 UK-COVID clips next to 86 TBScreen clips, and
they slightly change those TBScreen embeddings (details in
[reproducibility notes](docs/reproducibility.md)). UK-COVID is never used for
cleaning or evaluation.

Download [The UK COVID-19 Vocal Audio Dataset, Open Access Edition](https://zenodo.org/records/10043978)
(Open Government Licence v3.0), then copy the 32 recordings listed in the `raw`
column of `prep/manifest/qwen_context.csv`:

```bash
mkdir -p datasets/raw/ukcovid_context
tail -n +2 prep/manifest/qwen_context.csv | cut -d, -f2 | sort -u | while read f; do
  find /path/to/uk_covid_download -name "$f" -exec cp {} datasets/raw/ukcovid_context/ \;
done
ls datasets/raw/ukcovid_context | wc -l     # should print 32
```

Stage 01 checks the hash of every file and stops if one does not match.

## 3. Reproduce the paper

Run everything from the repository root. The commands below use HeAR, which needs
no Qwen weights. Every script has `--help`.

### Step 01: prepare the clips

```bash
python scripts/01_prepare_data.py --workers 8
# for the exact Qwen setup add: --include-uk-context
```

This selects the recordings, cuts or trims the clips, applies the cough filters
(TBScreen: our cough detector in `cad/`, then the AST cough-vs-speech
check) and writes `datasets/preprocess/` and `data/metadata/combine.csv`.
Expected: **23,638 clips** (CODA 9,279, CIDRZ 7,343, TBScreen 7,016) with
2,773 / 1,279 / 5,087 TB clips.

### Step 02: extract embeddings

```bash
python scripts/02_extract_embeddings.py --encoder hear --check-only   # checks paths, loads no model
python scripts/02_extract_embeddings.py --encoder hear
```

Writes `data/embeds/embeds_hear.npy` (23,638 × 512) and `metadata_hear.csv` in
the same row order. For Qwen (23,638 × 2048), exactly as in the paper:

```bash
python scripts/02_extract_embeddings.py --encoder qwen --include-uk-context --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```

or, without UK-COVID audio (one clip per forward pass; numbers differ from the paper):

```bash
python scripts/02_extract_embeddings.py --encoder qwen --qwen-mode independent --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```

### Step 03: clean the labels

```bash
python scripts/03_clean_labels.py --encoders hear
```

Runs Proposed, CL (PBNR) and CL (CL) per dataset and writes
`data/cleaning/decisions_hear.{csv,npz,json}`. The CSV has one row per clip with
its keep flag, TB-likelihood and Fisher coordinate, so you can inspect which TB
clips were marked Consistent or Ambiguous. Expected Consistent TB counts:

| | CODA | CIDRZ | TBScreen |
|---|---|---|---|
| HeAR | 1,453 | 578 | 3,457 |
| Qwen | 1,454 | 562 | 3,268 |

### Step 04: evaluate

```bash
python scripts/04_evaluate.py --encoders hear --workers 16
```

Trains LR, MLP and HistGradientBoosting with participant-grouped 5-fold CV for
Original, Random (15 repeats), CL (PBNR), CL (CL) and Proposed labels. Results
go to `result/table_i/`.

### Step 05: make Table I

```bash
python scripts/05_make_table.py --encoders hear
```

Writes `result/table_i/table_i.md`, `table_i.csv` and `table_i_comparison.csv`
(each cell next to the published value). With HeAR, and with Qwen using
`--include-uk-context`, all means and standard deviations match the paper.
Our run is in [result/reproduced_table_i](result/reproduced_table_i/table_i.md).

### Step 06: acoustic figures (Figs. 1 and 4)

```bash
python scripts/06_acoustic_figures.py --encoder qwen --dataset coda --workers 8
```

Recomputes 153 MFCC/spectral statistics for every CODA clip and plots the
MFCC-0 10th percentile, standard deviation and 25th percentile before cleaning
(Non-TB vs TB) and after cleaning (Non-TB / Ambiguous / Consistent TB). Output
in `result/acoustic_figures/`. Fig. 1 needs only the audio; Fig. 4 uses the
stage-03 decisions of the chosen encoder.

### Everything in one command

```bash
python run_pipeline.py --encoders hear --workers 8 --eval-workers 16
# both encoders, exact paper setup:
python run_pipeline.py --include-uk-context --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```

The root notebooks `01_*.ipynb` to `05_*.ipynb` run the same stages if you
prefer Jupyter.

## Notes

- Cleaning is fitted on each whole dataset before classifier CV, as in the
  paper. Sensitivity after removing Ambiguous TB is an upper bound, not
  screening sensitivity.
- A few details in the code differ from the paper text (TBScreen cap 25–29 per
  patient and device instead of 35; the Random baseline removes as many clips as
  Proposed keeps). The published numbers come from this code. See
  [reproducibility notes](docs/reproducibility.md).

## Repository map

```text
scripts/          Numbered stages 01–06
prep/             Dataset preparation and manifests
cad/              Cough detector (ONNX) used for TBScreen
lib/              Encoders, cleaning, evaluation and table code
01_*.ipynb … 05_*  Notebook versions of the stages
datasets/         raw/ and preprocess/ (contents not committed)
data/             Generated metadata, embeddings and cleaning decisions (not committed)
result/           Tables and figures (reproduced Table I included)
docs/             Overview, method walkthrough, workflow, reproducibility notes
extras/           Older combined notebook
```

`extract_embeddings.py` and `paper_study.py` are older entry points that still
work. Doxygen pages can be built with `doxygen Doxyfile`.

## Datasets and licences

Please cite the dataset papers and follow each provider's terms. CODA: CODA TB
DREAM Challenge (Synapse `syn31472953`). CIDRZ: Google + CIDRZ Health AI Evaluation
Zambia (Kaggle `googlehealthai/google-health-ai`). TBScreen: Sharma et al., *Science Advances* 2024,
doi:10.1126/sciadv.adi0282. UK-COVID: The UK COVID-19 Vocal Audio Dataset Open
Access Edition, UK Health Security Agency, 2023, Open Government Licence v3.0.
No audio is distributed in this repository.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{prasetyo2026weakly,
  title     = {Weakly Supervised Noisy Label Cleaning for Cough-Based {TB} Screening},
  author    = {Prasetyo, Aprianto Dwi and Arifianto, Dhany and Indriawati, Katherin and
               Atmaja, Bagus Tris and Sakti, Sakriani},
  booktitle = {Proceedings of the Asia-Pacific Signal and Information Processing
               Association Annual Summit and Conference (APSIPA ASC)},
  year      = {2026}
}
```

