"""DB3 TBScreen: passive (automatic) coughs.

Port of the original TBScreen preprocessing notebook, cells 28-38 (the path that
produced the paper's metadata_coughs_original.csv / metadata_coughs.csv):
  1. each passive clip: load 16 kHz, remove DC, librosa trim (top_db=25),
     write PCM_16 wav, CAD posterior mean on the trimmed clip (June CAD);
  2. keep cad_probs >= 0.75;
  3. AST (AudioSet) log p(Cough) - log p(Speech) margin on the written wav,
     keep margin >= -4;
  4. per (type, subject_device): keep all if < 25 clips, else a random
     25-29 (rng = default_rng(42), sample random_state=42).
     (The paper text says "cap 35 per patient per device"; the code that
     produced the numbers caps at 25-29.)
The paper only uses the passive ("longi") clips, so the forced-cough folder is
not needed. Groups are processed in (type, subject_device) order, so leaving
the forced clips out does not change the random draws for the passive ones.

Input  datasets/raw/tbscreen/Passive_coughs/{Passive_coughs.csv,Audio_files/}
Output datasets/preprocess/tbscreen/longi_*.wav,
       data/metadata/tbscreen_coughs_original.csv (after CAD + AST filters),
       data/metadata/tbscreen_coughs.csv (after the per-device cap)
"""
import argparse
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from paths import RAW, WAVS, META, CAD, SR, rel, absolute

sys.path.insert(0, str(CAD))
from infer import CADDetector  # noqa: E402

OUT = WAVS / "tbscreen"
_CAD = None


def load_passive():
    """@brief Read passive-cough metadata and resolve available raw recordings.
    @return Frame with absolute input paths, patient/device grouping and longi type.
    @note Missing recordings are omitted. Remaining missing metadata becomes
          "unknown" and is handled by the later schema harmonization.
    """
    root = RAW / "tbscreen" / "Passive_coughs"
    df = pd.read_csv(root / "Passive_coughs.csv")
    df["path"] = [str(root / "Audio_files" / f"{p}.wav") for p in df["path"]]
    df["subject_device"] = df["subject"] + "_" + df["device"]
    df["type"] = "longi"
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
    df = df.drop(columns=["S.NO", "all_device_cough_count", "single_device_cough_count"], errors="ignore")
    return df.fillna("unknown")


def process_row(row):
    """@brief Prepare one passive cough and measure its June-CAD confidence.
    @param row Recording metadata dictionary containing the raw audio path.
    @return Updated row with prepared path_file and mean cad_probs; also write a WAV.
    @details Load at 16 kHz, remove DC, trim at 25 dB, then score the trimmed waveform.
    @note Each worker initializes its CAD once. The caller applies the 0.75 filter;
          this function writes the candidate audio before that selection.
    """
    global _CAD
    if _CAD is None:
        _CAD = CADDetector(str(CAD))
    x, fs = librosa.load(row["path"], sr=SR)
    x = x - np.mean(x)
    x_trimmed, _ = librosa.effects.trim(x, top_db=25)
    probs = _CAD.posterior_from_wav(x_trimmed, chunk_sec=0.0)
    stem = os.path.splitext(os.path.basename(row["path"]))[0]
    out_path = OUT / f"{row['type']}_{stem}.wav"
    sf.write(out_path, x_trimmed, fs)
    row["path_file"] = rel(out_path)
    row["cad_probs"] = float(probs.mean())
    return row


def ast_margin(paths, batch_size=32, num_workers=8):
    """@brief Score prepared coughs by AST Cough-versus-Speech log margin.
    @param paths Prepared WAV paths, retained as keys in the returned mapping.
    @param batch_size Number of waveforms per AST inference call.
    @param num_workers DataLoader workers used to load audio.
    @return Mapping from each input path to logP(Cough) - logP(Speech).
    @note This value is a log margin, not a cough probability. The paper path
          retains margins at least -4; it reads the already written candidate WAVs.
    """
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = ASTForAudioClassification.from_pretrained(name).to(device).eval()
    cough_id, speech_id = model.config.label2id["Cough"], model.config.label2id["Speech"]

    def collate(batch):
        audios = [librosa.load(absolute(p), sr=SR)[0] for p in batch]
        return fe(audios, sampling_rate=SR, return_tensors="pt", padding=True), batch

    loader = DataLoader(paths, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    out = {}
    with torch.no_grad():
        for inputs, batch in tqdm(loader, desc="AST"):
            logp = F.log_softmax(model(**{k: v.to(device) for k, v in inputs.items()}).logits, dim=-1)
            margin = (logp[:, cough_id] - logp[:, speech_id]).cpu().numpy()
            out.update(zip(batch, margin.astype(float)))
    return out


def cap_per_device(df):
    """@brief Apply the original seeded cap independently within patient/device groups.
    @param df Eligible TBScreen rows with type and subject_device columns.
    @return Selected rows with a fresh index.
    @details Keep groups shorter than 25; otherwise sample a seeded cap of 25–29,
             limited by the group size. Preserve the original group iteration order.
    @note The paper text says35, but reproducing the code requires this 25–29 cap.
    """
    rng = np.random.default_rng(42)
    return (
        df.groupby(["type", "subject_device"], group_keys=False)
        .apply(lambda x: x if len(x) < 25 else x.sample(n=min(rng.integers(25, 30), len(x)), random_state=42))
        .reset_index(drop=True)
    )


def main():
    """@brief Trim passive coughs, apply June CAD/AST filters, cap per device, and save metadata.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = load_passive()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        rows = list(tqdm(pool.imap(process_row, df.to_dict("records")), total=len(df), desc="trim+CAD"))
    df = pd.DataFrame(rows)
    df = df[df["cad_probs"] >= 0.75]

    margin = ast_margin(df["path_file"].tolist())
    df["margin"] = df["path_file"].map(margin)
    df = df[df["margin"] >= -4].reset_index(drop=True)
    META.mkdir(parents=True, exist_ok=True)
    df.to_csv(META / "tbscreen_coughs_original.csv", index=False)
    print("TBScreen after CAD+AST:", len(df))

    df = cap_per_device(df)
    df = df[df["type"] == "longi"]
    df.to_csv(META / "tbscreen_coughs.csv", index=False)
    print("TBScreen after cap:", len(df), df["Label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
