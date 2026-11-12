"""DB2 CIDRZ: unsegmented recordings -> cough segments.

Port of the original CIDRZ cough-extraction notebook.
  1. metadata_cidrz_cough.csv: GHAI clinical table joined to every raw wav
     (notebook cell 2).
  2. Cough segments. The paper segments were cut by the June CAD with
     parameters that were later edited and are not recoverable, so the
     default (`--mode paper`) cuts them from the stored sample offsets
     (prep/manifest/cidrz_segments.csv). Each paper segment is a verbatim slice
     librosa.load(raw, 16 kHz, mono)[start:end] written as PCM_16; this
     reproduces all 7,343 paper segments bit-exact.

Input  datasets/raw/cidrz/  (official download: site/phone folders + "Metadata and Codebook/")
Output datasets/preprocess/cidrz/*.wav, data/metadata/cidrz_cough.csv, data/metadata/cidrz_segments.csv
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from paths import RAW, WAVS, META, MANIFEST, SR, rel

LABEL_FOLDER_MAP = {"Audio-Recorder": 0, "Phone A": 1, "Phone B": 2, "Phone C": 3}
OUT = WAVS / "cidrz"


def build_recording_metadata():
    """@brief Match available CIDRZ recordings to the official clinical metadata.
    @return Recording DataFrame with relative source paths and phone codes.
    @note Unmatched recording stems are skipped; segment selection comes later
          from the supplied paper-offset manifest, not from this file scan.
    """
    root = RAW / "cidrz"
    df = pd.read_csv(root / "Metadata and Codebook" / "GHAI_Final_Data_2023.csv")
    rows = []
    for wav in sorted(root.rglob("*.wav")):
        folder = wav.relative_to(root).parts[0]
        phone = next((v for k, v in LABEL_FOLDER_MAP.items() if k in folder), 9)
        sel = df[df["barcode"] == wav.stem]
        if sel.empty:
            continue
        sel = sel.copy()
        sel["phone_code"] = phone
        sel["raw"] = str(wav.relative_to(root))
        rows.append(sel)
    return pd.concat(rows, ignore_index=True)


def cut_recording(args):
    """@brief Write the manifest-selected segments from one CIDRZ recording.
    @param args Pair of raw relative path and (filename, start, end) segment tuples.
    @return Number of segments written.
    @details Load mono at 16 kHz, slice the stored sample offsets, and write PCM16.
          Offsets are sample indices at that rate, with end excluded by slicing.
    """
    raw, segs = args
    audio, _ = librosa.load(RAW / "cidrz" / raw, sr=SR, mono=True)
    for file, start, end in segs:
        sf.write(OUT / file, audio[start:end], SR, subtype="PCM_16")
    return len(segs)


def main():
    """@brief Prepare recording metadata and write the paper CIDRZ segments in parallel.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper"], default="paper")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rec = build_recording_metadata()
    META.mkdir(parents=True, exist_ok=True)
    rec.to_csv(META / "cidrz_cough.csv", index=False)
    print("CIDRZ recordings:", len(rec))

    seg = pd.read_csv(MANIFEST / "cidrz_segments.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(raw, list(g[["file", "start", "end"]].itertuples(index=False, name=None)))
            for raw, g in seg.groupby("raw", sort=False)]
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(cut_recording, jobs), total=len(jobs), desc="cut"):
            pass

    out = seg[["file", "raw"]].merge(rec, on="raw", how="left", validate="many_to_one")
    out.insert(0, "path_file", [rel(OUT / f) for f in out["file"]])
    out.to_csv(META / "cidrz_segments.csv", index=False)
    print("CIDRZ segments:", len(out), out["ground_truth_tb"].value_counts().to_dict())


if __name__ == "__main__":
    main()
