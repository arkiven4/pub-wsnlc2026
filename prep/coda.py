"""DB1 CODA TB DREAM Challenge: solicited coughs.

Port of the original CODA preprocessing notebook (cells 0, 7-8, 42-45)
in the form that produced the paper's `solicited_original.csv`:
  official solicited meta, sound_prediction_score >= 0.90, merged with the raw
  clinical table; one clip = one solicited recording (no segmentation).

Input  datasets/raw/coda/meta_data/{Clinical,Cough Metadata}/*.csv
       datasets/raw/coda/raw_data/solicited_data/*.wav
Output datasets/preprocess/coda/*.wav, data/metadata/coda_solicited.csv
"""
import filecmp
import shutil

import pandas as pd

from paths import RAW, WAVS, META, rel

TB_CONDITION_MAP = {"tb_prior": 1, "tb_prior_Pul": 2, "tb_prior_Extrapul": 4, "tb_prior_Unknown": 8}


def prior_code(row):
    """@brief Encode the supplied prior-TB indicators as the original metadata bit field.
    @param row Metadata row containing the TB_CONDITION_MAP Yes/No columns.
    @return Sum of the weights whose indicators are Yes.
    @note This metadata field is not a newly inferred clinical label.
    """
    return sum(w for c, w in TB_CONDITION_MAP.items() if row[c] == "Yes")


def main():
    """@brief Prepare the paper's solicited CODA clips and metadata.
    @return None; write CODA WAVs under datasets/preprocess and metadata under data/metadata.
    @details Join solicited/clinical metadata, retain sound_prediction_score >=0.90,
             discard missing source files, then copy selected WAVs verbatim.
    @note Replace stale processed files if they differ from the raw recording.
          Extra trimming or cough detection would change the paper input.
    """
    meta = RAW / "coda" / "meta_data"
    clinical = pd.read_csv(meta / "Clinical" / "CODA_TB_Clinical_Meta_Info.csv")
    solic = pd.read_csv(meta / "Cough Metadata" / "CODA_TB_Solicited_Meta_Info.csv")
    solic = solic[["participant", "filename", "sound_prediction_score"]].merge(clinical, on="participant", how="left")
    solic = solic[solic["sound_prediction_score"] >= 0.90]
    solic["tb_type"] = solic.apply(prior_code, axis=1)

    src = RAW / "coda" / "raw_data" / "solicited_data"
    solic = solic[solic["filename"].map(lambda f: (src / f).is_file())].reset_index(drop=True)

    out = WAVS / "coda"
    out.mkdir(parents=True, exist_ok=True)
    for f in solic["filename"]:
        # Replace stale processed clips left by earlier pipeline versions.
        # Paper solicited audio is copied verbatim from the raw download.
        if not (out / f).exists() or not filecmp.cmp(src / f, out / f, shallow=False):
            shutil.copy2(src / f, out / f)
    solic["path_file"] = [rel(out / f) for f in solic["filename"]]

    META.mkdir(parents=True, exist_ok=True)
    solic.to_csv(META / "coda_solicited.csv", index=False)
    print("CODA clips:", len(solic), solic["tb_status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
