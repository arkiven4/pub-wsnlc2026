"""Combine the three DBs into the paper clip list.

Port of the original build_db notebook cell 7 (NXSCough commit
869e1ba, the version that produced the paper list) plus the TBScreen column
mapping of the original TBScreen preprocessing notebook cell 38. UK-COVID (db 3 in
the original notebook) is not used in the paper and is left out.

Rows are then put in the paper's order (prep/manifest/paper_clips.csv). The
order matters only for Qwen extraction: the paper embedded clips in batches of
128 consecutive rows, and padded batches leak across clips (see 02).

Output data/metadata/combine.csv
"""
import warnings

import numpy as np
import pandas as pd
import soundfile as sf

from paths import RAW, META, MANIFEST, absolute

warnings.simplefilter("ignore", FutureWarning)  # pandas replace() downcasting notice

COLS = ["path_file", "participant", "gender", "disease_status", "weight_loss", "hemoptysis", "smoker", "night_sweats", "hiv", "db"]


def get_duration(path):
    """@brief Read a clip duration without decoding the complete waveform.
    @param path Absolute or repository-relative audio path.
    @return Duration in seconds, or None when the file cannot be read.
    @note Unreadable rows fail the later duration filter rather than acquiring a fabricated duration.
    """
    try:
        with sf.SoundFile(absolute(path)) as f:
            return len(f) / f.samplerate
    except Exception:
        return None


def coda():
    """@brief Map prepared CODA clinical fields to the shared study schema.
    @return Frame containing original binary labels, harmonized covariates and database ID 0.
    @details Join HIV metadata and encode participants before combining datasets.
    @note These are clinical metadata mappings, not acoustic label-cleaning decisions.
    """
    extra = pd.read_csv(RAW / "coda" / "meta_data" / "Clinical" / "CODA_TB_additional_variables_train.csv")
    df = pd.read_csv(META / "coda_solicited.csv").rename(columns={"tb_status": "disease_status"})
    df = df.merge(extra[["participant", "HIVstatus"]], on="participant", how="left")
    df = df.rename(columns={"sex": "gender", "smoke_lweek": "smoker", "HIVstatus": "hiv"})
    df["db"] = 0
    df = df[COLS]
    cols = ["weight_loss", "hemoptysis", "smoker", "night_sweats"]
    df[cols] = df[cols].replace({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].replace({"Male": 0, "Female": 1})
    df["hiv"] = df["hiv"].replace({"Negative": 0, "Positive": 1, "Unknown": np.nan})
    df["participant_original"] = "CODA_" + df["participant"]
    df["participant"] = df["participant"].astype("category").cat.codes
    return df


def cidrz():
    """@brief Map prepared CIDRZ segments and clinical fields to the shared schema.
    @return Complete-case rows with database ID 1 and participant codes offset by 1000.
    @note The original mapping treats both current and stopped tobacco use as 1.
          Preserve that convention rather than inferring a new smoker definition.
    """
    df = pd.read_csv(META / "cidrz_segments.csv")
    df = df.rename(columns={"ground_truth_tb": "disease_status", "anon_id": "participant", "sex": "gender",
                            "tobacco_use": "smoker", "haemoptysis": "hemoptysis", "ngtsweats": "night_sweats",
                            "weightloss": "weight_loss", "hiv_status": "hiv"})
    df["db"] = 1
    df["disease_status"] = df["disease_status"].replace({"neg": 0, "pos": 1})
    cols = ["weight_loss", "hemoptysis", "night_sweats"]
    df[cols] = df[cols].replace({"yes": 1, "no": 0})
    df["smoker"] = df["smoker"].replace({"current": 1, "never": 0, "stopped": 1})
    df["gender"] = df["gender"].replace({"m": 0, "f": 1})
    df["hiv"] = df["hiv"].replace({"neg": 0, "pos": 1})
    df = df.dropna(subset=["hemoptysis"])
    df = df[COLS]
    df["participant_original"] = "CIDRZ_" + df["participant"].astype(str)
    df["participant"] = 1000 + df["participant"].astype("category").cat.codes
    return df.dropna().reset_index(drop=True)


def tbscreen():
    """@brief Map TBScreen passive-cough fields to the shared study schema.
    @return Frame with database ID 2 and participant codes offset by 2000.
    @note The smoker remap is repeated after numeric conversion because earlier
          rows can contain the string value "2". Both passes are intentional.
    """
    df = pd.read_csv(META / "tbscreen_coughs.csv", low_memory=False)
    df = df.rename(columns={"subject": "participant", "gndr_a": "gender", "smkr_a": "smoker", "un_wght_loss": "weight_loss",
                            "nght_swts": "night_sweats", "cough_bld": "hemoptysis", "hiv_infected_a": "hiv",
                            "Label": "disease_status"})
    df["db"] = 2
    df["smoker"] = df["smoker"].replace({2: 1})
    df["disease_status"] = df["disease_status"].replace({"NTB": 0, "TB": 1})
    cols = ["gender", "disease_status", "weight_loss", "hemoptysis", "smoker", "night_sweats", "hiv"]
    df[cols] = df[cols].replace("unknown", 0).astype(float)
    df["smoker"] = df["smoker"].replace({2: 1})  # build_db repeats this; values are strings before astype
    df = df[COLS]
    df["participant_original"] = "TBScreen_" + df["participant"]
    df["participant"] = 2000 + df["participant"].astype("category").cat.codes
    return df


def main():
    """@brief Assemble the three prepared datasets into one paper-ordered clip list.
    @return None; write data/metadata/combine.csv.
    @details Harmonize clinical fields/participant IDs, retain durations0.3–1.0s,
             join paper clip IDs, reject missing/extra clips, and sort by paper_row.
    @note Row order defines the original Qwen batch membership. Metadata labels
          remain the supplied clinical labels; acoustic cleaning is a later stage.
    """
    df = pd.concat([coda(), cidrz(), tbscreen()], ignore_index=True)
    df["hiv"] = df["hiv"].fillna(0)
    print("before duration filter:", df["db"].value_counts().sort_index().to_dict())
    df["duration_sec"] = df["path_file"].map(get_duration)
    df = df[(df["duration_sec"] >= 0.3) & (df["duration_sec"] <= 1.0)].reset_index(drop=True)
    print("after 0.3-1.0 s filter:", df["db"].value_counts().sort_index().to_dict())

    clips = pd.read_csv(MANIFEST / "paper_clips.csv")
    df["clip"] = df["db"].astype(str) + "/" + df["path_file"].str.rsplit("/", n=1).str[-1]
    df = df.merge(clips[["clip", "paper_row", "qwen_batch"]], on="clip", how="left", validate="one_to_one")
    missing = set(clips["clip"]) - set(df["clip"])
    extra = df["paper_row"].isna().sum()
    print(f"paper clips missing: {len(missing)}, clips not in paper list: {extra}")
    if missing or extra:
        raise ValueError("Prepared clip list differs from the paper manifest; check raw inputs and filtering.")
    df = df.sort_values("paper_row", na_position="last", kind="stable").reset_index(drop=True)

    df = df[["clip", "path_file", "participant", "participant_original", "gender", "disease_status", "weight_loss",
             "hemoptysis", "smoker", "night_sweats", "hiv", "db", "duration_sec", "paper_row", "qwen_batch"]]
    df.to_csv(META / "combine.csv", index=False)
    print("TB clips per DB:", df[df["disease_status"] == 1].groupby("db").size().to_dict())


if __name__ == "__main__":
    main()
