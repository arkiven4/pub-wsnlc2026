"""Stage 04: evaluate previously generated cleaning decisions; do not refit cleaners."""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import utils as utils_indev
from .study_cleaning import STAGES, load_cleaning


def evaluate_study(encoder, embeddings_dir, cleaning_dir, output_dir, workers=16):
    """@brief Evaluate all five arms using the saved cleaning decisions.
    @param encoder Encoder name used in the matrix and decision filenames.
    @param embeddings_dir Stage-02 features and aligned metadata.
    @param cleaning_dir Stage-03 masks and their input fingerprints.
    @param output_dir Destination for overall, LODO and retained-TB-count CSVs.
    @param workers Number of parallel classifier workers.
    @return Experiment filename suffix, for example "emb hear_perdb".
    @details Validate the decisions, select rows from the original frozen features,
             queue grouped-CV classifier jobs, and aggregate each arm. Cleaning
             models are not refitted here. LODO is supplementary to Table I.
    @note The Random arm uses 15 seeds. CL-PBNR and CL-CL are each one cleaner
          with five-fold out-of-fold probability estimation, not 15 repetitions.
    """
    df, embeddings_raw, _, _, _, _ = utils_indev.load_data_precomp(
        db_filter=[0, 1, 2], name=encoder, base_dir=embeddings_dir)
    decisions = load_cleaning(encoder, embeddings_dir, cleaning_dir, len(df))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    CSV_DIR = str(output_dir)
    experiment_string = f'emb {encoder}_perdb'
    overall_metrics, overall_lodo_metrics, overall_tb_count = [], [], []
    jobs, job_specs = [], []

    def add_stage_jobs(eval_df, stage_name):
        """@brief Queue in-domain and leave-one-database-out jobs for one selection.
        @param eval_df Selected metadata rows with their original frozen embeddings.
        @param stage_name Label attached to the resulting summaries.
        @return None; append jobs and their tags to the enclosing lists.
        @note Warn and omit an evaluation target if selection leaves only one class.
        """
        present = sorted(eval_df['db'].unique())
        eval_dbs = [d for d in present if d != 3]   # db 3 kicked out before LODO/per-db

        target_dbs = []
        for d in eval_dbs:
            if eval_df.loc[eval_df['db'] == d, 'disease_status'].nunique() < 2:
                tqdm.write(f"[WARN] {experiment_string} | {stage_name}: db {d} is "
                           f"single-class after cleaning -> dropped from per-db & "
                           f"LODO targets (kept as training data)")
                continue
            target_dbs.append(d)

        for now_db in target_dbs:
            jobs.append({"df": eval_df, "train_db": now_db, "test_dbs": []})
            job_specs.append(("perdb", {"db": now_db, "stage": stage_name}))

        for held_out in target_dbs:
            train_dbs = [d for d in present if d != held_out]   # all other present dbs (incl db 3 extra data)
            jobs.append({"df": eval_df, "train_db": train_dbs, "test_dbs": [held_out]})
            job_specs.append(("lodo", {
                "held_out_db": held_out,
                "train_dbs": str(train_dbs),
                "stage": stage_name,
            }))

    def add_random_jobs(db_counts, do_refine, stage_name="Random 50% Cleaning", n_rep=15):
        """@brief Queue the paper's seeded random-removal baseline.
        @param db_counts Per-database TB counts returned by the Proposed selection.
        @param do_refine Tag retained for the original result grouping.
        @param stage_name Label attached to the queued random jobs.
        @param n_rep Number of independent seeds; the paper path uses 15.
        @return None; append one in-domain job per seed and eligible database.
        @note The original implementation passes retained TB counts as drop counts.
              Preserve that behavior for reproduction; the paper wording differs.
        """
        counts = dict(zip(db_counts["db"], db_counts["tb_count"]))
        tb_all = (df["disease_status"] == 1).values
        db_vals = df["db"].values
        for seed in range(n_rep):
            rng = np.random.default_rng(seed)
            rev = tb_all.astype(int)              # 1 = TB, 0 = non-TB
            for now_db, k in counts.items():
                if k <= 0:
                    continue
                idx = np.where((db_vals == now_db) & tb_all)[0]
                rev[rng.choice(idx, size=min(int(k), len(idx)), replace=False)] = 2
            keep = (rev != 2) & (db_vals != 3)
            df_eval = df[keep].copy()
            df_eval["embed"] = list(embeddings_raw[keep])
            df_eval = df_eval.reset_index(drop=True)
            for now_db in sorted(df_eval["db"].unique()):
                if now_db == 3:
                    continue
                jobs.append({"df": df_eval, "train_db": now_db, "test_dbs": []})
                job_specs.append(("random", {"db": now_db, "seed": seed,
                                             "stage": stage_name, "do_refine": do_refine}))

    # Classifier job order and random sampling are identical to paper_study.py.
    for key, stage_name in STAGES:
        mask = decisions[key]
        selected = df[mask].copy()
        selected['embed'] = list(embeddings_raw[mask])
        selected = selected.reset_index(drop=True)
        add_stage_jobs(selected, stage_name)
        counts = (df[(df.disease_status == 1) & decisions[f'consistent_{key}']].groupby('db')
                  .size().reset_index(name='tb_count'))
        counts['stage'] = stage_name
        overall_tb_count.append(counts)
        if key == 'proposed':
            # Preserve the original implementation: these retained TB counts
            # are passed as random drop counts (see docs/reproducibility.md).
            add_random_jobs(counts, False)

    # ---- One flattened parallel run over every collected job ----
    summaries = utils_indev.run_embedding_classifications_batched(
        jobs, label_col="disease_status", n_splits=5, n_components=256, n_jobs=workers,
    )
    random_rows = []
    for (kind, tag), summ in zip(job_specs, summaries):
        for col, val in tag.items():
            summ[col] = val
        if kind == "perdb":
            overall_metrics.append(summ)
        elif kind == "lodo":
            overall_lodo_metrics.append(summ)
        else:   # random -> aggregated over seeds after the loop
            random_rows.append(summ)

    # Aggregate the random baseline: mean +/- std over seeds, kept separate per
    # proposed n (do_refine) and db, matching the old two-block output.
    if random_rows:
        random_df = pd.concat(random_rows, ignore_index=True)
        metric_cols = ["acc", "bacc", "sens", "spec", "auroc", "auprc", "pauroc"]
        for col in metric_cols:
            random_df[col] = random_df[col].astype(str).str.split("±").str[0].astype(float)
        gb = random_df.groupby(["do_refine", "db"])
        mean_df, std_df = gb[metric_cols].mean(), gb[metric_cols].std()
        random_agg = pd.DataFrame(index=mean_df.index)
        for col in metric_cols:
            random_agg[col] = mean_df[col].round(2).astype(str) + " ± " + std_df[col].round(2).astype(str)
        random_agg = random_agg.reset_index().drop(columns=["do_refine"])
        random_agg["stage"] = "Random Cleaning"
        overall_metrics.append(random_agg)

    overall_df = pd.concat(overall_metrics, ignore_index=True)
    overall_lodo_df = pd.concat(overall_lodo_metrics, ignore_index=True)
    overall_tbcount_df = pd.concat(overall_tb_count, ignore_index=True)

    overall_df.to_csv(f"{CSV_DIR}/overall_{experiment_string}.csv", index=False)
    overall_lodo_df.to_csv(f"{CSV_DIR}/lodo_{experiment_string}.csv", index=False)
    overall_tbcount_df.to_csv(f"{CSV_DIR}/tbcount_{experiment_string}.csv", index=False)

    return experiment_string
