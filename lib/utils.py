import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    balanced_accuracy_score,
    roc_auc_score, roc_curve, average_precision_score
)
from sklearn.base import clone
from joblib import Parallel, delayed

import soundfile as sf
import librosa

# =============================================================================
# DATA LOADING
# =============================================================================

def load_audio(path: str, target_sr: int = 16000):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = np.mean(y, axis=1)  # mono

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = np.max(np.abs(y))
    y = y / max_val if max_val != 0 else y

    return y, sr

def prepare_labels(disease_status):
    if isinstance(disease_status, pd.Series):
        disease_status = disease_status.values
    label_map = {
        'TB': 1, 'tb': 1, 'positive': 1, 'pos': 1, '1': 1, 1: 1,
        'NTB': 0, 'ntb': 0, 'negative': 0, 'neg': 0, '0': 0, 0: 0,
        'healthy': 0, 'control': 0,
    }
    labels = np.array([label_map.get(str(int(s)).strip(), -1) for s in disease_status])
    if (labels == -1).sum() > 0:
        print(f"  WARNING: {(labels==-1).sum()} unknown labels")
    return labels

def load_data_precomp(db_filter=None, name=None, base_dir=None):
    """@brief Load stage-02 embeddings and metadata without changing their row alignment.
    @param db_filter Optional list of database IDs to retain.
    @param name Encoder suffix in the paired .npy and .csv filenames.
    @param base_dir Directory containing those two files.
    @return Metadata frame, embeddings, labels, gender, participant IDs and database IDs.
    @exception ValueError Rows, labels, finiteness or clip identities are invalid.
    @note These are features computed by extraction, not paper reference vectors.
    """
    from pathlib import Path
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1] / "data/embeds"
    suffix = f"_{name}" if name else ""
    df = pd.read_csv(base / f"metadata{suffix}.csv")
    embs = np.load(base / f"embeds{suffix}.npy")
    if embs.ndim != 2 or len(df) != len(embs) or not np.isfinite(embs).all():
        raise ValueError("Invalid matrix or metadata / embedding row mismatch")
    if df.path_file.duplicated().any():
        raise ValueError("Duplicate clips in embedding metadata")
    df["embed"] = list(embs)
    if db_filter is not None:
        df = df[df.db.isin(db_filter)].reset_index(drop=True)
    labels = prepare_labels(df.disease_status)
    if (labels < 0).any():
        raise ValueError("Unknown labels in metadata")
    return (df, np.stack(df.embed), labels, df.gender.to_numpy(),
            df.participant.to_numpy(), df.db.to_numpy())


# =============================================================================
# Other Funtion
# =============================================================================
def optimize_threshold_youden(y_true, y_prob):
    """@brief Choose the calibration threshold that balances sensitivity and specificity.
    @param y_true Binary calibration labels.
    @param y_prob Predicted TB probabilities for the same calibration clips.
    @return ROC threshold minimizing abs(TPR - TNR), using the first minimum.
    @note The function name says Youden, but the implementation does
          not maximize Youden's J. Keep the actual rule for paper reproduction.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # youden_j = tpr - fpr
    # best_idx = np.argmax(youden_j)

    specificity = 1 - fpr
    gap = np.abs(tpr - specificity)
    best_idx = np.argmin(gap)

    return thresholds[best_idx]

def _make_base_classifiers() -> dict:
    """@brief Construct the paper's LR, MLP and histogram-gradient-boosting estimators.
    @return Mapping from readable classifier names to unfitted estimators.
    @note Scaling and PCA belong to the fold preparation, not these estimators.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
        # SVM(RBF, probability=True) was the runtime bottleneck (~108s/fit on
        # 16k samples — 20-70x the other models — because probability=True adds
        # an internal 5-fold Platt CV on an O(n^2) kernel). HistGradientBoosting
        # is 16-84x faster and matches SVM on LODO AUROC (in-domain -0.03).
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }


def get_classifiers(n_components=None) -> dict:
    """
    Return a dict of classifiers to compare.
    All wrapped in a Pipeline with StandardScaler + PCA for fair comparison.
    PCA(n_components) reduces embeddings to a stable space for all classifiers.
    """
    scale_step = ("scaler", StandardScaler())

    def make(clf):
        steps = [scale_step]
        if n_components is not None:
            steps.append(("pca", PCA(n_components=n_components, random_state=42)))
        steps.append(("clf", clf))
        return Pipeline(steps)

    return {name: make(clone(clf)) for name, clf in _make_base_classifiers().items()}

def compute_metrics(y_true, y_pred, y_proba):
    classes = np.unique(y_true)
    assert len(classes) == 2, f"Expected binary"

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    acc = accuracy_score(y_true, y_pred)

    # clinical metrics
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # TB recall
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # non-TB recall

    # optional but valid
    b_acc = 0.5 * (sens + spec)

    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    spec_curve = 1 - fpr

    mask = (tpr >= 0.80) & (spec_curve >= 0.60)
    if mask.sum() > 1:
        p_auroc = np.trapz(tpr[mask], fpr[mask])
    else:
        p_auroc = 0.0

    return {
        "acc": acc,
        "bacc": b_acc,
        "sens": sens,
        "spec": spec,
        "auroc": auroc,
        "pauroc": p_auroc,
        "auprc": auprc,
    }

def build_summary_table(model_results, test_dbs):
    """@brief Summarize fold-averaged classifier results for one evaluation job.
    @param model_results Per-classifier metric means, including any external databases.
    @param test_dbs External database IDs to include after the in-domain row.
    @return DataFrame of formatted mean/std values across classifiers.
    @note These standard deviations are not pooled over every classifier-fold score.
    """
    metric_keys = ["acc", "bacc", "sens", "spec", "auroc", "auprc", "pauroc"]

    rows = []

    # ---- OOF aggregation ----
    row_oof = {"db": "oof"}
    for k in metric_keys:
        vals = [v[k] for v in model_results.values() if k in v]
        if vals:
            row_oof[k] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
    rows.append(row_oof)

    # ---- External DB aggregation ----
    for db in test_dbs:
        row = {"db": f"db_{db}"}
        for k in metric_keys:
            key = f"{k}_db{db}"
            vals = [v[key] for v in model_results.values() if key in v]
            if vals:
                row[k] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    # enforce column order
    df = df[["db"] + metric_keys]

    return df

def _prep_fold(X, y, train_idx, test_idx, external_sets, split_idx, n_components):
    """@brief Prepare one training/calibration/test split for all three classifiers.
    @param X Raw frozen embedding matrix for this evaluation job.
    @param y Binary labels aligned with X.
    @param train_idx Outer-CV training indices.
    @param test_idx Outer-CV test indices, grouped by participant.
    @param external_sets Optional database matrices/labels for LODO scoring.
    @param split_idx Fold number used to seed the calibration split.
    @param n_components Fold PCA dimension, or None to omit PCA.
    @return Transformed train/calibration/test/external arrays, or None for an unusable fold.
    @details Split outer training rows 80/20 into fit/calibration subsets. Fit both
             StandardScaler and PCA on the fit subset only, then transform all
             other subsets. Share this preparation across LR, MLP and HistGB.
    @note The inner calibration split is sample stratified, not participant grouped.
    """
    y_test = y[test_idx]
    y_train_full = y[train_idx]
    # Both classes must survive in train/test; calibration also needs at least
    # two training examples per class for a stratified split.
    if len(np.unique(y_test)) < 2 or len(np.unique(y_train_full)) < 2 \
            or np.min(np.bincount(y_train_full)) < 2:
        return None

    X_train_full, X_test = X[train_idx], X[test_idx]

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42 + split_idx,
    )

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xcal = scaler.transform(X_cal)
    Xte = scaler.transform(X_test)
    ext = {db: (scaler.transform(Xe), ye) for db, (Xe, ye) in external_sets.items()}

    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=42).fit(Xtr)
        Xtr, Xcal, Xte = pca.transform(Xtr), pca.transform(Xcal), pca.transform(Xte)
        ext = {db: (pca.transform(Xe), ye) for db, (Xe, ye) in ext.items()}

    return {
        "Xtr": Xtr, "y_train": y_train,
        "Xcal": Xcal, "y_cal": y_cal,
        "Xte": Xte, "y_test": y_test,
        "ext": ext,
    }


def _fit_clf_on_fold(clf_name, clf, fold, metric_keys):
    """@brief Train one classifier and score a fold using its calibration threshold.
    @param clf_name Name retained in the output summary.
    @param clf Unfitted estimator; clone it so jobs do not share fitted state.
    @param fold Arrays produced by _prep_fold.
    @param metric_keys Metric names requested by the caller.
    @return Classifier name, test metrics and optional external-database metrics.
    @details Fit on Xtr, select a threshold on Xcal, and use that same threshold
             for the test fold and all external sets. AUROC uses probabilities.
    """
    clf_local = clone(clf)
    clf_local.fit(fold["Xtr"], fold["y_train"])
    y_cal_proba = clf_local.predict_proba(fold["Xcal"])[:, 1]
    optimized_threshold = optimize_threshold_youden(fold["y_cal"], y_cal_proba)

    y_proba = clf_local.predict_proba(fold["Xte"])[:, 1]
    y_preds = (y_proba >= optimized_threshold).astype(int)
    metrics = compute_metrics(fold["y_test"], y_preds, y_proba)

    metrics_ext = {}
    for db, (X_ext, y_ext) in fold["ext"].items():
        y_proba_ext = clf_local.predict_proba(X_ext)[:, 1]
        y_preds_ext = (y_proba_ext >= optimized_threshold).astype(int)
        metrics_ext[db] = compute_metrics(y_ext, y_preds_ext, y_proba_ext)

    return {
        "clf_name": clf_name,
        "metrics": {k: metrics[k] for k in metric_keys},
        "metrics_ext": {db: {k: metrics_ext[db][k] for k in metric_keys} for db in metrics_ext},
    }


def run_embedding_classifications_batched(
    jobs, embed_col="embed", label_col="disease_status",
    participant_col="participant", db_col="db",
    n_splits=5, n_components=256, n_jobs=-1,
):
    """@brief Run grouped-CV evaluation jobs in one shared parallel pool.
    @param jobs List of dictionaries with df, train_db and optional test_dbs.
    @param embed_col Column containing the original frozen embedding vectors.
    @param label_col Column containing binary TB labels.
    @param participant_col Column used to keep participants within one outer fold.
    @param db_col Column identifying the source database.
    @param n_splits Number of participant-grouped outer folds; the paper uses 5.
    @param n_components Number of train-fitted PCA components; the paper uses 256.
    @param n_jobs Number of parallel workers.
    @return Summary DataFrames in the same order as jobs.
    @details Build participant-grouped splits, prepare each fold once, fit all
             three classifiers, then average folds within each classifier before
             computing the final across-classifier summary.
    """
    metric_keys = ["acc", "bacc", "sens", "spec", "auroc", "auprc", "pauroc"]
    base_clfs = _make_base_classifiers()

    # ---- Stage 0: build X/y/splits per job, collect per-fold prep tasks ----
    prep_specs = []          # (job_idx, X, y, external_sets, split_idx, train_idx, test_idx)
    job_test_dbs = []
    for ji, job in enumerate(jobs):
        df = job["df"]
        train_db = job["train_db"]
        test_dbs = job.get("test_dbs") or []
        if not isinstance(train_db, list):
            train_db = [train_db]

        df_train = df[df[db_col].isin(train_db)]
        X = np.vstack(df_train[embed_col].values).astype(np.float32)
        y = df_train[label_col].astype(int).values
        groups = df_train[participant_col].values

        external_sets = {}
        valid_test_dbs = []
        for db in test_dbs:
            df_ext = df[df[db_col] == db]
            y_ext = df_ext[label_col].astype(int).values
            # Safety net: an external db that is empty or single-class can't be
            # scored (compute_metrics needs both classes; vstack([]) would raise).
            # Callers should filter these out beforehand — warn so it is never a
            # silent drop that quietly shrinks a LODO experiment.
            if len(y_ext) == 0 or len(np.unique(y_ext)) < 2:
                print(f"[WARN] run_embedding_classifications_batched: external db "
                      f"{db!r} empty/single-class (n={len(y_ext)}) -> skipped")
                continue
            external_sets[db] = (
                np.vstack(df_ext[embed_col].values).astype(np.float32), y_ext)
            valid_test_dbs.append(db)

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
            prep_specs.append((ji, X, y, external_sets, split_idx, train_idx, test_idx))
        job_test_dbs.append(valid_test_dbs)

    # ---- Stage 1: compute scaler+PCA once per fold (parallel) ----
    folds = Parallel(n_jobs=n_jobs)(
        delayed(_prep_fold)(X, y, train_idx, test_idx, external_sets, split_idx, n_components)
        for (_, X, y, external_sets, split_idx, train_idx, test_idx) in prep_specs
    )

    # ---- Stage 2: fit every classifier on every prepared fold (parallel) ----
    fit_specs = []           # (job_idx, clf_name, clf, fold)
    for (ji, *_), fold in zip(prep_specs, folds):
        if fold is None:
            continue
        for clf_name, clf in base_clfs.items():
            fit_specs.append((ji, clf_name, clf, fold))

    fit_results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_clf_on_fold)(clf_name, clf, fold, metric_keys)
        for (_, clf_name, clf, fold) in fit_specs
    )

    # ---- Aggregate per job ----
    grouped = {ji: [] for ji in range(len(jobs))}
    for (ji, *_), res in zip(fit_specs, fit_results):
        grouped[ji].append(res)

    summaries = []
    for ji in range(len(jobs)):
        test_dbs = job_test_dbs[ji]
        agg = {cn: {k: [] for k in metric_keys} for cn in base_clfs}
        agg_ext = {cn: {db: {k: [] for k in metric_keys} for db in test_dbs} for cn in base_clfs}
        for res in grouped[ji]:
            cn = res["clf_name"]
            for k in metric_keys:
                agg[cn][k].append(res["metrics"][k])
            for db in test_dbs:
                for k in metric_keys:
                    agg_ext[cn][db][k].append(res["metrics_ext"][db][k])

        model_results = {}
        for cn in base_clfs:
            model_results[cn] = {
                k: (np.mean(v) if len(v) > 0 else np.nan) for k, v in agg[cn].items()
            }
            for db in test_dbs:
                for k in metric_keys:
                    vals = agg_ext[cn][db][k]
                    model_results[cn][f"{k}_db{db}"] = np.mean(vals) if len(vals) > 0 else np.nan

        summaries.append(build_summary_table(model_results, test_dbs))
    return summaries


def run_embedding_classification(
    df, train_db, test_dbs=None, embed_col="embed",
    label_col="disease_status", participant_col="participant",
    db_col="db", n_splits=5, n_components=256, n_jobs=-1
):
    """Single-job convenience wrapper around run_embedding_classifications_batched
    (preserves the original return: one summary DataFrame)."""
    job = {"df": df, "train_db": train_db, "test_dbs": test_dbs or []}
    return run_embedding_classifications_batched(
        [job], embed_col=embed_col, label_col=label_col,
        participant_col=participant_col, db_col=db_col,
        n_splits=n_splits, n_components=n_components, n_jobs=n_jobs,
    )[0]

def majority_vote(group):
    counts = group["disease_status_rev"].value_counts(normalize=True)
    if counts.get(1, 0) > 0.51:
        return 1
    elif counts.get(2, 0) > 0.51:
        return 2
    else:
        return 2

def print_tb_stats(df_temp, tb_mask0, target_column="disease_status_rev"):
    tb_df = df_temp[tb_mask0]

    vc = tb_df[target_column].value_counts()
    total = len(tb_df)

    print(f"Acoustically Consistent TB: {vc.get(1,0)} ({vc.get(1,0)/total*100:.1f}%) | "
          f"Acoustically Ambiguous TB: {vc.get(2,0)} ({vc.get(2,0)/total*100:.1f}%)")

    db_counts = tb_df.groupby(["db",target_column]).size().unstack(fill_value=0)

    print("Acoustically Consistent TB -> " + " | ".join(
        f"DB {db}: {int(db_counts.loc[db].get(1,0))} ({v:.1f}%)"
        for db, v in (db_counts.get(1, 0).div(db_counts.sum(1))*100).items()
    ) + "\n")


def plot_tb_score_distribution(
    df_temp,
    colors=None,
    score_col="acoustic_tb_score",
    figsize=None,
    include_boxplot=True,
    include_all=True,
    save_path=None,
    show=False,
):
    df_plot = df_temp.copy()
    if colors is None:
        colors = {"Non-TB": "#4E4CAF", "TB": "#F3213A"}

    # Set figsize based on include_boxplot if not provided
    if figsize is None:
        figsize = (16, 10) if include_boxplot else (16, 5)

    sel_scores = df_plot[score_col].values
    unique_dbs = sorted(df_plot['db'].unique())
    n_cols = len(unique_dbs) + (1 if include_all else 0)
    n_rows = 2 if include_boxplot else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    all_offset = 1 if include_all else 0
    for i in range(n_cols):
        if i == 0 and include_all:
            tb_mask = df_plot['disease_status'] == 1
            ntb_mask = df_plot['disease_status'] == 0
            title_suffix = "All"
        else:
            db_val = unique_dbs[i - all_offset]
            tb_mask = (df_plot['disease_status'] == 1) & (df_plot['db'] == db_val)
            ntb_mask = (df_plot['disease_status'] == 0) & (df_plot['db'] == db_val)
            title_suffix = f"DB{db_val + 1}"

        tb_scores = sel_scores[tb_mask]
        ntb_scores = sel_scores[ntb_mask]

        # ---- KDE PLOT ----
        ax = axes[0, i] if include_boxplot else axes[i]

        if len(ntb_scores) > 1:
            sns.kdeplot(
                ntb_scores,
                fill=True,
                alpha=0.4,
                color=colors["Non-TB"],
                label=f"Non-TB (μ={ntb_scores.mean():.2f})",
                ax=ax
            )

        if len(tb_scores) > 1:
            sns.kdeplot(
                tb_scores,
                fill=True,
                alpha=0.5,
                color=colors["TB"],
                label=f"TB (μ={tb_scores.mean():.2f})",
                ax=ax
            )

        ax.set_xlabel("Acoustic TB Likelihood Score")
        ax.set_ylabel("Density")
        ax.set_title(f"KDE Distribution ({title_suffix})")
        ax.legend()

        # ---- BOXPLOT ----
        if include_boxplot:
            ax = axes[1, i]

            data_plot = [ntb_scores, tb_scores]
            bp = ax.boxplot(
                data_plot,
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color="white", linewidth=2)
            )

            for patch, color in zip(bp["boxes"], [colors["Non-TB"], colors["TB"]]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Non-TB", "TB"])
            ax.set_ylabel("Score")
            ax.set_title(f"Boxplot ({title_suffix})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf",
                    dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_tb_split_filter(
    df_temp,
    score_columns="acoustic_tb_score",
    now_xlabel="Acoustic TB Likelihood Score",
    split_tb=True,
    include_all=True,
    save_path="harmonization.pdf",
    show=False,
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df_plot = df_temp.copy()

    if split_tb:
        disease_status_rev = df_plot["disease_status_rev"]
    else:
        disease_status_rev = df_plot["disease_status_rev"].replace(2, 1)

    sel_scores = df_plot[score_columns].values
    unique_dbs = sorted(df_plot["db"].unique())
    n_cols = len(unique_dbs) + (1 if include_all else 0)

    n_legend_entries = 3 if split_tb else 2
    width = min(7.16, 1.5 * n_cols)
    height = 2.5 + 0.15 * n_legend_entries

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(width, height),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.04)
    if n_cols == 1:
        axes = [axes]

    color_nontb = "#2980B9"
    color_acoustic = "#C0392B"
    color_labelonly = "#F39C12"

    all_offset = 1 if include_all else 0
    for i in range(n_cols):
        if i == 0 and include_all:
            tb_mask = df_plot["disease_status"] == 1
            ntb_mask = df_plot["disease_status"] == 0
            title = "All"
        else:
            db_val = unique_dbs[i - all_offset]
            tb_mask = (df_plot["disease_status"] == 1) & (df_plot["db"] == db_val)
            ntb_mask = (df_plot["disease_status"] == 0) & (df_plot["db"] == db_val)
            title = f"DB{db_val + 1}"

        acoustic_tb_mask = tb_mask & (disease_status_rev == 1)
        labelonly_tb_mask = tb_mask & (disease_status_rev == 2)

        non_tb_scores = sel_scores[ntb_mask]
        acoustic_tb_scores = sel_scores[acoustic_tb_mask]
        labelonly_tb_scores = sel_scores[labelonly_tb_mask]

        ax = axes[i]

        panel_handles = []
        panel_labels = []

        if len(non_tb_scores) > 0:
            sns.kdeplot(
                non_tb_scores,
                fill=True,
                alpha=0.35,
                color=color_nontb,
                ax=ax,
                linewidth=1.2,
            )
            panel_handles.append(Patch(facecolor=color_nontb, alpha=0.35))
            panel_labels.append(f"Non-TB (μ={non_tb_scores.mean():.2f})")

        if split_tb:
            if len(acoustic_tb_scores) > 0:
                sns.kdeplot(
                    acoustic_tb_scores,
                    fill=True,
                    alpha=0.5,
                    color=color_acoustic,
                    ax=ax,
                    linewidth=1.2,
                )
                panel_handles.append(Patch(facecolor=color_acoustic, alpha=0.5))
                panel_labels.append(f"Acoustically Consistent TB (μ={acoustic_tb_scores.mean():.2f})")
            if len(labelonly_tb_scores) > 0:
                sns.kdeplot(
                    labelonly_tb_scores,
                    bw_adjust=5.0,
                    fill=True,
                    alpha=0.5,
                    color=color_labelonly,
                    ax=ax,
                    linewidth=1.2,
                )
                panel_handles.append(Patch(facecolor=color_labelonly, alpha=0.5))
                panel_labels.append(f"Acoustically Ambiguous TB (μ={labelonly_tb_scores.mean():.2f})")
        else:
            if len(acoustic_tb_scores) > 0:
                sns.kdeplot(
                    acoustic_tb_scores,
                    bw_adjust=1.0,
                    fill=True,
                    alpha=0.5,
                    color=color_acoustic,
                    ax=ax,
                    linewidth=1.2,
                )
                panel_handles.append(Patch(facecolor=color_acoustic, alpha=0.5))
                panel_labels.append(f"TB (μ={acoustic_tb_scores.mean():.2f})")

        ax.set_title(title)
        if i == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        ax.legend(
            handles=panel_handles,
            labels=panel_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=1,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            borderaxespad=0.2,
            labelspacing=0.3,
        )

        fig.supxlabel(now_xlabel, fontsize=8, y=0.13)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close()
