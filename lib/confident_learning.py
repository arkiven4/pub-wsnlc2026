"""Confident Learning baselines with the same selection interface as Proposed.

The paper calls run_cleanlab for two filters: prune_by_noise_rate (PBNR) and
confident_learning (CL). Both use cleanlab's out-of-fold probability estimation
with balanced logistic regression. Only flagged TB-labelled rows are removed.

The earlier hand-written joint/threshold helpers and run_confident_learning are
kept for reference; they are not the implementation used by the paper runner.
The returned biomarker_mask/label_only_mask keys mean retained/removed TB clips,
matching the Proposed interface. They do not confirm clinical error.

Reference: Northcutt, Jiang, Chuang, JAIR2021. See docs/proposed_method.md for a
comparison of the two cleaning paths.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. CROSS-VALIDATED OUT-OF-FOLD PROBABILITIES
# =============================================================================

def cross_val_predict_proba(embeddings, labels, classifier=None,
                            n_splits=5, random_state=42, standardize=True):
    """@brief Build out-of-fold probabilities for the reference CL implementation.
    @param embeddings Feature matrix (N, D).
    @param labels Integer class labels (N,).
    @param classifier Optional estimator; defaults to balanced logistic regression.
    @param n_splits Number of stratified folds.
    @param random_state Seed for folds and the default estimator.
    @param standardize Fit feature scaling on each training fold only.
    @return Probability matrix (N, K), scattered back to input row order.
    @note The paper baseline calls run_cleanlab, which uses cleanlab's own CV helper.
    """
    N = len(labels)
    K = int(labels.max()) + 1
    proba = np.zeros((N, K))

    if classifier is None:
        classifier = LogisticRegression(
            max_iter=2000, random_state=random_state,
            class_weight='balanced', C=1.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for tr_idx, te_idx in skf.split(embeddings, labels):
        X_tr, X_te = embeddings[tr_idx], embeddings[te_idx]
        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        clf = _clone(classifier)
        clf.fit(X_tr, labels[tr_idx])
        proba[te_idx] = clf.predict_proba(X_te)

    return proba


def _clone(estimator):
    from sklearn.base import clone
    return clone(estimator)


# =============================================================================
# 2. CONFIDENT JOINT & NOISE MATRIX
# =============================================================================

def compute_per_class_thresholds(proba, labels):
    """t_k = mean P̂(y=k|x) over samples with noisy label k."""
    K = proba.shape[1]
    thresholds = np.zeros(K)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            thresholds[k] = proba[mask, k].mean()
        else:
            thresholds[k] = 1.0 / K
    return thresholds


def compute_confident_joint(proba, labels, thresholds):
    """
    C[i,j] = #{x : label=i, j = argmax_{k: p_k ≥ t_k} p_k}.
    Ties broken by highest probability.
    """
    K = proba.shape[1]
    N = len(labels)
    C = np.zeros((K, K), dtype=int)

    above = proba >= thresholds[np.newaxis, :]  # (N, K)
    for i in range(N):
        cand = np.where(above[i])[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(proba[i, cand])]
        C[labels[i], j] += 1
    return C


def calibrate_joint(C, labels):
    """Calibrate confident joint to estimate joint distribution P(y_noisy, y_true)."""
    K = C.shape[0]
    class_counts = np.array([(labels == k).sum() for k in range(K)])
    C_cal = C.astype(float)
    row_sums = C_cal.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C_cal = C_cal / row_sums * class_counts[:, np.newaxis]
    total = C_cal.sum()
    if total > 0:
        joint = C_cal / total
    else:
        joint = C_cal
    return joint


# =============================================================================
# 3. LABEL-ISSUE DETECTION
# =============================================================================

def find_label_issues(proba, labels, thresholds=None, method='prune_by_noise_rate'):
    """
    Return boolean mask of likely-mislabeled samples.

    method:
      'prune_by_noise_rate' — for each (i,j) off-diagonal in joint,
          take the n_ij samples with label i that have highest
          P̂(y=j|x) - P̂(y=i|x) margin.
      'self_confidence'    — flag samples with P̂(y=label|x) below threshold.
    """
    if thresholds is None:
        thresholds = compute_per_class_thresholds(proba, labels)

    if method == 'self_confidence':
        return proba[np.arange(len(labels)), labels] < thresholds[labels]

    C = compute_confident_joint(proba, labels, thresholds)
    joint = calibrate_joint(C, labels)

    N, K = proba.shape[0], proba.shape[1]
    issue = np.zeros(N, dtype=bool)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            n_ij = int(round(joint[i, j] * N))
            if n_ij <= 0:
                continue
            cand = np.where(labels == i)[0]
            if len(cand) == 0:
                continue
            margin = proba[cand, j] - proba[cand, i]
            n_take = min(n_ij, len(cand))
            top = cand[np.argsort(-margin)[:n_take]]
            issue[top] = True
    return issue


# =============================================================================
# 4. MAIN PIPELINE — matches helper.run_fisher_and_split interface
# =============================================================================

def run_confident_learning(embeddings, labels, classifier=None,
                           n_splits=5, method='prune_by_noise_rate',
                           do_refinement=True, verbose=True,
                           random_state=42):
    """
    Drop-in alternative to `helper.run_fisher_and_split`.

    Confident Learning identifies TB+ samples whose embeddings look like NTB
    → label-only TB. Remaining TB+ → biomarker TB.

    Returns dict with keys:
        has_natural_split (bool, True if any label issues found among TB+)
        biomarker_mask (N,) bool
        label_only_mask (N,) bool
        projected_1d (N,) = P(TB|x)  [analogue of Fisher projection]
        posterior_ntb (N,) = 1 - P(TB|x)
        proba (N, K) out-of-fold probabilities
        thresholds (K,) per-class CL thresholds
        confident_joint (K, K)
        joint_distribution (K, K)
        label_issue_mask (N,) any class
        tb_indices (n_tb,)
        n_label_only, n_biomarker
        is_refined (bool)
        gradient_percentiles (if no split found)
    """
    labels = np.asarray(labels).astype(int)

    if verbose:
        print(f"  CL: computing OOF probabilities (n_splits={n_splits})...")
    proba = cross_val_predict_proba(
        embeddings, labels, classifier=classifier,
        n_splits=n_splits, random_state=random_state)

    thresholds = compute_per_class_thresholds(proba, labels)
    C = compute_confident_joint(proba, labels, thresholds)
    joint = calibrate_joint(C, labels)

    if verbose:
        print(f"  Per-class thresholds: {np.round(thresholds, 3).tolist()}")
        print(f"  Confident joint:\n{C}")

    issue_mask = find_label_issues(proba, labels, thresholds, method=method)

    tb_indices = np.where(labels == 1)[0]
    tb_issue_mask = issue_mask & (labels == 1)  # TB+ flagged as NTB

    label_only_mask = tb_issue_mask
    biomarker_mask = (labels == 1) & ~tb_issue_mask

    n_lo = int(label_only_mask.sum())
    n_bio = int(biomarker_mask.sum())
    has_split = n_lo > 0

    if verbose:
        print(f"  Label issues among TB+: {n_lo}/{len(tb_indices)} "
              f"({n_lo / max(len(tb_indices), 1) * 100:.1f}%)")
        print(f"  Biomarker TB: {n_bio}, Label-only TB: {n_lo}")

    p_tb = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
    posterior_ntb = 1.0 - p_tb

    result = {
        'has_natural_split': has_split,
        'biomarker_mask': biomarker_mask,
        'label_only_mask': label_only_mask,
        'projected_1d': p_tb,
        'posterior_ntb': posterior_ntb,
        'proba': proba,
        'thresholds': thresholds,
        'confident_joint': C,
        'joint_distribution': joint,
        'label_issue_mask': issue_mask,
        'tb_indices': tb_indices,
        'n_label_only': n_lo,
        'n_biomarker': n_bio,
        'is_refined': False,
        'method': method,
    }

    if not has_split:
        pcts = np.percentile(posterior_ntb[labels == 1], [10, 25, 50, 75, 90])
        result['gradient_percentiles'] = {
            'p10': pcts[0], 'p25': pcts[1], 'p50': pcts[2],
            'p75': pcts[3], 'p90': pcts[4],
        }
        if verbose:
            print(f"  → No label issues found. P(NTB|x) gradient on TB+:")
            print(f"    p10={pcts[0]:.3f}, p25={pcts[1]:.3f}, "
                  f"median={pcts[2]:.3f}, p75={pcts[3]:.3f}, p90={pcts[4]:.3f}")

    if do_refinement and has_split:
        if verbose:
            print(f"\n  Refining: re-fit CL with label-only TB+ removed...")
        keep_mask = ~label_only_mask
        emb_clean = embeddings[keep_mask]
        lab_clean = labels[keep_mask]
        idx_map = np.where(keep_mask)[0]

        proba2 = cross_val_predict_proba(
            emb_clean, lab_clean, classifier=classifier,
            n_splits=n_splits, random_state=random_state)

        thr2 = compute_per_class_thresholds(proba2, lab_clean)
        issue2 = find_label_issues(proba2, lab_clean, thr2, method=method)
        tb_issue2 = issue2 & (lab_clean == 1)

        # Map back to global indices
        extra_lo_global = idx_map[tb_issue2]
        if verbose:
            print(f"  Refinement found {tb_issue2.sum()} additional "
                  f"label-only TB+ in cleaned set")

        refined_lo = label_only_mask.copy()
        refined_lo[extra_lo_global] = True
        refined_bio = (labels == 1) & ~refined_lo

        result['label_only_mask'] = refined_lo
        result['biomarker_mask'] = refined_bio
        result['n_label_only'] = int(refined_lo.sum())
        result['n_biomarker'] = int(refined_bio.sum())
        result['is_refined'] = True
        result['has_natural_split'] = bool(result['n_label_only'] > 0)

    return result


# =============================================================================
# 4b. CLEANLAB-BACKED PIPELINE — uses the `cleanlab` library
# =============================================================================

def run_cleanlab(embeddings, labels, classifier=None, n_splits=5,
                 filter_by='prune_by_noise_rate', do_refinement=True,
                 verbose=True, random_state=42, standardize=True,
                 dataset_labels=None):
    """@brief Run the paper's Confident Learning baseline through cleanlab.
    @param embeddings Input features (N, D), normally global PCA-256 output.
    @param labels Binary clip labels (N,).
    @param classifier Optional estimator; defaults to balanced logistic regression.
    @param n_splits Number of stratified out-of-fold probability folds, normally 5.
    @param filter_by cleanlab filter; paper arms use prune_by_noise_rate or confident_learning.
    @param do_refinement Optionally run a second pass after removing flagged TB clips.
    @param verbose Print label-issue diagnostics when true.
    @param random_state Seed for the probability folds and default estimator.
    @param standardize Wrap the classifier in a fold-fitted StandardScaler pipeline.
    @param dataset_labels Optional database IDs for separate cleaner fitting.
    @return Dictionary using the same score/mask interface as the Proposed cleaner.
    @details Estimate out-of-fold probabilities, apply the requested cleanlab
             filter, and remove only flagged TB-labelled clips. Keep every non-TB
             clip even if cleanlab flags it. Preserve original row order.
    @note The paper disables refinement. These probability folds are sample
          stratified; participant grouping is applied later in classifier evaluation.
          Label issues do not establish that a clinical TB diagnosis was incorrect.
    """
    if dataset_labels is not None:
        from .helper import run_per_database
        return run_per_database(
            run_cleanlab, embeddings, labels, dataset_labels, verbose=verbose,
            classifier=classifier, n_splits=n_splits, filter_by=filter_by,
            do_refinement=do_refinement, random_state=random_state,
            standardize=standardize)

    from sklearn.pipeline import make_pipeline
    from cleanlab.count import (
        estimate_cv_predicted_probabilities, compute_confident_joint)
    from cleanlab.filter import find_label_issues as cl_find_label_issues
    from cleanlab.rank import get_label_quality_scores

    labels = np.asarray(labels).astype(int)
    embeddings = np.asarray(embeddings)

    if classifier is None:
        classifier = LogisticRegression(
            max_iter=2000, random_state=random_state,
            class_weight='balanced', C=1.0)
    if standardize:
        classifier = make_pipeline(StandardScaler(), classifier)

    if verbose:
        print(f"  cleanlab: OOF probabilities (cv_n_folds={n_splits})...")
    proba = estimate_cv_predicted_probabilities(
        embeddings, labels, clf=classifier,
        cv_n_folds=n_splits, seed=random_state)

    issue_mask = cl_find_label_issues(labels, proba, filter_by=filter_by)
    quality = get_label_quality_scores(labels, proba)
    thresholds = compute_per_class_thresholds(proba, labels)
    C = compute_confident_joint(labels, proba, calibrate=False)
    joint = compute_confident_joint(labels, proba, calibrate=True).astype(float)
    joint = joint / max(joint.sum(), 1)

    tb_indices = np.where(labels == 1)[0]
    tb_issue_mask = issue_mask & (labels == 1)   # TB+ flagged as NTB

    label_only_mask = tb_issue_mask
    biomarker_mask = (labels == 1) & ~tb_issue_mask

    n_lo = int(label_only_mask.sum())
    n_bio = int(biomarker_mask.sum())
    has_split = n_lo > 0

    if verbose:
        print(f"  Confident joint:\n{C}")
        print(f"  Label issues among TB+: {n_lo}/{len(tb_indices)} "
              f"({n_lo / max(len(tb_indices), 1) * 100:.1f}%)")
        print(f"  Biomarker TB: {n_bio}, Label-only TB: {n_lo}")

    p_tb = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
    posterior_ntb = 1.0 - p_tb

    result = {
        'has_natural_split': has_split,
        'biomarker_mask': biomarker_mask,
        'label_only_mask': label_only_mask,
        'projected_1d': p_tb,
        'posterior_ntb': posterior_ntb,
        'proba': proba,
        'label_quality_scores': quality,
        'thresholds': thresholds,
        'confident_joint': C,
        'joint_distribution': joint,
        'label_issue_mask': issue_mask,
        'tb_indices': tb_indices,
        'n_label_only': n_lo,
        'n_biomarker': n_bio,
        'is_refined': False,
        'method': f'cleanlab:{filter_by}',
    }

    if not has_split:
        pcts = np.percentile(posterior_ntb[labels == 1], [10, 25, 50, 75, 90])
        result['gradient_percentiles'] = {
            'p10': pcts[0], 'p25': pcts[1], 'p50': pcts[2],
            'p75': pcts[3], 'p90': pcts[4],
        }
        if verbose:
            print(f"  → No label issues found. P(NTB|x) gradient on TB+:")
            print(f"    p10={pcts[0]:.3f}, p25={pcts[1]:.3f}, "
                  f"median={pcts[2]:.3f}, p75={pcts[3]:.3f}, p90={pcts[4]:.3f}")

    if do_refinement and has_split:
        if verbose:
            print(f"\n  Refining: re-fit cleanlab with label-only TB+ removed...")
        keep_mask = ~label_only_mask
        emb_clean = embeddings[keep_mask]
        lab_clean = labels[keep_mask]
        idx_map = np.where(keep_mask)[0]

        proba2 = estimate_cv_predicted_probabilities(
            emb_clean, lab_clean, clf=classifier,
            cv_n_folds=n_splits, seed=random_state)
        issue2 = cl_find_label_issues(lab_clean, proba2, filter_by=filter_by)
        tb_issue2 = issue2 & (lab_clean == 1)

        extra_lo_global = idx_map[tb_issue2]
        if verbose:
            print(f"  Refinement found {int(tb_issue2.sum())} additional "
                  f"label-only TB+ in cleaned set")

        refined_lo = label_only_mask.copy()
        refined_lo[extra_lo_global] = True
        refined_bio = (labels == 1) & ~refined_lo

        result['label_only_mask'] = refined_lo
        result['biomarker_mask'] = refined_bio
        result['n_label_only'] = int(refined_lo.sum())
        result['n_biomarker'] = int(refined_bio.sum())
        result['is_refined'] = True
        result['has_natural_split'] = bool(result['n_label_only'] > 0)

    return result


# =============================================================================
# 5. UTILITIES
# =============================================================================

def print_cl_summary(result, labels):
    """Pretty-print result analogous to Fisher pipeline."""
    print("\n=== CONFIDENT LEARNING SUMMARY ===")
    print(f"  Method: {result['method']}")
    print(f"  Refined: {result['is_refined']}")
    print(f"  Thresholds: {np.round(result['thresholds'], 3).tolist()}")
    print(f"  Confident joint:\n{result['confident_joint']}")
    print(f"  Joint distribution:\n"
          f"{np.round(result['joint_distribution'], 4)}")

    n_tb = int((labels == 1).sum())
    print(f"\n  TB+ total: {n_tb}")
    print(f"    Biomarker TB: {result['n_biomarker']} "
          f"({result['n_biomarker'] / max(n_tb, 1) * 100:.1f}%)")
    print(f"    Label-only TB: {result['n_label_only']} "
          f"({result['n_label_only'] / max(n_tb, 1) * 100:.1f}%)")
    print("=" * 40)
