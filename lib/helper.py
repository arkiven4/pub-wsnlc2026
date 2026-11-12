"""Proposed cleaner and additional research diagnostics.

The paper path is:
    fit_pca -> run_fisher_and_split -> compute_fisher_projection
            -> compute_posterior_ntb -> find_natural_split

Stage 03 fits PCA once over the study features, then calls the cleaner for each
source database. Only TB-labelled clips are separated into Acoustically
Consistent and Acoustically Ambiguous groups; clinical labels are not rewritten.

Older internal key names kept for compatibility:
    biomarker_mask  = Acoustically Consistent TB
    label_only_mask = Acoustically Ambiguous TB
    posterior_ntb   = acoustic non-TB posterior

Confounder removal, overlap analysis, k-NN and consensus functions below are
research utilities, not extra steps in the paper's five-stage workflow.
See docs/proposed_method.md for the equations, decision rules and reading order.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
import torch
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. NORMALIZATION
# =============================================================================

def normalize_embeddings(embeddings, method, dataset_labels=None):
    if method == 'none':
        return embeddings.copy()
    elif method == 'l2':
        norms = np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-10, None)
        return embeddings / norms
    elif method == 'zscore':
        assert dataset_labels is not None
        result = embeddings.copy()
        for ds in np.unique(dataset_labels):
            m = dataset_labels == ds
            mu = result[m].mean(axis=0)
            sigma = np.clip(result[m].std(axis=0), 1e-10, None)
            result[m] = (result[m] - mu) / sigma
        return result
    else:
        raise ValueError(f"Unknown: {method}")


# =============================================================================
# 2. PCA
# =============================================================================

def fit_pca(embeddings, n_components):
    """@brief Fit the global feature reduction used by the cleaning stage.
    @param embeddings Float matrix (N, D), in metadata row order.
    @param n_components Number of retained principal components; the paper uses 256.
    @return Fitted PCA, transformed matrix (N, n_components), and variance ratios.
    @note This PCA uses all study clips before cleaning. Classifier folds fit their
          own scaler/PCA later on the original frozen embeddings.
    """
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  PCA to {n_components} dims: {cumvar[-1]*100:.1f}% variance retained")
    return pca, transformed, pca.explained_variance_ratio_


# =============================================================================
# 3. CONFOUNDER REMOVAL
# =============================================================================

def coral_alignment(source, target):
    D = source.shape[1]
    C_s = np.cov(source, rowvar=False) + np.eye(D) * 1e-6
    C_t = np.cov(target, rowvar=False) + np.eye(D) * 1e-6
    es, vs = np.linalg.eigh(C_s)
    whiten = vs @ np.diag(1.0 / np.sqrt(np.clip(es, 1e-10, None))) @ vs.T
    et, vt = np.linalg.eigh(C_t)
    recolor = vt @ np.diag(np.sqrt(np.clip(et, 1e-10, None))) @ vt.T
    return (source - source.mean(0)) @ whiten @ recolor + target.mean(0)


def apply_coral(embeddings, dataset_labels, reference_ds=None):
    result = embeddings.copy()
    unique_ds = np.unique(dataset_labels)
    if reference_ds is None:
        counts = {ds: np.sum(dataset_labels == ds) for ds in unique_ds}
        reference_ds = max(counts, key=counts.get)
    ref = embeddings[dataset_labels == reference_ds]
    print(f"  CORAL: aligning to '{reference_ds}'")
    for ds in unique_ds:
        if ds != reference_ds:
            m = dataset_labels == ds
            result[m] = coral_alignment(embeddings[m], ref)
    return result


def apply_combat(embeddings, dataset_labels, disease_labels):
    from neuroCombat import neuroCombat
    covars = pd.DataFrame({'batch': dataset_labels, 'disease': disease_labels}) #
    print(covars.columns)
    result = neuroCombat(dat=embeddings.T, covars=covars,
                         batch_col='batch', categorical_cols=['disease']) #
    return result['data'].T


def apply_regression_residualization(embeddings, confound_matrix):
    result = np.zeros_like(embeddings)
    for i in range(embeddings.shape[1]):
        lr = LinearRegression()
        lr.fit(confound_matrix, embeddings[:, i])
        result[:, i] = embeddings[:, i] - lr.predict(confound_matrix)
    return result


def build_confound_matrix(dataset_labels, gender_labels):
    unique_ds = np.unique(dataset_labels)
    ds_oh = np.zeros((len(dataset_labels), len(unique_ds) - 1))
    for i, ds in enumerate(unique_ds[1:]):
        ds_oh[:, i] = (dataset_labels == ds).astype(float)
    g = (gender_labels == np.unique(gender_labels)[0]).astype(float).reshape(-1, 1)
    return np.hstack([ds_oh, g])


def remove_confounders(embeddings, method, dataset_labels, gender_labels,
                       disease_labels=None):
    if method == 'coral':
        aligned = apply_coral(embeddings, dataset_labels)
        g = (gender_labels == np.unique(gender_labels)[0]).astype(float).reshape(-1, 1)
        return apply_regression_residualization(aligned, g)
    elif method == 'combat':
        assert disease_labels is not None
        harmonized = apply_combat(embeddings, dataset_labels, disease_labels)
        g = (gender_labels == np.unique(gender_labels)[0]).astype(float).reshape(-1, 1)
        return apply_regression_residualization(harmonized, g)
    elif method == 'regression':
        return apply_regression_residualization(
            embeddings, build_confound_matrix(dataset_labels, gender_labels))
    else:
        raise ValueError(f"Unknown: {method}")


def verify_confounder_removal(embeddings, dataset_labels, gender_labels,
                              disease_labels):
    results = {}
    ds_scores = cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        embeddings, dataset_labels, cv=5, scoring='accuracy')
    results['db_accuracy'] = ds_scores.mean()
    results['db_chance'] = 1.0 / len(np.unique(dataset_labels))

    g_scores = cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        embeddings, gender_labels, cv=5, scoring='accuracy')
    results['gender_accuracy'] = g_scores.mean()
    results['gender_chance'] = 1.0 / len(np.unique(gender_labels))

    tb = (disease_labels == 'TB').astype(int) if disease_labels.dtype == object \
        else disease_labels
    tb_scores = cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        embeddings, tb, cv=5, scoring='roc_auc')
    results['tb_auroc'] = tb_scores.mean()
    return results


# =============================================================================
# 4. FISHER PROJECTION & POSTERIOR
# =============================================================================

def _silverman_bw(X):
    """@brief Choose a one-dimensional Gaussian KDE bandwidth.
    @param X Class-specific Fisher coordinates, shape (n,) or (n, 1).
    @return max(1.09 * std(X) * n**(-0.2), 0.001).
    @note The floor avoids a zero bandwidth for nearly constant coordinates.
    """
    return max(1.09 * X.std() * len(X) ** (-0.2), 1e-3)


def compute_fisher_projection(embeddings, labels, tb_mask_override=None):
    """@brief Fit a Fisher axis separating the supplied TB and non-TB labels.
    @param embeddings Feature matrix (N, D), normally the global PCA output.
    @param labels Binary clip labels: 0 = non-TB, 1 = TB.
    @param tb_mask_override Optional Boolean mask selecting TB clips for a refit.
    @return Projected coordinates (N,) and the fitted LDA coefficient vector (D,).
    @details Fit on the selected TB clips and all non-TB clips, then project every
             row. Orient the coordinates so the TB mean lies above the non-TB mean.
    @note The returned coefficient vector is the estimator's original coefficient;
          only the returned coordinates receive the optional sign flip.
    """
    if tb_mask_override is not None:
        custom_labels = np.zeros(len(labels))
        custom_labels[tb_mask_override] = 1
        fit_mask = (labels == 0) | tb_mask_override
    else:
        custom_labels = labels.copy()
        fit_mask = np.ones(len(labels), dtype=bool)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(embeddings[fit_mask], custom_labels[fit_mask])
    projected = lda.transform(embeddings).flatten()

    # Ensure higher = more TB-like
    if projected[labels == 1].mean() < projected[labels == 0].mean():
        projected = -projected

    return projected, lda.coef_.flatten()


def compute_posterior_ntb(projected_1d, labels):
    """@brief Estimate how non-TB-like each Fisher coordinate is.
    @param projected_1d Oriented Fisher coordinates (N,).
    @param labels Binary clip labels, aligned with the coordinates.
    @return P(non-TB | coordinate) for every clip, shape (N,).
    @details Fit one Gaussian KDE per labelled class and use observed class
             frequencies as priors. Apply Bayes' rule in log space to avoid
             underflow. The stored TB-likeness score is 1 - this posterior.
    @note This is an acoustic score estimated from the supplied labels, not a
          calibrated probability of clinical TB disease.
    """
    # Fit one density per supplied class along the same Fisher axis.
    tb_vals = projected_1d[labels == 1].reshape(-1, 1)
    ntb_vals = projected_1d[labels == 0].reshape(-1, 1)

    kde_tb = KernelDensity(bandwidth=_silverman_bw(tb_vals),
                           kernel='gaussian').fit(tb_vals)
    kde_ntb = KernelDensity(bandwidth=_silverman_bw(ntb_vals),
                            kernel='gaussian').fit(ntb_vals)

    # Weight the densities by the observed class frequencies, in log space.
    log_prior_tb = np.log(max(labels.mean(), 1e-10))
    log_prior_ntb = np.log(max(1 - labels.mean(), 1e-10))

    # Evaluate both class numerators at every clip coordinate.
    pts = projected_1d.reshape(-1, 1)
    log_p_tb = kde_tb.score_samples(pts) + log_prior_tb     # log of TB numerator
    log_p_ntb = kde_ntb.score_samples(pts) + log_prior_ntb  # log of NTB numerator

    # Log-sum-exp gives the shared evidence without exponentiating large logs.
    log_max = np.maximum(log_p_tb, log_p_ntb)
    log_sum = log_max + np.log(np.exp(log_p_tb - log_max) + np.exp(log_p_ntb - log_max))

    # Divide the non-TB numerator by that evidence and return ordinary scores.
    return np.exp(log_p_ntb - log_sum)


# =============================================================================
# 5. DATA-DRIVEN SPLIT (GMM on TB+ posteriors)
# =============================================================================

def find_natural_split(posterior_ntb, labels, verbose=True):
    """@brief Look for two acoustic groups within the TB-labelled population.
    @param posterior_ntb Per-clip P(non-TB | Fisher coordinate), shape (N,).
    @param labels Binary clip labels. Only TB-labelled rows enter the GMM fits.
    @param verbose Print mixture diagnostics when true.
    @return Dictionary containing masks, component summaries and BIC diagnostics.
    @details Fit one-, two- and three-component GMMs. The three-component result
             is diagnostic; the actual split always comes from the two-component
             model. Accept it only if BIC improves over one component, separation
             exceeds 0.5, and the means straddle 0.5 or separation exceeds 1.5.
             Lower mean P(non-TB) maps to Acoustically Consistent TB; higher mean
             maps to Acoustically Ambiguous TB. Assignment uses GMM membership,
             not a fixed posterior cutoff at 0.5.
    @note If the split is not supported, retain every TB clip. The original output
          keys biomarker_mask and label_only_mask mean Consistent and Ambiguous TB.
    """
    N = len(labels)
    tb_indices = np.where(labels == 1)[0]
    tb_post = posterior_ntb[tb_indices].reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(tb_post)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=10).fit(tb_post)
    # A third component is diagnostic; the accepted partition below remains binary.
    gmm3 = GaussianMixture(n_components=3, random_state=42, n_init=10).fit(tb_post)

    bic1, bic2, bic3 = gmm1.bic(tb_post), gmm2.bic(tb_post), gmm3.bic(tb_post)
    best_k = np.argmin([bic1, bic2, bic3]) + 1

    result = {
        'tb_posteriors': tb_post.flatten(),
        'tb_indices': tb_indices,
        'bic_1': bic1,
        'bic_2': bic2,
        'bic_3': bic3,
        'bic_diff_2v1': bic1 - bic2,  # positive = 2 better
        'bic_diff_3v2': bic2 - bic3,  # positive = 3 better
        'best_k': best_k,
    }

    if verbose:
        print(f"  BIC: 1-comp={bic1:.1f}, 2-comp={bic2:.1f}, 3-comp={bic3:.1f}")
        print(f"  Best K by BIC: {best_k}")

    # Check if 2-component split is meaningful
    means_2 = gmm2.means_.flatten()
    stds_2 = np.sqrt(gmm2.covariances_.flatten())
    weights_2 = gmm2.weights_
    order = np.argsort(means_2)
    means_2 = means_2[order]
    stds_2 = stds_2[order]
    weights_2 = weights_2[order]

    result['comp_means'] = means_2.tolist()
    result['comp_stds'] = stds_2.tolist()
    result['comp_weights'] = weights_2.tolist()

    # Measure separation in units of the average component standard deviation.
    separation = abs(means_2[1] - means_2[0])
    avg_std = (stds_2[0] + stds_2[1]) / 2
    separation_ratio = separation / max(avg_std, 1e-10)

    # Also check: does one component have mean > 0.5 (NTB-like)
    # and the other < 0.5 (TB-like)?
    straddles_05 = (means_2[0] < 0.5 and means_2[1] > 0.5) or \
                   (means_2[0] > 0.5 and means_2[1] < 0.5)

    # Natural split requires:
    # 1) BIC favors 2+ components (bic_diff > 0)
    # 2) Separation ratio exceeds 0.5 (the exact gate used by the paper code).
    # 3) The split is meaningful (straddles 0.5 OR large separation)
    has_split = (bic1 - bic2 > 0) and (separation_ratio > 0.5) and \
                (straddles_05 or separation_ratio > 1.5)

    result['separation_ratio'] = separation_ratio
    result['straddles_05'] = straddles_05
    result['has_natural_split'] = has_split

    if verbose:
        print(f"  Component means: {means_2[0]:.3f}, {means_2[1]:.3f}")
        print(f"  Component stds: {stds_2[0]:.3f}, {stds_2[1]:.3f}")
        print(f"  Component weights: {weights_2[0]:.3f}, {weights_2[1]:.3f}")
        print(f"  Separation ratio: {separation_ratio:.2f}")
        print(f"  Straddles 0.5: {straddles_05}")
        print(f"  Natural split found: {has_split}")

    if has_split:
        # Assign each TB+ subject to a component
        assignments = gmm2.predict(tb_post)
        # Re-order to match sorted means
        assignments = np.array([np.where(order == a)[0][0] for a in assignments])
        probs = gmm2.predict_proba(tb_post)[:, order]

        # Higher mean P(non-TB) identifies the acoustically Ambiguous TB component.
        lo_comp = int(np.argmax(means_2))  # higher posterior = more NTB-like
        bio_comp = 1 - lo_comp

        lo_mask_local = assignments == lo_comp
        bio_mask_local = assignments == bio_comp

        result['component_assignment'] = assignments
        result['component_probs'] = probs
        result['label_only_component'] = lo_comp
        result['biomarker_component'] = bio_comp

        result['label_only_indices'] = tb_indices[lo_mask_local]
        result['biomarker_indices'] = tb_indices[bio_mask_local]

        # Full-size masks
        lo_mask = np.zeros(N, dtype=bool)
        bio_mask = np.zeros(N, dtype=bool)
        lo_mask[result['label_only_indices']] = True
        bio_mask[result['biomarker_indices']] = True
        result['label_only_mask'] = lo_mask
        result['biomarker_mask'] = bio_mask

        result['label_only_fraction'] = lo_mask_local.mean()
        result['biomarker_fraction'] = bio_mask_local.mean()

        if verbose:
            print(f"  → Label-only: {lo_mask_local.sum()} "
                  f"({result['label_only_fraction']*100:.1f}%)")
            print(f"  → Biomarker: {bio_mask_local.sum()} "
                  f"({result['biomarker_fraction']*100:.1f}%)")
    else:
        # No natural split — report as gradient
        pcts = np.percentile(tb_post.flatten(), [10, 25, 50, 75, 90])
        result['gradient_percentiles'] = {
            'p10': pcts[0], 'p25': pcts[1], 'p50': pcts[2],
            'p75': pcts[3], 'p90': pcts[4],
        }
        # Still provide soft scores — just no hard boundary
        result['label_only_mask'] = np.zeros(N, dtype=bool)
        result['biomarker_mask'] = labels == 1  # all TB+ are "potential biomarker"

        if verbose:
            print(f"  → No natural split. P(NTB|x) gradient:")
            print(f"    p10={pcts[0]:.3f}, p25={pcts[1]:.3f}, "
                  f"median={pcts[2]:.3f}, p75={pcts[3]:.3f}, p90={pcts[4]:.3f}")

    return result


def refine_fisher_with_split(embeddings, labels, split_result, verbose=True):
    """
    If a natural split was found, recompute Fisher using ONLY biomarker
    TB+ subjects. Then recompute posteriors and the split.

    Returns:
        refined split_result (same structure), or original if no split
    """
    if not split_result['has_natural_split']:
        if verbose:
            print("  No natural split → skipping refinement")
        return split_result

    bio_mask = split_result['biomarker_mask']
    if verbose:
        print(f"  Refining Fisher with {bio_mask.sum()} biomarker TB+ "
              f"(excluding {split_result['label_only_mask'].sum()} label-only)...")

    projected, direction = compute_fisher_projection(embeddings, labels, tb_mask_override=bio_mask)
    posterior_ntb = compute_posterior_ntb(projected, labels)

    refined = find_natural_split(posterior_ntb, labels, verbose=verbose)
    refined['projected_1d'] = projected
    refined['posterior_ntb'] = posterior_ntb
    refined['fisher_direction'] = direction
    refined['is_refined'] = True

    return refined


# =============================================================================
# 5b. PER-DATABASE WRAPPER
# =============================================================================

# (N,)- or (N,K)-shaped per-sample outputs that must be stitched back into
# full-size arrays when cleaning each database independently. Everything else
# (TB-only arrays, matrices, scalars) is kept under per_database_results[ds].
_PERSAMPLE_KEYS = (
    'biomarker_mask', 'label_only_mask', 'projected_1d', 'posterior_ntb',
    'proba', 'label_issue_mask', 'label_quality_scores',
)


def run_per_database(run_fn, embeddings, labels, dataset_labels,
                     verbose=False, **kwargs):
    """@brief Fit a cleaner separately in each database and restore input row order.
    @param run_fn Cleaner callable, such as run_fisher_and_split or run_cleanlab.
    @param embeddings Shared feature matrix (N, D).
    @param labels Binary clip labels (N,).
    @param dataset_labels Database ID for each row (N,).
    @param verbose Print per-database diagnostics when true.
    @param kwargs Additional arguments forwarded to each cleaner.
    @return Merged per-clip arrays plus per_database_results for each subset.
    @details Gather each database's rows, call its cleaner without dataset_labels,
             and scatter masks/scores back to their original positions. Skip a
             database if it lacks one of the two labelled classes.
    @note Only cleaner fitting is per database. The input PCA was fitted globally;
          this function does not make the complete study an out-of-fold cleaner.
    """
    print("PerDB Cleaning")
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    dataset_labels = np.asarray(dataset_labels)
    N = len(labels)

    merged = {}
    per_db = {}
    has_split_any = False
    refined_any = False

    for ds in np.unique(dataset_labels):
        m = dataset_labels == ds
        sub_idx = np.where(m)[0]
        sub_labels = labels[m]

        if verbose:
            print(f"\n=== Database '{ds}' (n={int(m.sum())}, "
                  f"TB+={int((sub_labels == 1).sum())}) ===")

        # Need both classes present to fit a split.
        if len(np.unique(sub_labels)) < 2:
            if verbose:
                print(f"  Skipping '{ds}': only one class present")
            per_db[ds] = {'skipped': True, 'reason': 'single_class'}
            continue

        sub_res = run_fn(embeddings[m], sub_labels, verbose=verbose, **kwargs)
        per_db[ds] = sub_res
        has_split_any = has_split_any or bool(sub_res.get('has_natural_split', False))
        refined_any = refined_any or bool(sub_res.get('is_refined', False))

        for key in _PERSAMPLE_KEYS:
            val = sub_res.get(key)
            if not isinstance(val, np.ndarray) or val.shape[0] != m.sum():
                continue
            if key not in merged:
                shape = (N,) + val.shape[1:]
                merged[key] = (np.zeros(shape, dtype=bool) if val.dtype == bool
                               else np.full(shape, np.nan, dtype=float))
            merged[key][sub_idx] = val

    merged['per_database_results'] = per_db
    merged['per_database'] = True
    merged['has_natural_split'] = has_split_any
    merged['is_refined'] = refined_any
    merged['tb_indices'] = np.where(labels == 1)[0]
    merged['n_label_only'] = int(merged.get(
        'label_only_mask', np.zeros(N, dtype=bool)).sum())
    merged['n_biomarker'] = int(merged.get(
        'biomarker_mask', np.zeros(N, dtype=bool)).sum())

    # Ensure masks exist even if no database produced one.
    merged.setdefault('label_only_mask', np.zeros(N, dtype=bool))
    merged.setdefault('biomarker_mask', labels == 1)

    if verbose:
        print(f"\n=== Per-database total: label-only={merged['n_label_only']}, "
              f"biomarker={merged['n_biomarker']} ===")
    return merged


def run_fisher_and_split(embeddings, labels, do_refinement=True, verbose=False,
                         dataset_labels=None):
    """@brief Run the Proposed Fisher -> KDE posterior -> GMM cleaning sequence.
    @param embeddings Feature matrix (N, D); stage 03 supplies PCA-256 features.
    @param labels Binary clip labels (N,).
    @param do_refinement Optionally refit Fisher using Consistent TB only.
    @param verbose Print intermediate diagnostics when true.
    @param dataset_labels Optional database IDs; when provided, fit each subset separately.
    @return Dictionary with projected_1d, posterior_ntb, TB subgroup masks and diagnostics.
    @details Compute the Fisher coordinates, estimate non-TB posteriors, then test
             whether the TB-labelled scores support a two-group split. When an
             optional refit loses that split, retain the initial result.
    @note The paper path passes do_refinement=False, despite this helper's default.
          No encoder or evaluation classifier is trained inside this function.
    """
    if dataset_labels is not None:
        return run_per_database(
            run_fisher_and_split, embeddings, labels, dataset_labels,
            verbose=verbose, do_refinement=do_refinement)

    if verbose:
        print(f"  Computing initial Fisher projection...")

    projected, direction = compute_fisher_projection(embeddings, labels)
    posterior_ntb = compute_posterior_ntb(projected, labels)

    if verbose:
        print(f"\n  Finding natural split in TB+ posterior distribution...")

    split = find_natural_split(posterior_ntb, labels, verbose=verbose)
    split['projected_1d'] = projected
    split['posterior_ntb'] = posterior_ntb
    split['fisher_direction'] = direction
    split['is_refined'] = False

    if do_refinement and split['has_natural_split']:
        if verbose:
            print(f"\n  Attempting refinement...")
        refined = refine_fisher_with_split(embeddings, labels, split, verbose=verbose)
        # Keep refined if it still has a natural split
        if refined['has_natural_split']:
            if verbose:
                print("  Refinement successful — using refined split")
            return refined
        else:
            if verbose:
                print("  Refinement lost the split — keeping original")
            return split
    return split


# =============================================================================
# 6. DENSITY OVERLAP DIAGNOSTICS (outside the paper runner)
# =============================================================================

def estimate_class_densities(projected_1d, labels, n_grid=1000):
    """KDE densities for TB+ and NTB on Fisher axis."""
    tb = projected_1d[labels == 1].reshape(-1, 1)
    ntb = projected_1d[labels == 0].reshape(-1, 1)
    kde_tb = KernelDensity(bandwidth=_silverman_bw(tb), kernel='gaussian').fit(tb)
    kde_ntb = KernelDensity(bandwidth=_silverman_bw(ntb), kernel='gaussian').fit(ntb)
    margin = (projected_1d.max() - projected_1d.min()) * 0.1
    grid = np.linspace(projected_1d.min() - margin,
                       projected_1d.max() + margin, n_grid).reshape(-1, 1)
    return grid.flatten(), np.exp(kde_tb.score_samples(grid)), \
        np.exp(kde_ntb.score_samples(grid))


def compute_overlap_coefficient(grid, density_tb, density_ntb):
    """OVL = integral min(f_TB, f_NTB). 1=perfect overlap, 0=separated."""
    dx = grid[1] - grid[0]
    ovl = np.sum(np.minimum(density_tb, density_ntb)) * dx
    norm = (np.sum(density_tb) * dx + np.sum(density_ntb) * dx) / 2.0
    return np.clip(ovl / max(norm, 1e-10), 0.0, 1.0)


def fit_tb_mixture_on_fisher(projected_1d, labels):
    """
    Fit TB+ as 2-comp GMM on Fisher axis (not posteriors).
    Test if one component matches NTB via KS test.
    """
    tb = projected_1d[labels == 1].reshape(-1, 1)
    ntb = projected_1d[labels == 0].reshape(-1, 1)
    if len(tb) < 30:
        return {'error': 'Too few TB+'}

    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(tb)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=10).fit(tb)

    result = {'bic_1': gmm1.bic(tb), 'bic_2': gmm2.bic(tb),
              'bimodal_evidence': gmm1.bic(tb) - gmm2.bic(tb)}

    if result['bimodal_evidence'] > 0:
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())
        weights = gmm2.weights_
        order = np.argsort(means)
        means, stds, weights = means[order], stds[order], weights[order]

        result.update({
            'comp_means': means.tolist(), 'comp_stds': stds.tolist(),
            'comp_weights': weights.tolist(),
            'ntb_mean': float(ntb.mean()), 'ntb_std': float(ntb.std()),
        })

        for i in range(2):
            samp = np.random.RandomState(42).normal(
                means[i], stds[i], min(500, len(ntb)))
            ks_s, ks_p = stats.ks_2samp(samp, ntb.flatten())
            result[f'comp{i}_ks_stat'] = ks_s
            result[f'comp{i}_ks_pval'] = ks_p
            result[f'comp{i}_matches_ntb'] = ks_p > 0.05

        dists = np.abs(means - result['ntb_mean'])
        lo = int(np.argmin(dists))
        result['label_only_component'] = lo
        result['label_only_weight'] = float(weights[lo])
        result['biomarker_component'] = 1 - lo
        result['biomarker_weight'] = float(weights[1 - lo])

    return result


def run_density_overlap_analysis(projected_1d, labels, verbose=True):
    """Full density overlap analysis."""
    results = {}
    grid, d_tb, d_ntb = estimate_class_densities(projected_1d, labels)
    results['grid'] = grid
    results['density_tb'] = d_tb
    results['density_ntb'] = d_ntb

    ovl = compute_overlap_coefficient(grid, d_tb, d_ntb)
    results['overlap_coefficient'] = ovl
    results['separable_fraction'] = 1.0 - ovl
    if verbose:
        print(f"  OVL: {ovl:.4f} (separable: {(1-ovl)*100:.1f}%)")

    decomp = fit_tb_mixture_on_fisher(projected_1d, labels)
    results['decomposition'] = decomp
    if verbose:
        ev = decomp.get('bimodal_evidence', 0)
        tag = "YES" if ev > 10 else ("weak" if ev > 0 else "no")
        print(f"  Bimodal evidence on Fisher axis: {ev:.1f} ({tag})")
        if 'label_only_weight' in decomp:
            print(f"  Mixture: label-only={decomp['label_only_weight']*100:.1f}%, "
                  f"biomarker={decomp['biomarker_weight']*100:.1f}%")
            for i in range(2):
                if f'comp{i}_matches_ntb' in decomp:
                    m = "MATCHES" if decomp[f'comp{i}_matches_ntb'] else "differs"
                    print(f"  Comp {i} vs NTB: {m} "
                          f"(KS p={decomp[f'comp{i}_ks_pval']:.4f})")
    return results


# =============================================================================
# 7. k-NN ENRICHMENT
# =============================================================================

def compute_local_tb_enrichment(embeddings, labels, k_neighbors=20):
    """Fraction of k-NN that are TB+. Independent of Fisher."""
    k = min(k_neighbors, len(embeddings) // 3)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    nbr = indices[:, 1:]
    pop = labels.mean()
    enrichment = np.zeros(len(embeddings))
    pvalues = np.ones(len(embeddings))
    for i in range(len(embeddings)):
        tc = int(labels[nbr[i]].sum())
        enrichment[i] = tc / k
        pvalues[i] = stats.binomtest(tc, k, pop, alternative='greater').pvalue
    return enrichment, pvalues


# =============================================================================
# 8. LODO CONSISTENCY
# =============================================================================

def lodo_configurations(dataset_labels):
    from itertools import combinations
    unique_ds = np.unique(dataset_labels)
    yield 'all', np.ones(len(dataset_labels), dtype=bool)
    for pair in combinations(unique_ds, 2):
        yield '+'.join(sorted(pair)), np.isin(dataset_labels, pair)


def run_lodo_consistency(embeddings, labels, dataset_labels,
                         k_neighbors=20, verbose=True):
    """Run Fisher + split + k-NN on each config, check consistency."""
    N = len(embeddings)
    config_results = {}

    for config_name, mask in lodo_configurations(dataset_labels):
        if verbose:
            print(f"\n--- Config '{config_name}' (n={mask.sum()}) ---")

        sub_e, sub_l = embeddings[mask], labels[mask]

        # Fisher + split
        result = run_fisher_and_split(sub_e, sub_l, verbose=verbose)

        # Overlap
        overlap = run_density_overlap_analysis(
            result['projected_1d'], sub_l, verbose=verbose)

        # k-NN
        k = min(k_neighbors, len(sub_e) // 3)
        enrichment, epvals = compute_local_tb_enrichment(sub_e, sub_l, k)

        # Store in full-size arrays
        full_post = np.full(N, np.nan)
        full_proj = np.full(N, np.nan)
        full_enrich = np.full(N, np.nan)
        full_bio = np.zeros(N, dtype=bool)
        full_lo = np.zeros(N, dtype=bool)

        sub_idx = np.where(mask)[0]
        full_post[sub_idx] = result['posterior_ntb']
        full_proj[sub_idx] = result['projected_1d']
        full_enrich[sub_idx] = enrichment

        for j, gi in enumerate(sub_idx):
            if result['biomarker_mask'][j]:
                full_bio[gi] = True
            if result['label_only_mask'][j]:
                full_lo[gi] = True

        config_results[config_name] = {
            'posterior_ntb': full_post,
            'projected_1d': full_proj,
            'enrichment': full_enrich,
            'biomarker_mask': full_bio,
            'label_only_mask': full_lo,
            'overlap': overlap,
            'has_natural_split': result['has_natural_split'],
            'split_details': {k: v for k, v in result.items()
                              if k not in ('projected_1d', 'posterior_ntb',
                                           'fisher_direction', 'tb_posteriors',
                                           'component_probs')},
        }

    # Consistency
    if verbose:
        print("\n  Computing consistency...")
    cnames = list(config_results.keys())
    consistency = {}
    for st in ['posterior_ntb', 'enrichment']:
        pairs = []
        for i, a in enumerate(cnames):
            for b in cnames[i+1:]:
                sa, sb = config_results[a][st], config_results[b][st]
                valid = ~(np.isnan(sa) | np.isnan(sb))
                if valid.sum() > 10:
                    rho, pv = stats.spearmanr(sa[valid], sb[valid])
                    pairs.append({'a': a, 'b': b, 'rho': rho,
                                  'pval': pv, 'n': int(valid.sum())})
        consistency[st] = pairs

    # Classification agreement
    tb_idx = np.where(labels == 1)[0]
    agrees = []
    for idx in tb_idx:
        classes = []
        for cn in cnames:
            if not np.isnan(config_results[cn]['posterior_ntb'][idx]):
                if config_results[cn]['biomarker_mask'][idx]:
                    classes.append('BIO')
                elif config_results[cn]['label_only_mask'][idx]:
                    classes.append('LO')
                else:
                    classes.append('AMB')
        if len(classes) >= 2:
            matches = sum(1 for i in range(len(classes))
                          for j in range(i+1, len(classes))
                          if classes[i] == classes[j])
            total = len(classes) * (len(classes)-1) / 2
            agrees.append(matches / total)
    consistency['classification_agreement'] = np.mean(agrees) if agrees else 0.0

    # How many configs found a natural split?
    n_splits = sum(1 for cr in config_results.values()
                   if cr['has_natural_split'])
    consistency['configs_with_split'] = n_splits
    consistency['total_configs'] = len(cnames)

    return config_results, consistency


# =============================================================================
# 9. CONSENSUS
# =============================================================================

def compute_consensus(config_results, labels, dataset_labels,
                      gender, participant):
    """Aggregate LODO configs into per-subject verdict."""
    N = len(labels)
    cnames = list(config_results.keys())
    rows = []

    for i in range(N):
        row = {
            'subject_idx': i, 'label': int(labels[i]),
            'dataset': dataset_labels[i], 'gender': gender[i],
            'participant': participant[i],
        }
        posts, enrichs = [], []
        bio_v, lo_v, n_cfg = 0, 0, 0

        for cn in cnames:
            cr = config_results[cn]
            if not np.isnan(cr['posterior_ntb'][i]):
                posts.append(cr['posterior_ntb'][i])
                enrichs.append(cr['enrichment'][i])
                n_cfg += 1
                if cr['biomarker_mask'][i]:
                    bio_v += 1
                elif cr['label_only_mask'][i]:
                    lo_v += 1

        row['n_configs'] = n_cfg
        row['mean_posterior_ntb'] = np.mean(posts) if posts else np.nan
        row['std_posterior_ntb'] = np.std(posts) if len(posts) > 1 else np.nan
        row['mean_enrichment'] = np.mean(enrichs) if enrichs else np.nan
        row['biomarker_votes'] = bio_v
        row['label_only_votes'] = lo_v

        if labels[i] == 0:
            row['consensus'] = 'NTB'
            row['confidence'] = 1.0
        elif n_cfg == 0:
            row['consensus'] = 'UNKNOWN'
            row['confidence'] = 0.0
        else:
            mp = np.mean(posts)
            vf_bio = bio_v / n_cfg
            vf_lo = lo_v / n_cfg

            if mp < 0.4 and vf_bio >= 0.5:
                row['consensus'] = 'BIOMARKER_TB'
                row['confidence'] = (1.0 - mp) * vf_bio
            elif mp > 0.6 and vf_lo >= 0.5:
                row['consensus'] = 'LABEL_ONLY_TB'
                row['confidence'] = mp * vf_lo
            else:
                row['consensus'] = 'AMBIGUOUS_TB'
                row['confidence'] = 1.0 - 2 * abs(mp - 0.5)

        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 10. UTILITIES
# =============================================================================

def print_data_summary(embeddings, labels, dataset_labels, gender_labels):
    print(f"\n=== DATA SUMMARY ===")
    print(f"Total: {len(embeddings)}, Dim: {embeddings.shape[1]}")
    print(f"TB+: {(labels==1).sum()} ({(labels==1).mean()*100:.1f}%), "
          f"TB-: {(labels==0).sum()} ({(labels==0).mean()*100:.1f}%)")
    for ds in np.unique(dataset_labels):
        m = dataset_labels == ds
        print(f"  {ds}: n={m.sum()}, TB+={labels[m].mean()*100:.1f}%")
    print("=" * 40)


def print_consistency(consistency):
    print("\n=== LODO CONSISTENCY ===")
    print(f"  Configs with natural split: "
          f"{consistency['configs_with_split']}/{consistency['total_configs']}")
    for st, pairs in consistency.items():
        if st in ('classification_agreement', 'configs_with_split', 'total_configs'):
            continue
        if not pairs:
            continue
        rhos = [p['rho'] for p in pairs]
        print(f"\n  {st}:")
        for p in pairs:
            print(f"    {p['a']} vs {p['b']}: rho={p['rho']:.3f} "
                  f"(p={p['pval']:.2e}, n={p['n']})")
        print(f"    Mean rho: {np.mean(rhos):.3f}")
    print(f"\n  Classification agreement: "
          f"{consistency['classification_agreement']:.3f}")
    print("=" * 40)


def print_overlap_across_configs(config_results):
    print("\n=== OVERLAP ACROSS CONFIGS ===")
    for cn, cr in config_results.items():
        ov = cr['overlap']
        split_str = "SPLIT" if cr['has_natural_split'] else "gradient"
        print(f"  {cn}: OVL={ov['overlap_coefficient']:.4f} "
              f"(sep={ov['separable_fraction']*100:.1f}%) [{split_str}]")
        d = ov.get('decomposition', {})
        if 'label_only_weight' in d:
            print(f"    Fisher mixture: lo={d['label_only_weight']*100:.1f}%, "
                  f"bio={d['biomarker_weight']*100:.1f}%")
    ovls = [cr['overlap']['overlap_coefficient'] for cr in config_results.values()]
    if len(ovls) > 1:
        print(f"\n  OVL: mean={np.mean(ovls):.4f}, std={np.std(ovls):.4f}")
    print("=" * 40)


def print_consensus_summary(consensus_df):
    print("\n=== CONSENSUS ===")
    for cat, n in consensus_df['consensus'].value_counts().items():
        conf = consensus_df[consensus_df['consensus'] == cat]['confidence'].mean()
        print(f"  {cat}: {n} ({n/len(consensus_df)*100:.1f}%) "
              f"[conf={conf:.2f}]")
    tb = consensus_df[consensus_df['label'] == 1]
    if len(tb) > 0:
        print(f"\n  Within TB+ ({len(tb)}):")
        for cat, n in tb['consensus'].value_counts().items():
            print(f"    {cat}: {n} ({n/len(tb)*100:.1f}%)")
    print("=" * 40)


# =============================================================================
# MEL NORMALIZATION TECHNIQUES
# =============================================================================

def mel_normalize(input_features, method='none'):
    """
    Apply normalization to mel spectrogram tensor.

    Args:
        input_features: (B, F, T) — batch of mel spectrograms
                        OR (F, T_flat) — after masking/flattening
        method: normalization technique name

    Returns:
        normalized input_features, same shape
    """
    if method == 'none':
        return input_features

    elif method == 'freq_cmvn_mean':
        # Per-utterance, per-frequency-bin mean subtraction.
        # Removes the static device frequency response.
        # This is the single most impactful technique from DCASE.
        # For each mel bin, subtract its average across time.
        return input_features - input_features.mean(dim=-1, keepdim=True)

    elif method == 'freq_cmvn_full':
        # Per-utterance, per-frequency-bin mean+variance normalization.
        # More aggressive — also normalizes energy variation per bin.
        mean = input_features.mean(dim=-1, keepdim=True)
        std = input_features.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (input_features - mean) / std

    elif method == 'instance_norm_freq':
        # Instance normalization across frequency axis per time frame.
        # For each time frame, normalize across all mel bins.
        # Removes frame-level spectral shape entirely.
        mean = input_features.mean(dim=-2, keepdim=True)  # mean across F
        std = input_features.std(dim=-2, keepdim=True).clamp(min=1e-6)
        return (input_features - mean) / std

    elif method == 'instance_norm_time':
        # Instance normalization across time axis per frequency bin.
        # Same as freq_cmvn_full but using instance norm formulation.
        mean = input_features.mean(dim=-1, keepdim=True)  # mean across T
        std = input_features.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (input_features - mean) / std

    elif method == 'freq_cmvn_mean_alpha':
        # Relaxed frequency-wise mean subtraction (RFN-like).
        # alpha=0.0 = no normalization, alpha=1.0 = full subtraction.
        # Allows the model to retain SOME device info if it's useful.
        alpha = 0.7  # tunable
        mean = input_features.mean(dim=-1, keepdim=True)
        return input_features - alpha * mean

    elif method == 'global_cmvn':
        # Global mean+variance normalization (across both F and T).
        # Removes all first-order statistics. Very aggressive.
        mean = input_features.mean(dim=(-2, -1), keepdim=True)
        std = input_features.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (input_features - mean) / std

    elif method == 'freq_median_subtract':
        # Subtract median instead of mean per frequency bin.
        # More robust to outlier frames (e.g., cough onset transients).
        median = input_features.median(dim=-1, keepdim=True).values
        return input_features - median

    elif method == 'freq_cmvn_mean_logscale':
        # If input_features are in log-mel scale, device response is additive.
        # Mean subtraction directly removes it.
        # If already in log scale (which WhisperFeatureExtractor produces),
        # this is mathematically equivalent to dividing out the device
        # frequency response in the linear magnitude domain.
        # Same as freq_cmvn_mean but named explicitly for clarity.
        return input_features - input_features.mean(dim=-1, keepdim=True)

    elif method == 'freq_percentile_norm':
        # Normalize each frequency bin to [0, 1] range using percentiles.
        # Robust to outliers.
        p05 = torch.quantile(input_features, 0.05, dim=-1, keepdim=True)
        p95 = torch.quantile(input_features, 0.95, dim=-1, keepdim=True)
        denom = (p95 - p05).clamp(min=1e-6)
        return (input_features - p05) / denom

    else:
        raise ValueError(f"Unknown mel normalization method: {method}")



def get_outlier_mask(df_temp, contamination=0.05):
    """
    df_temp: already filtered dataframe (single class / subset)
    contamination: fraction of outliers (e.g. 0.05)

    Returns:
        outlier_mask (True = outlier)
    """

    # ─────────────────────────────────────────────
    # Embedding + normalize
    # ─────────────────────────────────────────────
    df_temp = df_temp.reset_index(drop=True)
    raw_emb = np.stack(df_temp["embed"].values)

    # ─────────────────────────────────────────────
    # PCA (with whitening for better distance behavior)
    # ─────────────────────────────────────────────
    pca = PCA(n_components=min(50, raw_emb.shape[1]), random_state=42, whiten=True)
    emb_pca = pca.fit_transform(raw_emb)

    # ─────────────────────────────────────────────
    # Isolation Forest
    # ─────────────────────────────────────────────
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(emb_pca)

    if_scores = -clf.score_samples(emb_pca)
    if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)

    # ─────────────────────────────────────────────
    # Centroid distance (robust)
    # ─────────────────────────────────────────────
    centroid = emb_pca.mean(axis=0)
    dists = np.linalg.norm(emb_pca - centroid, axis=1)

    med = np.median(dists)
    mad = np.median(np.abs(dists - med)) + 1e-8

    z = (dists - med) / mad
    z = np.clip(z, 0, None)

    cd_scores = (z - z.min()) / (z.max() - z.min() + 1e-8)
    # ─────────────────────────────────────────────
    # Ensemble
    # ─────────────────────────────────────────────
    scores = 0.5 * if_scores + 0.5 * cd_scores

    threshold = np.percentile(scores, 100 * (1 - contamination))
    outlier_mask = scores >= threshold

    return outlier_mask
