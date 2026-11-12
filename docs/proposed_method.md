# Proposed: from embeddings to decisions

Start at `lib/study_cleaning.py::compute_cleaning`. The paper calls
`fit_pca(features, 256)`, then `run_fisher_and_split(..., do_refinement=False,
dataset_labels=...)`. The encoder is already frozen; this stage fits statistical
models on its output. The unit of analysis is a cough clip.

## 1. Reduce the frozen features

`lib/helper.py::fit_pca` fits PCA once to all study embeddings. The input is
(N, D); the cleaning representation is (N, 256). This is the global cleaning
PCA. The evaluation classifiers later fit a different PCA on their own training
subsets, starting from the original frozen embeddings.

## 2. Fit the Fisher axis within each dataset

`run_per_database` selects each source database and calls the cleaner on that
subset. `compute_fisher_projection` fits binary LDA with the supplied non-TB/TB
labels. It projects every subset row to one coordinate and flips the projected
axis, if needed, so the TB mean is higher.

The optional TB mask supports a later refit. The paper's run does not use that
refinement. The returned LDA coefficient is not sign-flipped with the projected
coordinates; use the actual projected array when reproducing the following steps.

## 3. Fit two class densities

`compute_posterior_ntb` fits one Gaussian KDE to the TB coordinates and another
to non-TB coordinates. `_silverman_bw` uses the exact code rule:

```text
bandwidth(class) = max(1.09 * std(class_coordinates) * n_class**(-0.2), 0.001)
```

The class priors are the observed class fractions in that database. Both
labelled classes influence the score; this is not a density fit on TB alone.

## 4. Compute the acoustic non-TB posterior

Bayes' rule combines the two densities and their priors:

```text
p_NTB(x) = f_NTB(x) * prior_NTB
           ----------------------------------------------
           f_NTB(x) * prior_NTB + f_TB(x) * prior_TB

acoustic_tb_score(x) = 1 - p_NTB(x)
```

The code evaluates the numerator/denominator in log space to avoid underflow.
A high `p_NTB` means the clip is acoustically closer to the labelled non-TB
population along this axis. It is not a clinical disease probability.

For example, an illustrative `p_NTB=0.8` gives a stored TB-likeness score 0.2.
That single value does **not** determine removal; the next step fits a population
mixture to the TB-labelled scores.

## 5. Test for a split among TB-labelled clips

`find_natural_split` fits one-, two- and three-component GMMs to the TB-labelled
`p_NTB` values. The three-component BIC is diagnostic only. The actual partition
is always the two-component prediction, accepted when all of the following hold:

1. `BIC(one component) - BIC(two components) > 0`.
2. The component-mean distance divided by average component std exceeds 0.5.
3. The means straddle 0.5, or that separation ratio exceeds 1.5.

The lower-mean non-TB-posterior component is **Acoustically Consistent TB**;
the higher-mean component is **Acoustically Ambiguous TB**. A clip's component
is assigned by `gmm2.predict`, not by comparing its posterior against 0.5.
If the checks fail, retain all TB clips in that database.

## 6. Restore the metadata order and write decisions

`run_per_database` scatters subset scores/masks back into the full input order.
Stage 03 writes `keep_proposed`, the acoustic score, the Fisher coordinate, and
input fingerprints. The original labels remain unchanged. Stage 04 applies the
keep mask to the original frozen embeddings and evaluates that selected subset.

The code uses older internal names for these outputs:

| Internal key | Meaning used in the paper |
|---|---|
| `biomarker_mask` | Acoustically Consistent TB; retain |
| `label_only_mask` | Acoustically Ambiguous TB; remove |
| `posterior_ntb` | Acoustic non-TB posterior |
| `projected_1d` | Oriented Fisher coordinate |
| `has_natural_split` | The split passed the implementation's checks |

## Compare with Confident Learning

The paper uses `lib/confident_learning.py::run_cleanlab`, not the earlier manual
CL helpers in that file. A balanced logistic model estimates out-of-fold class
probabilities through five stratified folds. Each fold's model uses training-fold
scaling. cleanlab then flags label issues with PBNR or CL; the study wrapper only
removes flagged TB clips and retains every non-TB clip.

Proposed fits a one-dimensional acoustic-density/mixture model. CL fits an
auxiliary probability predictor and applies its label-quality rules. Neither
is repeated 15 times. The **Random** evaluation baseline is the one with 15 seeds;
its original retained-count-as-drop-count implementation is documented in
[reproducibility.md](reproducibility.md).

## Evaluation details to keep separate

Cleaning precedes classifier CV in this paper protocol. Outer classification
folds are participant grouped; the inner 80/20 calibration split is sample
stratified. The function called `optimize_threshold_youden` actually minimizes
`abs(TPR - TNR)` on calibration predictions. These facts matter when interpreting
the selected-subset results; the documentation describes the existing program
rather than silently changing the experiment.
