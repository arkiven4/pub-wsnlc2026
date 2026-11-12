"""Stage 03: fit the paper cleaners and save per-clip decisions, without classifiers."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import confident_learning as cl
from . import utils
from .helper import fit_pca, run_fisher_and_split

STAGES = [('raw', 'Raw'), ('pbnr', 'CL Cleaning (prune_by_noise_rate)'),
          ('cl', 'CL Cleaning (confident_learning)'),
          ('proposed', 'Proposed Cleaning (refine=False)')]


def file_sha256(path):
    """@brief Hash a file without loading the complete matrix into memory.
    @param path Input file path.
    @return SHA256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprints(encoder, embeddings_dir):
    """@brief Record the exact stage-02 files used to produce cleaning decisions.
    @param encoder Encoder name: qwen or hear.
    @param embeddings_dir Directory holding the matrix and its metadata CSV.
    @return Mapping from the two input filenames to their SHA256 digests.
    @note Metadata is hashed too: reordering clips must invalidate old masks.
    """
    root = Path(embeddings_dir)
    return {name: file_sha256(root / name) for name in
            [f'metadata_{encoder}.csv', f'embeds_{encoder}.npy']}


def compute_cleaning(encoder, embeddings_dir, cleaning_dir):
    """@brief Compute Original, CL-PBNR, CL-CL and Proposed selection masks.
    @param encoder Encoder name used in the stage-02 filenames.
    @param embeddings_dir Directory containing aligned extracted features and metadata.
    @param cleaning_dir Destination for per-clip masks, diagnostic CSV and provenance.
    @return Dictionary of Boolean keep masks and Consistent-TB count masks.
    @details Load the three study databases, fit global PCA-256, then fit each
             cleaner per database. Write scores and decisions so readers can
             inspect the selection before running evaluation.
    @note Non-TB clips are retained. Random Cleaning draws its 15 baseline seeds
          in stage 04 using the Proposed counts. No evaluation classifier runs here.
    """
    # Keep decisions tied to both the feature values and their metadata order.
    source_hashes = input_fingerprints(encoder, embeddings_dir)
    df, raw, labels, _, _, datasets = utils.load_data_precomp(
        db_filter=[0, 1, 2], name=encoder, base_dir=embeddings_dir)
    # Cleaners share this reduction; evaluation will select original features.
    _, reduced, _ = fit_pca(raw, 256)
    decisions = {'raw': np.ones(len(df), dtype=bool)}
    columns = df.drop(columns=['embed']).copy()
    tb = df.disease_status.to_numpy() == 1
    decisions['consistent_raw'] = tb.copy()
    for key, method in [('pbnr', 'prune_by_noise_rate'), ('cl', 'confident_learning'),
                        ('proposed', None)]:
        if method is None:
            split = run_fisher_and_split(reduced, labels, dataset_labels=datasets,
                                         do_refinement=False)
        else:
            split = cl.run_cleanlab(reduced, labels.astype(int), dataset_labels=datasets,
                                    n_splits=5, filter_by=method,
                                    do_refinement=False, verbose=False)
        # Internal codes: 0 non-TB, 1 Consistent TB, 2 Ambiguous TB.
        revised = np.zeros(len(df), dtype=int)
        if split['has_natural_split']:
            revised[tb & split['biomarker_mask']] = 1
            revised[tb & split['label_only_mask']] = 2
        else:
            revised[tb] = 1
        diagnostic = df.copy()
        diagnostic['disease_status_rev'] = revised
        utils.print_tb_stats(diagnostic, tb)
        decisions[key] = (revised != 2) & (datasets != 3)
        decisions[f'consistent_{key}'] = tb & (revised == 1)
        columns[f'keep_{key}'] = decisions[key]
        columns[f'acoustic_tb_score_{key}'] = 1.0 - split['posterior_ntb']
        columns[f'fisher_projection_{key}'] = split['projected_1d']
    # Write inspectable scores as well as the masks stage 04 consumes.
    target = Path(cleaning_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifact = target / f'decisions_{encoder}.npz'
    np.savez_compressed(artifact, **decisions)
    columns.to_csv(target / f'decisions_{encoder}.csv', index=False)
    manifest = {'schema': 1, 'encoder': encoder, 'rows': len(df),
                'inputs': source_hashes, 'decisions_sha256': file_sha256(artifact),
                'pca_components': 256, 'cleaning_before_classifier_cv': True,
                'source': 'Computed by stage 03 from extracted study embeddings; no reference features'}
    (target / f'decisions_{encoder}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return decisions


def load_cleaning(encoder, embeddings_dir, cleaning_dir, rows):
    """@brief Read decisions only when they match the current extraction inputs.
    @param encoder Encoder name used in the decision filenames.
    @param embeddings_dir Directory of the original matrix and metadata.
    @param cleaning_dir Directory written by stage 03.
    @param rows Expected number of study clips after database filtering.
    @return Validated Boolean decision arrays, in stage-02 row order.
    @exception FileNotFoundError Stage 03 has not produced a manifest.
    @exception ValueError Input hashes, decision hash, row count or array types differ.
    @note Reading generated decisions is not substituting stored reference features.
    """
    target = Path(cleaning_dir)
    manifest_path = target / f'decisions_{encoder}.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Run scripts/03_clean_labels.py first: missing {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    artifact = target / f'decisions_{encoder}.npz'
    if (manifest.get('schema') != 1 or manifest.get('encoder') != encoder or
            manifest.get('rows') != rows or
            manifest.get('inputs') != input_fingerprints(encoder, embeddings_dir) or
            manifest.get('decisions_sha256') != file_sha256(artifact)):
        raise ValueError('Cleaning decisions/input provenance differs; rerun stage 03 for these embeddings.')
    with np.load(artifact, allow_pickle=False) as saved:
        keys = [key for key, _ in STAGES]
        keys += [f'consistent_{key}' for key, _ in STAGES]
        decisions = {key: saved[key].copy() for key in keys}
    if any(mask.dtype != np.bool_ or mask.shape != (rows,) for mask in decisions.values()):
        raise ValueError('Invalid cleaning decision mask.')
    return decisions
