"""Format study outputs as Table I and compare with the published values."""
import json
from pathlib import Path
import pandas as pd

METHODS = {'Raw': 'Original', 'Random Cleaning': 'Random',
           'CL Cleaning (prune_by_noise_rate)': 'CL (PBNR)',
           'CL Cleaning (confident_learning)': 'CL (CL)',
           'Proposed Cleaning (refine=False)': 'Proposed'}
DATASETS = {0: 'CODA', 1: 'CIDRZ', 2: 'TBScreen'}
METRICS = {'auroc': 'AUROC', 'bacc': 'UAR', 'sens': 'Sensitivity', 'spec': 'Specificity'}


def write_table(output_dir, encoders):
    """@brief Format evaluated results and report differences from published Table I.
    @param output_dir Directory containing the stage-04 overall CSVs.
    @param encoders Encoder names whose results should be included.
    @return None; write table_i.csv, table_i.md and table_i_comparison.csv.
    @details Validate all 15 dataset/method rows per encoder, convert formatted
             proportions to percentages, and join published means/stds for comparison.
    @note Reference numbers are comparison targets only. They never replace a
          computed metric or alter the classifier outputs.
    """
    output_dir = Path(output_dir)
    reference = pd.DataFrame(json.loads(Path(__file__).with_name('table_i_reference.json').read_text()))
    table, comparison = [], []
    for encoder in encoders:
        df = pd.read_csv(output_dir / f'overall_emb {encoder}_perdb.csv')
        if len(df) != 15:
            raise ValueError(f'Expected 15 study rows for {encoder}, got {len(df)}')
        for db, dataset in DATASETS.items():
            for stage, method in METHODS.items():
                selected = df[(df.db == db) & (df.stage == stage)]
                if len(selected) != 1:
                    raise ValueError(f'Missing or duplicate result: {encoder}/{dataset}/{method}')
                result = {'Encoder': encoder, 'Dataset': dataset, 'Method': method}
                row = selected.iloc[0]
                for column, metric in METRICS.items():
                    mean, std = [round(float(v.strip())*100) for v in row[column].split('±')]
                    result[metric] = f'{mean} ± {std}'
                    target = reference[(reference.encoder == encoder) & (reference.dataset == dataset) &
                                       (reference.method == method) & (reference.metric == metric)].iloc[0]
                    comparison.append(dict(encoder=encoder, dataset=dataset, method=method, metric=metric,
                                           paper_mean=int(target['mean']), paper_std=int(target['std']),
                                           local_mean=mean, local_std=std, delta_pp=mean-int(target['mean'])))
                table.append(result)
    table = pd.DataFrame(table)
    table.to_csv(output_dir / 'table_i.csv', index=False)
    compare = pd.DataFrame(comparison)
    compare.to_csv(output_dir / 'table_i_comparison.csv', index=False)
    columns = list(table.columns)
    lines = ['# Regenerated Table I', '', 'All values are percentages (mean ± standard deviation), using the paper study aggregation.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, values)) + ' |' for values in table.itertuples(index=False, name=None))
    lines += ['', 'Published-value differences are in `table_i_comparison.csv`.',
              'Regenerated embeddings may differ from the paper; see the repository README.']
    (output_dir / 'table_i.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {output_dir / "table_i.md"}', flush=True)
