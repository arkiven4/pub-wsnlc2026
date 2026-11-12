"""Step 06: reproduce paper Figs. 1 and 4 (MFCC-0 statistics before/after cleaning).

Loads every clip of one dataset (CODA by default) from the prepared WAVs,
recomputes 153 MFCC/spectral statistics, and plots KDE + box plots for the
three paper features:

* before cleaning: Non-TB vs TB (sputum labels);
* after cleaning: Non-TB vs Acoustically Ambiguous TB vs Acoustically Consistent TB,
  using the Proposed decisions written by stage 03.

It also ranks all 153 statistics by single-feature AUROC (Consistent TB vs
Non-TB, as in the original analysis) to show why these three were chosen.
This is a descriptive analysis on the cleaned groups, not held-out validation.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from _common import ROOT, positive_int

DATASETS = {'coda': 0, 'cidrz': 1, 'tbscreen': 2}
# Same colours/labels as the paper figures (original notebook cells 15 and 17).
BEFORE = [('Non-TB', 'Non-TB', '#2196F3'), ('TB', 'TB', '#F44336')]
AFTER = [('Non-TB', 'Non-TB', '#2196F3'), ('Ambiguous TB', 'Ambiguous', '#FF9800'),
         ('Consistent TB', 'Consistent', '#F44336')]
PAPER_STYLE = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42}


def _extract(path):
    from lib.acoustic_features import extract
    return extract(str(ROOT / path))


def load_features(frame, cache, workers):
    """@brief Extract (or reuse cached) acoustic statistics for every row.
    @param frame Decision rows of one dataset; 'clip' and ``path_file`` used.
    @param cache .npz path; reused only when its clip list equals frame['clip'].
    @param workers Number of extraction processes.
    @return (n_clips, 153) float32 matrix aligned to ``frame``.
    """
    import numpy as np
    clips = frame['clip'].to_numpy().astype(str)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if np.array_equal(saved['clips'], clips):
                print(f'Reusing acoustic features {cache}', flush=True)
                return saved['features']
    missing = [p for p in frame.path_file if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} WAVs missing, e.g. {missing[0]}; run stage 01')
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, vector in enumerate(pool.map(_extract, frame.path_file, chunksize=32), 1):
            rows.append(vector)
            if i % 1000 == 0 or i == len(frame):
                print(f'Extracted {i}/{len(frame)}', flush=True)
    features = np.vstack(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, clips=clips, features=features)
    return features


def rank_features(features, groups, names):
    """@brief Single-feature AUROC of Consistent TB vs Non-TB (Ambiguous excluded).
    @return DataFrame sorted by AUROC = max(AUC, 1-AUC), with Cohen's d.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    pool = groups != 'Ambiguous TB'
    y = (groups[pool] == 'Consistent TB').astype(int)
    rows = []
    for i, name in enumerate(names):
        a = features[pool, i]
        ok = np.isfinite(a)
        auc = roc_auc_score(y[ok], a[ok])
        tb, ntb = a[ok & (y == 1)], a[ok & (y == 0)]
        d = (tb.mean() - ntb.mean()) / (np.sqrt((tb.var(ddof=1) + ntb.var(ddof=1)) / 2) + 1e-12)
        rows.append(dict(feature=name, auroc=max(auc, 1 - auc), cohen_d=d))
    return pd.DataFrame(rows).sort_values('auroc', ascending=False, ignore_index=True)


def plot(features, groups, names, selected, styles, title, path):
    """@brief Draw one figure in the paper layout: KDE row on top, box-plot row below.
    @param styles (group, box label, colour) per group, in drawing order.
    @return List of per-group summary dictionaries for the CSV report.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from lib.acoustic_features import pretty_name
    fig, axs = plt.subplots(2, len(selected), figsize=(7.16, 4.7),
                            constrained_layout=True, squeeze=False)
    summary = []
    for j, feature in enumerate(selected):
        column = features[:, names.index(feature)]
        values = []
        for label, _, color in styles:
            a = column[(groups == label) & np.isfinite(column)]
            values.append(a)
            grid = np.linspace(a.min(), a.max(), 400)
            axs[0, j].plot(grid, gaussian_kde(a, bw_method='scott')(grid), color=color,
                           linewidth=1.5, label=label)
            summary.append(dict(view=title, feature=feature, group=label, n=len(a),
                                mean=a.mean(), std=a.std(), p25=np.percentile(a, 25),
                                median=np.median(a), p75=np.percentile(a, 75)))
        axs[0, j].set_title(pretty_name(feature))
        axs[0, j].set(xlabel='Value', ylabel='Density')
        box = axs[1, j].boxplot(values, tick_labels=[s[1] for s in styles], patch_artist=True,
                                medianprops=dict(color='black', linewidth=1.2), widths=0.6,
                                flierprops=dict(marker='o', markersize=2, alpha=0.4))
        for patch, (_, _, color) in zip(box['boxes'], styles):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axs[1, j].set_ylabel('Value')
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='center', ncol=len(styles),
               bbox_to_anchor=(0.5, -0.03 if len(styles) == 2 else -0.04), frameon=False)
    for suffix in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return summary


def main():
    """@brief Parse CLI options, extract features, rank them and write both figures.
    @return None; outputs go to ``--output-dir``.
    @see docs/workflow.md
    """
    from lib.acoustic_features import PAPER_FEATURES
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--encoder', choices=['qwen', 'hear'], default='qwen',
                        help='whose stage-03 Proposed decisions define the subgroups')
    parser.add_argument('--dataset', choices=list(DATASETS), default='coda')
    parser.add_argument('--cleaning-dir', type=Path, default=ROOT / 'data/cleaning')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/acoustic_figures')
    parser.add_argument('--features', nargs='+', default=PAPER_FEATURES,
                        help='statistics to plot (default: the three paper features)')
    parser.add_argument('--top', type=int, default=0,
                        help='plot the top-N ranked statistics instead of --features')
    parser.add_argument('--workers', type=positive_int, default=8)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lib.acoustic_features import feature_names

    source = args.cleaning_dir / f'decisions_{args.encoder}.csv'
    if not source.is_file():
        raise FileNotFoundError(f'Run scripts/03_clean_labels.py first: missing {source}')
    frame = pd.read_csv(source)
    frame = frame[frame.db == DATASETS[args.dataset]].reset_index(drop=True)
    tb = frame.disease_status.to_numpy() == 1
    keep = frame.keep_proposed.astype(bool).to_numpy()
    after = np.where(~tb, 'Non-TB', np.where(keep, 'Consistent TB', 'Ambiguous TB'))
    before = np.where(tb, 'TB', 'Non-TB')
    print(pd.Series(after).value_counts().to_string(), flush=True)

    cache = ROOT / f'data/acoustic/features_{args.dataset}.npz'
    features = load_features(frame, cache, args.workers)
    names = feature_names()
    ranking = rank_features(features, after, names)
    selected = ranking.feature.head(args.top).tolist() if args.top else args.features
    unknown = set(selected) - set(names)
    if unknown:
        raise SystemExit(f'Unknown feature(s): {sorted(unknown)}')

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PAPER_STYLE)
    summary = plot(features, before, names, selected, BEFORE, 'before',
                   out / f'fig1_before_{args.dataset}_{args.encoder}')
    summary += plot(features, after, names, selected, AFTER, 'after',
                    out / f'fig4_after_{args.dataset}_{args.encoder}')
    tag = f'{args.dataset}_{args.encoder}'
    ranking.to_csv(out / f'feature_ranking_{tag}.csv', index=False)
    pd.DataFrame(summary).to_csv(out / f'distribution_summary_{tag}.csv', index=False)
    top3 = ranking.head(3)
    report = dict(dataset=args.dataset, encoder=args.encoder, clips=len(frame),
                  groups={k: int(v) for k, v in pd.Series(after).value_counts().items()},
                  plotted=selected, top3_ranked=top3.to_dict('records'),
                  top3_equals_paper_features=top3.feature.tolist() == PAPER_FEATURES,
                  note='Descriptive AUROC on the cleaned groups; not held-out validation.')
    (out / f'summary_{tag}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(top3.to_string(index=False))
    print(f'Wrote figures and tables to {out}')


if __name__ == '__main__':
    main()
