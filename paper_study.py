"""Compatibility entry point: cleaning -> evaluation -> Table I.

For separate, inspectable steps use scripts/03_clean_labels.py,
scripts/04_evaluate.py and scripts/05_make_table.py.
"""
import argparse
from pathlib import Path
from joblib import parallel_config
from lib.study_cleaning import compute_cleaning
from lib.study_evaluation import evaluate_study
from lib.table_i import write_table

ROOT = Path(__file__).resolve().parent


def run_study(encoder, embeddings_dir, output_dir, workers=16, cleaning_dir=None):
    """@brief Compatibility shortcut for cleaning followed by classifier evaluation.
    @param encoder Encoder name, qwen or hear.
    @param embeddings_dir Extracted matrix and aligned metadata directory.
    @param output_dir Destination for aggregate evaluation CSVs.
    @param workers Number of classifier workers.
    @param cleaning_dir Optional generated-decision directory; defaults to data/cleaning.
    @return Experiment filename suffix returned by evaluate_study.
    @note You can run scripts 03 and 04 separately to inspect decisions first.
    """
    cleaning_dir = Path(cleaning_dir) if cleaning_dir is not None else ROOT / 'data/cleaning'
    compute_cleaning(encoder, embeddings_dir, cleaning_dir)
    return evaluate_study(encoder, embeddings_dir, cleaning_dir, output_dir, workers)


def main():
    """@brief Run cleaning, evaluation and table formatting through the compatibility CLI.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', nargs='+', choices=['qwen', 'hear'], default=['qwen', 'hear'])
    parser.add_argument('--embeddings-dir', type=Path, default=ROOT / 'data/embeds')
    parser.add_argument('--cleaning-dir', type=Path, default=ROOT / 'data/cleaning')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/table_i')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for encoder in args.encoders:
        with parallel_config(backend='loky', inner_max_num_threads=1, verbose=10):
            run_study(encoder, args.embeddings_dir, args.output_dir, args.workers, args.cleaning_dir)
    write_table(args.output_dir, args.encoders)


if __name__ == '__main__':
    main()
