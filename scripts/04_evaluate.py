"""Step 04: evaluate saved cleaning decisions; no cleaning models are refitted."""
import argparse
from pathlib import Path
from _common import ROOT, positive_int


def main():
    """@brief Read CLI paths and evaluate matching decisions with bounded worker threads.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', nargs='+', choices=['qwen', 'hear'], default=['qwen', 'hear'])
    parser.add_argument('--embeddings-dir', type=Path, default=ROOT / 'data/embeds')
    parser.add_argument('--cleaning-dir', type=Path, default=ROOT / 'data/cleaning')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/table_i')
    parser.add_argument('--workers', type=positive_int, default=16)
    args = parser.parse_args()
    from joblib import parallel_config
    from lib.study_evaluation import evaluate_study
    for encoder in args.encoders:
        print(f'Evaluating {encoder}', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1, verbose=10):
            evaluate_study(encoder, args.embeddings_dir, args.cleaning_dir, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
