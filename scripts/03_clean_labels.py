"""Step 03: compute cleaning decisions only; classifiers run in step 04."""
import argparse
from pathlib import Path
from _common import ROOT, positive_int


def main():
    """@brief Read CLI paths and compute cleaner decisions for each selected encoder.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', nargs='+', choices=['qwen', 'hear'], default=['qwen', 'hear'])
    parser.add_argument('--embeddings-dir', type=Path, default=ROOT / 'data/embeds')
    parser.add_argument('--cleaning-dir', type=Path, default=ROOT / 'data/cleaning')
    args = parser.parse_args()
    from lib.study_cleaning import compute_cleaning
    for encoder in args.encoders:
        print(f'Cleaning {encoder}', flush=True)
        compute_cleaning(encoder, args.embeddings_dir, args.cleaning_dir)


if __name__ == '__main__':
    main()
