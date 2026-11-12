"""Step 05: format the evaluation CSVs and compare every metric to Table I."""
import argparse
from pathlib import Path
from _common import ROOT


def main():
    """@brief Read evaluated CSVs and format the selected encoders into Table I.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', nargs='+', choices=['qwen', 'hear'], default=['qwen', 'hear'])
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'result/table_i')
    args = parser.parse_args()
    from lib.table_i import write_table
    write_table(args.results_dir, args.encoders)


if __name__ == '__main__':
    main()
