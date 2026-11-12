"""Step 01: prepare three datasets and build the paper-ordered metadata."""
import argparse
from _common import run, positive_int


def main():
    """@brief Run dataset preparation in the required CODA, CIDRZ, TBScreen, metadata order.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=positive_int, default=8)
    parser.add_argument('--include-uk-context', action='store_true', help='prepare the UK-COVID clips needed for the exact paper Qwen extraction')
    args = parser.parse_args()
    run('prep/coda.py')
    run('prep/cidrz.py', '--workers', args.workers)
    run('prep/tbscreen.py', '--workers', args.workers)
    run('prep/build_db.py')
    if args.include_uk_context:
        run('prep/ukcovid_context.py')


if __name__ == '__main__':
    main()
