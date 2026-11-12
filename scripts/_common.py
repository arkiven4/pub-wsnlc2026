"""Repository paths and process helpers shared by the numbered entry points."""
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')


def run(script, *options):
    subprocess.run([sys.executable, '-u', str(ROOT / script), *map(str, options)],
                   cwd=ROOT, check=True)


def positive_int(value):
    import argparse
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number
