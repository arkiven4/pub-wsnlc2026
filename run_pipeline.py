"""Run raw-data preparation, fresh extraction and the paper study end to end."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    """@brief Run the five numbered stages with consistent encoder and output settings.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, help='Local Qwen3-Omni weights')
    parser.add_argument('--workers', type=int, default=8, help='Preparation workers')
    parser.add_argument('--eval-workers', type=int, default=16)
    parser.add_argument('--embeddings-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--include-uk-context', action='store_true', help='Restore original extraction context for exact Qwen/HeAR replay')
    parser.add_argument('--qwen-mode', choices=['paper', 'independent'], default='paper', help='Independent mode removes cross-clip dependence and changes results; writes separate outputs by default')
    parser.add_argument('--encoders', nargs='+', choices=['qwen', 'hear'], default=['qwen', 'hear'])
    parser.add_argument('--cleaning-dir', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.eval_workers < 1:
        parser.error('--workers and --eval-workers must be positive')
    if 'qwen' in args.encoders and args.model_path is None:
        parser.error('--model-path is required when extracting Qwen')
    if args.qwen_mode == 'independent' and args.include_uk_context:
        parser.error('This mode does not load UK audio; omit --include-uk-context.')
    if args.embeddings_dir is None:
        args.embeddings_dir = ROOT / ('data/embeds/independent' if args.qwen_mode == 'independent' else 'data/embeds')
    if args.output_dir is None:
        args.output_dir = ROOT / ('result/independent_table_i' if args.qwen_mode == 'independent' else 'result/table_i')
    env = os.environ.copy()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[name] = '1'
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    def run(script, *options):
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT / script), *map(str, options)],
                       cwd=ROOT, env=env, check=True)

    if args.cleaning_dir is None:
        args.cleaning_dir = ROOT / ('data/cleaning/independent' if args.qwen_mode == 'independent' else 'data/cleaning')
    context_options = ['--include-uk-context'] if args.include_uk_context else []
    run('scripts/01_prepare_data.py', '--workers', args.workers, *context_options)
    for encoder in args.encoders:
        options = ['--encoder', encoder, '--output', args.embeddings_dir.resolve() / f'embeds_{encoder}.npy']
        if encoder == 'qwen':
            options += ['--model-path', args.model_path.resolve(), '--qwen-mode', args.qwen_mode]
        run('scripts/02_extract_embeddings.py', *options, *context_options)
    shared = ['--encoders', *args.encoders, '--embeddings-dir', args.embeddings_dir.resolve(),
              '--cleaning-dir', args.cleaning_dir.resolve()]
    run('scripts/03_clean_labels.py', *shared)
    run('scripts/04_evaluate.py', *shared, '--output-dir', args.output_dir.resolve(),
        '--workers', args.eval_workers)
    run('scripts/05_make_table.py', '--encoders', *args.encoders,
        '--results-dir', args.output_dir.resolve())


if __name__ == '__main__':
    main()
