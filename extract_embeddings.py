"""Extract paper-aligned embeddings; run --help for usage.

Qwen deliberately replays packed SDPA batches: changing batch composition changes
features. --include-uk-context restores the original boundary batch without
including UK-COVID in the study. HeAR needs only zero filler, not UK audio.
Independent Qwen processes one study clip per call and changes paper features.
"""
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent


def repo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


from lib.qwen_extractor import QwenReplay


def load_wave(path):
    """@brief Read a prepared clip for Qwen inference.
    @param path Absolute path or repository-relative audio path.
    @return Mono, 16 kHz, peak-normalized float32 waveform.
    @exception ValueError The waveform is empty, silent or non-finite.
    """
    wave, _ = librosa.load(repo_path(path), sr=16000, mono=True)
    peak = np.abs(wave).max() if len(wave) else 0
    if peak == 0 or not np.isfinite(wave).all():
        raise ValueError(f'Empty, silent or nonfinite audio: {path}')
    return (wave / peak).astype(np.float32)


def read_metadata(path):
    """@brief Validate study metadata before loading any encoder weights.
    @param path Combined metadata CSV, absolute or relative to the repository.
    @return DataFrame retaining the input row order.
    @exception ValueError Database IDs or paper row/batch mapping are invalid.
    @exception FileNotFoundError A prepared clip is missing.
    @note Only database IDs 0, 1 and 2 are study rows.
    """
    df = pd.read_csv(repo_path(path))
    required = {'path_file', 'paper_row', 'qwen_batch', 'db'}
    if not required.issubset(df.columns):
        raise ValueError(f'Metadata must contain {sorted(required)}; run prep/build_db.py.')
    if df.empty or not df.db.isin([0, 1, 2]).all():
        raise ValueError('Expected nonempty metadata for CODA, CIDRZ and TBScreen only.')
    if df.paper_row.duplicated().any() or not (df.qwen_batch == df.paper_row // 128).all():
        raise ValueError('Invalid paper row / batch mapping.')
    missing = [p for p in df.path_file if not repo_path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing clips; first: {missing[0]}')
    return df


def read_context(df):
    """@brief Load the optional UK-COVID batch companions from the context manifest.
    @param df Validated three-dataset study metadata.
    @return Context rows with resolved audio paths and original paper row positions.
    @exception ValueError Context overlaps study rows or leaves a partial batch.
    @exception FileNotFoundError A required context WAV is absent.
    @note Used only for the exact paper replay (--include-uk-context). Context rows are not
          written into study output metadata or used as classifier rows.
    """
    context = pd.read_csv(ROOT / 'prep/manifest/qwen_context.csv')
    if context.paper_row.duplicated().any() or context.paper_row.isin(df.paper_row).any():
        raise ValueError('Duplicate context/study paper rows.')
    if not (context.qwen_batch == context.paper_row // 128).all():
        raise ValueError('Invalid context batch mapping.')
    context['path_file'] = context.file.map(lambda name: str(ROOT / 'datasets/preprocess/ukcovid_context' / name))
    missing = [p for p in context.path_file if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing UK extraction context; run prep/ukcovid_context.py. First: {missing[0]}')
    for batch in context.qwen_batch.unique():
        if (df.qwen_batch == batch).sum() + (context.qwen_batch == batch).sum() != 128:
            raise ValueError(f'Incomplete restored context batch {batch}')
    return context


def main():
    """@brief Extract selected study features from prepared WAVs and save aligned row metadata.
    @return None; parse command-line arguments and write the documented stage outputs.
    @see docs/workflow.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['qwen', 'hear'], default='qwen')
    parser.add_argument('--qwen-mode', choices=['paper', 'independent'], default='paper', help='paper: original packed replay; independent: one clip per forward pass, no external context (different features/results)')
    parser.add_argument('--hear-profile', choices=['paper', 'legacy'], default='paper', help='paper: original batch-64 centered zero padding; legacy: approximate repeated audio')
    parser.add_argument('--hear-autotune-cache', default='lib/hear_autotune_rtx4090_tf219.textproto', help='Pinned XLA GPU kernels for paper replay (RTX 4090 / TF 2.19); use none to allow autotuning')
    parser.add_argument('--model-path', help='Local Qwen3-Omni-30B-A3B-Thinking weights')
    parser.add_argument('--metadata', default='data/metadata/combine.csv')
    parser.add_argument('--output', help='Default: data/embeds/embeds_{encoder}.npy')
    parser.add_argument('--check-only', action='store_true', help='Validate metadata and audio paths without loading models')
    parser.add_argument('--include-uk-context', action='store_true', help='Restore the 42 original extraction companions for Qwen/HeAR; exclude them from study output')
    args = parser.parse_args()
    if args.encoder == 'qwen' and args.qwen_mode == 'independent' and args.include_uk_context:
        parser.error('This Qwen mode does not read UK audio; omit --include-uk-context.')
    df = read_metadata(args.metadata)
    context = read_context(df) if args.include_uk_context else None
    if context is not None:
        print(f'{len(context)} UK extraction companions; excluded from study output')
    print(f'{len(df)} clips; {df.qwen_batch.nunique()} paper batches')
    if args.check_only:
        return
    if args.encoder == 'qwen':
        if not args.model_path:
            parser.error('--model-path is required for Qwen')
        extractor = QwenReplay(args.model_path)
        embeds = np.empty((len(df), 2048), dtype=np.float32)
        groups = (((index, df.loc[[index]]) for index in df.index) if args.qwen_mode == 'independent'
                  else df.groupby('qwen_batch', sort=True))
        for batch, group in tqdm(groups, total=len(df) if args.qwen_mode == 'independent' else df.qwen_batch.nunique(), desc=f'Qwen {args.qwen_mode}'):
            # Merge original companions before stable length sorting in embed().
            records = [(int(row.paper_row), row.path_file, index)
                       for index, row in group.iterrows()]
            if context is not None:
                records.extend((int(row.paper_row), row.path_file, None)
                               for row in context[context.qwen_batch == batch].itertuples())
            records.sort(key=lambda record: record[0])
            if args.qwen_mode == 'paper' and len(records) != 128:
                print(f'Batch {batch}: {len(records)} clips (UK context not restored).')
            features = extractor.embed([load_wave(record[1]) for record in records])
            for record, feature in zip(records, features):
                if record[2] is not None:
                    embeds[record[2]] = feature
    else:
        if args.hear_profile == 'paper' and args.hear_autotune_cache != 'none':
            cache = repo_path(args.hear_autotune_cache).resolve()
            if not cache.is_file():
                raise FileNotFoundError(cache)
            flags = os.environ.get('XLA_FLAGS', '')
            if '--xla_gpu_load_autotune_results_from' in flags:
                raise ValueError('Set the HeAR kernel cache via --hear-autotune-cache, not a second XLA_FLAGS entry.')
            os.environ['XLA_FLAGS'] = flags + f' --xla_gpu_load_autotune_results_from={cache}'
            print(f'Pinned HeAR XLA kernels: {cache}')
        from lib.hear_extractor import HearExtractor, HearPaperExtractor
        if args.hear_profile == 'paper':
            records = [(int(row.paper_row), row.path_file, index) for index, row in df.iterrows()]
            if context is not None:
                records.extend((int(row.paper_row), row.path_file, None) for row in context.itertuples())
            records.sort(key=lambda record: record[0])
            # The paper's row slots preserve the GPU arithmetic. Missing slots
            # need no dataset: zero filler is pointwise-independent in HeAR.
            by_row = {record[0]: record for record in records}
            size = ((max(by_row) + 1 + 63) // 64) * 64
            records = [by_row.get(i, (i, None, None)) for i in range(size)]
            filler_count = sum(record[1] is None for record in records)
            print(f'HeAR: {filler_count} synthetic zero slots; no UK audio needed.')
            with HearPaperExtractor(use_gpu=True) as extractor:
                features = extractor.extract([str(repo_path(record[1])) if record[1] is not None else None
                                              for record in records], batch_size=64, num_workers=24)
            embeds = np.empty((len(df), 512), dtype=np.float32)
            for record, feature in zip(records, features):
                if record[2] is not None:
                    embeds[record[2]] = feature
        else:
            with HearExtractor(pad='repeat', place='right', norm='peak', use_gpu=True) as extractor:
                embeds = extractor.extract([str(repo_path(p)) for p in df.path_file], batch_size=32, num_workers=24)
    if not np.isfinite(embeds).all():
        raise RuntimeError('Nonfinite embeddings; output was not saved.')
    default_output = (f'data/embeds/independent/embeds_qwen.npy'
                      if args.encoder == 'qwen' and args.qwen_mode == 'independent' else
                      f'data/embeds/embeds_{args.encoder}.npy')
    output = repo_path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeds)
    # Preserve row provenance alongside the matrix.
    df.to_csv(output.with_name(output.stem.replace('embeds_', 'metadata_') + '.csv'), index=False)
    print(f'Saved {embeds.shape} to {output}')


if __name__ == '__main__':
    main()
