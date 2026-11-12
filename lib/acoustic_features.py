"""Hand-crafted MFCC/spectral statistics used for paper Figs. 1 and 4.

Port of the development helper that produced the published acoustic figures,
reduced to the settings actually used there (no openSMILE, no CMVN, no
band-limit, no RMS normalisation). Each clip becomes 153 numbers:
nine time summaries of 13 MFCC streams plus four spectral streams.
"""
import re
import numpy as np
import soundfile as sf
import librosa
from scipy.stats import skew, kurtosis

SAMPLE_RATE = 16000
N_MFCC = 13
SPEC_STREAMS = ['spec_centroid', 'spec_bandwidth', 'spec_rolloff', 'spec_flatness']
STATS = ['mean', 'std', 'skew', 'kurt', 'p10', 'p25', 'p50', 'p75', 'p90']
# Exact settings of the paper's acoustic analysis notebook.
PARAMS = dict(n_fft=2048, hop_length=64, win_length=512, fmin=50, fmax=5000)
PAPER_FEATURES = ['mfcc_0_p10', 'mfcc_0_std', 'mfcc_0_p25']

_STAT_LABEL = {'mean': 'Mean', 'std': 'Std', 'skew': 'Skew', 'kurt': 'Kurtosis',
               'p10': '10th Pct', 'p25': '25th Pct', 'p50': 'Median',
               'p75': '75th Pct', 'p90': '90th Pct'}
_STREAM_LABEL = {'spec_centroid': 'Spectral Centroid', 'spec_bandwidth': 'Spectral Bandwidth',
                 'spec_rolloff': 'Spectral Rolloff', 'spec_flatness': 'Spectral Flatness'}


def feature_names():
    """@brief Column names in extraction order (stat-major, then stream).
    @return List of 153 names such as ``mfcc_0_p10``.
    """
    streams = [f'mfcc_{i}' for i in range(N_MFCC)] + SPEC_STREAMS
    return [f'{s}_{st}' for st in STATS for s in streams]


def pretty_name(name):
    """@brief Plot label, e.g. ``mfcc_0_p10`` -> ``MFCC-0 [10th Pct]``."""
    stream, stat = re.fullmatch(r'(.+)_(' + '|'.join(STATS) + ')', name).groups()
    match = re.fullmatch(r'mfcc_(\d+)', stream)
    head = f'MFCC-{match.group(1)}' if match else _STREAM_LABEL.get(stream, stream)
    return f'{head} [{_STAT_LABEL[stat]}]'


def load_audio(path):
    """@brief Read WAV, average to mono, resample to 16 kHz, peak-normalise.
    @param path WAV file path.
    @return float32 waveform in [-1, 1].
    """
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


def extract(path):
    """@brief Compute the 153 summary statistics for one clip.
    @param path WAV file path.
    @return float32 vector ordered as feature_names().
    """
    y = load_audio(path)
    sr = SAMPLE_RATE
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, **PARAMS)
    stft = dict(n_fft=PARAMS['n_fft'], win_length=PARAMS['win_length'],
                hop_length=PARAMS['hop_length'])
    S = np.abs(librosa.stft(y, **stft))
    frame = dict(win_length=PARAMS['win_length'], hop_length=PARAMS['hop_length'])
    spectral = np.vstack([
        librosa.feature.spectral_centroid(S=S, sr=sr, **frame)[0],
        librosa.feature.spectral_bandwidth(S=S, sr=sr, **frame)[0],
        librosa.feature.spectral_rolloff(S=S, sr=sr, **frame)[0],
        librosa.feature.spectral_flatness(S=S, **frame)[0]])
    x = np.concatenate([mfcc, spectral], axis=0)
    summaries = [x.mean(1), x.std(1), skew(x, axis=1, bias=False),
                 kurtosis(x, axis=1, fisher=True, bias=False)]
    summaries += [np.percentile(x, q, axis=1) for q in (10, 25, 50, 75, 90)]
    return np.concatenate(summaries).astype(np.float32)
