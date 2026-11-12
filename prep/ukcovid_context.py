"""Restore the 42 UK-COVID clips needed only as Qwen extraction context.

Taken from the original UK-COVID preprocessing notebook. Detection uses RMS=0.1;
returned slices retain the original demeaned amplitude. These clips do not enter
combine.csv, cleaning or evaluation. Only the manifest-selected context is saved.

Input: datasets/raw/ukcovid_context/{raw_filename}.wav
Output: datasets/preprocess/ukcovid_context/{segment_filename}.wav
Native-rate downloads are converted to the expected 16 kHz PCM16 representation
before segmentation; a manifest hash check rejects non-matching source audio.
"""
import hashlib
import io
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from paths import RAW, WAVS, MANIFEST, SR


def pcm16(wave):
    buffer = io.BytesIO()
    sf.write(buffer, wave, SR, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return sf.read(buffer, dtype="float32")[0]


def digest(wave):
    return hashlib.sha256(np.asarray(wave, dtype="<i2").tobytes()).hexdigest()


def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
    x_original = x.copy()

    target_rms = 0.1
    rms_norm = np.sqrt(np.mean(x**2))
    x = x * (target_rms / (rms_norm + 1e-8))

    cough_mask = np.array([False]*len(x))

    #Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h =  th_h_multiplier*rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0

    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x_original[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x_original[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True

    return coughSegments, cough_mask


def main():
    manifest = pd.read_csv(MANIFEST / "qwen_context.csv")
    source = RAW / "ukcovid_context"
    output = WAVS / "ukcovid_context"
    missing = [name for name in manifest.raw.unique() if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} UK context recordings under {source}; first: {missing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    saved = 0
    for raw, group in manifest.groupby("raw", sort=False):
        wave, _ = librosa.load(source / raw, sr=SR, mono=True)
        wave = pcm16(wave)
        integers = np.round(wave * 32768).astype(np.int16)
        if digest(integers) != group.raw_pcm16_sha256.iloc[0]:
            raise ValueError(f"Raw UK context audio differs from the paper source: {raw}")
        wave = wave - np.mean(wave)
        segments, _ = segment_cough(wave, SR, cough_padding=0.1,
                                    min_cough_len=0.2, th_l_multiplier=0.10,
                                    th_h_multiplier=3)
        for row in group.itertuples(index=False):
            index = int(row.file.rsplit("_", 1)[1].split(".")[0])
            if index >= len(segments):
                raise ValueError(f"Missing recovered UK segment: {row.file}")
            target = output / row.file
            sf.write(target, segments[index], SR, subtype="PCM_16")
            integers, _ = sf.read(target, dtype="int16")
            if digest(integers) != row.clip_pcm16_sha256:
                raise ValueError(f"Recovered UK context samples differ: {row.file}")
            saved += 1
    print(f"Restored {saved} UK-COVID context clips from {manifest.raw.nunique()} raw recordings; all sample hashes match.")
    print("These are extraction context only, excluded from combine.csv and study evaluation.")


if __name__ == "__main__":
    main()
