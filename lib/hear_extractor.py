"""HeAR waveform preparation, frozen inference and retained legacy options.

The paper profile is HearPaperExtractor: soundfile loading, mono16kHz, peak
normalization without DC removal, centred silence padding and batch 64 inference.
Missing row slots use synthetic zeros; no UK audio is required. The CLI loads
matching XLA kernel choices before importing this module. Other BaseExtractor
framing options are kept for comparison with other HeAR recipes.
"""
import gc
from concurrent.futures import ThreadPoolExecutor
import librosa
import numpy as np
import soundfile as sf
import torch
import tensorflow as tf
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
SR, WINDOW_SEC = 16000, 2.0
N_SAMPLES = int(SR * WINDOW_SEC)
PADS = ("zero", "mirror", "repeat", "noise")
PLACES = ("center", "right")
NORMS = ("peak", "rms", "zscore", "none")

class BaseExtractor:
    """@brief Shared waveform loading, framing and ordered extraction driver.
    Subclasses supply inference and can override the paper audio recipe.
    """
    name = "base"
    dim = None
    def __init__(self, pad="zero", place="center", norm="peak", target_rms=0.05,
                 sr=SR, n_samples=N_SAMPLES, seed=0):
        if pad not in PADS:
            raise ValueError(f"pad must be one of {PADS}, got {pad!r}")
        if place not in PLACES:
            raise ValueError(f"place must be one of {PLACES}, got {place!r}")
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.pad, self.place, self.norm = pad, place, norm
        self.target_rms, self.sr, self.n_samples, self.seed = target_rms, sr, n_samples, seed

    def normalize(self, wav):
        """@brief Apply the configured gain scaling to one waveform.
        @param wav Waveform array, converted to float32.
        @return Scaled waveform; silence is left unchanged.
        @details peak divides by maximum absolute amplitude; rms targets target_rms;
                 none preserves gain. The zscore branch uses RMS scaling.
        @note This method does not subtract a mean. Centering, where applicable, is
              performed by the loader. The paper HeAR profile uses peak scaling.
        """
        wav = np.asarray(wav, dtype=np.float32)
        if self.norm == "none":
            return wav
        if self.norm == "peak":
            scale = float(np.abs(wav).max())
        else:
            scale = float(np.sqrt((wav ** 2).mean()))
            if self.norm == "rms":
                scale /= self.target_rms
        return (wav / scale if scale > 0 else wav).astype(np.float32)

    def load_clip(self, path):
        """16 kHz mono, DC-removed, gain-normalized. No windowing - that is
        `frame_clip`."""
        wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if file_sr != self.sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=self.sr)
        return self.normalize(wav - wav.mean())

    def frame_clip(self, wav, rng=None):
        """Fit `wav` into an `n_samples` window.

        pad     what surrounds the clip
                "zero"    digital silence
                "mirror"  the clip reflected back and forth
                "repeat"  the clip tiled end to end
                "noise"   gaussian at the clip's own noise floor (room tone)
        place   where the original clip sits
                "center"  filler split evenly on both sides (default)
                "right"   filler all on the right, clip at the start
                Audible only for zero/noise; mirror and repeat fill the window
                either way, so there `place` only shifts the phase.
        """
        n = self.n_samples
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) >= n:
            return wav[:n]

        gap = n - len(wav)
        left = gap // 2 if self.place == "center" else 0
        right = gap - left
        if self.pad == "mirror":
            return np.pad(wav, (left, right), mode="reflect").astype(np.float32)
        if self.pad == "repeat":
            return np.pad(wav, (left, right), mode="wrap").astype(np.float32)
        out = np.pad(wav, (left, right))
        if self.pad == "noise":
            # Noise floor = RMS of the quietest 10% of 10 ms frames, so the
            # filler sits at the clip's own room tone, not an arbitrary level.
            k = len(wav) // 160 * 160
            sd = 1e-4
            if k >= 160:
                r = np.sqrt((wav[:k].reshape(-1, 160) ** 2).mean(1))
                sd = max(float(np.percentile(r, 10)), 1e-5)
            rng = rng or np.random.default_rng(self.seed)
            out[:left] = rng.normal(0, sd, left)
            if right:
                out[left + len(wav):] = rng.normal(0, sd, right)
        return out.astype(np.float32)

    def prepare(self, path, index=0):
        loaded_wav = self.load_clip(path)
        preprocess_wav = self.frame_clip(loaded_wav, rng=np.random.default_rng([self.seed, index]))
        return preprocess_wav

    def _embed(self, batch):
        raise NotImplementedError

    def close(self):
        """Hand cached GPU blocks back; subclasses drop their model first."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free, total = torch.cuda.mem_get_info()
            print(f"[{self.name}] cuda free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- driver ----------------------------------------------------------
    def extract(self, paths, batch_size=32, num_workers=8, progress=True):
        """@brief Prepare and embed paths in batches without reordering their outputs.
        @param paths Audio paths; the paper subclass also accepts None for zero filler.
        @param batch_size Number of model inputs per inference call.
        @param num_workers Threads used to load/prepare audio; zero runs serially.
        @param progress Show a progress bar when true.
        @return Float32 matrix (number of paths, encoder dimension).
        @note Loading can run in parallel; model inference follows input batch order.
        """
        paths = list(paths)
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        desc = f"{self.name}/{self.pad}/{self.place}/{self.norm}"
        pool = ThreadPoolExecutor(num_workers) if num_workers else None
        try:
            for start in tqdm(range(0, len(paths), batch_size), disable=not progress, desc=desc):
                chunk = list(enumerate(paths[start:start + batch_size], start))
                work = lambda item: self.prepare(item[1], item[0])
                batch = list(pool.map(work, chunk)) if pool else [work(i) for i in chunk]
                out[start:start + len(chunk)] = self._embed(batch)
        finally:
            if pool is not None:
                pool.shutdown()
        return out
    
class HearExtractor(BaseExtractor):
    """@brief Frozen TensorFlow HeAR model with configurable legacy framing.
    The paper workflow uses the more specific HearPaperExtractor subclass.
    """
    name = "hear"
    dim = 512
    def __init__(self, repo="google/hear", use_gpu=False, **framing):
        super().__init__(**framing)
        if use_gpu:
            # TF claims the whole card up front, which starves the torch model later.
            for gpu in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as err:
                    print(f"memory growth already fixed: {err}")
        else:
            print("HEAR on CPU")
            tf.config.set_visible_devices([], "GPU")
        self._model = tf.saved_model.load(snapshot_download(repo, revision='9b2eb2853c426676255cc6ac5804b7f1fe8e563f' if repo == 'google/hear' else None))
        self._infer = self._model.signatures["serving_default"]

    def _embed(self, batch):
        out = self._infer(x=tf.constant(np.stack(batch)))["output_0"]
        return np.array(out, dtype=np.float32)

    def close(self):
        self._infer = None
        self._model = None

        tf.keras.backend.clear_session()
        super().close()



class HearPaperExtractor(HearExtractor):
    """June 24/25 direct loader: peak gain, centered silence, batch size 64.

    No DC removal. The original notebook iterates in Qwen output order.
    The CLI loads pinned XLA kernel choices before TensorFlow is imported.
    """
    def __init__(self, **kwargs):
        super().__init__(pad="zero", place="center", norm="peak", **kwargs)

    def load_clip(self, path):
        """@brief Load the paper HeAR input without removing its DC component.
        @param path Input WAV path.
        @return Mono, 16 kHz, finite, peak-normalized float32 waveform.
        @details Read with soundfile, average stereo channels, resample only when
                 needed, replace non-finite samples, then normalize gain.
        """
        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim == 2:
            wave = np.mean(wave, axis=1)
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
        wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
        return self.normalize(wave)

    def frame_clip(self, wav, rng=None):
        """@brief Place one cough in the centre of the two-second HeAR window.
        @param wav Loaded waveform.
        @param rng Accepted for the shared interface; zero padding is deterministic.
        @return Float32 waveform containing exactly 32,000 samples at the paper rate.
        @note Short clips receive symmetric zero padding; longer clips are centre-cropped.
        """
        if len(wav) >= self.n_samples:
            offset = (len(wav) - self.n_samples) // 2
            return np.asarray(wav[offset:offset + self.n_samples], dtype=np.float32)
        return super().frame_clip(wav, rng=rng)

    def prepare(self, path, index=0):
        # Synthetic filler preserves the fixed numerical batch shape. HeAR's
        # examples are independent, so filler content cannot affect study clips.
        """@brief Prepare a real clip or a synthetic empty slot for paper-shaped batches.
        @param path WAV path, or None for a missing row slot.
        @param index Input position passed through to the shared preparation interface.
        @return Fixed-length model waveform; None produces a fresh all-zero array.
        @note HeAR is independent per example. Zero slots preserve batch 64 numerical
              shape without reading UK audio or supplying any reference embeddings.
        """
        if path is None:
            return np.zeros(self.n_samples, dtype=np.float32)
        return super().prepare(path, index=index)

    def extract(self, paths, batch_size=64, **kwargs):
        """@brief Run the paper HeAR framing with the required inference batch size.
        @param paths Ordered clip paths or synthetic None slots.
        @param batch_size Must be 64 for this profile.
        @param kwargs Loading/progress arguments forwarded to the shared driver.
        @return Float32 matrix with one 512-dimensional vector per input slot.
        @exception ValueError A different batch size was requested.
        @note The CLI configures matching XLA kernels before importing TensorFlow.
        """
        if batch_size != 64:
            raise ValueError('The paper HeAR extraction requires inference batch_size=64.')
        return super().extract(paths, batch_size=64, **kwargs)
