"""Standalone frame-level cough detection / segmentation (CAD).

Depends only on numpy, scipy, librosa, soundfile, onnxruntime (+ the .onnx and
config.json here). Runs the dense model to get a per-frame cough posterior
(~10 ms/frame), smooths it, thresholds, and returns cough events. No sliding
window, no peak hacks. Extracted segment audio is cut from the ORIGINAL
untouched file (native sample rate + channels).

Feature handling follows config.json "input_type":
    "wav"    - model has a learnable frontend (LEAF) baked into the ONNX;
               the raw waveform is fed directly.
    "logmel" - legacy bundles: log-mel is computed here with librosa first.
Old config.json without "input_type" defaults to "logmel".

CLI:
    python infer.py input.wav
    python infer.py input.wav --save_dir out/          # + per-cough wavs + mask.json
    python infer.py input.wav --threshold 0.6
"""
import argparse
import json
import os

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import find_peaks, medfilt

HERE = os.path.dirname(os.path.abspath(__file__))


class CADDetector:
    def __init__(self, model_dir=HERE, threshold=None):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.c = json.load(f)
        self.session = ort.InferenceSession(
            os.path.join(model_dir, self.c["onnx_file"]), providers=["CPUExecutionProvider"])
        self.threshold = self.c["threshold"] if threshold is None else threshold
        self.input_type = self.c.get("input_type", "logmel")   # old bundles: log-mel

    def _norm_wav(self, wav):
        """Waveform normalization steps, mirroring the training dataset."""
        for step in self.c.get("normalization", []):
            if step in ("max_wav_value", "peak_norm"):
                peak = np.max(np.abs(wav))
                if peak > 0:
                    wav = wav / peak
            elif step == "rms_norm":
                wav = wav * (self.c.get("target_rms", 0.1) / (np.sqrt(np.mean(wav ** 2)) + 1e-8))
            elif step == "cmvn_norm":
                wav = (wav - wav.mean()) / (wav.std() + 1e-16)
        return wav.astype(np.float32)

    def _logmel(self, wav):
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.c["sample_rate"], n_mels=self.c["n_mels"], n_fft=self.c["n_fft"],
            hop_length=self.c["hop_length"], win_length=self.c["win_length"],
            fmin=self.c["fmin"], fmax=self.c["fmax"])
        return np.log(mel + self.c["log_offset"]).astype(np.float32)

    def posterior_from_wav(self, wav, chunk_sec=0.0):
        """Per-frame cough probability for an in-memory mono waveform at the model's
        sample rate. chunk_sec>0 splits the time axis (Pi / low-RAM); 0 = single pass."""
        wav = self._norm_wav(np.asarray(wav, dtype=np.float32))

        if self.input_type == "wav":                            # LEAF frontend inside ONNX
            if chunk_sec <= 0:
                return self._run(wav[None])[0]
            spc = max(1, int(round(chunk_sec * self.c["sample_rate"])))
            # chunk boundaries reset the frontend's conv/PCEN context (~1 frame edge blur)
            return np.concatenate(
                [self._run(wav[None, s:s + spc])[0] for s in range(0, len(wav), spc)])

        feats = self._logmel(wav)                               # (n_mels, T)
        T = feats.shape[1]
        if chunk_sec <= 0:
            return self._run(feats[None])[0]
        fpc = max(1, int(round(chunk_sec / self.c["frame_hop_seconds"])))
        probs = np.zeros(T, dtype=np.float32)
        for s in range(0, T, fpc):
            e = min(s + fpc, T)
            probs[s:e] = self._run(feats[None, :, s:e])[0]
        return probs

    def posterior(self, path, chunk_sec=0.0):
        """Per-frame cough probability over a whole file."""
        wav, _ = librosa.load(path, sr=self.c["sample_rate"], mono=True)
        return self.posterior_from_wav(wav, chunk_sec=chunk_sec)

    def _run(self, feats):
        return self.session.run([self.c["output_name"]], {self.c["input_name"]: feats})[0]

    def _split_bout(self, sm_run, min_gap_frames):
        """Split a long active run into sub-coughs at posterior valleys between peaks.

        sm_run: smoothed posterior over the run. Returns list of (a, b) frame
        offsets relative to the run. One peak (or none) -> no split.
        """
        peaks, _ = find_peaks(sm_run, height=self.threshold, distance=max(1, min_gap_frames))
        if len(peaks) <= 1:
            return [(0, len(sm_run))]
        bounds = [0]
        for a, b in zip(peaks[:-1], peaks[1:]):
            bounds.append(a + int(np.argmin(sm_run[a:b + 1])))  # valley between peaks
        bounds.append(len(sm_run))
        return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]

    def events_from_wav(self, wav, chunk_sec=0.0, split_bouts=None, max_cough_dur=None,
                        min_cough_gap=None, energy_gate=None, energy_gate_db=None,
                        trim_silence=None, trim_top_db=None):
        """Cough events for an in-memory mono waveform: list of {start, end, score} sec.

        split_bouts:  split runs longer than max_cough_dur into individual coughs.
        trim_silence: tighten each event's boundaries to the energetic part
                      (removes leading/trailing silence; top_db = trim_top_db).
        energy_gate:  drop events whose RMS is more than energy_gate_db below the
                      loudest event — kills silence false positives.
        """
        if split_bouts is None:
            split_bouts = bool(self.c.get("split_bouts", False))
        max_cough_dur = (self.c.get("max_cough_dur_ms", 700) / 1000.0) if max_cough_dur is None else max_cough_dur
        min_cough_gap = (self.c.get("min_cough_gap_ms", 200) / 1000.0) if min_cough_gap is None else min_cough_gap
        if energy_gate is None:
            energy_gate = bool(self.c.get("energy_gate", False))
        energy_gate_db = self.c.get("energy_gate_db", 30.0) if energy_gate_db is None else energy_gate_db
        if trim_silence is None:
            trim_silence = bool(self.c.get("trim_silence", False))
        trim_top_db = self.c.get("trim_top_db", 25.0) if trim_top_db is None else trim_top_db

        wav = np.asarray(wav, dtype=np.float32)
        probs = self.posterior_from_wav(wav, chunk_sec=chunk_sec)
        hop = self.c["frame_hop_seconds"]

        k = max(1, int(round(self.c["median_filter_ms"] / 1000.0 / hop)))
        if k % 2 == 0:
            k += 1                                             # medfilt needs odd kernel
        sm = medfilt(probs, k) if k > 1 else probs

        hits = sm >= self.threshold
        merge_gap = self.c["merge_gap_ms"] / 1000.0
        min_dur = self.c["min_dur_ms"] / 1000.0

        # 1) contiguous active runs, merged across small gaps -> (i, j) frame spans.
        runs = []
        i, n = 0, len(hits)
        while i < n:
            if not hits[i]:
                i += 1
                continue
            j = i
            while j < n and hits[j]:
                j += 1
            if runs and (i - runs[-1][1]) * hop <= merge_gap:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j

        # 2) optionally split long runs (bouts) into individual coughs.
        out = []
        for i, j in runs:
            if split_bouts and (j - i) * hop > max_cough_dur:
                for a, b in self._split_bout(sm[i:j], int(round(min_cough_gap / hop))):
                    out.append((i + a, i + b))
            else:
                out.append((i, j))

        evs = [{"start": float(a * hop), "end": float(b * hop), "score": float(probs[a:b].max())}
               for a, b in out]
        evs = [ev for ev in evs if ev["end"] - ev["start"] >= min_dur]
        sr = self.c["sample_rate"]

        # Tighten boundaries: strip leading/trailing silence inside each event.
        if trim_silence and evs:
            tightened = []
            for ev in evs:
                a, b = int(ev["start"] * sr), int(ev["end"] * sr)
                seg = wav[a:b]
                if seg.size:
                    _, idx = librosa.effects.trim(seg, top_db=trim_top_db)
                    if idx[1] > idx[0]:
                        ev = {**ev, "start": (a + idx[0]) / sr, "end": (a + idx[1]) / sr}
                tightened.append(ev)
            evs = [ev for ev in tightened if ev["end"] - ev["start"] >= min_dur]

        # Energy gate: a real cough is loud; silence false positives are not.
        if energy_gate and evs:
            def _rms(ev):
                s = wav[int(ev["start"] * sr):int(ev["end"] * sr)]
                return float(np.sqrt(np.mean(s * s) + 1e-12))
            rms = [_rms(ev) for ev in evs]
            ref = max(rms)                                      # loudest event = reference
            evs = [ev for ev, r in zip(evs, rms)
                   if 20.0 * np.log10(r / (ref + 1e-12)) >= -energy_gate_db]
        return evs

    def events(self, path, chunk_sec=0.0, split_bouts=None, max_cough_dur=None, min_cough_gap=None,
               energy_gate=None, energy_gate_db=None, trim_silence=None, trim_top_db=None):
        """Cough events over a whole file (loads, then delegates to events_from_wav)."""
        wav, _ = librosa.load(path, sr=self.c["sample_rate"], mono=True)
        return self.events_from_wav(wav, chunk_sec=chunk_sec, split_bouts=split_bouts,
                                    max_cough_dur=max_cough_dur, min_cough_gap=min_cough_gap,
                                    energy_gate=energy_gate, energy_gate_db=energy_gate_db,
                                    trim_silence=trim_silence, trim_top_db=trim_top_db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model_dir", default=HERE)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--chunk_sec", type=float, default=0.0, help=">0 splits time axis for low-RAM (Pi)")
    ap.add_argument("--split_bouts", dest="split_bouts", action="store_true", default=None,
                    help="split long runs into individual coughs")
    ap.add_argument("--no_split_bouts", dest="split_bouts", action="store_false",
                    help="keep bouts as single events")
    ap.add_argument("--max_cough_dur", type=float, default=None, help="[split] run longer than this (s) is split")
    ap.add_argument("--min_cough_gap", type=float, default=None, help="[split] min spacing between sub-coughs (s)")
    ap.add_argument("--energy_gate", dest="energy_gate", action="store_true", default=None,
                    help="drop silence false positives by loudness")
    ap.add_argument("--no_energy_gate", dest="energy_gate", action="store_false")
    ap.add_argument("--energy_gate_db", type=float, default=None, help="dB below loudest event to drop (default 30)")
    ap.add_argument("--trim_silence", dest="trim_silence", action="store_true", default=None,
                    help="tighten event boundaries (strip leading/trailing silence)")
    ap.add_argument("--no_trim_silence", dest="trim_silence", action="store_false")
    ap.add_argument("--trim_top_db", type=float, default=None, help="trim threshold dB below event peak (default 25)")
    ap.add_argument("--save_dir", default=None, help="write per-cough wavs (untouched original) + mask.json")
    args = ap.parse_args()

    det = CADDetector(args.model_dir, threshold=args.threshold)
    segs = det.events(args.audio, chunk_sec=args.chunk_sec, split_bouts=args.split_bouts,
                      max_cough_dur=args.max_cough_dur, min_cough_gap=args.min_cough_gap,
                      energy_gate=args.energy_gate, energy_gate_db=args.energy_gate_db,
                      trim_silence=args.trim_silence, trim_top_db=args.trim_top_db)

    print(f"threshold={det.threshold:.3f}  frame={1000 * det.c['frame_hop_seconds']:.0f}ms  coughs={len(segs)}")
    for i, s in enumerate(segs):
        print(f"  [{i:03d}] {s['start']:7.2f} - {s['end']:7.2f} s  ({s['end']-s['start']:.2f}s, score {s['score']:.3f})")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        orig, osr = sf.read(args.audio, always_2d=True)        # native SR + channels, untouched
        for i, s in enumerate(segs):
            a, b = int(s["start"] * osr), int(s["end"] * osr)
            sf.write(os.path.join(args.save_dir, f"cough_{i:03d}.wav"), orig[a:b], osr)
        with open(os.path.join(args.save_dir, "mask.json"), "w") as f:
            json.dump({"threshold": det.threshold, "events": segs}, f, indent=2)
        print(f"saved {len(segs)} coughs + mask.json -> {args.save_dir}")


if __name__ == "__main__":
    main()
