"""Repo-relative locations shared by the prep scripts.

datasets/raw/{dataset}/      original dataset downloads (git-ignored; see README)
datasets/preprocess/{dataset}/     clips written by the prep scripts (git-ignored)
data/metadata/      per-DB and combined metadata CSVs (git-ignored)
prep/manifest/      paper clip list / CIDRZ offsets / checksums (tracked)
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "datasets" / "raw"
WAVS = REPO / "datasets" / "preprocess"
META = REPO / "data" / "metadata"
MANIFEST = REPO / "prep" / "manifest"
CAD = REPO / "cad"   # cough detector used for the paper (June 2026 model)

SR = 16000


def rel(p):
    """Store paths in metadata relative to the repo root."""
    return str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else str(p)


def absolute(p):
    return str(REPO / p)
