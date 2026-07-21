"""Atomic, checksum-verified downloads for EyeLine runtime models."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

WEIGHTS_URL = (
    "https://github.com/WangWilly/gaze-correction-cam/releases/download/v0.1.1/weights.zip"
)
WEIGHTS_SHA256 = "07279dd9072f784e32c26e0c40bdf10270f98c6f3e9b3effc3254c4ce05fa76a"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class ChecksumMismatch(RuntimeError):
    """A downloaded artifact did not match its pinned SHA-256."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str,
    destination: str | Path,
    expected_sha256: str,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> Path:
    """Download to a sibling temporary file, verify it, then atomically replace."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_sha256.lower()
    if target.is_file() and sha256_file(target) == expected:
        return target

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, opener(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise ChecksumMismatch(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def extract_weights(archive: str | Path, model_dir: str | Path) -> Path:
    """Safely extract the pinned archive and return its checkpoint root."""

    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"refusing symlink in weights archive: {info.filename}")
            output = (destination / info.filename).resolve()
            if output != destination_root and destination_root not in output.parents:
                raise ValueError(f"unsafe path in weights archive: {info.filename}")
        bundle.extractall(destination)
    checkpoint_root = destination / "weights" / "warping_model" / "flx" / "12"
    for side in ("L", "R"):
        if not (checkpoint_root / side / f"{side}.index").is_file():
            raise ValueError(f"weights archive is missing the {side} checkpoint")
    return checkpoint_root


def download_runtime_models(model_dir: str | Path) -> tuple[Path, Path]:
    """Fetch both pinned artifacts, returning checkpoint root and landmarker path."""

    root = Path(model_dir)
    archive = download_verified(WEIGHTS_URL, root / "downloads" / "weights.zip", WEIGHTS_SHA256)
    checkpoints = extract_weights(archive, root)
    landmarker = download_verified(
        FACE_LANDMARKER_URL,
        root / "face_landmarker.task",
        FACE_LANDMARKER_SHA256,
    )
    return checkpoints, landmarker
