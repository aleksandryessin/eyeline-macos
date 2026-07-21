from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from eyeline.models.downloads import (
    ChecksumMismatch,
    download_verified,
    extract_weights,
    sha256_file,
)


def test_verified_download_is_atomic_and_reuses_valid_file(tmp_path) -> None:
    payload = b"pinned model bytes"
    expected = hashlib.sha256(payload).hexdigest()
    calls = 0

    def opener(url, timeout):
        nonlocal calls
        calls += 1
        return io.BytesIO(payload)

    destination = tmp_path / "model.task"
    downloaded = download_verified(
        "https://example.invalid/model", destination, expected, opener=opener
    )
    assert downloaded == destination
    assert destination.read_bytes() == payload
    assert sha256_file(destination) == expected
    download_verified("https://example.invalid/model", destination, expected, opener=opener)
    assert calls == 1


def test_checksum_mismatch_does_not_publish_partial_file(tmp_path) -> None:
    destination = tmp_path / "model.task"
    with pytest.raises(ChecksumMismatch):
        download_verified(
            "https://example.invalid/model",
            destination,
            "0" * 64,
            opener=lambda url, timeout: io.BytesIO(b"wrong"),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".model.task.*"))


def test_weight_extraction_validates_layout(tmp_path) -> None:
    archive = tmp_path / "weights.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for side in ("L", "R"):
            bundle.writestr(f"weights/warping_model/flx/12/{side}/{side}.index", b"index")
    root = extract_weights(archive, tmp_path / "models")
    assert root.name == "12"
    assert (root / "L" / "L.index").is_file()


def test_weight_extraction_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../../outside", b"bad")
    with pytest.raises(ValueError, match="unsafe path"):
        extract_weights(archive, tmp_path / "models")
    assert not (tmp_path / "outside").exists()
