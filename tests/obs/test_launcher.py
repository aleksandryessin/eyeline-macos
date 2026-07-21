from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _prepare_launcher(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    source = Path(__file__).parents[2] / "run-eyeline.command"
    launcher = project / "run-eyeline.command"
    shutil.copy2(source, launcher)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "$UV_LOG"
printf '%s\\n' "$PYTHONPATH" > "${UV_LOG}.pythonpath"
case "$*" in
  *tools/download_models.py*)
    if [ "${UV_FAIL_DOWNLOAD:-0}" = 1 ]; then exit 42; fi
    mkdir -p "$EYELINE_MODEL_DIR/weights/warping_model/flx/12/L"
    mkdir -p "$EYELINE_MODEL_DIR/weights/warping_model/flx/12/R"
    : > "$EYELINE_MODEL_DIR/face_landmarker.task"
    : > "$EYELINE_MODEL_DIR/weights/warping_model/flx/12/L/L.index"
    : > "$EYELINE_MODEL_DIR/weights/warping_model/flx/12/R/R.index"
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    model_dir = tmp_path / "models"
    log = tmp_path / "uv.log"
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("EYELINE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EYELINE_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("UV_LOG", str(log))
    return launcher, model_dir, log


def _create_models(model_dir: Path) -> None:
    for side in ("L", "R"):
        checkpoint = model_dir / "weights" / "warping_model" / "flx" / "12" / side
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / f"{side}.index").touch()
    (model_dir / "face_landmarker.task").touch()


def test_launcher_bootstraps_missing_models_once(tmp_path, monkeypatch) -> None:
    launcher, model_dir, log = _prepare_launcher(tmp_path, monkeypatch)
    result = subprocess.run([str(launcher), "--passthrough"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "tools/download_models.py" in calls[0]
    assert f"--model-dir {model_dir}" in calls[0]
    assert calls[1].endswith("eyeline run --passthrough")


def test_launcher_reuses_complete_models_without_download(tmp_path, monkeypatch) -> None:
    launcher, model_dir, log = _prepare_launcher(tmp_path, monkeypatch)
    _create_models(model_dir)
    result = subprocess.run([str(launcher)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "tools/download_models.py" not in calls[0]
    assert calls[0].endswith("eyeline run")
    source_path = str(tmp_path / "project" / "src")
    python_path = Path(f"{log}.pythonpath").read_text(encoding="utf-8")
    assert python_path.split(":", maxsplit=1)[0] == source_path


def test_launcher_stops_on_download_or_checksum_failure(tmp_path, monkeypatch) -> None:
    launcher, _, log = _prepare_launcher(tmp_path, monkeypatch)
    monkeypatch.setenv("UV_FAIL_DOWNLOAD", "1")
    result = subprocess.run([str(launcher)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "could not download or verify" in result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "tools/download_models.py" in calls[0]
