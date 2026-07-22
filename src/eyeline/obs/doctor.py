"""Read-only diagnostics plus explicit opt-in camera/output probes."""

from __future__ import annotations

import glob
import importlib.metadata
import json
import os
import platform
import plistlib
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from eyeline.config import CameraConfig
from eyeline.obs.capture import create_camera_capture
from eyeline.obs.processes import is_obs_running
from eyeline.obs.sink import OBSVirtualCameraSink


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    detail: str


def _model_directory() -> Path:
    configured = os.environ.get("EYELINE_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    home = Path(os.environ.get("EYELINE_HOME", "~/.local/share/eyeline")).expanduser()
    return home / "models"


def _obs_version(app: Path) -> str | None:
    try:
        with (app / "Contents/Info.plist").open("rb") as plist_file:
            info = plistlib.load(plist_file)
        return str(info.get("CFBundleShortVersionString") or info.get("CFBundleVersion"))
    except (OSError, plistlib.InvalidFileException):
        return None


def _extension_state() -> tuple[bool, str]:
    identifier = "com.obsproject.obs-studio.mac-camera-extension"
    try:
        result = subprocess.run(
            ["systemextensionsctl", "list"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        output = f"{result.stdout}\n{result.stderr}"
        lines = [line.strip() for line in output.splitlines() if identifier in line]
        if lines:
            active = any("activated enabled" in line.lower() for line in lines)
            return active, lines[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    installed = glob.glob(
        "/Library/SystemExtensions/*/"
        "com.obsproject.obs-studio.mac-camera-extension.systemextension"
    )
    if installed:
        return False, f"installed but activation could not be confirmed: {installed[0]}"
    return False, "OBS camera Media Extension is not activated"


def _dependency_check(distribution: str) -> Check:
    try:
        version = importlib.metadata.version(distribution)
        return Check(distribution, "ok", version)
    except importlib.metadata.PackageNotFoundError:
        return Check(distribution, "error", "not installed in the active Python environment")


def collect_checks(
    camera: CameraConfig,
    *,
    probe_camera: bool = False,
    probe_output: bool = False,
) -> list[Check]:
    checks: list[Check] = []
    is_macos = platform.system() == "Darwin"
    checks.append(
        Check(
            "system",
            "ok" if is_macos else "error",
            f"{platform.platform()} ({platform.machine()})",
        )
    )
    python_ok = sys.version_info[:2] == (3, 12)
    checks.append(Check("python", "ok" if python_ok else "error", platform.python_version()))
    for distribution in ("opencv-contrib-python", "pyvirtualcam"):
        checks.append(_dependency_check(distribution))

    for variable in ("EYELINE_HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR"):
        raw = os.environ.get(variable)
        if not raw:
            checks.append(Check(variable, "warning", "not set; use run-eyeline.command"))
            continue
        path = Path(raw).expanduser()
        writable = path.is_dir() and os.access(path, os.W_OK)
        checks.append(Check(variable, "ok" if writable else "error", str(path)))

    model_dir = _model_directory()
    landmarker = model_dir / "face_landmarker.task"
    checks.append(
        Check(
            "face landmarker",
            "ok" if landmarker.is_file() else "warning",
            str(landmarker) if landmarker.is_file() else f"missing from {model_dir}",
        )
    )
    checkpoint_files = list(model_dir.rglob("*.index")) if model_dir.is_dir() else []
    checks.append(
        Check(
            "gaze weights",
            "ok" if checkpoint_files else "warning",
            (
                str(checkpoint_files[0])
                if checkpoint_files
                else f"no TensorFlow checkpoint in {model_dir}"
            ),
        )
    )

    obs_app = Path("/Applications/OBS.app")
    version = _obs_version(obs_app) if obs_app.is_dir() else None
    checks.append(
        Check(
            "OBS Studio",
            "ok" if obs_app.is_dir() else "error",
            f"{obs_app} version {version or 'unknown'}" if obs_app.is_dir() else "not installed",
        )
    )
    extension_active, extension_detail = _extension_state()
    checks.append(
        Check("OBS Media Extension", "ok" if extension_active else "error", extension_detail)
    )
    running = is_obs_running()
    checks.append(
        Check(
            "OBS process",
            "error" if running else "ok",
            "running; quit OBS before EyeLine" if running else "not running",
        )
    )

    if os.environ.get("CI") and (probe_camera or probe_output):
        checks.append(Check("live probes", "skipped", "disabled because CI is set"))
        return checks

    if probe_camera:
        try:
            with create_camera_capture(
                camera.index, camera.width, camera.height, camera.fps
            ) as source:
                frame = source.read()
            checks.append(Check("physical camera probe", "ok", f"BGR frame {frame.shape}"))
        except Exception as exc:
            checks.append(Check("physical camera probe", "error", str(exc)))
    else:
        checks.append(Check("physical camera probe", "skipped", "pass --probe-camera to opt in"))

    if probe_output:
        if running:
            checks.append(Check("OBS backend probe", "error", "quit OBS before probing"))
        else:
            try:
                with OBSVirtualCameraSink(camera.width, camera.height, camera.fps) as sink:
                    detail = sink.device or "OBS backend opened"
                checks.append(Check("OBS backend probe", "ok", detail))
            except Exception as exc:
                checks.append(Check("OBS backend probe", "error", str(exc)))
    else:
        checks.append(Check("OBS backend probe", "skipped", "pass --probe-output to opt in"))
    return checks


def format_checks(checks: Iterable[Check], *, json_output: bool = False) -> str:
    checks = list(checks)
    if json_output:
        return json.dumps([asdict(check) for check in checks], indent=2)
    width = max((len(check.name) for check in checks), default=0)
    return "\n".join(
        f"[{check.status.upper():7}] {check.name:<{width}}  {check.detail}" for check in checks
    )


def checks_succeeded(checks: Iterable[Check]) -> bool:
    return all(check.status != "error" for check in checks)
