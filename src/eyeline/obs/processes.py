"""macOS process checks used before claiming the OBS camera extension."""

from __future__ import annotations

import subprocess


def is_obs_running() -> bool:
    """Return true only for an OBS application process, without requiring psutil."""

    try:
        result = subprocess.run(
            ["pgrep", "-f", r"(^|/)OBS\.app/Contents/MacOS/OBS($| )"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def require_obs_closed() -> None:
    if is_obs_running():
        raise RuntimeError(
            "OBS Studio is running and may own OBS Virtual Camera. Stop Virtual Camera, quit OBS "
            "completely, and then run EyeLine again. OBS must remain closed while EyeLine runs."
        )
