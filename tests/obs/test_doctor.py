from __future__ import annotations

from eyeline.config import CameraConfig
from eyeline.obs import doctor


def test_ci_never_opens_live_camera_or_output(monkeypatch) -> None:
    monkeypatch.setenv("CI", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("live device was touched")

    monkeypatch.setattr(doctor, "create_camera_capture", forbidden)
    monkeypatch.setattr(doctor, "OBSVirtualCameraSink", forbidden)
    monkeypatch.setattr(doctor, "is_obs_running", lambda: False)
    monkeypatch.setattr(doctor, "_extension_state", lambda: (False, "not active"))
    checks = doctor.collect_checks(
        CameraConfig(width=16, height=8), probe_camera=True, probe_output=True
    )
    assert any(check.name == "live probes" and check.status == "skipped" for check in checks)


def test_json_report_is_machine_readable() -> None:
    rendered = doctor.format_checks([doctor.Check("one", "ok", "ready")], json_output=True)
    assert '"name": "one"' in rendered
