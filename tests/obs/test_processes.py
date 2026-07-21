from __future__ import annotations

from types import SimpleNamespace

import pytest

from eyeline.obs import processes


def test_obs_running_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        processes.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="123\n"),
    )
    assert processes.is_obs_running()
    with pytest.raises(RuntimeError, match="quit OBS"):
        processes.require_obs_closed()


def test_obs_not_running_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        processes.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert not processes.is_obs_running()
