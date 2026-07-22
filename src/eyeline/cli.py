"""EyeLine command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from eyeline.config import EyeLineConfig, load_config
from eyeline.obs.capture import create_camera_capture
from eyeline.obs.doctor import checks_succeeded, collect_checks, format_checks
from eyeline.obs.fixtures import NullSink, PassthroughProcessor, SyntheticCapture
from eyeline.obs.processes import require_obs_closed
from eyeline.obs.runner import PipelineRunner
from eyeline.obs.sink import OBSVirtualCameraSink

LOGGER = logging.getLogger("eyeline")
DEFAULT_CONFIG = Path("config/default.yaml")


def _processor(config: EyeLineConfig, *, passthrough: bool):
    if passthrough:
        return PassthroughProcessor()
    try:
        from eyeline.engine import create_frame_processor
    except ImportError:
        try:
            from eyeline.engine.factory import create_frame_processor
        except ImportError as exc:
            raise RuntimeError(
                "EyeLine processing engine is unavailable. Install the complete project or pass "
                "--passthrough for a camera/output transport test."
            ) from exc
    return create_frame_processor(config)


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    require_obs_closed()
    processor = _processor(config, passthrough=args.passthrough)
    source = create_camera_capture(
        config.camera.index, config.camera.width, config.camera.height, config.camera.fps
    )
    sink = OBSVirtualCameraSink(config.camera.width, config.camera.height, config.camera.fps)
    LOGGER.info("Starting EyeLine at %dx%d @ %d FPS", *(
        config.camera.width, config.camera.height, config.camera.fps
    ))
    stats = PipelineRunner(source, processor, sink).run()
    LOGGER.info("Stopped after %d frames (%.1f FPS average)", stats.frames_sent, stats.average_fps)
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checks = collect_checks(
        config.camera, probe_camera=args.probe_camera, probe_output=args.probe_output
    )
    print(format_checks(checks, json_output=args.json))
    return 0 if checks_succeeded(checks) else 1


def _soak(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    processor = _processor(config, passthrough=args.passthrough)
    source = SyntheticCapture(
        config.camera.width, config.camera.height, config.camera.fps, realtime=not args.no_realtime
    )
    if args.obs:
        require_obs_closed()
        sink = OBSVirtualCameraSink(config.camera.width, config.camera.height, config.camera.fps)
    else:
        sink = NullSink()
    stats = PipelineRunner(source, processor, sink).run(
        duration_seconds=args.duration,
        max_frames=args.frames,
    )
    print(
        f"soak complete: frames={stats.frames_sent} fps={stats.average_fps:.1f} "
        f"processor_failures={stats.processor_failures}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eyeline")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="publish the physical camera through EyeLine to OBS")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--passthrough", action="store_true", help="test transport without correction")
    run.set_defaults(handler=_run)

    doctor = subparsers.add_parser("doctor", help="check the local OBS runtime")
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--probe-camera", action="store_true")
    doctor.add_argument("--probe-output", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_doctor)

    soak = subparsers.add_parser("soak", help="run a synthetic fixture without physical camera")
    soak.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    soak.add_argument("--duration", type=float, default=1800.0)
    soak.add_argument("--frames", type=int)
    soak.add_argument("--no-realtime", action="store_true")
    soak.add_argument("--passthrough", action="store_true")
    soak.add_argument(
        "--obs", action="store_true", help="publish fixture to OBS instead of null sink"
    )
    soak.set_defaults(handler=_soak)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; camera and virtual output released")
        return 130
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
