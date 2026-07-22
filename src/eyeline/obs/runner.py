"""Fail-open camera-to-processor-to-OBS pipeline."""

from __future__ import annotations

import logging
import signal
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from eyeline.contracts import FrameProcessor, OutputSink, ProcessedFrame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunStats:
    frames_sent: int
    processor_failures: int
    source_failures: int
    elapsed_seconds: float

    @property
    def average_fps(self) -> float:
        return self.frames_sent / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


class PipelineRunner:
    """Own lifecycle and preserve the original frame whenever processing fails."""

    def __init__(self, source: Any, processor: FrameProcessor, sink: OutputSink) -> None:
        self.source = source
        self.processor = processor
        self.sink = sink

    def run(
        self,
        *,
        stop_event: threading.Event | None = None,
        max_frames: int | None = None,
        duration_seconds: float | None = None,
        install_signal_handlers: bool = True,
    ) -> RunStats:
        stop = stop_event or threading.Event()
        started = time.monotonic()
        frames_sent = processor_failures = source_failures = 0
        face_frames = corrected_frames = 0
        effective_strength_total = 0.0
        correction_ready_logged = False
        last_status_at = started
        last_status_frames = 0
        old_handlers: dict[signal.Signals, Any] = {}

        def render(original_bgr: np.ndarray) -> np.ndarray:
            nonlocal processor_failures
            nonlocal face_frames, corrected_frames, effective_strength_total
            nonlocal correction_ready_logged

            timestamp = time.monotonic()
            output_bgr = original_bgr
            try:
                processed = self.processor.process(original_bgr, timestamp)
                if self._valid_processed(processed, original_bgr):
                    output_bgr = processed.frame_bgr
                    face_frames += int(processed.face_detected)
                    corrected_frames += int(processed.correction_applied)
                    effective_strength_total += processed.effective_strength
                    if processed.correction_applied and not correction_ready_logged:
                        LOGGER.info(
                            "EyeLine correction ready: effective strength %.2f",
                            processed.effective_strength,
                        )
                        correction_ready_logged = True
                else:
                    processor_failures += 1
                    LOGGER.warning("processor returned an invalid frame; publishing original")
            except Exception:
                processor_failures += 1
                LOGGER.exception("processor failed; publishing original frame")
            return np.ascontiguousarray(output_bgr[..., ::-1])

        def log_status() -> None:
            nonlocal last_status_at, last_status_frames
            nonlocal face_frames, corrected_frames, effective_strength_total

            now = time.monotonic()
            interval = now - last_status_at
            if interval < 5.0:
                return
            interval_frames = frames_sent - last_status_frames
            correction_rate = 100.0 * corrected_frames / max(interval_frames, 1)
            face_rate = 100.0 * face_frames / max(interval_frames, 1)
            average_strength = effective_strength_total / max(corrected_frames, 1)
            LOGGER.info(
                "EyeLine active: %.1f FPS, face %.0f%%, correction %.0f%%, strength %.2f",
                interval_frames / interval,
                face_rate,
                correction_rate,
                average_strength,
            )
            last_status_at = now
            last_status_frames = frames_sent
            face_frames = corrected_frames = 0
            effective_strength_total = 0.0

        if install_signal_handlers and threading.current_thread() is threading.main_thread():
            def request_stop(signum: int, frame: object) -> None:
                del signum, frame
                stop.set()

            for signal_number in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, request_stop)

        try:
            with ExitStack() as stack:
                processor_close = getattr(self.processor, "close", None)
                if processor_close is not None:
                    stack.callback(processor_close)
                source = stack.enter_context(self.source)
                first_rgb: np.ndarray | None = None
                should_start = not stop.is_set() and (max_frames is None or max_frames > 0)
                if duration_seconds is not None and duration_seconds <= 0.0:
                    should_start = False
                if should_start:
                    try:
                        original_bgr = source.read()
                    except Exception:
                        if not stop.is_set():
                            source_failures += 1
                            raise
                    else:
                        # Load landmarks/checkpoints and prepare a real frame before claiming
                        # OBS. Zoom can otherwise see a black virtual camera during TF startup.
                        first_rgb = render(original_bgr)

                if first_rgb is not None and not stop.is_set():
                    sink = stack.enter_context(self.sink)
                    sink.send(first_rgb)
                    frames_sent += 1
                    LOGGER.info(
                        "EyeLine video ready: publishing %dx%d RGB to %s",
                        first_rgb.shape[1],
                        first_rgb.shape[0],
                        getattr(sink, "device", None) or type(sink).__name__,
                    )

                    while not stop.is_set():
                        if max_frames is not None and frames_sent >= max_frames:
                            break
                        if (
                            duration_seconds is not None
                            and time.monotonic() - started >= duration_seconds
                        ):
                            break
                        try:
                            original_bgr = source.read()
                        except Exception:
                            if stop.is_set():
                                break
                            source_failures += 1
                            raise
                        sink.send(render(original_bgr))
                        frames_sent += 1
                        log_status()
        finally:
            for signal_number, handler in old_handlers.items():
                signal.signal(signal_number, handler)

        return RunStats(
            frames_sent=frames_sent,
            processor_failures=processor_failures,
            source_failures=source_failures,
            elapsed_seconds=max(time.monotonic() - started, 0.0),
        )

    @staticmethod
    def _valid_processed(processed: object, original: np.ndarray) -> bool:
        return (
            isinstance(processed, ProcessedFrame)
            and processed.frame_bgr.shape == original.shape
            and processed.frame_bgr.dtype == np.uint8
        )
