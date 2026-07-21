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
        old_handlers: dict[signal.Signals, Any] = {}

        if install_signal_handlers and threading.current_thread() is threading.main_thread():
            def request_stop(signum: int, frame: object) -> None:
                del signum, frame
                stop.set()

            for signal_number in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, request_stop)

        try:
            with ExitStack() as stack:
                source = stack.enter_context(self.source)
                sink = stack.enter_context(self.sink)
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
                    timestamp = time.monotonic()
                    output_bgr = original_bgr
                    try:
                        processed = self.processor.process(original_bgr, timestamp)
                        if self._valid_processed(processed, original_bgr):
                            output_bgr = processed.frame_bgr
                        else:
                            processor_failures += 1
                            LOGGER.warning(
                                "processor returned an invalid frame; publishing original"
                            )
                    except Exception:
                        processor_failures += 1
                        LOGGER.exception("processor failed; publishing original frame")
                    rgb = np.ascontiguousarray(output_bgr[..., ::-1])
                    sink.send(rgb)
                    frames_sent += 1
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
