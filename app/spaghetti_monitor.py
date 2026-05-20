from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .events import Event

log = logging.getLogger(__name__)


class CameraClient(Protocol):
    def capture_frame(self, timeout_sec: float = 10.0) -> bytes | None: ...


class Detector(Protocol):
    def detect(self, jpeg_bytes: bytes): ...


AlertCallback = Callable[[Event, bytes | None], None]
TickCallback = Callable[[bytes | None, Any, Any, bool, bool], None]
AutoPauseCallback = Callable[[], None]


@dataclass(frozen=True)
class DecisionState:
    frame_num: int = 0
    current_score: float = 0.0
    ewm_mean: float = 0.0
    rolling_mean_short: float = 0.0
    rolling_mean_long: float = 0.0
    adjusted_score: float = 0.0
    warning_threshold: float = 0.38
    failure_threshold: float = 0.665
    normalized_score: float = 0.0
    safe_frames_remaining: int = 0
    should_alert: bool = False
    should_pause: bool = False


class ObicoDecisionEngine:
    _EWM_ALPHA = 2 / (12 + 1)
    _ROLLING_WIN_SHORT = 310
    _ROLLING_WIN_LONG = 7200
    _ESCALATING_FACTOR = 1.0

    def __init__(
        self,
        *,
        sensitivity: float,
        threshold_low: float,
        threshold_high: float,
        rolling_mean_short_multiple: float,
    ) -> None:
        self._sensitivity = sensitivity
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._rolling_mean_short_multiple = rolling_mean_short_multiple
        self.reset()

    def reset(self) -> DecisionState:
        self._frame_num = 0
        self._ewm_mean = 0.0
        self._rolling_mean_short = 0.0
        self._rolling_mean_long = 0.0
        self._last = DecisionState(
            warning_threshold=self._threshold_low,
            failure_threshold=self._threshold_low * self._ESCALATING_FACTOR,
        )
        return self._last

    def current(self) -> DecisionState:
        return self._last

    def update(self, score: float) -> DecisionState:
        self._frame_num += 1
        self._ewm_mean = score * self._EWM_ALPHA + self._ewm_mean * (1 - self._EWM_ALPHA)
        self._rolling_mean_short = self._next_rolling_mean(
            score, self._rolling_mean_short, self._frame_num, self._ROLLING_WIN_SHORT,
        )
        self._rolling_mean_long = self._next_rolling_mean(
            score, self._rolling_mean_long, self._frame_num, self._ROLLING_WIN_LONG,
        )

        adjusted_score = (self._ewm_mean - self._rolling_mean_long) * self._sensitivity
        warning_threshold = min(
            self._threshold_high,
            max(
                self._threshold_low,
                (self._rolling_mean_short - self._rolling_mean_long) * self._rolling_mean_short_multiple,
            ),
        )
        failure_threshold = warning_threshold * self._ESCALATING_FACTOR

        self._last = DecisionState(
            frame_num=self._frame_num,
            current_score=score,
            ewm_mean=self._ewm_mean,
            rolling_mean_short=self._rolling_mean_short,
            rolling_mean_long=self._rolling_mean_long,
            adjusted_score=adjusted_score,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            normalized_score=self._calc_normalized_score(adjusted_score, warning_threshold, failure_threshold),
            should_alert=self._is_failing(1.0),
            should_pause=self._is_failing(self._ESCALATING_FACTOR),
        )
        return self._last

    def _is_failing(self, escalating_factor: float) -> bool:
        adjusted = (self._ewm_mean - self._rolling_mean_long) * self._sensitivity / escalating_factor
        if adjusted < self._threshold_low:
            return False
        if adjusted > self._threshold_high:
            return True
        return adjusted > (
            (self._rolling_mean_short - self._rolling_mean_long) * self._rolling_mean_short_multiple
        )

    @staticmethod
    def _next_rolling_mean(score: float, current: float, count: int, win_size: int) -> float:
        denom = float(win_size if win_size <= count else count + 1)
        return current + (score - current) / denom

    @staticmethod
    def _scale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
        if old_max == old_min:
            return new_max if value >= old_max else new_min
        scaled = (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
        return min(new_max, max(new_min, scaled))

    def _calc_normalized_score(
        self,
        adjusted_score: float,
        warning_threshold: float,
        failure_threshold: float,
    ) -> float:
        if adjusted_score > failure_threshold:
            return self._scale(adjusted_score, failure_threshold, failure_threshold * 1.5, 2.0 / 3.0, 1.0)
        if adjusted_score > warning_threshold:
            return self._scale(adjusted_score, warning_threshold, failure_threshold, 1.0 / 3.0, 2.0 / 3.0)
        return self._scale(adjusted_score, 0.0, warning_threshold, 0.0, 1.0 / 3.0)


class SpaghettiMonitor:
    def __init__(
        self,
        camera: CameraClient,
        detector: Detector,
        on_alert: AlertCallback,
        *,
        interval_sec: float = 30.0,
        sensitivity: float = 1.0,
        threshold_low: float = 0.38,
        threshold_high: float = 0.78,
        rolling_mean_short_multiple: float = 3.8,
        on_tick: TickCallback | None = None,
        auto_pause: AutoPauseCallback | None = None,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._on_alert = on_alert
        self._on_tick = on_tick
        self._auto_pause = auto_pause
        self._interval_sec = interval_sec
        self._decision_engine = ObicoDecisionEngine(
            sensitivity=sensitivity,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            rolling_mean_short_multiple=rolling_mean_short_multiple,
        )
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._alerted = False
        self._auto_paused = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="spaghetti-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = dict(snapshot)

    def tick_once(self) -> None:
        with self._lock:
            snapshot = dict(self._snapshot)

        try:
            frame = self._camera.capture_frame()
        except Exception as exc:
            log.warning("camera capture failed: %s", exc)
            frame = None
        if not frame:
            diag = getattr(self._camera, "last_capture_diagnostic", None)
            log.warning("camera capture returned no frame%s", f": {diag}" if diag else "")

        now_active = snapshot.get("gcode_state") == "RUNNING"
        if not now_active:
            self._reset_state(now_active=False)
            self._emit_tick(frame, None, False)
            return

        if not self._active:
            self._reset_state(now_active=True)

        if not frame:
            self._emit_tick(None, None, True)
            return

        try:
            result = self._detector.detect(frame)
        except Exception as exc:
            log.warning("spaghetti detection failed: %s", exc)
            self._emit_tick(frame, None, True)
            return

        decision = self._decision_engine.update(result.score)
        self._emit_tick(frame, result, True)

        if decision.should_alert and not self._alerted:
            detail = (
                f"{result.failure_type} (risk {decision.normalized_score:.2f}; "
                f"frame {decision.current_score:.2f}; top {result.confidence:.2f}) — {result.summary}"
            )
            event = Event(
                kind="spaghetti_detected",
                title="Possible spaghetti / blob failure detected",
                detail=detail,
                file=snapshot.get("subtask_name"),
                layer=self._maybe_int(snapshot.get("layer_num")),
                layer_total=self._maybe_int(snapshot.get("total_layer_num")),
                percent=self._maybe_int(snapshot.get("mc_percent")),
            )
            self._on_alert(event, frame)
            self._alerted = True

        if decision.should_pause and self._auto_pause is not None and not self._auto_paused:
            try:
                self._auto_pause()
                self._auto_paused = True
            except Exception as exc:
                log.warning("auto-pause failed: %s", exc)

    def _emit_tick(self, frame: bytes | None, result: Any, active: bool) -> None:
        if self._on_tick is None:
            return
        try:
            self._on_tick(frame, result, self._decision_engine.current(), active, self._alerted)
        except Exception as exc:
            log.debug("on_tick callback failed: %s", exc)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.tick_once()
            self._stop_event.wait(self._interval_sec)

    def _reset_state(self, *, now_active: bool) -> None:
        self._active = now_active
        self._alerted = False
        self._auto_paused = False
        self._decision_engine.reset()

    @staticmethod
    def _maybe_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
