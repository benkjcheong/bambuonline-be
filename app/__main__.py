from __future__ import annotations

import logging
import os
import signal
import sys

from .api import ApiServer, AppState, build_app
from .camera_stream import BambuCameraClient, CameraConfig, CameraUnavailableError
from .config import Config, load_config, setup_logging
from .failure_detector import DetectionError, LocalFailureDetector
from .mqtt_client import build_client
from .printer_control import PrinterControl
from .spaghetti_monitor import SpaghettiMonitor
from .state import StateTracker
from .telegram import TelegramNotifier

log = logging.getLogger("app")


def run() -> int:
    cfg = load_config()
    setup_logging(cfg.log_level)
    log.info("starting app v1 (printer=%s)", cfg.printer_serial)

    app_state = AppState()

    tracker = StateTracker()
    notifier = TelegramNotifier(cfg.tg_bot_token, cfg.tg_chat_id)

    control_holder: dict[str, PrinterControl | None] = {"ctrl": None}

    def auto_pause() -> None:
        ctrl = control_holder["ctrl"]
        if ctrl is not None:
            ctrl.publish_command("pause", source="auto_pause")

    monitor = _build_spaghetti_monitor(cfg, notifier, app_state, auto_pause)

    def handle_payload(payload):
        events = tracker.ingest(payload)
        app_state.update_snapshot(tracker.snapshot)
        for ev in events:
            _deliver_monitor_event(notifier, ev, None)
            app_state.push_event(ev)
        if monitor is not None:
            monitor.update_snapshot(tracker.snapshot)

    client = build_client(
        printer_ip=cfg.printer_ip,
        printer_serial=cfg.printer_serial,
        access_code=cfg.access_code,
        on_payload=handle_payload,
    )

    control = PrinterControl(
        client,
        cfg.printer_serial,
        app_state,
        printer_ip=cfg.printer_ip,
        access_code=cfg.access_code,
    )
    control_holder["ctrl"] = control
    app_state.bind_control(control)

    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    api_server = ApiServer(build_app(app_state), host=api_host, port=api_port)
    api_server.start()
    log.info("api server listening on %s:%d", api_host, api_port)

    if monitor is not None:
        monitor.start()

    def _stop(_signum, _frame):
        log.info("shutting down")
        if monitor is not None:
            monitor.stop()
        api_server.stop()
        client.disconnect()
        client.loop_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        client.loop_forever(retry_first_connection=True)
    finally:
        if monitor is not None:
            monitor.stop()
        api_server.stop()
    return 0


def _build_spaghetti_monitor(
    cfg: Config,
    notifier: TelegramNotifier,
    app_state: AppState,
    auto_pause,
) -> SpaghettiMonitor | None:
    try:
        camera = BambuCameraClient(
            CameraConfig(
                printer_ip=cfg.printer_ip,
                access_code=cfg.access_code,
                library_path=cfg.camera_library_path,
            )
        )
        detector = LocalFailureDetector(
            model_url=cfg.spaghetti_model_url,
            model_path=cfg.spaghetti_model_path,
            threshold=cfg.spaghetti_detection_threshold,
        )
    except CameraUnavailableError as exc:
        log.warning("AI monitoring disabled: %s", exc)
        return None
    except DetectionError as exc:
        log.warning("AI monitoring disabled: %s", exc)
        return None

    def on_tick(frame, result, decision, active, alerted):
        app_state.set_frame(frame)
        app_state.update_detection(result, decision, active, alerted)

    return SpaghettiMonitor(
        camera,
        detector,
        lambda event, image: _deliver_monitor_event(notifier, event, image, app_state),
        interval_sec=cfg.spaghetti_interval_sec,
        sensitivity=cfg.spaghetti_sensitivity,
        threshold_low=cfg.spaghetti_threshold_low,
        threshold_high=cfg.spaghetti_threshold_high,
        rolling_mean_short_multiple=cfg.spaghetti_rolling_mean_short_multiple,
        on_tick=on_tick,
        auto_pause=auto_pause,
    )


def _deliver_monitor_event(notifier: TelegramNotifier, event, image_bytes, app_state: AppState | None = None) -> None:
    log.info("event: %s — %s", event.kind, event.title)
    ok = notifier.send_photo(image_bytes, event.format()) if image_bytes else notifier.send(event.format())
    if not ok:
        log.error("failed to deliver event: %s", event.kind)
    if app_state is not None:
        app_state.set_frame(image_bytes)
        app_state.push_event(event)


if __name__ == "__main__":
    raise SystemExit(run())
