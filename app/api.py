from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

import uvicorn
from fastapi import FastAPI, Request, Response
from starlette.datastructures import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .events import Event
from .failure_detector import DetectionResult
from .ftps_upload import FtpsUploadError
from .printer_control import (
    FAN_INDEX,
    SPEED_LEVELS,
    VALID_ACTIONS,
    VALID_LIGHT_MODES,
    PrinterControl,
)
from .slicer import SUPPORTED_INPUTS, Slicer, SlicerError, summarize_result

log = logging.getLogger(__name__)

_EVENT_BUFFER_SIZE = 200
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB cap on 3MF parts


def _form_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _safe_remote_name(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".", " ") else "_" for ch in base)
    cleaned = cleaned.strip().replace(" ", "_") or "upload.3mf"
    if not cleaned.lower().endswith(".3mf"):
        cleaned += ".3mf"
    return cleaned[:120]


class AppState:
    """Thread-safe shared state surfaced to the HTTP API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=_EVENT_BUFFER_SIZE)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_bytes: bytes | None = None
        self._frame_ts: float | None = None
        self._detector: dict[str, Any] = {
            "last_tick_ts": None,
            "last_confidence": None,
            "last_score": None,
            "last_box_count": 0,
            "last_summary": None,
            "failure_detected": None,
            "monitoring_active": False,
            "decision_policy": "obico_like",
            "frame_num": 0,
            "normalized_score": 0.0,
            "adjusted_score": 0.0,
            "warning_threshold": 0.38,
            "failure_threshold": 0.665,
            "safe_frames_remaining": 0,
            "alerted": False,
        }
        self._control: PrinterControl | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        snap_copy = dict(snapshot)
        with self._lock:
            self._snapshot = snap_copy
        self._broadcast({"type": "snapshot", "ts": time.time(), "data": snap_copy})

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def push_event(self, event: Event) -> None:
        record = {"ts": time.time(), **asdict(event)}
        with self._lock:
            self._events.append(record)
        self._broadcast({"type": "event", "ts": record["ts"], "data": record})

    def _broadcast(self, msg: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers)
            loop = self._loop
        if loop is None:
            return
        for q in subs:
            loop.call_soon_threadsafe(self._safe_put, q, msg)

    @staticmethod
    def _safe_put(q: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def set_frame(self, jpeg: bytes | None) -> None:
        ts: float | None = None
        with self._lock:
            if jpeg:
                self._frame_bytes = jpeg
                self._frame_ts = time.time()
                ts = self._frame_ts
        if ts is not None:
            self._broadcast({"type": "frame", "ts": ts})

    def frame(self) -> tuple[bytes | None, float | None]:
        with self._lock:
            return self._frame_bytes, self._frame_ts

    def update_detection(self, result: DetectionResult | None, decision: Any, active: bool, alerted: bool) -> None:
        with self._lock:
            self._detector["last_tick_ts"] = time.time()
            self._detector["monitoring_active"] = active
            self._detector["alerted"] = alerted
            self._detector["frame_num"] = getattr(decision, "frame_num", 0)
            self._detector["normalized_score"] = getattr(decision, "normalized_score", 0.0)
            self._detector["adjusted_score"] = getattr(decision, "adjusted_score", 0.0)
            self._detector["warning_threshold"] = getattr(decision, "warning_threshold", 0.38)
            self._detector["failure_threshold"] = getattr(decision, "failure_threshold", 0.665)
            self._detector["safe_frames_remaining"] = getattr(decision, "safe_frames_remaining", 0)
            if result is not None:
                self._detector["last_confidence"] = result.confidence
                self._detector["last_score"] = result.score
                self._detector["last_box_count"] = result.box_count
                self._detector["last_summary"] = result.summary
                self._detector["failure_detected"] = result.failure_detected
            else:
                self._detector["last_confidence"] = None
                self._detector["last_score"] = None
                self._detector["last_box_count"] = 0
                self._detector["last_summary"] = None
                self._detector["failure_detected"] = None
            det_copy = dict(self._detector)
        self._broadcast({"type": "detector", "ts": time.time(), "data": det_copy})

    def detector(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._detector)

    def bind_control(self, ctrl: PrinterControl) -> None:
        with self._lock:
            self._control = ctrl

    def control(self) -> PrinterControl | None:
        with self._lock:
            return self._control


def build_app(state: AppState) -> FastAPI:
    app = FastAPI(title="App API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        state.bind_loop(asyncio.get_running_loop())

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "ts": time.time()}

    @app.get("/api/snapshot")
    def snapshot() -> dict[str, Any]:
        return {"snapshot": state.snapshot()}

    @app.get("/api/events")
    def events(limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(limit, _EVENT_BUFFER_SIZE))
        return {"events": state.recent_events(limit)}

    @app.get("/api/detector")
    def detector() -> dict[str, Any]:
        return state.detector()

    @app.get("/api/frame.jpg")
    def frame() -> Response:
        jpeg, ts = state.frame()
        if not jpeg:
            return Response(status_code=404, content=b"no frame yet")
        headers = {"Cache-Control": "no-store"}
        if ts is not None:
            headers["X-Frame-Timestamp"] = f"{ts:.3f}"
        return Response(content=jpeg, media_type="image/jpeg", headers=headers)

    @app.post("/api/print/start")
    async def print_start(request: Request) -> Response:
        try:
            form = await request.form(max_part_size=_MAX_UPLOAD_BYTES)
        except Exception as exc:
            log.warning("multipart parse failed: %s", exc)
            return JSONResponse({"ok": False, "error": "bad_form", "detail": str(exc)}, status_code=400)

        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return JSONResponse({"ok": False, "error": "missing_file"}, status_code=400)

        log.info(
            "print_start filename=%s size=%s content_type=%s",
            upload.filename, getattr(upload, "size", "?"), upload.content_type,
        )

        if not upload.filename or not upload.filename.lower().endswith(".3mf"):
            log.warning("reject: not 3mf filename=%s", upload.filename)
            return JSONResponse({"ok": False, "error": "must_be_3mf"}, status_code=400)
        ctrl = state.control()
        if ctrl is None or not ctrl.is_connected():
            log.warning("reject: mqtt disconnected")
            return JSONResponse({"ok": False, "error": "mqtt_disconnected"}, status_code=503)
        snap = state.snapshot()
        gcode = str(snap.get("gcode_state") or "").upper()
        if gcode not in {"IDLE", "FINISH", "FAILED", ""}:
            log.warning("reject: busy gcode_state=%s", gcode)
            return JSONResponse(
                {"ok": False, "error": "busy", "gcode_state": gcode},
                status_code=409,
            )

        try:
            plate = int(form.get("plate", 1))
        except (TypeError, ValueError):
            plate = 1
        use_ams = _form_bool(form.get("use_ams"), False)
        bed_leveling = _form_bool(form.get("bed_leveling"), True)
        flow_cali = _form_bool(form.get("flow_cali"), False)
        vibration_cali = _form_bool(form.get("vibration_cali"), True)
        layer_inspect = _form_bool(form.get("layer_inspect"), False)
        timelapse = _form_bool(form.get("timelapse"), False)

        safe_name = _safe_remote_name(upload.filename)
        subtask = safe_name.rsplit(".", 1)[0]
        try:
            upload.file.seek(0)
        except Exception:
            pass
        try:
            ok, seq = await run_in_threadpool(
                ctrl.upload_and_start,
                src=upload.file,
                remote_name=safe_name,
                subtask_name=subtask,
                plate=plate,
                use_ams=use_ams,
                bed_leveling=bed_leveling,
                flow_cali=flow_cali,
                vibration_cali=vibration_cali,
                layer_inspect=layer_inspect,
                timelapse=timelapse,
            )
        except FtpsUploadError as exc:
            return JSONResponse({"ok": False, "error": "ftps_failed", "detail": str(exc)}, status_code=502)
        finally:
            await upload.close()
        return JSONResponse(
            {"ok": ok, "sequence_id": seq, "remote_name": safe_name},
            status_code=200 if ok else 502,
        )

    @app.post("/api/print/{action}")
    def print_control(action: str) -> Response:
        if action not in VALID_ACTIONS:
            return JSONResponse({"ok": False, "error": "bad_action", "action": action}, status_code=400)
        ctrl = state.control()
        if ctrl is None or not ctrl.is_connected():
            return JSONResponse({"ok": False, "error": "mqtt_disconnected"}, status_code=503)
        snap = state.snapshot()
        gcode = str(snap.get("gcode_state") or "").upper()
        allowed = {"pause": {"RUNNING"}, "resume": {"PAUSE"}, "stop": {"RUNNING", "PAUSE"}}
        if gcode not in allowed[action]:
            return JSONResponse(
                {"ok": False, "error": "bad_state", "gcode_state": gcode},
                status_code=409,
            )
        ok, seq = ctrl.publish_command(action, source="api")
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    @app.post("/api/light/{action}")
    def light_control(action: str) -> Response:
        ctrl = state.control()
        if ctrl is None or not ctrl.is_connected():
            return JSONResponse({"ok": False, "error": "mqtt_disconnected"}, status_code=503)
        if action == "toggle":
            lights = state.snapshot().get("lights_report") or []
            current = "off"
            for item in lights:
                if isinstance(item, dict) and item.get("node") == "chamber_light":
                    current = str(item.get("mode") or "off").lower()
                    break
            mode = "off" if current == "on" else "on"
        elif action in VALID_LIGHT_MODES:
            mode = action
        else:
            return JSONResponse({"ok": False, "error": "bad_action"}, status_code=400)
        ok, seq = ctrl.set_light(mode)
        return JSONResponse({"ok": ok, "mode": mode, "sequence_id": seq}, status_code=200 if ok else 502)

    # ---- device controls (Bambu Studio device-tab parity) -------------------

    def _ctrl() -> PrinterControl | None:
        ctrl = state.control()
        if ctrl is None or not ctrl.is_connected():
            return None
        return ctrl

    _DISCONNECTED = JSONResponse({"ok": False, "error": "mqtt_disconnected"}, status_code=503)

    async def _json_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
            return body if isinstance(body, dict) else {}
        except Exception:
            return {}

    @app.post("/api/temp/{heater}")
    async def set_temp(heater: str, request: Request) -> Response:
        if heater not in ("nozzle", "bed"):
            return JSONResponse({"ok": False, "error": "bad_heater"}, status_code=400)
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        body = await _json_body(request)
        try:
            target = int(body.get("target"))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "bad_target"}, status_code=400)
        ok, seq = ctrl.set_nozzle_temp(target) if heater == "nozzle" else ctrl.set_bed_temp(target)
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    @app.post("/api/fan/{fan}")
    async def set_fan(fan: str, request: Request) -> Response:
        if fan not in FAN_INDEX:
            return JSONResponse({"ok": False, "error": "bad_fan"}, status_code=400)
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        body = await _json_body(request)
        try:
            percent = int(body.get("percent"))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "bad_percent"}, status_code=400)
        ok, seq = ctrl.set_fan(fan, percent)
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    @app.post("/api/speed")
    async def set_speed(request: Request) -> Response:
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        body = await _json_body(request)
        try:
            level = int(body.get("level"))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "bad_level"}, status_code=400)
        if level not in SPEED_LEVELS:
            return JSONResponse({"ok": False, "error": "bad_level"}, status_code=400)
        ok, seq = ctrl.set_speed_level(level)
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    def _movement_blocked() -> Response | None:
        gcode = str(state.snapshot().get("gcode_state") or "").upper()
        if gcode in {"RUNNING", "PREPARE", "PAUSE"}:
            return JSONResponse({"ok": False, "error": "busy", "gcode_state": gcode}, status_code=409)
        return None

    @app.post("/api/move/home")
    def move_home() -> Response:
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        blocked = _movement_blocked()
        if blocked is not None:
            return blocked
        ok, seq = ctrl.home()
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    @app.post("/api/move")
    async def move_jog(request: Request) -> Response:
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        blocked = _movement_blocked()
        if blocked is not None:
            return blocked
        body = await _json_body(request)
        axis = str(body.get("axis") or "").upper()
        try:
            dist = float(body.get("dist"))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "bad_dist"}, status_code=400)
        if axis not in ("X", "Y", "Z") or not dist:
            return JSONResponse({"ok": False, "error": "bad_axis_or_dist"}, status_code=400)
        ok, seq = ctrl.jog(axis, dist)
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    @app.post("/api/extrude")
    async def extrude(request: Request) -> Response:
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        blocked = _movement_blocked()
        if blocked is not None:
            return blocked
        try:
            nozzle = float(state.snapshot().get("nozzle_temper") or 0.0)
        except (TypeError, ValueError):
            nozzle = 0.0
        if nozzle < 170.0:
            return JSONResponse({"ok": False, "error": "nozzle_cold", "nozzle_temper": nozzle}, status_code=409)
        body = await _json_body(request)
        try:
            dist = float(body.get("dist"))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "bad_dist"}, status_code=400)
        if not dist:
            return JSONResponse({"ok": False, "error": "bad_dist"}, status_code=400)
        ok, seq = ctrl.extrude(dist)
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    # ---- SD card files -------------------------------------------------------

    @app.get("/api/files")
    async def list_files() -> Response:
        ctrl = state.control()
        if ctrl is None:
            return _DISCONNECTED
        try:
            files = await run_in_threadpool(ctrl.list_sd_files)
        except FtpsUploadError as exc:
            return JSONResponse({"ok": False, "error": "ftps_failed", "detail": str(exc)}, status_code=502)
        return JSONResponse({"ok": True, "files": files})

    def _start_busy() -> Response | None:
        gcode = str(state.snapshot().get("gcode_state") or "").upper()
        if gcode not in {"IDLE", "FINISH", "FAILED", ""}:
            return JSONResponse({"ok": False, "error": "busy", "gcode_state": gcode}, status_code=409)
        return None

    def _print_opts(body: dict[str, Any]) -> dict[str, Any]:
        try:
            plate = int(body.get("plate", 1))
        except (TypeError, ValueError):
            plate = 1
        return {
            "plate": plate,
            "use_ams": _form_bool(body.get("use_ams"), False),
            "bed_leveling": _form_bool(body.get("bed_leveling"), True),
            "flow_cali": _form_bool(body.get("flow_cali"), False),
            "vibration_cali": _form_bool(body.get("vibration_cali"), True),
            "layer_inspect": _form_bool(body.get("layer_inspect"), False),
            "timelapse": _form_bool(body.get("timelapse"), False),
        }

    @app.post("/api/files/print")
    async def print_sd_file(request: Request) -> Response:
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        busy = _start_busy()
        if busy is not None:
            return busy
        body = await _json_body(request)
        path = str(body.get("path") or "").strip().lstrip("/")
        if not path.lower().endswith(".3mf") or ".." in path:
            return JSONResponse({"ok": False, "error": "bad_path"}, status_code=400)
        ok, seq = ctrl.start_existing(remote_path=path, **_print_opts(body))
        return JSONResponse({"ok": ok, "sequence_id": seq}, status_code=200 if ok else 502)

    @app.delete("/api/files")
    async def delete_sd_file(path: str = "") -> Response:
        ctrl = state.control()
        if ctrl is None:
            return _DISCONNECTED
        clean = path.strip().lstrip("/")
        if not clean.lower().endswith(".3mf") or ".." in clean:
            return JSONResponse({"ok": False, "error": "bad_path"}, status_code=400)
        try:
            await run_in_threadpool(ctrl.delete_sd_file, clean)
        except FtpsUploadError as exc:
            return JSONResponse({"ok": False, "error": "ftps_failed", "detail": str(exc)}, status_code=502)
        return JSONResponse({"ok": True})

    # ---- slicing (Bambu Studio CLI) -----------------------------------------

    slicer = Slicer()

    @app.get("/api/slicer/presets")
    def slicer_presets() -> dict[str, Any]:
        return slicer.list_presets()

    @app.post("/api/slice")
    async def slice_model(request: Request) -> Response:
        if not slicer.available():
            return JSONResponse({"ok": False, "error": "slicer_unavailable"}, status_code=501)
        try:
            form = await request.form(max_part_size=_MAX_UPLOAD_BYTES)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": "bad_form", "detail": str(exc)}, status_code=400)
        upload = form.get("file")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return JSONResponse({"ok": False, "error": "missing_file"}, status_code=400)
        suffix = "." + upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if suffix not in SUPPORTED_INPUTS:
            return JSONResponse(
                {"ok": False, "error": "bad_type", "supported": list(SUPPORTED_INPUTS)},
                status_code=400,
            )

        tmpdir = tempfile.mkdtemp(prefix="spaghetteye-upload-")
        src_path = Path(tmpdir) / ("model" + suffix)
        try:
            with open(src_path, "wb") as fh:
                upload.file.seek(0)
                while True:
                    chunk = upload.file.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        finally:
            await upload.close()

        machine = str(form.get("machine") or "") or None
        process = str(form.get("process") or "") or None
        filament = str(form.get("filament") or "") or None
        try:
            scale = float(form.get("scale")) if form.get("scale") else None
        except (TypeError, ValueError):
            scale = None
        arrange = str(form.get("arrange") or "1")
        orient = str(form.get("orient") or "2")

        try:
            job = await run_in_threadpool(
                slicer.slice_file,
                src_path,
                upload.filename,
                machine=machine,
                process=process,
                filament=filament,
                scale=scale,
                arrange=arrange,
                orient=orient,
            )
        except SlicerError as exc:
            return JSONResponse({"ok": False, "error": "slice_failed", "detail": str(exc)}, status_code=502)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        state.push_event(Event(kind="slice_done", title="Model sliced", file=upload.filename))
        return JSONResponse({"ok": True, "job_id": job.job_id, "name": job.out_name, **summarize_result(job)})

    @app.post("/api/slice/{job_id}/print")
    async def slice_print(job_id: str, request: Request) -> Response:
        job = slicer.get_job(job_id)
        if job is None or not job.path.is_file():
            return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
        ctrl = _ctrl()
        if ctrl is None:
            return _DISCONNECTED
        busy = _start_busy()
        if busy is not None:
            return busy
        body = await _json_body(request)
        opts = _print_opts(body)
        safe_name = _safe_remote_name(job.out_name)

        def _upload_and_start():
            with open(job.path, "rb") as fh:
                return ctrl.upload_and_start(
                    src=fh,
                    remote_name=safe_name,
                    subtask_name=safe_name.rsplit(".", 1)[0],
                    **opts,
                )

        try:
            ok, seq = await run_in_threadpool(_upload_and_start)
        except FtpsUploadError as exc:
            return JSONResponse({"ok": False, "error": "ftps_failed", "detail": str(exc)}, status_code=502)
        return JSONResponse({"ok": ok, "sequence_id": seq, "remote_name": safe_name}, status_code=200 if ok else 502)

    @app.get("/api/slice/{job_id}/download")
    def slice_download(job_id: str) -> Response:
        job = slicer.get_job(job_id)
        if job is None or not job.path.is_file():
            return Response(status_code=404, content=b"unknown job")
        data = job.path.read_bytes()
        return Response(
            content=data,
            media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
            headers={"Content-Disposition": f'attachment; filename="{job.out_name}"'},
        )

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        q = state.subscribe()

        async def gen():
            try:
                yield ": connected\n\n"
                # Send initial state so client doesn't need to GET snapshot/detector first.
                init_snap = {"type": "snapshot", "ts": time.time(), "data": state.snapshot()}
                init_det = {"type": "detector", "ts": time.time(), "data": state.detector()}
                yield f"data: {json.dumps(init_snap)}\n\n"
                yield f"data: {json.dumps(init_det)}\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {json.dumps(item)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                state.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


class ApiServer:
    """Run uvicorn in a background thread."""

    def __init__(self, app: FastAPI, *, host: str = "0.0.0.0", port: int = 8000) -> None:
        config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=True)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="api-server", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
