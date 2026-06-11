from __future__ import annotations

import itertools
import json
import logging
import threading
from typing import TYPE_CHECKING, BinaryIO

import paho.mqtt.client as mqtt

from .events import Event
from .ftps_upload import FtpsUploadError, delete_file, list_3mf_files, upload_3mf

if TYPE_CHECKING:
    from .api import AppState

log = logging.getLogger(__name__)

VALID_ACTIONS = ("pause", "resume", "stop")
VALID_LIGHT_MODES = ("on", "off")

# Bambu fan indices for M106 P<n>
FAN_INDEX = {"part": 1, "aux": 2, "chamber": 3}

# print_speed levels: 1 silent, 2 standard, 3 sport, 4 ludicrous
SPEED_LEVELS = {1: "Silent", 2: "Standard", 3: "Sport", 4: "Ludicrous"}

JOG_AXES = ("X", "Y", "Z")


class PrinterControl:
    def __init__(
        self,
        client: mqtt.Client,
        serial: str,
        app_state: "AppState",
        *,
        printer_ip: str,
        access_code: str,
    ) -> None:
        self._client = client
        self._app_state = app_state
        self._printer_ip = printer_ip
        self._access_code = access_code
        self._topic = f"device/{serial}/request"
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    def is_connected(self) -> bool:
        try:
            return bool(self._client.is_connected())
        except Exception:
            return False

    def publish_command(self, action: str, *, source: str = "api") -> tuple[bool, str]:
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid action: {action}")
        with self._lock:
            seq = str(next(self._seq))
        payload = {"print": {"sequence_id": seq, "command": action}}
        info = self._client.publish(self._topic, json.dumps(payload), qos=0)
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        log.info("control %s seq=%s source=%s ok=%s", action, seq, source, ok)
        self._app_state.push_event(
            Event(
                kind=f"control_{action}",
                title=f"Print {action}" + (" (auto)" if source == "auto_pause" else ""),
                detail=f"source={source} seq={seq} ok={ok}",
            )
        )
        return ok, seq

    def set_light(self, mode: str, *, node: str = "chamber_light") -> tuple[bool, str]:
        if mode not in VALID_LIGHT_MODES:
            raise ValueError(f"invalid light mode: {mode}")
        with self._lock:
            seq = str(next(self._seq))
        payload = {
            "system": {
                "sequence_id": seq,
                "command": "ledctrl",
                "led_node": node,
                "led_mode": mode,
                "led_on_time": 500,
                "led_off_time": 500,
                "loop_times": 0,
                "interval_time": 0,
            }
        }
        info = self._client.publish(self._topic, json.dumps(payload), qos=0)
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        log.info("ledctrl node=%s mode=%s seq=%s ok=%s", node, mode, seq, ok)
        self._app_state.push_event(
            Event(
                kind=f"light_{mode}",
                title=f"Light {mode}",
                detail=f"node={node} seq={seq} ok={ok}",
            )
        )
        return ok, seq

    def send_gcode(self, lines: str | list[str], *, source: str = "api", event_title: str | None = None) -> tuple[bool, str]:
        if isinstance(lines, str):
            lines = [lines]
        param = "\n".join(line.strip() for line in lines if line.strip()) + "\n"
        with self._lock:
            seq = str(next(self._seq))
        payload = {"print": {"sequence_id": seq, "command": "gcode_line", "param": param}}
        info = self._client.publish(self._topic, json.dumps(payload), qos=0)
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        log.info("gcode_line seq=%s source=%s ok=%s param=%r", seq, source, ok, param)
        if event_title:
            self._app_state.push_event(
                Event(kind="control_gcode", title=event_title, detail=f"source={source} seq={seq} ok={ok}")
            )
        return ok, seq

    def set_nozzle_temp(self, target: int) -> tuple[bool, str]:
        target = max(0, min(int(target), 300))
        return self.send_gcode(f"M104 S{target}", event_title=f"Nozzle target {target}°C")

    def set_bed_temp(self, target: int) -> tuple[bool, str]:
        target = max(0, min(int(target), 110))
        return self.send_gcode(f"M140 S{target}", event_title=f"Bed target {target}°C")

    def set_fan(self, fan: str, percent: int) -> tuple[bool, str]:
        idx = FAN_INDEX[fan]
        percent = max(0, min(int(percent), 100))
        s = round(percent * 255 / 100)
        return self.send_gcode(f"M106 P{idx} S{s}", event_title=f"{fan.capitalize()} fan {percent}%")

    def set_speed_level(self, level: int) -> tuple[bool, str]:
        if level not in SPEED_LEVELS:
            raise ValueError(f"invalid speed level: {level}")
        with self._lock:
            seq = str(next(self._seq))
        payload = {"print": {"sequence_id": seq, "command": "print_speed", "param": str(level)}}
        info = self._client.publish(self._topic, json.dumps(payload), qos=0)
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        log.info("print_speed level=%s seq=%s ok=%s", level, seq, ok)
        self._app_state.push_event(
            Event(kind="control_speed", title=f"Speed: {SPEED_LEVELS[level]}", detail=f"level={level} seq={seq} ok={ok}")
        )
        return ok, seq

    def home(self) -> tuple[bool, str]:
        return self.send_gcode("G28", event_title="Homing axes")

    def jog(self, axis: str, dist: float, *, feed: int | None = None) -> tuple[bool, str]:
        axis = axis.upper()
        if axis not in JOG_AXES:
            raise ValueError(f"invalid axis: {axis}")
        dist = max(-100.0, min(float(dist), 100.0))
        if feed is None:
            feed = 900 if axis == "Z" else 3000
        return self.send_gcode([
            "M211 S",
            "M211 X1 Y1 Z1",
            "M1002 push_ref_mode",
            "G91",
            f"G1 {axis}{dist:.1f} F{feed}",
            "M1002 pop_ref_mode",
            "G90",
        ])

    def extrude(self, dist: float) -> tuple[bool, str]:
        dist = max(-50.0, min(float(dist), 50.0))
        return self.send_gcode(["M83", f"G1 E{dist:.1f} F300"])

    def list_sd_files(self) -> list[dict]:
        return list_3mf_files(host=self._printer_ip, access_code=self._access_code)

    def delete_sd_file(self, path: str) -> None:
        delete_file(host=self._printer_ip, access_code=self._access_code, path=path)
        self._app_state.push_event(Event(kind="file_delete", title="File deleted", file=path))

    def _publish_project_file(
        self,
        *,
        remote_path: str,
        subtask_name: str,
        plate: int,
        use_ams: bool,
        bed_leveling: bool,
        flow_cali: bool,
        vibration_cali: bool,
        layer_inspect: bool,
        timelapse: bool,
    ) -> tuple[bool, str]:
        with self._lock:
            seq = str(next(self._seq))
        payload = {
            "print": {
                "sequence_id": seq,
                "command": "project_file",
                "param": f"Metadata/plate_{plate}.gcode",
                "subtask_name": subtask_name,
                "url": f"ftp://{remote_path}",
                "bed_type": "auto",
                "bed_leveling": bed_leveling,
                "flow_cali": flow_cali,
                "vibration_cali": vibration_cali,
                "layer_inspect": layer_inspect,
                "timelapse": timelapse,
                "use_ams": use_ams,
                "ams_mapping": [],
                "profile_id": "0",
                "project_id": "0",
                "subtask_id": "0",
                "task_id": "0",
            }
        }
        info = self._client.publish(self._topic, json.dumps(payload), qos=0)
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        log.info("project_file seq=%s path=%s ok=%s", seq, remote_path, ok)
        self._app_state.push_event(
            Event(
                kind="print_start",
                title="Print started",
                detail=f"plate={plate} use_ams={use_ams} seq={seq} ok={ok}",
                file=remote_path,
            )
        )
        return ok, seq

    def start_existing(
        self,
        *,
        remote_path: str,
        plate: int = 1,
        use_ams: bool = False,
        bed_leveling: bool = True,
        flow_cali: bool = False,
        vibration_cali: bool = True,
        layer_inspect: bool = False,
        timelapse: bool = False,
    ) -> tuple[bool, str]:
        subtask = remote_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return self._publish_project_file(
            remote_path=remote_path,
            subtask_name=subtask,
            plate=plate,
            use_ams=use_ams,
            bed_leveling=bed_leveling,
            flow_cali=flow_cali,
            vibration_cali=vibration_cali,
            layer_inspect=layer_inspect,
            timelapse=timelapse,
        )

    def upload_and_start(
        self,
        *,
        src: BinaryIO,
        remote_name: str,
        subtask_name: str | None = None,
        plate: int = 1,
        use_ams: bool = False,
        bed_leveling: bool = True,
        flow_cali: bool = False,
        vibration_cali: bool = True,
        layer_inspect: bool = False,
        timelapse: bool = False,
    ) -> tuple[bool, str]:
        try:
            upload_3mf(
                host=self._printer_ip,
                access_code=self._access_code,
                src=src,
                remote_name=remote_name,
            )
        except FtpsUploadError as exc:
            self._app_state.push_event(
                Event(
                    kind="print_start_failed",
                    title="Print upload failed",
                    detail=str(exc),
                    file=remote_name,
                )
            )
            raise

        return self._publish_project_file(
            remote_path=remote_name,
            subtask_name=subtask_name or remote_name.rsplit(".", 1)[0],
            plate=plate,
            use_ams=use_ams,
            bed_leveling=bed_leveling,
            flow_cali=flow_cali,
            vibration_cali=vibration_cali,
            layer_inspect=layer_inspect,
            timelapse=timelapse,
        )
