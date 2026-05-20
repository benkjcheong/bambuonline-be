from __future__ import annotations

import itertools
import json
import logging
import threading
from typing import TYPE_CHECKING, BinaryIO

import paho.mqtt.client as mqtt

from .events import Event
from .ftps_upload import FtpsUploadError, upload_3mf

if TYPE_CHECKING:
    from .api import AppState

log = logging.getLogger(__name__)

VALID_ACTIONS = ("pause", "resume", "stop")
VALID_LIGHT_MODES = ("on", "off")


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

        with self._lock:
            seq = str(next(self._seq))
        payload = {
            "print": {
                "sequence_id": seq,
                "command": "project_file",
                "param": f"Metadata/plate_{plate}.gcode",
                "subtask_name": subtask_name or remote_name.rsplit(".", 1)[0],
                "url": f"ftp://{remote_name}",
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
        log.info("project_file seq=%s name=%s ok=%s", seq, remote_name, ok)
        self._app_state.push_event(
            Event(
                kind="print_start",
                title="Print started",
                detail=f"plate={plate} use_ams={use_ams} seq={seq} ok={ok}",
                file=remote_name,
            )
        )
        return ok, seq
