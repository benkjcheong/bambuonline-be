from __future__ import annotations

from typing import Any

from .events import Event

# gcode_state values seen in the wild: IDLE, PREPARE, RUNNING, PAUSE, FINISH, FAILED
_ACTIVE_STATES = {"RUNNING", "PAUSE", "PREPARE"}

# Subset of common Bambu HMS codes. Full list: https://e.bambulab.com/query.php?lang=en
# Codes are normalized to 4-block uppercase form: AAAA_BBBB_CCCC_DDDD.
_HMS_DESCRIPTIONS: dict[str, str] = {
    "0300_0E00_0002_0001": "Filament runout",
    "0300_0F00_0003_0001": "Filament tangle / extruder clog",
    "0500_0100_0001_0001": "Bed leveling failed",
    "0700_8003_0001_0001": "Nozzle temperature abnormal",
    "0C00_0100_0001_0001": "Door open",
    "1200_0100_0001_0001": "AMS filament runout",
}


def format_hms_code(raw: dict | str) -> str:
    """Normalize HMS entry to 'AAAA_BBBB_CCCC_DDDD' uppercase hex string."""
    if isinstance(raw, str):
        return raw.upper()
    attr = int(raw.get("attr", 0))
    code = int(raw.get("code", 0))
    a = (attr >> 16) & 0xFFFF
    b = attr & 0xFFFF
    c = (code >> 16) & 0xFFFF
    d = code & 0xFFFF
    return f"{a:04X}_{b:04X}_{c:04X}_{d:04X}"


def describe(code: str) -> str:
    return _HMS_DESCRIPTIONS.get(code, "Unknown HMS code — see e.bambulab.com/query.php")


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


class StateTracker:
    """Merge partial MQTT updates into a running snapshot, emit diff events."""

    def __init__(self) -> None:
        self.snapshot: dict[str, Any] = {}
        self.active_hms: set[str] = set()
        self.primed: bool = False  # silence first-message events (cold start)

    def ingest(self, payload: dict[str, Any]) -> list[Event]:
        """payload is the full MQTT message body. Returns events triggered by the diff."""
        print_section = payload.get("print")
        if not isinstance(print_section, dict):
            return []

        prev_gcode = self.snapshot.get("gcode_state")
        prev_print_error = self.snapshot.get("print_error", 0)

        _deep_merge(self.snapshot, print_section)
        events: list[Event] = []

        if not self.primed:
            self.primed = True
            # seed active_hms from initial snapshot without firing
            self.active_hms = self._current_hms_codes()
            return events

        # gcode_state transitions
        new_gcode = self.snapshot.get("gcode_state")
        if new_gcode != prev_gcode and new_gcode is not None:
            ev = self._state_change_event(new_gcode, prev_gcode)
            if ev is not None:
                events.append(ev)

        # print_error (non-zero, newly set)
        new_err = self.snapshot.get("print_error", 0)
        if new_err and new_err != prev_print_error:
            events.append(
                Event(
                    kind="print_error",
                    title="Print error code raised",
                    detail=f"print_error={new_err}",
                    file=self.snapshot.get("subtask_name"),
                    layer=self._maybe_int("layer_num"),
                    layer_total=self._maybe_int("total_layer_num"),
                    percent=self._maybe_int("mc_percent"),
                )
            )

        # HMS diff
        current = self._current_hms_codes()
        new_codes = current - self.active_hms
        for code in sorted(new_codes):
            events.append(
                Event(
                    kind="hms",
                    title="HMS alert",
                    detail=describe(code),
                    hms_code=code,
                    file=self.snapshot.get("subtask_name"),
                    layer=self._maybe_int("layer_num"),
                    layer_total=self._maybe_int("total_layer_num"),
                    percent=self._maybe_int("mc_percent"),
                )
            )
        self.active_hms = current

        return events

    def _current_hms_codes(self) -> set[str]:
        hms = self.snapshot.get("hms")
        if not isinstance(hms, list):
            return set()
        return {format_hms_code(item) for item in hms if isinstance(item, (dict, str))}

    def _maybe_int(self, key: str) -> int | None:
        v = self.snapshot.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _state_change_event(self, new: str, prev: str | None) -> Event | None:
        titles = {
            "RUNNING": ("print_start", "Print started")
            if prev in (None, "IDLE", "FINISH", "FAILED", "PREPARE")
            else ("print_resume", "Print resumed"),
            "PAUSE": ("print_pause", "Print paused"),
            "FINISH": ("print_finish", "Print finished"),
            "FAILED": ("print_fail", "Print failed"),
        }
        entry = titles.get(new)
        if entry is None:
            return None
        kind, title = entry
        return Event(
            kind=kind,
            title=title,
            file=self.snapshot.get("subtask_name"),
            layer=self._maybe_int("layer_num"),
            layer_total=self._maybe_int("total_layer_num"),
            percent=self._maybe_int("mc_percent"),
        )
