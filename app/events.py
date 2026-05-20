from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    kind: str
    title: str
    detail: str = ""
    file: str | None = None
    layer: int | None = None
    layer_total: int | None = None
    percent: int | None = None
    hms_code: str | None = None

    def format(self) -> str:
        lines = [f"[App] {self.title}"]
        if self.file:
            lines.append(f"File: {self.file}")
        if self.layer is not None and self.layer_total is not None:
            pct = f" ({self.percent}%)" if self.percent is not None else ""
            lines.append(f"Layer: {self.layer} / {self.layer_total}{pct}")
        elif self.percent is not None:
            lines.append(f"Progress: {self.percent}%")
        if self.hms_code:
            lines.append(f"HMS: {self.hms_code}{(' — ' + self.detail) if self.detail else ''}")
        elif self.detail:
            lines.append(self.detail)
        return "\n".join(lines)
