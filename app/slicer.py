from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_BIN = "/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"
_DEFAULT_PROFILE_DIR = "/Applications/BambuStudio.app/Contents/Resources/profiles/BBL"
_DEFAULT_MODEL_NAME = "Bambu Lab A1 mini"
_DEFAULT_MODEL_CODE = "A1M"
_DEFAULT_MACHINE = "Bambu Lab A1 mini 0.4 nozzle"
_DEFAULT_PROCESS = "0.20mm Standard @BBL A1M"
_DEFAULT_FILAMENT = "Alt Tab PLA Basic @Bambu Lab A1 mini 0.4 nozzle"
# Custom (user) filament presets live here, not in the app bundle's BBL profiles.
_DEFAULT_USER_PROFILE_DIR = str(Path.home() / "Library/Application Support/BambuStudio/user")
# Presets vendored alongside the repo (bambuonline-be/presets/), so the app
# doesn't depend on a local Bambu Studio install having them.
_DEFAULT_REPO_PRESET_DIR = str(Path(__file__).resolve().parent.parent / "presets")

SUPPORTED_INPUTS = (".stl", ".3mf", ".step", ".stp", ".obj")


class SlicerError(RuntimeError):
    pass


@dataclass
class SliceJob:
    job_id: str
    path: Path
    source_name: str
    out_name: str
    result: dict[str, Any] = field(default_factory=dict)
    workdir: Path | None = None
    filament_density: float = 1.24  # g/cm³, PLA-ish fallback


class Slicer:
    """Wraps the Bambu Studio CLI to slice models into printable gcode-3MF files."""

    def __init__(self) -> None:
        self.binary = os.environ.get("SLICER_BIN", "").strip() or _DEFAULT_BIN
        self.profile_dir = Path(os.environ.get("SLICER_PROFILE_DIR", "").strip() or _DEFAULT_PROFILE_DIR)
        self.user_profile_dir = Path(
            os.environ.get("SLICER_USER_PROFILE_DIR", "").strip() or _DEFAULT_USER_PROFILE_DIR
        ).expanduser()
        # Filament presets vendored into the repo — the canonical source of truth,
        # surviving a Bambu Studio reinstall/cloud-sync. Overrides same-named user presets.
        self.repo_preset_dir = Path(
            os.environ.get("SLICER_REPO_PRESET_DIR", "").strip() or _DEFAULT_REPO_PRESET_DIR
        ).expanduser()
        self.model_code = os.environ.get("SLICER_MODEL_CODE", "").strip() or _DEFAULT_MODEL_CODE
        self.model_name = os.environ.get("SLICER_MODEL_NAME", "").strip() or _DEFAULT_MODEL_NAME
        self.default_machine = os.environ.get("SLICER_MACHINE", "").strip() or _DEFAULT_MACHINE
        self.default_process = os.environ.get("SLICER_PROCESS", "").strip() or _DEFAULT_PROCESS
        self.default_filament = os.environ.get("SLICER_FILAMENT", "").strip() or _DEFAULT_FILAMENT
        self.timeout_sec = float(os.environ.get("SLICER_TIMEOUT_SEC", "300"))
        self._run_lock = threading.Lock()
        self._jobs: OrderedDict[str, SliceJob] = OrderedDict()
        self._jobs_lock = threading.Lock()
        self._max_jobs = 8
        self._settings_cache: dict[tuple[str, str, str], bytes] = {}

    def available(self) -> bool:
        return os.path.exists(self.binary) and self.profile_dir.is_dir()

    # ---- presets -----------------------------------------------------------

    def list_presets(self) -> dict[str, Any]:
        code = re.escape(self.model_code)
        # "@BBL A1" must not match "@BBL A1M" — require end or nozzle qualifier.
        suffix = re.compile(rf"@BBL {code}( \d+\.\d+ nozzle)?\.json$")
        machine_re = re.compile(rf"^{re.escape(self.model_name)} \d+\.\d+ nozzle\.json$")

        def names(subdir: str, pred) -> list[str]:
            d = self.profile_dir / subdir
            if not d.is_dir():
                return []
            return sorted(p.stem for p in d.glob("*.json") if pred(p.name))

        return {
            "available": self.available(),
            "machine": names("machine", lambda n: machine_re.match(n)),
            "process": names("process", lambda n: suffix.search(n)),
            # Filament list = the user's own custom presets only; stock Bambu/eSun
            # profiles are intentionally hidden.
            "filament": sorted(self._filament_preset_map()),
            "defaults": {
                "machine": self.default_machine,
                "process": self.default_process,
                "filament": self.default_filament,
            },
        }

    def _filament_preset_map(self) -> dict[str, Path]:
        """Map custom filament preset name -> json path.

        Scans two sources: the user's Bambu Studio profile dir and the presets
        vendored in this repo. Repo presets win on a name collision, so the
        committed copy is authoritative regardless of local Studio state. Only
        presets compatible with the configured machine are included.
        """
        out: dict[str, Path] = {}
        # User dir first (lower priority), then repo presets override by name.
        sources = [
            self.user_profile_dir.glob("*/filament/**/*.json") if self.user_profile_dir.is_dir() else [],
            (self.repo_preset_dir / "filament").glob("*.json") if self.repo_preset_dir.is_dir() else [],
        ]
        for paths in sources:
            for p in paths:
                try:
                    data = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                compatible = data.get("compatible_printers") or []
                if isinstance(compatible, str):
                    compatible = [compatible]
                if compatible and self.default_machine not in compatible:
                    continue
                name = data.get("name") or p.stem
                out[str(name)] = p
        return out

    def _filament_path(self, name: str) -> Path:
        safe = (name or "").strip()
        if not safe:
            raise SlicerError("filament preset name is empty")
        presets = self._filament_preset_map()
        p = presets.get(safe)
        if p is None or not p.is_file():
            raise SlicerError(f"unknown filament preset: {safe}")
        return p

    def _filament_density(self, name: str) -> float:
        """Walk the preset's 'inherits' chain for filament_density; CLI output drops it."""
        seen: set[str] = set()
        custom_presets = self._filament_preset_map()
        current = name
        while current and current not in seen:
            seen.add(current)
            # Custom presets live in the user dir; inherited parents may fall
            # back to the stock BBL profiles.
            p = custom_presets.get(current) or self.profile_dir / "filament" / f"{current}.json"
            if not p.is_file():
                break
            try:
                data = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                break
            val = data.get("filament_density")
            if isinstance(val, list) and val:
                val = val[0]
            try:
                density = float(val)
                if density > 0:
                    return density
            except (TypeError, ValueError):
                pass
            current = data.get("inherits") or ""
        return 1.24

    # ---- project settings injection -----------------------------------------

    def _export_settings_json(self, machine_p: Path, process_p: Path, filament_p: Path) -> bytes:
        """Merged preset dump via CLI --export-settings; cached per preset triple.

        Project 3mfs missing Metadata/project_settings.config segfault the CLI
        importer, so every uploaded 3mf gets one injected before slicing.
        """
        key = (str(machine_p), str(process_p), str(filament_p))
        cached = self._settings_cache.get(key)
        if cached is not None:
            return cached
        with tempfile.TemporaryDirectory(prefix="bambuonline-ps-") as tmp:
            out = Path(tmp) / "ps.json"
            cmd = [
                self.binary,
                "--load-settings", f"{machine_p};{process_p}",
                "--load-filaments", str(filament_p),
                "--export-settings", str(out),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                raise SlicerError("settings export timed out")
            if proc.returncode != 0 or not out.is_file():
                detail = (proc.stderr or proc.stdout or "").strip()[-300:]
                raise SlicerError(f"settings export failed (rc={proc.returncode}): {detail}")
            data = out.read_bytes()
        if len(self._settings_cache) > 16:
            self._settings_cache.clear()
        self._settings_cache[key] = data
        return data

    def _inject_project_settings(self, src_path: Path, machine_p: Path, process_p: Path, filament_p: Path) -> None:
        """Add Metadata/project_settings.config to a 3mf that lacks it (in place)."""
        import zipfile

        try:
            with zipfile.ZipFile(src_path) as z:
                if "Metadata/project_settings.config" in z.namelist():
                    return
        except zipfile.BadZipFile:
            return  # let the CLI produce its own error
        data = self._export_settings_json(machine_p, process_p, filament_p)
        with zipfile.ZipFile(src_path, "a", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("Metadata/project_settings.config", data)

    def _preset_path(self, subdir: str, name: str) -> Path:
        safe = name.strip()
        if not safe or "/" in safe or "\\" in safe or safe.startswith("."):
            raise SlicerError(f"invalid preset name: {name!r}")
        p = self.profile_dir / subdir / f"{safe}.json"
        if not p.is_file():
            raise SlicerError(f"unknown {subdir} preset: {safe}")
        return p

    # ---- slicing -----------------------------------------------------------

    def slice_file(
        self,
        src_path: Path,
        source_name: str,
        *,
        machine: str | None = None,
        process: str | None = None,
        filament: str | None = None,
        scale: float | None = None,
        arrange: str = "1",
        orient: str = "2",
    ) -> SliceJob:
        if not self.available():
            raise SlicerError(f"slicer binary not found at {self.binary}")
        machine_p = self._preset_path("machine", machine or self.default_machine)
        process_p = self._preset_path("process", process or self.default_process)
        filament_p = self._filament_path(filament or self.default_filament)

        if src_path.suffix.lower() == ".3mf":
            self._inject_project_settings(src_path, machine_p, process_p, filament_p)

        workdir = Path(tempfile.mkdtemp(prefix="spaghetteye-slice-"))
        out_dir = workdir / "out"
        base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(source_name).stem) or "model"
        out_name = f"{base}_sliced.3mf"

        cmd = [
            self.binary,
            str(src_path),
            "--load-settings", f"{machine_p};{process_p}",
            "--load-filaments", str(filament_p),
            "--arrange", arrange if arrange in ("0", "1", "2") else "1",
            "--orient", orient if orient in ("0", "1", "2") else "2",
            "--ensure-on-bed",
            "--slice", "0",
            "--export-3mf", out_name,
            "--outputdir", str(out_dir),
            "--debug", "1",
        ]
        if scale and scale > 0 and abs(scale - 1.0) > 1e-6:
            cmd += ["--scale", str(scale)]

        log.info("slicing %s (machine=%s process=%s filament=%s)", source_name, machine_p.stem, process_p.stem, filament_p.stem)
        with self._run_lock:  # Bambu CLI is heavyweight; one slice at a time
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                shutil.rmtree(workdir, ignore_errors=True)
                raise SlicerError(f"slicer timed out after {self.timeout_sec:.0f}s")

        result: dict[str, Any] = {}
        result_file = out_dir / "result.json"
        if result_file.is_file():
            try:
                result = json.loads(result_file.read_text())
            except (OSError, json.JSONDecodeError):
                result = {}

        out_path = out_dir / out_name
        if proc.returncode != 0 or not out_path.is_file():
            detail = result.get("error_string") or (proc.stderr or proc.stdout or "").strip()[-500:]
            shutil.rmtree(workdir, ignore_errors=True)
            raise SlicerError(f"slicing failed (rc={proc.returncode}): {detail}")

        job = SliceJob(
            job_id=uuid.uuid4().hex[:12],
            path=out_path,
            source_name=source_name,
            out_name=out_name,
            result=result,
            workdir=workdir,
            filament_density=self._filament_density((filament or self.default_filament).strip()),
        )
        self._store(job)
        return job

    # ---- job store ---------------------------------------------------------

    def _store(self, job: SliceJob) -> None:
        with self._jobs_lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > self._max_jobs:
                _, old = self._jobs.popitem(last=False)
                if old.workdir is not None:
                    shutil.rmtree(old.workdir, ignore_errors=True)

    def get_job(self, job_id: str) -> SliceJob | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)


def _gcode_filament_grams(threemf_path: Path, density: float) -> float:
    """result.json under-reports filament use; derive from the gcode header inside the 3mf."""
    import zipfile

    try:
        with zipfile.ZipFile(threemf_path) as z:
            names = [n for n in z.namelist() if re.match(r"Metadata/plate_\d+\.gcode$", n)]
            total = 0.0
            for n in names:
                with z.open(n) as fh:
                    head = fh.read(16384).decode("utf-8", "replace")
                m = re.search(r"total filament weight \[g\] : ([\d., ]+)", head)
                if m:
                    grams = sum(float(x) for x in m.group(1).replace(",", " ").split())
                    if grams > 0:
                        total += grams
                        continue
                # weight header is 0 when density got lost — compute from length
                m = re.search(r"total filament length \[mm\] : ([\d.]+)", head)
                if m:
                    length_mm = float(m.group(1))
                    area_mm2 = 3.14159265 * (1.75 / 2) ** 2
                    total += length_mm * area_mm2 * density / 1000.0
            return total
    except Exception:
        return 0.0


def summarize_result(job: SliceJob) -> dict[str, Any]:
    """Extract the FE-relevant bits of the CLI's result.json."""
    result = job.result
    plates = result.get("sliced_plates") or []
    plate = plates[0] if plates else {}
    filaments = plate.get("filaments") or []
    grams = sum(float(f.get("total_used_g") or 0.0) for f in filaments if isinstance(f, dict))
    if not grams:
        grams = _gcode_filament_grams(job.path, job.filament_density)
    return {
        "estimate_sec": round(float(plate.get("total_predication") or 0.0)),
        "filament_g": round(grams, 2),
        "layer_height": result.get("layer_height"),
        "objects": [o.get("name") for o in (plate.get("objects") or []) if isinstance(o, dict)],
        "warning": plate.get("warning_message") or "",
    }
