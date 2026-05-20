from __future__ import annotations

import ctypes as c
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_BAMBU_SUCCESS = 0
_BAMBU_STREAM_END = 1
_BAMBU_WOULD_BLOCK = 2
_BAMBU_BUFFER_LIMIT = 3
_BAMBU_STREAM_TYPE_VIDEO = 0
_BAMBU_VIDEO_SUBTYPE_AVC1 = 0
_BAMBU_VIDEO_SUBTYPE_MJPG = 1
_BAMBU_FORMAT_VIDEO_AVC_PACKET = 0
_BAMBU_FORMAT_VIDEO_AVC_BYTE_STREAM = 1
_BAMBU_FORMAT_VIDEO_JPEG = 2
_BAMBU_FLAG_SYNC = 1
_MAX_PENDING_VIDEO_BYTES = 4 * 1024 * 1024
_MIN_DECODE_ATTEMPT_BYTES = 4 * 1024


class CameraUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    printer_ip: str
    access_code: str
    user: str = "bblp"
    port: int = 6000
    library_path: str | None = None


class _Sample(c.Structure):
    _fields_ = [
        ("itrack", c.c_int),
        ("size", c.c_int),
        ("flags", c.c_int),
        ("buffer", c.POINTER(c.c_ubyte)),
        ("decode_time", c.c_ulonglong),
    ]


class _VideoFormat(c.Structure):
    _fields_ = [
        ("width", c.c_int),
        ("height", c.c_int),
        ("frame_rate", c.c_int),
    ]


class _AudioFormat(c.Structure):
    _fields_ = [
        ("sample_rate", c.c_int),
        ("channel_count", c.c_int),
        ("sample_size", c.c_int),
    ]


class _StreamFormat(c.Union):
    _fields_ = [
        ("video", _VideoFormat),
        ("audio", _AudioFormat),
    ]


class _StreamInfo(c.Structure):
    _fields_ = [
        ("type", c.c_int),
        ("sub_type", c.c_int),
        ("format", _StreamFormat),
        ("format_type", c.c_int),
        ("format_size", c.c_int),
        ("max_frame_size", c.c_int),
        ("format_buffer", c.POINTER(c.c_ubyte)),
    ]


@dataclass(frozen=True)
class _VideoStreamInfo:
    sub_type: int
    format_type: int
    width: int
    height: int
    frame_rate: int
    max_frame_size: int


class BambuCameraClient:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._lib = self._load_library(config.library_path)
        self._configure_api()
        self.last_capture_diagnostic: str | None = None

    @staticmethod
    def default_library_path() -> str:
        return str(Path.home() / "Library/Application Support/BambuStudio/plugins/libBambuSource.dylib")

    def _load_library(self, override: str | None):
        path = Path(override or self.default_library_path())
        if not path.exists():
            raise CameraUnavailableError(
                f"Bambu camera library not found at {path}. Install Bambu Studio or set CAMERA_LIBRARY_PATH."
            )
        try:
            return c.CDLL(str(path))
        except OSError as exc:
            raise CameraUnavailableError(f"Failed to load camera library at {path}: {exc}") from exc

    def _configure_api(self) -> None:
        self._lib.Bambu_Init.restype = c.c_int
        self._lib.Bambu_Deinit.restype = None
        self._lib.Bambu_Create.argtypes = [c.POINTER(c.c_void_p), c.c_char_p]
        self._lib.Bambu_Create.restype = c.c_int
        self._lib.Bambu_Open.argtypes = [c.c_void_p]
        self._lib.Bambu_Open.restype = c.c_int
        self._lib.Bambu_StartStream.argtypes = [c.c_void_p, c.c_bool]
        self._lib.Bambu_StartStream.restype = c.c_int
        self._lib.Bambu_GetStreamCount.argtypes = [c.c_void_p]
        self._lib.Bambu_GetStreamCount.restype = c.c_int
        self._lib.Bambu_GetStreamInfo.argtypes = [c.c_void_p, c.c_int, c.POINTER(_StreamInfo)]
        self._lib.Bambu_GetStreamInfo.restype = c.c_int
        self._lib.Bambu_ReadSample.argtypes = [c.c_void_p, c.POINTER(_Sample)]
        self._lib.Bambu_ReadSample.restype = c.c_int
        self._lib.Bambu_Close.argtypes = [c.c_void_p]
        self._lib.Bambu_Close.restype = None
        self._lib.Bambu_Destroy.argtypes = [c.c_void_p]
        self._lib.Bambu_Destroy.restype = None
        self._lib.Bambu_GetLastErrorMsg.restype = c.c_char_p

    def capture_frame(self, timeout_sec: float = 10.0) -> bytes | None:
        self.last_capture_diagnostic = None
        handle = c.c_void_p()
        url = (
            f"bambu:///local/{self._config.printer_ip}"
            f"?port={self._config.port}&user={self._config.user}&passwd={self._config.access_code}"
        ).encode("utf-8")
        rc = self._lib.Bambu_Init()
        if rc != _BAMBU_SUCCESS:
            raise CameraUnavailableError(f"Bambu_Init failed with rc={rc}")
        try:
            self._require_ok(self._lib.Bambu_Create(c.byref(handle), url), "Bambu_Create")
            self._require_ok(self._lib.Bambu_Open(handle), "Bambu_Open")
            deadline = time.monotonic() + timeout_sec
            if not self._start_stream(handle, deadline):
                return None
            stream_info = self._get_video_stream_info(handle)
            if stream_info:
                log.info(
                    "camera stream format: subtype=%s format_type=%s size=%sx%s fps=%s max_frame_size=%s",
                    self._describe_video_subtype(stream_info.sub_type),
                    self._describe_format_type(stream_info.format_type),
                    stream_info.width,
                    stream_info.height,
                    stream_info.frame_rate,
                    stream_info.max_frame_size,
                )

            pending_video = bytearray()
            saw_sample_bytes = False
            while time.monotonic() < deadline:
                sample = _Sample()
                sample_rc = self._lib.Bambu_ReadSample(handle, c.byref(sample))
                if sample_rc == _BAMBU_SUCCESS and sample.size > 0:
                    saw_sample_bytes = True
                    sample_bytes = c.string_at(sample.buffer, sample.size)
                    if self._is_jpeg_sample(sample_bytes, stream_info):
                        return sample_bytes

                    pending_video.extend(sample_bytes)
                    if len(pending_video) > _MAX_PENDING_VIDEO_BYTES:
                        del pending_video[:-_MAX_PENDING_VIDEO_BYTES]

                    if self._should_attempt_video_decode(sample, pending_video, stream_info):
                        frame = self._decode_h264_to_jpeg(bytes(pending_video))
                        if frame:
                            return frame
                        log.debug(
                            "camera sample was not directly decodable yet (%d buffered bytes, prefix=%r)",
                            len(pending_video),
                            sample_bytes[:8],
                        )
                elif sample_rc == _BAMBU_STREAM_END:
                    break
                elif sample_rc not in {_BAMBU_WOULD_BLOCK, _BAMBU_BUFFER_LIMIT}:
                    self.last_capture_diagnostic = f"Bambu_ReadSample failed with rc={sample_rc}"
                    self._raise_last_error("Bambu_ReadSample", sample_rc)
                time.sleep(0.1)

            if pending_video:
                frame = self._decode_h264_to_jpeg(bytes(pending_video))
                if frame:
                    return frame

            if not saw_sample_bytes:
                self.last_capture_diagnostic = (
                    "Timed out waiting for liveview bytes from Bambu_ReadSample. "
                    "Close Bambu Studio/Handy/Obico or any other liveview client, and make sure liveview is enabled."
                )
            else:
                self.last_capture_diagnostic = (
                    "Received liveview video samples, but they were not JPEG frames and could not be decoded. "
                    "Bambu/Orca liveview typically delivers H.264, not standalone JPEGs."
                )
            return None
        finally:
            if handle.value:
                self._lib.Bambu_Close(handle)
                self._lib.Bambu_Destroy(handle)
            self._lib.Bambu_Deinit()

    def _start_stream(self, handle: c.c_void_p, deadline: float) -> bool:
        while time.monotonic() < deadline:
            start_rc = self._lib.Bambu_StartStream(handle, True)
            if start_rc == _BAMBU_SUCCESS:
                return True
            if start_rc != _BAMBU_WOULD_BLOCK:
                self._raise_last_error("Bambu_StartStream", start_rc)
            time.sleep(0.1)

        self.last_capture_diagnostic = (
            "Bambu_StartStream never became ready before timeout. "
            "The printer camera may be busy, disabled, or already in use by another client."
        )
        return False

    def _get_video_stream_info(self, handle: c.c_void_p) -> _VideoStreamInfo | None:
        stream_count = self._lib.Bambu_GetStreamCount(handle)
        for idx in range(max(0, stream_count)):
            info = _StreamInfo()
            if self._lib.Bambu_GetStreamInfo(handle, idx, c.byref(info)) != _BAMBU_SUCCESS:
                continue
            if info.type != _BAMBU_STREAM_TYPE_VIDEO:
                continue
            return _VideoStreamInfo(
                sub_type=info.sub_type,
                format_type=info.format_type,
                width=info.format.video.width,
                height=info.format.video.height,
                frame_rate=info.format.video.frame_rate,
                max_frame_size=info.max_frame_size,
            )
        return None

    @staticmethod
    def _is_jpeg_sample(sample_bytes: bytes, stream_info: _VideoStreamInfo | None) -> bool:
        if sample_bytes[:2] == b"\xff\xd8":
            return True
        return bool(stream_info and stream_info.format_type == _BAMBU_FORMAT_VIDEO_JPEG)

    @staticmethod
    def _should_attempt_video_decode(
        sample: _Sample, pending_video: bytearray, stream_info: _VideoStreamInfo | None
    ) -> bool:
        if len(pending_video) < _MIN_DECODE_ATTEMPT_BYTES:
            return False
        if sample.flags & _BAMBU_FLAG_SYNC:
            return True
        if pending_video[:4] == b"\x00\x00\x00\x01" or pending_video[:3] == b"\x00\x00\x01":
            return True
        if stream_info and stream_info.max_frame_size > 0 and len(pending_video) >= stream_info.max_frame_size:
            return True
        return len(pending_video) >= 256 * 1024

    def _decode_h264_to_jpeg(self, video_bytes: bytes) -> bytes | None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise CameraUnavailableError(
                "ffmpeg is required to decode Bambu liveview H.264 samples into a JPEG frame."
            )

        proc = subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ],
            input=video_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout[:2] == b"\xff\xd8":
            return proc.stdout
        if proc.stderr:
            log.debug("ffmpeg could not decode buffered liveview sample: %s", proc.stderr.decode("utf-8", "ignore"))
        return None

    @staticmethod
    def _describe_video_subtype(sub_type: int) -> str:
        return {
            _BAMBU_VIDEO_SUBTYPE_AVC1: "AVC1",
            _BAMBU_VIDEO_SUBTYPE_MJPG: "MJPG",
        }.get(sub_type, str(sub_type))

    @staticmethod
    def _describe_format_type(format_type: int) -> str:
        return {
            _BAMBU_FORMAT_VIDEO_AVC_PACKET: "video_avc_packet",
            _BAMBU_FORMAT_VIDEO_AVC_BYTE_STREAM: "video_avc_byte_stream",
            _BAMBU_FORMAT_VIDEO_JPEG: "video_jpeg",
        }.get(format_type, str(format_type))

    def _require_ok(self, rc: int, call_name: str) -> None:
        if rc != _BAMBU_SUCCESS:
            self._raise_last_error(call_name, rc)

    def _raise_last_error(self, call_name: str, rc: int) -> None:
        msg = self._lib.Bambu_GetLastErrorMsg()
        detail = msg.decode("utf-8", errors="ignore") if msg else "unknown error"
        raise CameraUnavailableError(f"{call_name} failed with rc={rc}: {detail}")
