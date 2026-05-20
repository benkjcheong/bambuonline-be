from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import requests

_DEFAULT_MODEL_NAME = "obico-failure.onnx"


class DetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectionResult:
    failure_detected: bool
    confidence: float
    score: float
    box_count: int
    failure_type: str
    summary: str


class LocalFailureDetector:
    def __init__(
        self,
        *,
        model_url: str,
        model_path: str | None = None,
        threshold: float = 0.08,
        nms_threshold: float = 0.45,
    ) -> None:
        self._threshold = threshold
        self._nms_threshold = nms_threshold
        self._np = self._import_numpy()
        self._ort = self._import_onnxruntime()
        self._Image = self._import_pillow_image()
        self._model_path = self._ensure_model(model_url, model_path)
        self._session = self._ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._input_h = int(self._session.get_inputs()[0].shape[2])
        self._input_w = int(self._session.get_inputs()[0].shape[3])

    def detect(self, jpeg_bytes: bytes) -> DetectionResult:
        image = self._Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        width, height = image.size
        resized = image.resize((self._input_w, self._input_h), resample=self._Image.Resampling.BILINEAR)
        arr = self._np.asarray(resized, dtype=self._np.float32) / 255.0
        arr = self._np.transpose(arr, (2, 0, 1))[self._np.newaxis, ...]
        outputs = self._session.run(None, {self._input_name: arr})
        detections = self._post_process(outputs, width, height)
        if not detections:
            return DetectionResult(False, 0.0, 0.0, 0, "none", "No local failure detections")
        top = max(detections, key=lambda item: item[1])
        conf = float(top[1])
        boxes = len(detections)
        score = float(sum(item[1] for item in detections))
        return DetectionResult(
            failure_detected=True,
            confidence=conf,
            score=score,
            box_count=boxes,
            failure_type="failure",
            summary=(
                f"Local model flagged {boxes} failure region(s); "
                f"frame score {score:.2f}; top confidence {conf:.2f}"
            ),
        )

    def _post_process(self, outputs, width: int, height: int):
        np = self._np
        box_array = outputs[0]
        confs = outputs[1]
        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        box_array = box_array[:, :, 0]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        detections = []
        for i in range(box_array.shape[0]):
            mask = max_conf[i] > self._threshold
            boxes = box_array[i, mask, :]
            scores = max_conf[i, mask]
            classes = max_id[i, mask]
            for cls_id in range(confs.shape[2]):
                cls_mask = classes == cls_id
                cls_boxes = boxes[cls_mask, :]
                cls_scores = scores[cls_mask]
                if cls_boxes.size == 0:
                    continue
                keep = self._nms(cls_boxes, cls_scores, self._nms_threshold)
                for idx in keep:
                    box = cls_boxes[idx]
                    detections.append(
                        (
                            "failure",
                            float(cls_scores[idx]),
                            (
                                float(0.5 * width * (box[0] + box[2])),
                                float(0.5 * height * (box[1] + box[3])),
                                float(width * (box[2] - box[0])),
                                float(height * (box[3] - box[1])),
                            ),
                        )
                    )
        return detections

    def _nms(self, boxes, confs, nms_thresh):
        np = self._np
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            keep.append(idx_self)
            idx_other = order[1:]
            if idx_other.size == 0:
                break
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            over = inter / (areas[idx_self] + areas[idx_other] - inter)
            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def _ensure_model(self, model_url: str, override: str | None) -> Path:
        if override:
            path = Path(override).expanduser()
            if not path.exists():
                raise DetectionError(f"Configured local model path does not exist: {path}")
            return path

        cache_dir = Path.home() / ".cache" / "app"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / _DEFAULT_MODEL_NAME
        if path.exists():
            return path

        response = requests.get(model_url, stream=True, timeout=60)
        response.raise_for_status()
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        tmp_path.replace(path)
        return path

    @staticmethod
    def _import_numpy():
        try:
            import numpy
        except ImportError as exc:
            raise DetectionError("numpy is required for local failure detection") from exc
        return numpy

    @staticmethod
    def _import_onnxruntime():
        try:
            import onnxruntime
        except ImportError as exc:
            raise DetectionError("onnxruntime is required for local failure detection") from exc
        return onnxruntime

    @staticmethod
    def _import_pillow_image():
        try:
            from PIL import Image
        except ImportError as exc:
            raise DetectionError("Pillow is required for local failure detection") from exc
        return Image
