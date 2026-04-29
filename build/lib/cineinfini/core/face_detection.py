"""Face detection (Haar cascades + YuNet) and ArcFace embedding.

ArcFace runs through ONNX Runtime when ``arcface.onnx`` is available in
``MODELS_DIR``. When weights or onnxruntime are missing, the embedder
falls back to deterministic hashed unit vectors so downstream gates / DTW
keep producing finite numbers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..core.config import get_config

logger = logging.getLogger("cineinfini.face_detection")

MODELS_DIR: "Path | None" = None


def set_models_dir(path) -> None:
    global MODELS_DIR
    MODELS_DIR = Path(path)


def _resolve_models_dir() -> Path:
    """Return MODELS_DIR if set, else fall back to get_config().models_dir()."""
    if MODELS_DIR is not None:
        return Path(MODELS_DIR)
    return get_config().models_dir()


def _clip_box(x, y, w, h, img_w, img_h):
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


class CascadeFaceDetector:
    def __init__(self):
        self.names = []
        self.yunet = None
        self.haar_frontal = self.haar_profile = self.haar_alt2 = None
        models_dir = _resolve_models_dir()
        yunet_path = models_dir / "yunet.onnx"
        if yunet_path.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (0, 0))
                self.names.append("yunet")
            except Exception as e:
                logger.debug("YuNet load failed: %s", e)

        haar = cv2.data.haarcascades
        try:
            self.haar_frontal = cv2.CascadeClassifier(haar + "haarcascade_frontalface_default.xml")
            self.names.append("haar_frontalface_default")
        except Exception:
            pass
        try:
            self.haar_profile = cv2.CascadeClassifier(haar + "haarcascade_profileface.xml")
            self.names.append("haar_profileface")
        except Exception:
            pass
        try:
            self.haar_alt2 = cv2.CascadeClassifier(haar + "haarcascade_frontalface_alt2.xml")
            self.names.append("haar_frontalface_alt2")
        except Exception:
            pass

        if not self.names:
            raise RuntimeError("No face detector available")

    def detect(self, image):
        img_h, img_w = image.shape[:2]
        boxes = []

        if self.yunet is not None:
            self.yunet.setInputSize((img_w, img_h))
            _, detections = self.yunet.detect(image)
            if detections is not None and len(detections) > 0:
                for det in detections:
                    x, y, w_box, h_box = det[:4].astype(int)
                    clipped = _clip_box(x, y, w_box, h_box, img_w, img_h)
                    if clipped is not None:
                        boxes.append(clipped)
                if boxes:
                    return boxes

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for classifier in (self.haar_frontal, self.haar_alt2, self.haar_profile):
            if classifier is not None:
                for (x, y, w, h) in classifier.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)):
                    clipped = _clip_box(x, y, w, h, img_w, img_h)
                    if clipped is not None:
                        boxes.append(clipped)
        return boxes


class ArcFaceEmbedder:
    """ArcFace 512-D embedder with deterministic fallback."""

    def __init__(self, model_path=None):
        self.embedding_dim = 512
        self.session = None
        self.input_name = None
        self.name = "arcface_onnx_fallback"

        try:
            import onnxruntime as ort  # noqa: F401
        except Exception as e:
            logger.info("onnxruntime not available, using fallback embedder: %s", e)
            return

        if model_path is None:
            model_path = _resolve_models_dir() / "arcface.onnx"

        try:
            mp = Path(model_path)
            if not mp.exists():
                logger.info("ArcFace weights not found at %s, using fallback", mp)
                return
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            self.input_name = inputs[0].name if inputs else "input.1"
            self.name = "arcface_onnx"
        except Exception as e:
            logger.warning("Failed to load ArcFace, using fallback: %s", e)
            self.session = None

    def embed(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None
        if face_crop.shape[0] < 4 or face_crop.shape[1] < 4:
            return None
        if self.session is not None:
            try:
                img = cv2.resize(face_crop, (112, 112))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5
                img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
                emb = self.session.run(None, {self.input_name: img})[0].flatten()
                norm = np.linalg.norm(emb)
                if norm < 1e-6:
                    return None
                return emb / norm
            except Exception as e:
                logger.debug("ArcFace inference failed, falling back: %s", e)

        # Deterministic fallback: hash crop bytes -> seed a generator
        h = hash(face_crop.tobytes()) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        emb = rng.standard_normal(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None
        return emb / norm


def identity_within_shot(frames, detector, embedder, n_samples: int = 5):
    """Sample ``n_samples`` frames, embed the largest face per frame, return
    the mean ``1 - cos(sim)`` versus the first sampled frame's embedding."""
    if len(frames) < 2:
        return None
    idxs = np.linspace(0, len(frames) - 1, min(n_samples, len(frames)), dtype=int)
    embs = []
    for idx in idxs:
        f = frames[idx]
        boxes = detector.detect(f)
        if not boxes:
            continue
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        crop = f[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        emb = embedder.embed(crop)
        if emb is not None:
            embs.append(emb)
    if len(embs) < 2:
        return None
    ref = embs[0]
    dists = [1.0 - float(np.dot(ref, e)) for e in embs[1:]]
    return float(np.mean(dists))


def get_face_detector():
    return CascadeFaceDetector()


def get_face_embedder():
    return ArcFaceEmbedder()
