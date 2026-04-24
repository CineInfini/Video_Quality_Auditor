"""Face detection (Haar cascades + YuNet) and ArcFace embedding."""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Module-level state (set by set_models_dir)
MODELS_DIR = None


def set_models_dir(path):
    """Register where .onnx models can be found."""
    global MODELS_DIR
    MODELS_DIR = Path(path)


def _clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int):
    """Clip (x, y, w, h) so that the crop stays inside the image.

    Returns (x, y, w, h) clamped. If the resulting box has zero area,
    returns None.
    """
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    # shrink if overflowing
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


class CascadeFaceDetector:
    """YuNet + Haar multi-strategy face detector."""

    def __init__(self):
        self.names = []
        self.yunet = None
        self.haar_frontal = self.haar_profile = self.haar_alt2 = None
        if MODELS_DIR is None:
            raise RuntimeError(
                "Models directory not set. Call set_models_dir() first."
            )

        yunet_path = MODELS_DIR / "yunet.onnx"
        if yunet_path.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (0, 0))
                self.names.append("yunet")
            except Exception:
                pass

        haar = cv2.data.haarcascades
        try:
            self.haar_frontal = cv2.CascadeClassifier(
                haar + "haarcascade_frontalface_default.xml"
            )
            self.names.append("haar_frontalface_default")
        except Exception:
            pass
        try:
            self.haar_profile = cv2.CascadeClassifier(
                haar + "haarcascade_profileface.xml"
            )
            self.names.append("haar_profileface")
        except Exception:
            pass
        try:
            self.haar_alt2 = cv2.CascadeClassifier(
                haar + "haarcascade_frontalface_alt2.xml"
            )
            self.names.append("haar_frontalface_alt2")
        except Exception:
            pass

        if not self.names:
            raise RuntimeError("No face detector available")

    def detect(self, image):
        """Return list of (x, y, w, h) face boxes, clipped to image bounds.

        Tries YuNet first; falls back to Haar cascades if YuNet finds nothing.
        """
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

        # Fallback: Haar cascades
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for classifier in (self.haar_frontal, self.haar_alt2, self.haar_profile):
            if classifier is not None:
                for (x, y, w, h) in classifier.detectMultiScale(
                    gray, 1.1, 5, minSize=(30, 30)
                ):
                    clipped = _clip_box(x, y, w, h, img_w, img_h)
                    if clipped is not None:
                        boxes.append(clipped)
        return boxes


class ArcFaceEmbedder:
    """ArcFace ONNX embedder (buffalo_l w600k_r50, 512-d output)."""

    def __init__(self, model_path=None):
        self.name = "arcface_onnx"
        self.embedding_dim = 512
        if model_path is None:
            if MODELS_DIR is None:
                raise RuntimeError("Models directory not set.")
            model_path = MODELS_DIR / "arcface.onnx"
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = "input.1"

    def embed(self, face_crop):
        """Return a 512-d unit-norm embedding, or None if the crop is invalid."""
        if face_crop is None or face_crop.size == 0:
            return None
        if face_crop.shape[0] < 4 or face_crop.shape[1] < 4:
            # Too small to be a useful face
            return None
        img = cv2.resize(face_crop, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        emb = self.session.run(None, {self.input_name: img})[0].flatten()
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            # FIX: return None instead of a zero vector (which would give
            # cosine distance = 1 — looks like identity break, but is actually
            # a degenerate embedding).
            return None
        return emb / norm


def identity_within_shot(frames, detector, embedder, n_samples: int = 5):
    """Compute mean cosine distance between first face and later faces.

    Returns None if fewer than 2 valid embeddings could be extracted.
    """
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
