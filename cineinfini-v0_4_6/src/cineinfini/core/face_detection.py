"""Face detection (Haar cascades + YuNet) and ArcFace embedding."""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Module-level state (set by set_models_dir or read from get_config())
MODELS_DIR = None


def set_models_dir(path):
    """Override the models directory.

    DEPRECATED in v0.4.6: prefer setting `paths.models_dir` in your config.yaml,
    or use `get_config().paths["models_dir"] = path` and `set_config(cfg)`.
    Kept for backward compatibility.
    """
    global MODELS_DIR
    MODELS_DIR = Path(path)


def _resolve_models_dir():
    """Resolve the models directory from override → config → default."""
    if MODELS_DIR is not None:
        return MODELS_DIR
    try:
        from .config import get_config
        return get_config().models_dir()
    except Exception:
        return None


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
        if models_dir is None:
            raise RuntimeError(
                "Models directory not set. Either:\n"
                "  - call set_models_dir(path), or\n"
                "  - set paths.models_dir in ~/.cineinfini/config.yaml"
            )

        yunet_path = models_dir / "yunet.onnx"
        if yunet_path.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (0, 0))
                self.names.append("yunet")
            except Exception:
                pass

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
    def __init__(self, model_path=None):
        self.name = "arcface_onnx"
        self.embedding_dim = 512
        if model_path is None:
            models_dir = _resolve_models_dir()
            if models_dir is None:
                raise RuntimeError(
                    "Models directory not set. Either pass model_path explicitly, "
                    "call set_models_dir(), or set paths.models_dir in config.yaml"
                )
            model_path = models_dir / "arcface.onnx"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = "input.1"

    def embed(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None
        if face_crop.shape[0] < 4 or face_crop.shape[1] < 4:
            return None
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


def identity_within_shot(frames, detector, embedder, n_samples=5):
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
    """Get or initialize the face detector."""
    from .face_detection import CascadeFaceDetector
    return CascadeFaceDetector()

def get_face_embedder():
    """Get or initialize the face embedder."""
    from .face_detection import ArcFaceEmbedder
    return ArcFaceEmbedder()
