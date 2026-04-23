"""Face detection (Haar cascades + YuNet) and ArcFace embedding"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Global model paths (to be set during initialization)
MODELS_DIR = None

def set_models_dir(path):
    global MODELS_DIR
    MODELS_DIR = Path(path)

class CascadeFaceDetector:
    def __init__(self):
        self.names = []
        self.yunet = None
        self.haar_frontal = self.haar_profile = self.haar_alt2 = None
        if MODELS_DIR is None:
            raise RuntimeError("Models directory not set. Call set_models_dir() first.")
        yunet_path = MODELS_DIR / "yunet.onnx"
        if yunet_path.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (0,0))
                self.names.append("yunet")
            except:
                pass
        haar = cv2.data.haarcascades
        try:
            self.haar_frontal = cv2.CascadeClassifier(haar + 'haarcascade_frontalface_default.xml')
            self.names.append("haar_frontalface_default")
        except:
            pass
        try:
            self.haar_profile = cv2.CascadeClassifier(haar + 'haarcascade_profileface.xml')
            self.names.append("haar_profileface")
        except:
            pass
        try:
            self.haar_alt2 = cv2.CascadeClassifier(haar + 'haarcascade_frontalface_alt2.xml')
            self.names.append("haar_frontalface_alt2")
        except:
            pass
        if not self.names:
            raise RuntimeError("No face detector available")

    def detect(self, image):
        if self.yunet is not None:
            h,w = image.shape[:2]
            self.yunet.setInputSize((w,h))
            _, detections = self.yunet.detect(image)
            if detections is not None and len(detections) > 0:
                boxes = []
                for det in detections:
                    x,y,w_box,h_box = det[:4].astype(int)
                    if w_box>0 and h_box>0:
                        boxes.append((x,y,w_box,h_box))
                if boxes:
                    return boxes
        boxes = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.haar_frontal:
            boxes.extend([(x,y,w,h) for (x,y,w,h) in self.haar_frontal.detectMultiScale(gray, 1.1, 5, minSize=(30,30))])
        if self.haar_alt2:
            boxes.extend([(x,y,w,h) for (x,y,w,h) in self.haar_alt2.detectMultiScale(gray, 1.1, 5, minSize=(30,30))])
        if self.haar_profile:
            boxes.extend([(x,y,w,h) for (x,y,w,h) in self.haar_profile.detectMultiScale(gray, 1.1, 5, minSize=(30,30))])
        return boxes

class ArcFaceEmbedder:
    def __init__(self, model_path=None):
        self.name = "arcface_onnx"
        self.embedding_dim = 512
        if model_path is None:
            if MODELS_DIR is None:
                raise RuntimeError("Models directory not set.")
            model_path = MODELS_DIR / "arcface.onnx"
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = 'input.1'

    def embed(self, face_crop):
        img = cv2.resize(face_crop, (112,112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2,0,1))[np.newaxis, ...]
        emb = self.session.run(None, {self.input_name: img})[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

def identity_within_shot(frames, detector, embedder, n_samples=5):
    if len(frames) < 2:
        return None
    idxs = np.linspace(0, len(frames)-1, min(n_samples, len(frames)), dtype=int)
    embs = []
    for idx in idxs:
        f = frames[idx]
        boxes = detector.detect(f)
        if not boxes:
            continue
        x,y,w,h = max(boxes, key=lambda b: b[2]*b[3])
        crop = f[y:y+h, x:x+w]
        if crop.size > 0:
            embs.append(embedder.embed(crop))
    if len(embs) < 2:
        return None
    ref = embs[0]
    dists = [1.0 - np.dot(ref, e) for e in embs[1:]]
    return float(np.mean(dists))
