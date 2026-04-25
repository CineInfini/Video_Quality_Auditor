"""Animal face/body detection and identity embedding (added in v0.3.0)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

COCO_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

@dataclass
class AnimalDetection:
    x: int; y: int; w: int; h: int; confidence: float
    class_name: str = "animal"; detector_used: str = ""
    def crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x0 = max(0, int(self.x)); y0 = max(0, int(self.y))
        x1 = min(w, int(self.x+self.w)); y1 = min(h, int(self.y+self.h))
        if x1<=x0 or y1<=y0: return None
        return image[y0:y1, x0:x1]

class AnimalDetector(ABC):
    name: str = "abstract"
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[AnimalDetection]: ...
    def is_available(self) -> bool: return False

class YoloAnimalDetector(AnimalDetector):
    name = "yolo_animal"
    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.25, classes: Optional[set[str]] = None):
        self.weights_path = str(weights_path)
        self.conf_threshold = conf_threshold
        self.classes = classes if classes is not None else COCO_ANIMAL_CLASSES
        self.model = None
        self._load()
    def _load(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
        except Exception as e:
            print(f"[YoloAnimalDetector] failed: {e}"); self.model=None
    def is_available(self): return self.model is not None
    def detect(self, image: np.ndarray) -> list[AnimalDetection]:
        if self.model is None: return []
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        if not results: return []
        boxes = results[0].boxes
        if boxes is None or len(boxes)==0: return []
        names = self.model.names
        out = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2), cf, cid in zip(xyxy, confs, cls_ids):
            class_name = names[cid] if isinstance(names,dict) else str(cid)
            if class_name not in self.classes: continue
            out.append(AnimalDetection(x=int(x1),y=int(y1),w=int(x2-x1),h=int(y2-y1),confidence=float(cf),class_name=class_name,detector_used=self.name))
        return out

class DinoSaliencyDetector(AnimalDetector):
    name = "dinov2_saliency"
    def __init__(self, dinov2_model=None, dinov2_processor=None, device: str = "cpu", patch_quantile: float = 0.9):
        self.model = dinov2_model; self.processor = dinov2_processor; self.device = device; self.patch_quantile = patch_quantile
    def is_available(self): return self.model is not None and self.processor is not None
    def detect(self, image: np.ndarray) -> list[AnimalDetection]:
        if not self.is_available(): return []
        try:
            import torch
            from PIL import Image
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=False)
            patches = outputs.last_hidden_state[0, 1:, :]
            saliency = patches.norm(dim=-1).cpu().numpy()
            grid_n = int(np.sqrt(len(saliency)))
            if grid_n*grid_n != len(saliency): return []
            grid = saliency.reshape(grid_n, grid_n)
            thresh = np.quantile(grid, self.patch_quantile)
            mask = grid >= thresh
            if mask.sum()==0: return []
            ys, xs = np.where(mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            h,w = image.shape[:2]
            cell_h = h/grid_n; cell_w = w/grid_n
            box_x = int(x_min*cell_w); box_y = int(y_min*cell_h)
            box_w = int((x_max-x_min+1)*cell_w); box_h = int((y_max-y_min+1)*cell_h)
            return [AnimalDetection(x=box_x,y=box_y,w=box_w,h=box_h,confidence=float(thresh),class_name="salient_region",detector_used=self.name)]
        except Exception as e: return []

class CompositeAnimalDetector(AnimalDetector):
    name = "composite"
    def __init__(self, detectors: list[AnimalDetector]):
        self.detectors = [d for d in detectors if d.is_available()]
    def is_available(self): return len(self.detectors)>0
    def detect(self, image: np.ndarray) -> list[AnimalDetection]:
        for d in self.detectors:
            results = d.detect(image)
            if results: return results
        return []

class AnimalEmbedder(ABC):
    name: str = "abstract"
    embedding_dim: int = 0
    @abstractmethod
    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]: ...

class DinoV2BodyEmbedder(AnimalEmbedder):
    name = "dinov2_body"; embedding_dim = 768
    def __init__(self, dinov2_model=None, dinov2_processor=None, device: str = "cpu"):
        self.model = dinov2_model; self.processor = dinov2_processor; self.device = device
    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or self.processor is None: return None
        if crop is None or crop.size==0 or crop.shape[0]<16 or crop.shape[1]<16: return None
        try:
            import torch
            from PIL import Image
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            if norm<1e-6: return None
            return (emb/norm).astype(np.float64)
        except Exception as e: return None

class ArcFaceCropEmbedder(AnimalEmbedder):
    name = "arcface_on_crop"
    def __init__(self, arcface_embedder):
        self.arcface = arcface_embedder
        self.embedding_dim = getattr(arcface_embedder, "embedding_dim", 512)
    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if self.arcface is None or crop is None or crop.size==0: return None
        return self.arcface.embed(crop)

class AnimalDetectorAdapter:
    def __init__(self, animal_detector: AnimalDetector):
        self._d = animal_detector
        self.name = f"animal_adapter:{animal_detector.name}"
    def detect(self, image: np.ndarray) -> list[tuple]:
        results = self._d.detect(image)
        return [(d.x, d.y, d.w, d.h) for d in results]

def build_default_animal_pipeline(yolo_weights: Optional[str|Path]=None, use_dinov2_fallback: bool=True, device: str="cpu"):
    dinov2_model = None; dinov2_processor = None
    try:
        from .embedding import get_dinov2
        dinov2_processor, dinov2_model = get_dinov2()
    except Exception: pass
    detectors = []
    if yolo_weights is not None:
        yolo_det = YoloAnimalDetector(yolo_weights)
        if yolo_det.is_available(): detectors.append(yolo_det)
    if use_dinov2_fallback and dinov2_model is not None:
        sal_det = DinoSaliencyDetector(dinov2_model, dinov2_processor, device)
        if sal_det.is_available(): detectors.append(sal_det)
    if not detectors: return None, None
    detector = CompositeAnimalDetector(detectors) if len(detectors)>1 else detectors[0]
    embedder = None
    if dinov2_model is not None:
        embedder = DinoV2BodyEmbedder(dinov2_model, dinov2_processor, device)
    return detector, embedder
