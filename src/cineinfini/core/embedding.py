"""CLIP and DINOv2 models for semantic and narrative coherence."""
import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# DINOv2 singleton
# ---------------------------------------------------------------------------

class _DinoV2State:
    processor = None
    model = None
    device = None

    @classmethod
    def is_loaded(cls):
        return cls.model is not None

    @classmethod
    def load(cls, device: str = "cpu") -> bool:
        try:
            cls.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            cls.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            cls.model.eval()
            cls.device = device
            return True
        except Exception as e:
            print(f"DINOv2 not available: {e}")
            cls.processor = None
            cls.model = None
            return False


def load_dinov2(device: str = "cpu") -> bool:
    return _DinoV2State.load(device)


def get_dinov2():
    return _DinoV2State.processor, _DinoV2State.model


# ---------------------------------------------------------------------------
# CLIP scorer
# ---------------------------------------------------------------------------

class CLIPSemanticScorer:
    def __init__(self, model_path=None, device: str = "cpu"):
        self.name = "clip_similarity"
        self.device = device
        self.available = False
        try:
            if model_path is None:
                from .face_detection import MODELS_DIR
                if MODELS_DIR is None:
                    raise RuntimeError("MODELS_DIR not set")
                model_path = MODELS_DIR / "ViT-B-32.pt"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=str(model_path), device=device
            )
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"CLIP not available: {e}")

    def score(self, frames, description: str, n_samples: int = 6):
        if not self.available or not frames:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        idxs = np.linspace(0, len(frames) - 1, min(n_samples, len(frames)), dtype=int)
        images = [frames[i] for i in idxs]
        img_tensors = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            for f in images
        ]).to(self.device)
        text_tokens = open_clip.tokenize([description]).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensors)
            txt_feat = self.model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze()
        scores = sim.cpu().numpy()
        return {"mean": float(scores.mean()), "min": float(scores.min()), "max": float(scores.max())}


def clip_semantic_consistency(frames, clip_model, clip_preprocess, device: str):
    if len(frames) < 2 or clip_model is None:
        return None
    img0 = clip_preprocess(Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    img1 = clip_preprocess(Image.fromarray(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        emb0 = clip_model.encode_image(img0)
        emb1 = clip_model.encode_image(img1)
        emb0 = emb0 / emb0.norm(dim=-1, keepdim=True)
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        return float((emb0 @ emb1.T).item())


# ---------------------------------------------------------------------------
# Deprecated globals (kept for compatibility, will warn)
# ---------------------------------------------------------------------------
import warnings

dinov2_processor = _DinoV2State.processor
dinov2_model = _DinoV2State.model

def __getattr__(name):
    if name in ("dinov2_processor", "dinov2_model"):
        warnings.warn(
            f"`embedding.{name}` is deprecated and will be removed in v0.3.0. "
            "Use `load_dinov2()` and `get_dinov2()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
