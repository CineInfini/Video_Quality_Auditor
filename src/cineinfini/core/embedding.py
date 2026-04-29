"""CLIP and DINOv2 helpers for semantic / narrative coherence.

Heavy ML imports are lazy so simply importing the module does not require
``torch`` or ``open_clip``. Each scorer exposes ``self.available``.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("cineinfini.embedding")


class _DinoV2State:
    processor = None
    model = None
    device: Optional[str] = None

    @classmethod
    def is_loaded(cls) -> bool:
        return cls.model is not None

    @classmethod
    def load(cls, device: str = "cpu") -> bool:
        try:
            from transformers import AutoImageProcessor, AutoModel
            cls.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            cls.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            cls.model.eval()
            cls.device = device
            return True
        except Exception as e:
            logger.info("DINOv2 not available: %s", e)
            cls.processor = None
            cls.model = None
            return False


def load_dinov2(device: str = "cpu") -> bool:
    return _DinoV2State.load(device)


def get_dinov2():
    return _DinoV2State.processor, _DinoV2State.model


class CLIPSemanticScorer:
    def __init__(self, model_path=None, device: str = "cpu"):
        self.name = "clip_similarity"
        self.device = device
        self.available = False
        self.model = None
        self.preprocess = None
        try:
            import torch  # noqa: F401
            import open_clip
            from PIL import Image  # noqa: F401

            if model_path is None:
                from .config import get_config
                model_path = get_config().models_dir() / "ViT-B-32.pt"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=str(model_path), device=device
            )
            self.model.eval()
            self.available = True
        except Exception as e:
            logger.info("CLIP not available: %s", e)

    def score(self, frames, description: str, n_samples: int = 6):
        if not self.available or not frames:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        import torch
        import open_clip
        from PIL import Image

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
        return {
            "mean": float(scores.mean()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }

    def extract_features(self, image):
        """Return a 1-D normalised CLIP image embedding (or None)."""
        if not self.available or image is None:
            return None
        try:
            import torch
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil = image
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy().squeeze()
        except Exception as e:
            logger.debug("CLIPSemanticScorer.extract_features failed: %s", e)
            return None


def clip_semantic_consistency(frames, clip_model, clip_preprocess, device: str):
    if len(frames) < 2 or clip_model is None:
        return None
    import torch
    from PIL import Image

    img0 = clip_preprocess(
        Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    ).unsqueeze(0).to(device)
    img1 = clip_preprocess(
        Image.fromarray(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        emb0 = clip_model.encode_image(img0)
        emb1 = clip_model.encode_image(img1)
        emb0 = emb0 / emb0.norm(dim=-1, keepdim=True)
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        return float((emb0 @ emb1.T).item())


# Deprecated globals kept for backward compatibility
dinov2_processor = _DinoV2State.processor
dinov2_model = _DinoV2State.model


def __getattr__(name):
    if name in ("dinov2_processor", "dinov2_model"):
        warnings.warn(
            f"`embedding.{name}` is deprecated. Use load_dinov2() / get_dinov2().",
            DeprecationWarning, stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
