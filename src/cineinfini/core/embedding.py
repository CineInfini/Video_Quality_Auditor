"""CLIP and DINOv2 models for semantic and narrative coherence"""
import torch
import open_clip
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class CLIPSemanticScorer:
    def __init__(self, model_path=None, device="cpu"):
        self.name = "clip_similarity"
        self.device = device
        try:
            if model_path is None:
                # Assume models are in MODELS_DIR (global)
                from .face_detection import MODELS_DIR
                model_path = MODELS_DIR / "ViT-B-32.pt"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=str(model_path), device=device, weights_only=False)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"CLIP not available: {e}")
            self.available = False

    def score(self, frames, description, n_samples=6):
        if not self.available or not frames:
            return {"mean":0.0, "min":0.0, "max":0.0}
        idxs = np.linspace(0, len(frames)-1, min(n_samples, len(frames)), dtype=int)
        images = [frames[i] for i in idxs]
        img_tensors = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in images]).to(self.device)
        text_tokens = open_clip.tokenize([description]).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensors)
            txt_feat = self.model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze()
        scores = sim.cpu().numpy()
        return {"mean": float(scores.mean()), "min": float(scores.min()), "max": float(scores.max())}

def clip_semantic_consistency(frames, clip_model, clip_preprocess, device):
    if len(frames) < 2 or clip_model is None:
        return None
    img0 = clip_preprocess(Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    img1 = clip_preprocess(Image.fromarray(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        emb0 = clip_model.encode_image(img0)
        emb1 = clip_model.encode_image(img1)
        emb0 = emb0 / emb0.norm(dim=-1, keepdim=True)
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        return (emb0 @ emb1.T).item()

# DINOv2 global variable (to be set once)
dinov2_processor = None
dinov2_model = None

def load_dinov2(device="cpu"):
    global dinov2_processor, dinov2_model
    try:
        dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        dinov2_model.eval()
        return True
    except Exception as e:
        print(f"DINOv2 not available: {e}")
        return False
