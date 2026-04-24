"""Inter-shot coherence and narrative coherence."""
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_2d


def compute_inter_shot_coherence(shot_frames, clip_model, clip_preprocess, device):
    """Compute 3-component coherence (structure, style, semantic) between each
    pair of adjacent shots. Returns a list of dicts."""
    shot_ids = sorted(shot_frames.keys())
    results = []
    for i in range(len(shot_ids) - 1):
        a = shot_ids[i]
        b = shot_ids[i + 1]
        frames_a = shot_frames[a]
        frames_b = shot_frames[b]
        mid_a = frames_a[len(frames_a) // 2]
        mid_b = frames_b[len(frames_b) // 2]
        gray_a = cv2.cvtColor(mid_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(mid_b, cv2.COLOR_BGR2GRAY)
        # Handle size mismatch (defensive)
        if gray_a.shape != gray_b.shape:
            gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
        struct_val = ssim_2d(gray_a, gray_b, data_range=255)
        hsv_a = cv2.cvtColor(mid_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(mid_b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_a = cv2.normalize(hist_a, hist_a).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        style_val = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        if clip_model is not None:
            img_a = clip_preprocess(
                Image.fromarray(cv2.cvtColor(mid_a, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(device)
            img_b = clip_preprocess(
                Image.fromarray(cv2.cvtColor(mid_b, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                emb_a = clip_model.encode_image(img_a)
                emb_b = clip_model.encode_image(img_b)
                emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
                emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
                sem_val = float((emb_a @ emb_b.T).item())
        else:
            sem_val = None  # FIX: was 0.5 (silently neutral)
        weights = {"structure": 0.5, "style": 1.0, "semantic": 1.5}
        # Compute weighted total excluding None components
        if sem_val is None:
            total = (
                struct_val * weights["structure"] + style_val * weights["style"]
            ) / (weights["structure"] + weights["style"])
        else:
            total = (
                struct_val * weights["structure"]
                + style_val * weights["style"]
                + sem_val * weights["semantic"]
            ) / sum(weights.values())
        results.append({
            "shot_a": a,
            "shot_b": b,
            "structure": float(struct_val),
            "style": float(style_val),
            "semantic": sem_val,
            "total": float(total),
            "clip_available": clip_model is not None,
        })
    return results


def compute_narrative_coherence(shot_frames, dinov2_model, dinov2_processor, device):
    """Narrative coherence = DINOv2 cosine similarity between mid-frames of
    adjacent shots.

    Returns None (not [1.0, 1.0, ...]) when DINOv2 is not loaded, so callers
    can distinguish "not computed" from "perfect coherence".
    """
    if dinov2_model is None or dinov2_processor is None:
        return None  # FIX: was [1.0] * N (silently reported perfect)
    shot_ids = sorted(shot_frames.keys())
    results = []
    with torch.no_grad():
        for i in range(len(shot_ids) - 1):
            a = shot_ids[i]
            b = shot_ids[i + 1]
            mid_a = shot_frames[a][len(shot_frames[a]) // 2]
            mid_b = shot_frames[b][len(shot_frames[b]) // 2]
            inputs_a = dinov2_processor(
                images=Image.fromarray(cv2.cvtColor(mid_a, cv2.COLOR_BGR2RGB)),
                return_tensors="pt",
            ).to(device)
            inputs_b = dinov2_processor(
                images=Image.fromarray(cv2.cvtColor(mid_b, cv2.COLOR_BGR2RGB)),
                return_tensors="pt",
            ).to(device)
            emb_a = dinov2_model(**inputs_a).last_hidden_state.mean(dim=1)
            emb_b = dinov2_model(**inputs_b).last_hidden_state.mean(dim=1)
            emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
            emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
            sim = float((emb_a @ emb_b.T).item())
            results.append(sim)
    return results
