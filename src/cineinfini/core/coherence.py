"""Inter-shot coherence and narrative coherence.

v0.3.0 changes:
  - Fixed signature mismatch between audit.py and compute_inter_shot_coherence.
    The audit.py call site has been passing (shot_repr_frames, detector,
    embedder, clip_scorer, subsample) since v0.2.0 but the function signature
    was (shot_frames, clip_model, clip_preprocess, device). Calling the audit
    pipeline on any video with 2+ shots therefore crashed with TypeError.
    The new signature is unified and accepts both the old kwarg-style and
    the new positional-style calls.
  - Added optional identity_dtw_inter computation (v0.2.1 promise).
"""
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_2d

def compute_inter_shot_coherence(
    shot_frames,
    detector_or_clip_model=None,
    embedder_or_clip_preprocess=None,
    clip_scorer_or_device=None,
    subsample=5,
    *,
    detector=None,
    embedder=None,
    clip_scorer=None,
    clip_model=None,
    clip_preprocess=None,
    device=None,
    compute_identity_dtw=True,
):
    """Compute coherence between adjacent shots.

    Tolerates THREE call signatures for backward compatibility:
    1) v0.3.0 audit.py style: (shot_repr_frames, detector, embedder, clip_scorer, subsample)
    2) Pre-v0.2.0 explicit: (shot_frames, clip_model, clip_preprocess, device)
    3) Kwargs only.
    """
    arg2 = detector_or_clip_model
    arg3 = embedder_or_clip_preprocess
    arg4 = clip_scorer_or_device

    if arg2 is not None and hasattr(arg2, "detect"):
        detector = detector or arg2
        embedder = embedder or arg3
        clip_scorer = clip_scorer or arg4
        if clip_scorer is not None:
            clip_model = clip_model or getattr(clip_scorer, "model", None)
            clip_preprocess = clip_preprocess or getattr(clip_scorer, "preprocess", None)
            device = device or getattr(clip_scorer, "device", "cpu")
    elif arg2 is not None and hasattr(arg2, "encode_image"):
        clip_model = clip_model or arg2
        clip_preprocess = clip_preprocess or arg3
        device = device or arg4 or "cpu"

    if isinstance(shot_frames, list):
        shot_frames_dict = {i: f for i, f in enumerate(shot_frames) if f is not None}
    elif isinstance(shot_frames, dict):
        shot_frames_dict = shot_frames
    else:
        raise TypeError(f"shot_frames must be list or dict, got {type(shot_frames).__name__}")

    _dtw_fn = None
    if compute_identity_dtw and detector is not None and embedder is not None:
        try:
            from .identity_dtw import identity_between_shots_dtw
            _dtw_fn = identity_between_shots_dtw
        except ImportError:
            _dtw_fn = None

    shot_ids = sorted(shot_frames_dict.keys())
    results = []
    for i in range(len(shot_ids) - 1):
        a = shot_ids[i]
        b = shot_ids[i + 1]
        frames_a = shot_frames_dict[a]
        frames_b = shot_frames_dict[b]
        if frames_a is None or frames_b is None or len(frames_a) == 0 or len(frames_b) == 0:
            results.append({
                "shot_a": a, "shot_b": b,
                "structure": None, "style": None, "semantic": None,
                "total": None, "clip_available": clip_model is not None,
                "skipped_reason": "missing_frames",
            })
            continue

        if isinstance(frames_a, np.ndarray):
            frames_a = [frames_a]
        if isinstance(frames_b, np.ndarray):
            frames_b = [frames_b]

        mid_a = frames_a[len(frames_a) // 2]
        mid_b = frames_b[len(frames_b) // 2]

        gray_a = cv2.cvtColor(mid_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(mid_b, cv2.COLOR_BGR2GRAY)
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

        if clip_model is not None and clip_preprocess is not None:
            img_a = clip_preprocess(Image.fromarray(cv2.cvtColor(mid_a, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device or "cpu")
            img_b = clip_preprocess(Image.fromarray(cv2.cvtColor(mid_b, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device or "cpu")
            with torch.no_grad():
                emb_a = clip_model.encode_image(img_a)
                emb_b = clip_model.encode_image(img_b)
                emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
                emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
                sem_val = float((emb_a @ emb_b.T).item())
        else:
            sem_val = None

        weights = {"structure": 0.5, "style": 1.0, "semantic": 1.5}
        if sem_val is None:
            total = (struct_val * weights["structure"] + style_val * weights["style"]) / (weights["structure"] + weights["style"])
        else:
            total = (struct_val * weights["structure"] + style_val * weights["style"] + sem_val * weights["semantic"]) / sum(weights.values())

        entry = {
            "shot_a": a, "shot_b": b,
            "structure": float(struct_val), "style": float(style_val),
            "semantic": sem_val, "total": float(total),
            "clip_available": clip_model is not None,
        }

        if _dtw_fn is not None:
            try:
                dtw_res = _dtw_fn(frames_a, frames_b, detector, embedder, max_samples=8)
                entry["identity_dtw_inter"] = dtw_res.normalized
                entry["identity_dtw_n"] = dtw_res.n_a + dtw_res.n_b
            except Exception as e:
                entry["identity_dtw_inter"] = None
                entry["identity_dtw_n"] = 0
                entry["identity_dtw_error"] = str(e)

        results.append(entry)
    return results

def compute_narrative_coherence(shot_frames, dinov2_model, dinov2_processor, device):
    """Cosine similarity between DINOv2 embeddings of mid-frames of adjacent shots."""
    if dinov2_model is None or dinov2_processor is None:
        return None
    if isinstance(shot_frames, list):
        shot_frames_dict = {i: f for i, f in enumerate(shot_frames) if f is not None}
    else:
        shot_frames_dict = shot_frames
    shot_ids = sorted(shot_frames_dict.keys())
    results = []
    with torch.no_grad():
        for i in range(len(shot_ids) - 1):
            a = shot_ids[i]; b = shot_ids[i+1]
            frames_a = shot_frames_dict[a]; frames_b = shot_frames_dict[b]
            if not frames_a or not frames_b:
                results.append(None)
                continue
            if isinstance(frames_a, np.ndarray):
                frames_a = [frames_a]
            if isinstance(frames_b, np.ndarray):
                frames_b = [frames_b]
            mid_a = frames_a[len(frames_a)//2]
            mid_b = frames_b[len(frames_b)//2]
            inputs_a = dinov2_processor(images=Image.fromarray(cv2.cvtColor(mid_a, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(device)
            inputs_b = dinov2_processor(images=Image.fromarray(cv2.cvtColor(mid_b, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(device)
            emb_a = dinov2_model(**inputs_a).last_hidden_state.mean(dim=1)
            emb_b = dinov2_model(**inputs_b).last_hidden_state.mean(dim=1)
            emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
            emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
            sim = float((emb_a @ emb_b.T).item())
            results.append(sim)
    return results
