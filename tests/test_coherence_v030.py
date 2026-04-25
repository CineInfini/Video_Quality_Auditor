"""Tests for compute_inter_shot_coherence v0.3.0."""
import numpy as np
import pytest
from cineinfini.core.coherence import compute_inter_shot_coherence, compute_narrative_coherence

class _MockDetector:
    def detect(self, frame): h,w=frame.shape[:2]; return [(w//4, h//4, w//2, h//2)]
class _MockEmbedder:
    def __init__(self, vector=None):
        v = np.array([1,0.1,0,0,0,0,0,0]) if vector is None else np.asarray(vector)
        self.vector = v/np.linalg.norm(v)
    def embed(self, crop): return self.vector.copy()
def _make_frames(n=8, h=64,w=64,fill=128): return [np.full((h,w,3),fill,dtype=np.uint8) for _ in range(n)]

def test_backward_compat_dict_input_no_models():
    shot_frames = {0:_make_frames(8,fill=100),1:_make_frames(8,fill=200)}
    results = compute_inter_shot_coherence(shot_frames, None, None, "cpu")
    assert len(results)==1 and results[0]["semantic"] is None

def test_audit_style_list_input_with_detector():
    shots = [_make_frames(8,fill=100), _make_frames(8,fill=200)]
    det = _MockDetector(); emb = _MockEmbedder()
    results = compute_inter_shot_coherence(shots, det, emb, None, 5)
    assert "identity_dtw_inter" in results[0] and results[0]["identity_dtw_inter"] is not None

def test_dtw_disabled_explicitly():
    results = compute_inter_shot_coherence([_make_frames(), _make_frames()], detector=_MockDetector(), embedder=_MockEmbedder(), compute_identity_dtw=False)
    assert "identity_dtw_inter" not in results[0]

def test_skipped_pair_when_frames_missing():
    results = compute_inter_shot_coherence([_make_frames(8), [], _make_frames(8)], None,None,"cpu")
    assert results[0].get("skipped_reason")=="missing_frames"
