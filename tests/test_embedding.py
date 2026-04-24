"""
CLIP and DINOv2 tests.
"""
from cineinfini.core.embedding import CLIPSemanticScorer, load_dinov2

def test_clip_scorer_init():
    scorer = CLIPSemanticScorer(device="cpu")
    assert scorer.available is False
    assert scorer.score([], "test")["mean"] == 0.0

def test_dinov2_load():
    assert isinstance(load_dinov2("cpu"), bool)
