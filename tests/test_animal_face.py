"""Tests for animal_face module."""
import numpy as np
import pytest
from cineinfini.core.animal_face import (
    AnimalDetection, YoloAnimalDetector, CompositeAnimalDetector,
    AnimalDetectorAdapter, build_default_animal_pipeline
)
def _img(): return np.full((256,256,3),128,dtype=np.uint8)

def test_animal_detection_crop():
    d = AnimalDetection(x=10,y=20,w=100,h=100,confidence=0.9)
    crop = d.crop(_img())
    assert crop.shape == (100,100,3)

def test_yolo_missing_weights():
    det = YoloAnimalDetector("/nonexistent.pt")
    assert not det.is_available()
    assert det.detect(_img()) == []

def test_composite_falls_through():
    class FakeDetector:
        name="fake"; available=True
        def detect(self,img): return []
        def is_available(self): return True
    class FakeDetector2:
        def detect(self,img): return [AnimalDetection(0,0,10,10,0.9,"cat")]
        def is_available(self): return True
    comp = CompositeAnimalDetector([FakeDetector(), FakeDetector2()])
    assert len(comp.detect(_img())) == 1

def test_adapter_converts():
    class Fake:
        def detect(self,img): return [AnimalDetection(1,2,30,40,0.8,"dog")]
    adapter = AnimalDetectorAdapter(Fake())
    assert adapter.detect(_img()) == [(1,2,30,40)]

def test_factory_returns_none():
    d,e = build_default_animal_pipeline(yolo_weights=None, use_dinov2_fallback=False)
    assert d is None and e is None
