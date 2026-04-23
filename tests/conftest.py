
---

## `tests/conftest.py`

```python
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

@pytest.fixture
def sample_video():
    """Create a temporary synthetic video for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        out = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*'mp4v'), 24, (320, 240))
        for _ in range(120):  # 5 seconds
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        yield Path(tmp.name)

@pytest.fixture
def sample_shot_frames():
    """Return dummy shot frames for testing coherence functions."""
    frames = []
    for _ in range(10):
        frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        frames.append(frame)
    return {1: frames, 2: frames}
