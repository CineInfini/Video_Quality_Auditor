"""CineInfini core package.

Eager imports of metrics / face_detection / embedding / coherence are
attempted first; if any of those modules need a heavy dep that isn't
installed (torch / open_clip / onnxruntime) we swallow the ImportError so
that lightweight subcommands like ``cineinfini bootstrap`` still work on
a fresh environment.
"""
from __future__ import annotations

import logging as _logging

_log = _logging.getLogger("cineinfini.core")

try:
    from .metrics import *  # noqa: F401,F403
except Exception as _e:  # noqa: BLE001
    _log.debug("core.metrics deferred import: %s", _e)

try:
    from .face_detection import *  # noqa: F401,F403
except Exception as _e:  # noqa: BLE001
    _log.debug("core.face_detection deferred import: %s", _e)

try:
    from .embedding import *  # noqa: F401,F403
except Exception as _e:  # noqa: BLE001
    _log.debug("core.embedding deferred import: %s", _e)

try:
    from .coherence import *  # noqa: F401,F403
except Exception as _e:  # noqa: BLE001
    _log.debug("core.coherence deferred import: %s", _e)
