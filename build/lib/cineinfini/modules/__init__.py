"""CineInfini audit modules.

Importing this package registers each module with the audit registry
via the ``@register_module`` decorator. The registry honours
``cfg.is_module_enabled(mod_id)`` at run-time, so an imported-but-
disabled module is registered yet skipped during audits.

Modules grouped by readiness:

**Pure-CV (work out of the box, no extra weights):**
- motion_coherence, identity_consistency, semantic_consistency,
- background_consistency, aesthetic_cinematic, causal_reasoning,
- temporal_signature, physics_plausibility, trustworthiness,
- world_model_surprise, creative_composition, long_term_narrative,
- subject_consistency_long, benchmark_forensic, explainability,
- benchmark_fusion.

**Require a trained classifier or VLM (report ``available: false`` until weights are present):**
- origin_detection (needs origin_classifier.npz + DINOv2),
- multi_modal_safety (needs CLIP weights),
- prompt_alignment_fine (needs CLIP weights + per-shot prompts).
"""
# Pure CV (always work)
from . import motion_coherence
from . import identity_consistency
from . import semantic_consistency
from . import background_consistency
from . import aesthetic_cinematic
from . import causal_reasoning
from . import temporal_signature
from . import physics_plausibility
from . import trustworthiness
from . import world_model_surprise
from . import creative_composition
from . import long_term_narrative
from . import subject_consistency_long
from . import benchmark_forensic
from . import explainability
from . import benchmark_fusion

# Require ML weights (graceful fallback)
from . import origin_detection
from . import multi_modal_safety
from . import prompt_alignment_fine

# Cross-benchmark wrappers (subsume DOVER, FAST-VQA into our output).
from . import dover_score
from . import fastvqa_score
from . import temporal_smoothness_via_interp_artifacts

__all__ = [
    "motion_coherence", "identity_consistency", "semantic_consistency",
    "background_consistency", "aesthetic_cinematic", "causal_reasoning",
    "temporal_signature", "physics_plausibility", "trustworthiness",
    "world_model_surprise", "creative_composition", "long_term_narrative",
    "subject_consistency_long", "benchmark_forensic", "explainability",
    "benchmark_fusion",
    "origin_detection", "multi_modal_safety", "prompt_alignment_fine",
    "dover_score", "fastvqa_score",
]
