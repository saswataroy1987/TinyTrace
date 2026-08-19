"""Stage 0 data contracts for TinyTrace Phase B v2.

This package is intentionally separate from the preserved Phase B v1 model and
contains no detector, captioner, or training implementation.
"""

from .config import Stage0Config
from .data import ActivityNetV2Dataset, activitynet_v2_collate_fn
from .manifest import prepare_stage0

__all__ = [
    "ActivityNetV2Dataset",
    "Stage0Config",
    "activitynet_v2_collate_fn",
    "prepare_stage0",
]
