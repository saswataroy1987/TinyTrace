"""Read-only data, staged detector, and FLAN-T5 captioning for TinyTrace Phase B v2."""

from .config import PhaseBV2Config, Stage0Config
from .data import ActivityNetV2Dataset, activitynet_v2_collate_fn
from .manifest import prepare_stage0
from .model import PhaseBV2Model
from .temporal import TemporalEventDetector, filter_events, hungarian_match, localization_loss

__all__ = [
    "ActivityNetV2Dataset",
    "PhaseBV2Config",
    "PhaseBV2Model",
    "Stage0Config",
    "TemporalEventDetector",
    "activitynet_v2_collate_fn",
    "filter_events",
    "hungarian_match",
    "localization_loss",
    "prepare_stage0",
]
