"""Alternative Stage 2 visual bridges for controlled resampler ablations."""

from .bridges import build_bridge
from .model import BridgeAblationCaptionModel

__all__ = ["BridgeAblationCaptionModel", "build_bridge"]
