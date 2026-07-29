from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_metadata
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_transforms

from .config import TinyTraceConfig


@contextmanager
def _patch_package_metadata_versions():
    """Work around broken dist-info metadata in ad-hoc student envs.

    Some local venvs report ``importlib.metadata.version(...)`` as ``None``
    even though the module imports and exposes ``__version__`` correctly.
    The MobileCLIP import path pulls in transformers via open_clip, and that
    code expects a real version string. We patch only the import window needed
    to construct the backbone.
    """

    original_version = importlib_metadata.version
    module_fallbacks = {
        "torch": "torch",
        "numpy": "numpy",
        "pillow": "PIL",
        "Pillow": "PIL",
    }

    def patched_version(name: str) -> str:
        version = original_version(name)
        if version is None and name in module_fallbacks:
            module = importlib.import_module(module_fallbacks[name])
            fallback_version = getattr(module, "__version__", None)
            if fallback_version is not None:
                return str(fallback_version)
        return version

    importlib_metadata.version = patched_version
    try:
        yield
    finally:
        importlib_metadata.version = original_version


class MobileCLIPSpatialEncoder(nn.Module):
    """Frozen MobileCLIP-S0 image tower exposing pre-pooling spatial tokens.

    Apple's public ``encode_image`` API returns one pooled vector per image.
    TinyTRACE needs a token sequence for slot compression, so this adapter uses
    the S0 tower's embedding, backbone, and expansion layers before its global
    pooling head. For a 256x256 image this is expected to produce an 8x8 map
    with 1024 channels, represented as 64 spatial tokens.
    """

    def __init__(
        self,
        config: TinyTraceConfig,
        backbone: nn.Module | None = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.trainable_strategy = "frozen"
        self.backbone = (
            backbone if backbone is not None else self._load_official_backbone(load_pretrained=load_pretrained)
        )

        self.set_trainable(not config.freeze_visual_encoder)

    def _load_official_backbone(self, load_pretrained: bool) -> nn.Module:
        checkpoint = Path(self.config.mobileclip_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[1] / checkpoint
        if load_pretrained and not checkpoint.is_file():
            raise FileNotFoundError(
                "MobileCLIP-S0 checkpoint not found at "
                f"{checkpoint}. Download the official checkpoint and set "
                "mobileclip_checkpoint in the TinyTrace config."
            )
        if load_pretrained and self.config.mobileclip_checkpoint_sha256:
            digest = hashlib.sha256()
            with checkpoint.open("rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != self.config.mobileclip_checkpoint_sha256.lower():
                raise ValueError(
                    "MobileCLIP checkpoint SHA-256 mismatch: expected "
                    f"{self.config.mobileclip_checkpoint_sha256}, received {actual_sha256}."
                )

        try:
            with _patch_package_metadata_versions():
                import mobileclip

                clip_model, _, _ = mobileclip.create_model_and_transforms(
                    self.config.mobileclip_model_name,
                    pretrained=str(checkpoint) if load_pretrained else None,
                    reparameterize=False,
                )
        except ImportError as exc:
            raise ImportError(
                "The official Apple MobileCLIP package is required. Install "
                "apple/ml-mobileclip before constructing TinyTraceModel."
            ) from exc
        return clip_model.image_encoder.model

    def train(self, mode: bool = True) -> "MobileCLIPSpatialEncoder":
        super().train(mode)
        # MobileCLIP-S0 contains BatchNorm layers and must stay in evaluation
        # mode while frozen, even when the surrounding TinyTrace model trains.
        if self.trainable_strategy == "frozen":
            self.backbone.eval()
        elif self.trainable_strategy == "conv_exp":
            self.backbone.eval()
            self.backbone.conv_exp.train(mode)
        return self

    def set_trainable(self, trainable: bool, strategy: str = "full") -> None:
        if not trainable:
            self.trainable_strategy = "frozen"
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            return

        if strategy not in {"full", "conv_exp"}:
            raise ValueError(f"Unsupported MobileCLIP trainable strategy: {strategy}")
        self.trainable_strategy = strategy
        self.backbone.requires_grad_(False)
        if strategy == "full":
            self.backbone.requires_grad_(True)
            self.backbone.train()
        else:
            self.backbone.conv_exp.requires_grad_(True)
            self.backbone.eval()
            self.backbone.conv_exp.train()

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.size(1) != 3:
            raise ValueError("MobileCLIP frames must have shape [batch, 3, height, width].")
        frames = frames.float().clamp(0.0, 1.0)
        frames = vision_transforms.resize(
            frames,
            size=self.config.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        frames = vision_transforms.center_crop(
            frames,
            output_size=[self.config.image_size, self.config.image_size],
        )
        if self.config.mobileclip_apply_normalization:
            frames = self._normalize(
                frames,
                self.config.mobileclip_image_mean,
                self.config.mobileclip_image_std,
            )
        return frames

    @staticmethod
    def _normalize(
        frames: torch.Tensor,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> torch.Tensor:
        mean_tensor = torch.tensor(mean, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
        return (frames - mean_tensor) / std_tensor

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self.preprocess(frames)
        context = torch.no_grad() if self.trainable_strategy == "frozen" else torch.enable_grad()
        with context:
            features = self.backbone.forward_embeddings(frames)
            features = self.backbone.forward_tokens(features)
            features = self.backbone.conv_exp(features)

        if features.ndim != 4:
            raise RuntimeError(
                "MobileCLIP spatial extraction expected [batch, channels, height, width], "
                f"but received {tuple(features.shape)}."
            )
        if features.size(1) != self.config.visual_hidden_dim:
            raise RuntimeError(
                f"Configured visual_hidden_dim={self.config.visual_hidden_dim}, but MobileCLIP "
                f"produced {features.size(1)} channels."
            )
        return features.flatten(2).transpose(1, 2)


class SlotCompressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_slots: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_slots, input_dim))
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(patch_tokens.size(0), -1, -1)
        scores = torch.matmul(queries, patch_tokens.transpose(-1, -2)) / patch_tokens.size(-1) ** 0.5
        weights = torch.softmax(scores, dim=-1)
        return self.projector(torch.matmul(weights, patch_tokens))
