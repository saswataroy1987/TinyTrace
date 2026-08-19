from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .cache import load_and_validate_cache
from .config import Stage0Config
from .manifest import MANIFEST_SCHEMA


class ActivityNetV2Dataset(Dataset):
    """Read-only ActivityNet V2 dataset backed by validated MobileCLIP caches."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        cache_root: str | Path,
        config: Stage0Config,
        split: str | None = None,
    ) -> None:
        if split not in {None, "train", "val"}:
            raise ValueError("split must be None, 'train', or 'val'.")
        self.manifest_path = Path(manifest_path)
        self.cache_root = Path(cache_root)
        self.config = config
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported V2 manifest schema in {self.manifest_path}")
        if payload.get("cache_read_only") is not True:
            raise ValueError("V2 manifest must declare cache_read_only=true.")
        samples = payload.get("samples")
        if not isinstance(samples, list) or not all(isinstance(item, dict) for item in samples):
            raise ValueError("V2 manifest samples must be a list of objects.")
        self.items = [item for item in samples if split is None or item.get("split") == split]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        feature_path = Path(str(item["visual_feature_path"]))
        if not feature_path.is_absolute():
            feature_path = self.cache_root / feature_path
        duration = float(item["duration"])
        features, frame_times, _ = load_and_validate_cache(
            feature_path,
            duration=duration,
            config=self.config,
            compute_statistics=False,
        )
        events = item["events"]
        seconds = torch.tensor(
            [[float(event["start"]), float(event["end"])] for event in events],
            dtype=torch.float32,
        )
        segments = (seconds / duration).clamp(0.0, 1.0)
        return {
            "video_id": str(item["video_id"]),
            "duration": duration,
            "visual_features": features,
            "frame_times": frame_times,
            "segments": segments,
            "segments_seconds": seconds,
            "captions": [str(event["caption"]) for event in events],
            "frame_mask": torch.ones(features.size(0), dtype=torch.bool),
            "event_mask": torch.ones(len(events), dtype=torch.bool),
        }


def activitynet_v2_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty V2 batch.")
    max_frames = max(int(sample["visual_features"].shape[0]) for sample in batch)  # type: ignore[union-attr]
    max_events = max(int(sample["segments"].shape[0]) for sample in batch)  # type: ignore[union-attr]
    feature_tail = tuple(batch[0]["visual_features"].shape[1:])  # type: ignore[union-attr]
    feature_dtype = batch[0]["visual_features"].dtype  # type: ignore[union-attr]

    features_batch = []
    times_batch = []
    frame_masks = []
    segments_batch = []
    seconds_batch = []
    event_masks = []
    captions: list[list[str]] = []
    for sample in batch:
        features = sample["visual_features"]
        times = sample["frame_times"]
        segments = sample["segments"]
        seconds = sample["segments_seconds"]
        if tuple(features.shape[1:]) != feature_tail:  # type: ignore[union-attr]
            raise ValueError("All V2 feature tensors must share patch/channel dimensions.")
        frame_pad = max_frames - int(features.shape[0])  # type: ignore[union-attr]
        event_pad = max_events - int(segments.shape[0])  # type: ignore[union-attr]
        features_batch.append(
            torch.cat(
                [features, torch.zeros((frame_pad, *feature_tail), dtype=feature_dtype)], dim=0  # type: ignore[list-item]
            )
        )
        times_batch.append(torch.cat([times, torch.zeros(frame_pad, dtype=torch.float32)]))  # type: ignore[list-item]
        frame_masks.append(
            torch.cat(
                [torch.ones(features.shape[0], dtype=torch.bool), torch.zeros(frame_pad, dtype=torch.bool)]  # type: ignore[union-attr]
            )
        )
        segments_batch.append(torch.cat([segments, torch.zeros((event_pad, 2), dtype=torch.float32)]))  # type: ignore[list-item]
        seconds_batch.append(torch.cat([seconds, torch.zeros((event_pad, 2), dtype=torch.float32)]))  # type: ignore[list-item]
        event_masks.append(
            torch.cat(
                [torch.ones(segments.shape[0], dtype=torch.bool), torch.zeros(event_pad, dtype=torch.bool)]  # type: ignore[union-attr]
            )
        )
        captions.append(list(sample["captions"]))  # type: ignore[arg-type]

    return {
        "video_id": [str(sample["video_id"]) for sample in batch],
        "duration": torch.tensor([float(sample["duration"]) for sample in batch], dtype=torch.float32),
        "visual_features": torch.stack(features_batch),
        "frame_times": torch.stack(times_batch),
        "segments": torch.stack(segments_batch),
        "segments_seconds": torch.stack(seconds_batch),
        "captions": captions,
        "frame_mask": torch.stack(frame_masks),
        "event_mask": torch.stack(event_masks),
    }
