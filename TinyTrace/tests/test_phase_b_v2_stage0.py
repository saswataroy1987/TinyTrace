from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tinytrace.phase_b_v2 import (
    ActivityNetV2Dataset,
    Stage0Config,
    activitynet_v2_collate_fn,
    prepare_stage0,
)
from tinytrace.phase_b_v2.cache import CacheValidationError, load_and_validate_cache
from tinytrace.phase_b_v2.cache import legacy_v1_cache_filename
from tinytrace.config import TinyTraceConfig
from tinytrace.data import JsonTinyTraceDataset


class PhaseBV2Stage0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_root = self.root / "cache"
        self.cache_root.mkdir()
        self.output_root = self.root / "phase_b_activitynet_v2_run"
        self.train_json = self.root / "train.json"
        self.val_json = self.root / "val_1.json"
        self.mapping_json = self.root / "cache_mapping.json"
        self.config = Stage0Config(representative_sample_count=2)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_cache(
        self,
        name: str,
        *,
        frames: int,
        duration: float,
        invalid_value: float | None = None,
    ) -> Path:
        features = torch.arange(frames * 64 * 1024, dtype=torch.float32).reshape(frames, 64, 1024)
        features = (features.remainder(101) / 100.0).to(torch.float16)
        if invalid_value is not None:
            features[0, 0, 0] = invalid_value
        path = self.cache_root / f"{name}.pt"
        torch.save(
            {
                "format_version": 2,
                "patch_features": features,
                "frame_times": torch.linspace(0.0, duration - 0.25, frames, dtype=torch.float32),
            },
            path,
        )
        return path

    def write_inputs(self, *, scope: str = "full") -> Stage0Config:
        self.write_cache("train", frames=3, duration=10.0)
        self.write_cache("val", frames=2, duration=8.0)
        train_events = [
            (f"event {index}", [float(index), float(index) + 0.5])
            for index in reversed(range(7))
        ]
        # One invalid event is reported while the seven valid events are kept.
        train_events.append(("", [8.0, 9.0]))
        train = {
            "v_train": {
                "duration": 10.0,
                "sentences": [item[0] for item in train_events],
                "timestamps": [item[1] for item in train_events],
            },
            "v_invalid": {
                "duration": 5.0,
                "sentences": ["reversed"],
                "timestamps": [[4.0, 2.0]],
            },
        }
        val = {
            "v_val": {
                "duration": 8.0,
                "sentences": ["validation event"],
                "timestamps": [[1.0, 4.0]],
            }
        }
        mapping = {
            "schema_version": "tinytrace.phase-b-v2.cache-map.v1",
            "entries": [
                {"video_id": "v_train", "visual_feature_path": "train.pt"},
                {"video_id": "v_val", "visual_feature_path": "val.pt"},
            ],
        }
        self.train_json.write_text(json.dumps(train), encoding="utf-8")
        self.val_json.write_text(json.dumps(val), encoding="utf-8")
        self.mapping_json.write_text(json.dumps(mapping), encoding="utf-8")
        return Stage0Config(representative_sample_count=2, validation_scope=scope)

    def prepare(self, config: Stage0Config | None = None) -> dict[str, object]:
        return prepare_stage0(
            train_annotations=self.train_json,
            val_annotations=self.val_json,
            cache_mapping=self.mapping_json,
            cache_root=self.cache_root,
            output_root=self.output_root,
            config=config or self.config,
            repository_root=self.root,
        )

    def test_manifest_preserves_all_events_and_reports_every_skip(self) -> None:
        config = self.write_inputs()
        result = self.prepare(config)

        report = result["dataset_validation"]
        self.assertTrue(report["validation_passed"])
        self.assertTrue(report["ready_for_training"])
        self.assertEqual(report["retained_samples"], {"train": 1, "val": 1, "total": 2})
        self.assertEqual(len(report["representative_cache_statistics"]), 2)
        self.assertEqual(report["expected_feature_shape"], ["T", 64, 1024])
        self.assertEqual(report["dataset_statistics"]["train"]["events"], 7)
        self.assertEqual(report["dataset_statistics"]["total"]["events"], 8)

        manifest = result["manifest"]
        train_item = next(item for item in manifest["samples"] if item["video_id"] == "v_train")
        self.assertEqual(len(train_item["events"]), 7)
        self.assertEqual([event["start"] for event in train_item["events"]], sorted(event["start"] for event in train_item["events"]))

        skips = result["skipped_samples"]
        self.assertEqual(skips["counts_by_reason"]["empty_caption"], 1)
        self.assertEqual(skips["counts_by_reason"]["reversed_or_empty_event"], 1)
        self.assertEqual(skips["counts_by_reason"]["no_valid_events"], 1)

    def test_dataset_and_collator_return_v2_contract_and_masks(self) -> None:
        config = self.write_inputs()
        self.prepare(config)
        dataset = ActivityNetV2Dataset(
            self.output_root / "manifests" / "activitynet_v2_manifest.json",
            cache_root=self.cache_root,
            config=config,
        )
        train_sample = next(dataset[index] for index in range(len(dataset)) if dataset[index]["video_id"] == "v_train")
        val_sample = next(dataset[index] for index in range(len(dataset)) if dataset[index]["video_id"] == "v_val")

        self.assertEqual(train_sample["visual_features"].shape, (3, 64, 1024))
        self.assertEqual(train_sample["frame_times"].shape, (3,))
        self.assertEqual(train_sample["segments"].shape, (7, 2))
        self.assertEqual(train_sample["frame_mask"].tolist(), [True, True, True])
        self.assertEqual(train_sample["event_mask"].sum().item(), 7)
        self.assertTrue(torch.allclose(train_sample["segments"] * 10.0, train_sample["segments_seconds"]))

        batch = activitynet_v2_collate_fn([train_sample, val_sample])
        self.assertEqual(batch["visual_features"].shape, (2, 3, 64, 1024))
        self.assertEqual(batch["frame_times"].shape, (2, 3))
        self.assertEqual(batch["segments"].shape, (2, 7, 2))
        self.assertEqual(batch["frame_mask"].tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(batch["event_mask"].sum(dim=1).tolist(), [7, 1])

    def test_non_finite_cache_is_rejected_with_machine_reason(self) -> None:
        path = self.write_cache("bad", frames=2, duration=4.0, invalid_value=float("nan"))
        with self.assertRaises(CacheValidationError) as raised:
            load_and_validate_cache(path, duration=4.0, config=self.config)
        self.assertEqual(raised.exception.code, "non_finite_features")

    def test_split_leakage_fails_validation_without_reassigning_video(self) -> None:
        self.write_cache("shared", frames=2, duration=4.0)
        entry = {"duration": 4.0, "sentences": ["event"], "timestamps": [[0.0, 2.0]]}
        self.train_json.write_text(json.dumps({"v_shared": entry}), encoding="utf-8")
        self.val_json.write_text(json.dumps({"v_shared": entry}), encoding="utf-8")
        self.mapping_json.write_text(
            json.dumps(
                {
                    "schema_version": "tinytrace.phase-b-v2.cache-map.v1",
                    "entries": [{"video_id": "v_shared", "visual_feature_path": "shared.pt"}],
                }
            ),
            encoding="utf-8",
        )
        result = self.prepare()
        self.assertFalse(result["dataset_validation"]["validation_passed"])
        self.assertEqual(result["manifest"]["samples"], [])
        self.assertEqual(
            result["skipped_samples"]["counts_by_reason"]["train_validation_leakage"], 2
        )

    def test_representative_subset_can_validate_but_is_not_training_ready(self) -> None:
        config = self.write_inputs(scope="representative_subset")
        result = self.prepare(config)
        self.assertTrue(result["dataset_validation"]["validation_passed"])
        self.assertFalse(result["dataset_validation"]["ready_for_training"])

    def test_legacy_cache_filename_exactly_matches_v1_dataset_contract(self) -> None:
        video = self.root / "video.mp4"
        video.write_bytes(b"stable-video-identity")
        annotation = self.root / "verified.json"
        annotation.write_text(
            json.dumps([{"video_id": "v_exact", "video_path": str(video), "events": []}]),
            encoding="utf-8",
        )
        v1_config = TinyTraceConfig(max_frames=32)
        v1_dataset = JsonTinyTraceDataset(
            annotation,
            v1_config,
            visual_feature_cache_dir=self.cache_root,
        )
        expected = v1_dataset.visual_feature_cache_path(0).name
        actual = legacy_v1_cache_filename(
            video_path=video,
            num_frames=32,
            image_size=v1_config.image_size,
            mobileclip_model_name=v1_config.mobileclip_model_name,
            mobileclip_checkpoint_sha256=v1_config.mobileclip_checkpoint_sha256,
            mobileclip_apply_normalization=v1_config.mobileclip_apply_normalization,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
