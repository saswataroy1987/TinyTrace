from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_phase_a_qvhighlights import (
    AnnotationValidationError,
    prepare_phase_a_dataset,
)


def source_block(start: float, score: float) -> tuple[list[list[float]], list[list[float]]]:
    return (
        [[start + offset * 0.5] for offset in range(4)],
        [[score] for _ in range(4)],
    )


class PhaseADatasetPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotations"
        self.annotations.mkdir()
        (self.root / "videos" / "train").mkdir(parents=True)
        (self.root / "videos" / "val").mkdir(parents=True)

        for split, name in (("train", "train-good.mp4"), ("train", "train-short.mp4"), ("val", "val-good.mp4")):
            (self.root / "videos" / split / name).touch()

        train_times, train_scores = source_block(2.0, 3.0)
        # This positive block is early, so the old "last annotation fits"
        # policy would incorrectly accept the 100-second media file. Phase A
        # requires the complete 150-second temporal window regardless of where
        # positives happen to occur.
        short_times, short_scores = source_block(2.0, 4.0)
        val_times, val_scores = source_block(0.0, 2.0)
        self.raw = [
            self.raw_item(1, "train-good.mp4", train_times, train_scores),
            self.raw_item(2, "train-short.mp4", short_times, short_scores),
            self.raw_item(3, "val-good.mp4", val_times, val_scores),
        ]
        self.train = [
            self.split_item(1, "train", "train-good.mp4"),
            self.split_item(2, "train", "train-short.mp4"),
        ]
        self.val = [self.split_item(3, "val", "val-good.mp4")]
        self.write_inputs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def raw_item(
        source_id: int,
        video_name: str,
        times: list[list[float]],
        scores: list[list[float]],
    ) -> dict:
        return {
            "id": source_id,
            "video": f"qvhighlights/videos/train/{video_name}",
            "times": times,
            "scores": scores,
        }

    @staticmethod
    def split_item(source_id: int, split: str, video_name: str) -> dict:
        return {
            "source_id": source_id,
            "video_path": f"videos/{split}/{video_name}",
            "instruction": f"Find highlights for query {source_id}",
            "query": f"query {source_id}",
            "task_mode": "highlight",
            "events": [{"timestamp": [0.0, 149.5], "score": [4.0]}],
        }

    def write_inputs(self) -> None:
        for name, payload in (
            ("qvh_raw_valid.json", self.raw),
            ("tinytrace_train.json", self.train),
            ("tinytrace_val.json", self.val),
        ):
            (self.annotations / name).write_text(json.dumps(payload), encoding="utf-8")

    def paths(self) -> dict[str, Path]:
        return {
            "raw_json": self.annotations / "qvh_raw_valid.json",
            "train_json": self.annotations / "tinytrace_train.json",
            "val_json": self.annotations / "tinytrace_val.json",
            "output_train_json": self.annotations / "tinytrace_phase_a_v3_train.json",
            "output_val_json": self.annotations / "tinytrace_phase_a_v3_val.json",
            "exclusions_json": self.annotations / "phase_a_v3_exclusions.json",
            "manifest_json": self.annotations / "phase_a_v3_manifest.json",
        }

    @staticmethod
    def fake_probe(path: Path, _timeout: float) -> float:
        if path.name == "train-short.mp4":
            return 100.0
        return 150.0

    def test_builds_direct_dense_bins_and_excludes_truncated_media(self) -> None:
        paths = self.paths()
        input_hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
            if name in {"raw_json", "train_json", "val_json"}
        }

        manifest = prepare_phase_a_dataset(**paths, probe_fn=self.fake_probe)

        prepared_train = json.loads(paths["output_train_json"].read_text())
        prepared_val = json.loads(paths["output_val_json"].read_text())
        exclusions = json.loads(paths["exclusions_json"].read_text())
        self.assertEqual(len(prepared_train), 1)
        self.assertEqual(len(prepared_val), 1)
        self.assertEqual(len(prepared_train[0]["dense_saliency_scores"]), 75)
        self.assertEqual(prepared_train[0]["dense_saliency_scores"][0], 0.0)
        self.assertEqual(prepared_train[0]["dense_saliency_scores"][1], 3.0)
        self.assertEqual(prepared_val[0]["dense_saliency_scores"][0], 2.0)
        self.assertNotIn("events", prepared_train[0])
        self.assertEqual(prepared_train[0]["source_id"], 1)
        self.assertEqual(prepared_train[0]["query"], "query 1")
        self.assertEqual(exclusions["count"], 1)
        self.assertEqual(exclusions["items"][0]["source_id"], 2)
        self.assertEqual(exclusions["items"][0]["reason"], "truncated_media")
        self.assertEqual(manifest["counts"]["output_train"], 1)
        self.assertEqual(manifest["schema_version"], "tinytrace.qvhighlights.phase-a.v3")
        self.assertEqual(manifest["media_validation"]["expected_duration_seconds"], 150.0)
        self.assertEqual(manifest["media_validation"]["minimum_valid_duration_seconds"], 149.5)
        self.assertIn("no -1 offset", manifest["target_contract"]["trace_offset_warning"])
        for name, expected in input_hashes.items():
            self.assertEqual(hashlib.sha256(paths[name].read_bytes()).hexdigest(), expected)

        with self.assertRaisesRegex(FileExistsError, "Immutable"):
            prepare_phase_a_dataset(**paths, probe_fn=self.fake_probe)

    def test_rejects_inconsistent_four_point_source_block_without_outputs(self) -> None:
        self.raw[0]["times"][3] = [4.0]
        self.write_inputs()
        paths = self.paths()

        with self.assertRaisesRegex(AnnotationValidationError, "four-point"):
            prepare_phase_a_dataset(**paths, probe_fn=self.fake_probe)

        self.assertFalse(paths["output_train_json"].exists())
        self.assertFalse(paths["manifest_json"].exists())

    def test_rejects_malformed_source_score(self) -> None:
        self.raw[0]["scores"][0] = [float("nan")]
        self.write_inputs()

        with self.assertRaisesRegex(AnnotationValidationError, "finite"):
            prepare_phase_a_dataset(**self.paths(), probe_fn=self.fake_probe)

    def test_rejects_split_overlap(self) -> None:
        self.val.append(self.split_item(1, "val", "train-good.mp4"))
        self.write_inputs()

        with self.assertRaisesRegex(AnnotationValidationError, "overlap"):
            prepare_phase_a_dataset(**self.paths(), probe_fn=self.fake_probe)

    def test_rejects_source_id_collision(self) -> None:
        self.raw.append(dict(self.raw[0]))
        self.write_inputs()

        with self.assertRaisesRegex(AnnotationValidationError, "collision"):
            prepare_phase_a_dataset(**self.paths(), probe_fn=self.fake_probe)

    def test_rejects_duration_tolerance_that_erases_the_window_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "below 150"):
            prepare_phase_a_dataset(
                **self.paths(),
                duration_tolerance_seconds=150.0,
                probe_fn=self.fake_probe,
            )


if __name__ == "__main__":
    unittest.main()
