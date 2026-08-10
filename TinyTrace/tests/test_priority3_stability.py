import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.train_tinytrace import (
    build_dataset,
    build_loader,
    collect_predictions,
    configure_cuda_determinism,
    run_epoch,
    save_checkpoint,
    summarize_prediction_artifacts,
)
from test_vision import FakeMobileCLIPBackbone
from tinytrace.config import TinyTraceConfig
from tinytrace.data import JsonTinyTraceDataset, SyntheticTinyTraceDataset, tinytrace_collate_fn
from tinytrace.model import TinyTraceModel
from tinytrace.parsing import decode_event_sequence
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer
from tinytrace.training import (
    CHECKPOINT_FORMAT_VERSION,
    EarlyStopping,
    JsonlLogger,
    TrainingConfig,
    TrainingProfile,
    build_named_optimizer,
    build_warmup_cosine_scheduler,
    capture_rng_state,
    prune_periodic_checkpoints,
    resolve_amp_settings,
    restore_rng_state,
    validate_checkpoint_version,
)


class RepeatedDataset(Dataset):
    def __init__(self, sample: dict, size: int) -> None:
        self.sample = sample
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        del index
        return self.sample


def make_model_and_dataset(
    size: int = 2,
    dropout: float = 0.0,
) -> tuple[TinyTraceModel, Dataset, TinyTraceConfig]:
    config = TinyTraceConfig(
        image_size=16,
        max_frames=1,
        max_events=1,
        max_caption_tokens=5,
        min_caption_tokens=1,
        max_generated_tokens=25,
        d_model=24,
        num_heads=4,
        num_layers=1,
        dropout=dropout,
    )
    sample = SyntheticTinyTraceDataset(config, size=1, seed=7)[0]
    model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
    return model, RepeatedDataset(sample, size), config


class Priority3StabilityTests(unittest.TestCase):
    def test_deterministic_cuda_configures_cublas_before_training(self) -> None:
        original = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            configured = configure_cuda_determinism(
                deterministic=True,
                device=torch.device("cuda"),
            )
            self.assertEqual(configured, ":4096:8")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertIsNone(
                configure_cuda_determinism(
                    deterministic=False,
                    device=torch.device("cuda"),
                )
            )
        finally:
            if original is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = original

    def test_training_profile_defaults_and_unknown_field_rejection(self) -> None:
        profile = TrainingProfile.from_dict(
            {"epochs": 2, "init_checkpoint": "bootstrap/best-primary-metric.pt"}
        )

        self.assertEqual(profile.accumulation_steps, 1)
        self.assertTrue(profile.deterministic)
        self.assertEqual(profile.monitor, "val_loss")
        self.assertEqual(profile.init_checkpoint, "bootstrap/best-primary-metric.pt")
        with self.assertRaisesRegex(ValueError, "Unknown training profile"):
            TrainingProfile.from_dict({"future_option": True})

    def test_training_profile_rejects_invalid_optimization(self) -> None:
        with self.assertRaisesRegex(ValueError, "accumulation_steps"):
            TrainingProfile(accumulation_steps=0)
        with self.assertRaisesRegex(ValueError, "monitor must be one of"):
            TrainingProfile(monitor="unknown_metric")
        with self.assertRaisesRegex(ValueError, "cannot be used with stage-2"):
            TrainingProfile(
                visual_feature_cache_dir="features",
                require_visual_feature_cache=True,
                stage2_start_epoch=2,
            )

    def test_final_training_profile_is_canonical_and_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "final_train_qvh500.json"
        profile = TrainingProfile.from_json(path)

        self.assertEqual(profile.monitor, "qvh_mAP")
        self.assertEqual(profile.monitor_mode, "max")
        self.assertEqual(
            profile.model_config,
            "TinyTrace/configs/tinytrace_qvhighlights_phase_a.json",
        )
        self.assertEqual(profile.checkpoint_keep, 3)
        self.assertFalse(profile.allow_random_frames)

    def test_phase_a_v3_profile_locks_dense_frozen_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs"
        profile = TrainingProfile.from_json(root / "train_qvhighlights_phase_a_v3.json")
        config = TinyTraceConfig.from_json(root / "tinytrace_qvhighlights_phase_a_v3.json")

        self.assertEqual(config.max_frames, 128)
        self.assertTrue(config.phase_a_dense_saliency)
        self.assertEqual(config.phase_a_bin_count, 75)
        self.assertEqual(config.phase_a_bin_size_seconds, 2.0)
        self.assertEqual(config.max_caption_tokens, 0)
        self.assertEqual(config.time_loss_weight, 0.0)
        self.assertEqual(config.score_loss_weight, 0.0)
        self.assertEqual(config.boundary_loss_weight, 0.0)
        self.assertEqual(profile.lr, 0.0001)
        self.assertEqual(profile.gradient_clip, 5.0)
        self.assertEqual(profile.accumulation_steps, 8)
        self.assertEqual(profile.stage2_start_epoch, 0)
        self.assertTrue(profile.require_visual_feature_cache)
        self.assertEqual(profile.monitor, "qvh_mean_score_proxy_Good_mAP")

    def test_phase_a_v4_warmstart_profile_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs"
        profile = TrainingProfile.from_json(root / "train_qvhighlights_phase_a_v4_warmstart.json")
        config = TinyTraceConfig.from_json(root / "tinytrace_qvhighlights_phase_a_v4.json")

        self.assertEqual(profile.lr, 0.00005)
        self.assertEqual(profile.epochs, 12)
        self.assertEqual(profile.monitor, "qvh_mean_score_proxy_Good_mAP")
        self.assertEqual(profile.init_checkpoint, "")
        self.assertTrue(profile.require_visual_feature_cache)
        self.assertTrue(config.phase_a_dense_saliency)
        self.assertAlmostEqual(config.saliency_positive_weight, 4.68293643)

    def test_dropout_candidate_does_not_change_baseline_default(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs"
        baseline = TinyTraceConfig.from_json(root / "tinytrace_baseline.json")
        candidate = TinyTraceConfig.from_json(root / "tinytrace_dropout_010.json")

        self.assertEqual(baseline.dropout, 0.0)
        self.assertEqual(candidate.dropout, 0.1)
        baseline_without_dropout = baseline.to_dict()
        candidate_without_dropout = candidate.to_dict()
        baseline_without_dropout.pop("dropout")
        candidate_without_dropout.pop("dropout")
        self.assertEqual(baseline_without_dropout, candidate_without_dropout)

    def test_epoch_seeded_loaders_have_repeatable_order(self) -> None:
        config = TinyTraceConfig(image_size=16, max_frames=1)
        dataset = SyntheticTinyTraceDataset(config, size=8, seed=11)

        def order(seed: int) -> list[float]:
            loader = build_loader(dataset, 1, True, 0, seed, False)
            return [float(batch["frame_times"][0, 0]) + float(batch["token_ids"][0].sum()) for batch in loader]

        self.assertEqual(order(13), order(13))
        self.assertNotEqual(order(13), order(14))

    def test_real_validation_dataset_can_disable_random_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps([{"events": []}]), encoding="utf-8")
            dataset = build_dataset(path.as_posix(), TinyTraceConfig(max_frames=1), directory, False)

            with self.assertRaisesRegex(ValueError, "random fallback frames are disabled"):
                dataset[0]

    def test_cached_and_uncached_validation_frames_are_identical(self) -> None:
        class DeterministicDataset(JsonTinyTraceDataset):
            decode_calls = 0

            def _load_video_frames(self, video_path: str, num_frames: int):
                del video_path
                self.decode_calls += 1
                return (
                    torch.full(
                        (num_frames, 3, self.config.image_size, self.config.image_size),
                        0.25,
                    ),
                    torch.arange(num_frames, dtype=torch.float32),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "validation.json"
            annotations.write_text(
                json.dumps([{"video_path": "video.mp4", "num_frames": 2, "events": []}]),
                encoding="utf-8",
            )
            dataset = DeterministicDataset(
                annotations,
                TinyTraceConfig(image_size=16, max_frames=2),
                frame_cache_dir=root / "cache",
                allow_random_frames=False,
            )
            uncached = dataset[0]
            cached = dataset[0]

        self.assertEqual(dataset.decode_calls, 1)
        self.assertTrue(torch.equal(uncached["frames"], cached["frames"]))
        self.assertTrue(torch.equal(uncached["frame_times"], cached["frame_times"]))

    def test_gradient_accumulation_matches_equivalent_batch(self) -> None:
        model, dataset, _ = make_model_and_dataset(size=2)
        accumulated = copy.deepcopy(model)
        direct = copy.deepcopy(model)
        accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=1e-3)
        direct_optimizer = torch.optim.SGD(direct.parameters(), lr=1e-3)
        micro_loader = DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn)
        direct_loader = DataLoader(dataset, batch_size=2, collate_fn=tinytrace_collate_fn)

        accumulated_metrics, accumulated_steps = run_epoch(
            accumulated,
            micro_loader,
            torch.device("cpu"),
            accumulated_optimizer,
            0.0,
            1,
            0,
            "train",
            accumulation_steps=2,
        )
        direct_metrics, direct_steps = run_epoch(
            direct,
            direct_loader,
            torch.device("cpu"),
            direct_optimizer,
            0.0,
            1,
            0,
            "train",
        )

        self.assertEqual((accumulated_steps, direct_steps), (1, 1))
        self.assertEqual(accumulated_metrics["optimizer_steps"], 1)
        for left, right in zip(accumulated.parameters(), direct.parameters()):
            self.assertTrue(torch.allclose(left, right, rtol=1e-5, atol=1e-6))

    def test_partial_accumulation_window_matches_direct_windows(self) -> None:
        model, dataset, _ = make_model_and_dataset(size=3)
        accumulated = copy.deepcopy(model)
        direct = copy.deepcopy(model)
        accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=1e-3)
        direct_optimizer = torch.optim.SGD(direct.parameters(), lr=1e-3)

        accumulated_metrics, accumulated_steps = run_epoch(
            accumulated,
            DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
            torch.device("cpu"),
            accumulated_optimizer,
            0.0,
            1,
            0,
            "train",
            accumulation_steps=2,
        )
        direct_metrics, direct_steps = run_epoch(
            direct,
            DataLoader(dataset, batch_size=2, collate_fn=tinytrace_collate_fn),
            torch.device("cpu"),
            direct_optimizer,
            0.0,
            1,
            0,
            "train",
        )

        self.assertEqual((accumulated_steps, direct_steps), (2, 2))
        self.assertEqual(accumulated_metrics["optimizer_steps"], 2)
        self.assertEqual(direct_metrics["optimizer_steps"], 2)
        for left, right in zip(accumulated.parameters(), direct.parameters()):
            self.assertTrue(torch.allclose(left, right, rtol=1e-5, atol=1e-6))

    def test_accumulation_advances_scheduler_only_at_boundaries_and_logs_counters(self) -> None:
        model, dataset, _ = make_model_and_dataset(size=3)
        optimizer = build_named_optimizer(model, TrainingConfig(learning_rate=1e-3))
        scheduler = build_warmup_cosine_scheduler(optimizer, 2, 0, 0.1)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "events.jsonl")
            metrics, global_step = run_epoch(
                model,
                DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
                torch.device("cpu"),
                optimizer,
                0.0,
                1,
                0,
                "train",
                scheduler=scheduler,
                event_logger=logger,
                global_micro_step=10,
                accumulation_steps=2,
            )
            records = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]

        step_records = [record for record in records if record["record_type"] == "step"]
        self.assertEqual(global_step, 2)
        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual([record["global_micro_step"] for record in step_records], [11, 12, 13])
        self.assertNotIn("optimizer_step_performed", step_records[0])
        self.assertTrue(step_records[1]["optimizer_step_performed"])
        self.assertTrue(step_records[2]["optimizer_step_performed"])

    def test_prediction_artifacts_are_complete_and_restore_model_mode(self) -> None:
        model, dataset, config = make_model_and_dataset(size=1)
        model.train()

        predictions = collect_predictions(
            model,
            dataset,
            config,
            torch.device("cpu"),
            checkpoint_identity="run:epoch-0001",
        )

        self.assertTrue(model.training)
        prediction = predictions[0]
        self.assertIn("raw_generated_token_ids", prediction)
        self.assertIn("generation", prediction)
        self.assertIn("parser_warnings", prediction)
        self.assertEqual(prediction["checkpoint_identity"], "run:epoch-0001")
        self.assertIn(prediction["generation"]["termination_reason"], {"eos", "max_tokens"})

    def test_parser_collects_warnings_without_changing_safe_behavior(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        warnings: list[str] = []
        parsed = decode_event_sequence(
            [config.score_token_base],
            config,
            CharTokenizer(config.text_vocab_size),
            NumericTokenizer(config.time_vocab, width=6),
            NumericTokenizer(config.score_vocab, width=3),
            warnings=warnings,
        )

        self.assertEqual(parsed, [])
        self.assertTrue(warnings)

    def test_malformed_generated_timestamp_survives_for_parser_diagnostics(self) -> None:
        model, _, _ = make_model_and_dataset(size=1)
        malformed = [2, 12, 12, 1]

        self.assertEqual(model._constrain_time_ids(malformed, clip_end=10.0), malformed)

    def test_prediction_diagnostics_report_termination_and_parser_rates(self) -> None:
        diagnostics = summarize_prediction_artifacts(
            [
                {
                    "generation": {
                        "termination_reason": "eos",
                        "forced_caption_termination": False,
                    },
                    "parser_warnings": [],
                },
                {
                    "generation": {
                        "termination_reason": "max_tokens",
                        "forced_caption_termination": True,
                    },
                    "parser_warnings": ["malformed"],
                },
            ]
        )

        self.assertEqual(diagnostics["eos_termination_rate"], 0.5)
        self.assertEqual(diagnostics["maximum_budget_termination_rate"], 0.5)
        self.assertEqual(diagnostics["forced_caption_termination_rate"], 0.5)
        self.assertEqual(diagnostics["parser_warning_rate"], 0.5)

    def test_rng_state_round_trip(self) -> None:
        random.seed(123)
        torch.manual_seed(123)
        state = capture_rng_state()
        expected = (random.random(), torch.rand(3))
        restore_rng_state(state)

        self.assertEqual(random.random(), expected[0])
        self.assertTrue(torch.equal(torch.rand(3), expected[1]))

    def test_checkpoint_schema_and_version_validation(self) -> None:
        model, _, config = make_model_and_dataset(size=1)
        training_config = TrainingConfig(learning_rate=1e-3)
        optimizer = build_named_optimizer(model, training_config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                config,
                1,
                1.0,
                [],
                training_config=training_config,
                global_step=2,
                global_micro_step=4,
                global_examples=8,
                selection_state={"monitor": "val_loss"},
                run_metadata={"run_id": "test"},
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(validate_checkpoint_version(payload), CHECKPOINT_FORMAT_VERSION)
        self.assertEqual(payload["global_micro_step"], 4)
        self.assertIn("rng_state", payload)
        self.assertEqual(payload["artifact_type"], "resumable_training_checkpoint")
        with self.assertRaisesRegex(ValueError, "newer than supported"):
            validate_checkpoint_version({"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION + 1})

    def test_checkpoint_rng_restoration_reproduces_next_dropout_update(self) -> None:
        base, dataset, config = make_model_and_dataset(size=1, dropout=0.2)
        uninterrupted = copy.deepcopy(base)
        interrupted = copy.deepcopy(base)
        training_config = TrainingConfig(learning_rate=1e-3, weight_decay=0.0)
        initial_rng = capture_rng_state()

        restore_rng_state(initial_rng)
        full_optimizer = build_named_optimizer(uninterrupted, training_config)
        full_scheduler = build_warmup_cosine_scheduler(full_optimizer, 2, 0, 0.1)
        for epoch in (1, 2):
            run_epoch(
                uninterrupted,
                DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
                torch.device("cpu"),
                full_optimizer,
                0.0,
                epoch,
                0,
                "train",
                scheduler=full_scheduler,
                global_step=epoch - 1,
            )

        restore_rng_state(initial_rng)
        split_optimizer = build_named_optimizer(interrupted, training_config)
        split_scheduler = build_warmup_cosine_scheduler(split_optimizer, 2, 0, 0.1)
        run_epoch(
            interrupted,
            DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
            torch.device("cpu"),
            split_optimizer,
            0.0,
            1,
            0,
            "train",
            scheduler=split_scheduler,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "resume.pt"
            save_checkpoint(
                checkpoint_path,
                interrupted,
                split_optimizer,
                config,
                1,
                1.0,
                [],
                scheduler=split_scheduler,
                training_config=training_config,
                global_step=1,
                global_micro_step=1,
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        resumed = copy.deepcopy(base)
        resumed.load_state_dict(payload["model_state"])
        resumed_optimizer = build_named_optimizer(resumed, training_config)
        resumed_scheduler = build_warmup_cosine_scheduler(resumed_optimizer, 2, 0, 0.1)
        resumed_optimizer.load_state_dict(payload["optimizer_state"])
        resumed_scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload["rng_state"])
        run_epoch(
            resumed,
            DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
            torch.device("cpu"),
            resumed_optimizer,
            0.0,
            2,
            0,
            "train",
            scheduler=resumed_scheduler,
            global_step=1,
            global_micro_step=1,
        )

        for left, right in zip(uninterrupted.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(full_scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_checkpoint_retention_preserves_newest_periodic_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in range(1, 6):
                (root / f"epoch-{epoch:04d}.pt").write_bytes(b"checkpoint")
            (root / "latest.pt").write_bytes(b"latest")
            removed = prune_periodic_checkpoints(root, keep=2)

            self.assertEqual(len(removed), 3)
            self.assertEqual(
                sorted(path.name for path in root.glob("epoch-*.pt")),
                ["epoch-0004.pt", "epoch-0005.pt"],
            )
            self.assertTrue((root / "latest.pt").is_file())

    def test_json_logger_surfaces_io_failure_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(directory)
            success = logger.log({"record_type": "test"})

            self.assertFalse(success)
            self.assertTrue(logger.failures)

    def test_nonfinite_gradients_are_logged_before_training_aborts(self) -> None:
        model, dataset, _ = make_model_and_dataset(size=1)
        with torch.no_grad():
            model.text_head.weight.fill_(float("nan"))
        optimizer = build_named_optimizer(model, TrainingConfig(learning_rate=1e-3))
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "events.jsonl")
            with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient norm"):
                run_epoch(
                    model,
                    DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
                    torch.device("cpu"),
                    optimizer,
                    1.0,
                    1,
                    0,
                    "train",
                    event_logger=logger,
                )
            records = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[-1]["record_type"], "nonfinite_gradient")

    def test_max_mode_early_stopping_tracks_best_step(self) -> None:
        stopping = EarlyStopping(patience=1, mode="max", monitor="caption_exact_match")

        self.assertEqual(stopping.update(0.2, 1, global_step=3), (True, False))
        self.assertEqual(stopping.update(0.1, 2, global_step=6), (False, True))
        self.assertEqual(stopping.best_step, 3)
        self.assertIn("caption_exact_match", stopping.stop_reason)

    def test_cpu_bfloat16_amp_has_finite_loss_and_matching_shapes(self) -> None:
        model, dataset, _ = make_model_and_dataset(size=1)
        batch = tinytrace_collate_fn([dataset[0]])
        model.eval()
        with torch.no_grad():
            fp32 = model(
                batch["frames"],
                batch["frame_times"],
                batch["token_ids"],
                labels=batch["token_ids"],
                label_types=batch["label_types"],
                frame_mask=batch["frame_mask"],
            )
            settings = resolve_amp_settings("bf16", torch.device("cpu"))
            with torch.autocast("cpu", dtype=settings.dtype):
                mixed = model(
                    batch["frames"],
                    batch["frame_times"],
                    batch["token_ids"],
                    labels=batch["token_ids"],
                    label_types=batch["label_types"],
                    frame_mask=batch["frame_mask"],
                )

        self.assertEqual(fp32.logits.shape, mixed.logits.shape)
        self.assertTrue(torch.isfinite(fp32.loss))
        self.assertTrue(torch.isfinite(mixed.loss))

    def test_dropout_is_disabled_in_eval_and_active_in_training(self) -> None:
        config = TinyTraceConfig(
            image_size=16,
            max_frames=1,
            dropout=0.5,
            d_model=24,
            num_heads=4,
            num_layers=1,
        )
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        sample = tinytrace_collate_fn([SyntheticTinyTraceDataset(config, size=1, seed=7)[0]])
        arguments = (
            sample["frames"],
            sample["frame_times"],
            sample["token_ids"],
        )

        model.eval()
        first_eval = model(*arguments, frame_mask=sample["frame_mask"]).logits
        second_eval = model(*arguments, frame_mask=sample["frame_mask"]).logits
        model.train()
        first_train = model(*arguments, frame_mask=sample["frame_mask"]).logits
        second_train = model(*arguments, frame_mask=sample["frame_mask"]).logits

        self.assertTrue(torch.equal(first_eval, second_eval))
        self.assertFalse(torch.equal(first_train, second_train))


if __name__ == "__main__":
    unittest.main()
