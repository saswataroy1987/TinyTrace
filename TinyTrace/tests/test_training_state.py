import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_tinytrace import run_epoch, save_checkpoint
from tinytrace.config import TinyTraceConfig
from tinytrace.data import SyntheticTinyTraceDataset
from tinytrace.model import TinyTraceModel
from tinytrace.training import (
    OPTIMIZER_FORMAT_VERSION,
    EarlyStopping,
    JsonlLogger,
    TrainingConfig,
    build_named_optimizer,
    build_warmup_cosine_scheduler,
    load_optimizer_state_compat,
    optimizer_group_summary,
    resolve_amp_settings,
    warmup_cosine_multiplier,
)
from torch.utils.data import DataLoader

from test_vision import FakeMobileCLIPBackbone
from tinytrace.data import tinytrace_collate_fn


class TrainingStateTests(unittest.TestCase):
    def test_named_optimizer_assigns_every_parameter_once(self) -> None:
        model = TinyTraceModel(
            TinyTraceConfig(max_frames=1),
            mobileclip_backbone=FakeMobileCLIPBackbone(),
        )
        optimizer = build_named_optimizer(
            model,
            TrainingConfig(learning_rate=1e-3, weight_decay=0.01),
        )

        names = [group["group_name"] for group in optimizer.param_groups]
        assigned = [parameter for group in optimizer.param_groups for parameter in group["params"]]

        self.assertEqual(len(assigned), len({id(parameter) for parameter in assigned}))
        self.assertEqual({id(parameter) for parameter in assigned}, {id(p) for p in model.parameters()})
        self.assertIn("mobileclip.decay", names)
        self.assertIn("task_heads.no_decay", names)
        self.assertTrue(
            all(group["weight_decay"] == 0 for group in optimizer.param_groups if group["group_name"].endswith("no_decay"))
        )

    def test_stage_transition_keeps_optimizer_and_existing_adam_state(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        optimizer = build_named_optimizer(model, TrainingConfig(learning_rate=1e-3))
        tracked_parameter = model.text_head.weight
        tracked_parameter.square().mean().backward()
        optimizer.step()
        previous_step = optimizer.state[tracked_parameter]["step"].clone()
        optimizer_identifier = id(optimizer)

        model.set_visual_encoder_trainable(True, strategy="conv_exp")

        self.assertEqual(id(optimizer), optimizer_identifier)
        self.assertTrue(torch.equal(optimizer.state[tracked_parameter]["step"], previous_step))
        visual_parameters = {
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("visual_encoder.mobileclip.") and parameter.requires_grad
        }
        optimizer_parameters = {
            parameter for group in optimizer.param_groups for parameter in group["params"]
        }
        self.assertTrue(visual_parameters.issubset(optimizer_parameters))

    def test_legacy_optimizer_state_migrates_to_named_groups(self) -> None:
        model = TinyTraceModel(
            TinyTraceConfig(max_frames=1),
            mobileclip_backbone=FakeMobileCLIPBackbone(),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        legacy = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)
        tracked = model.text_head.weight
        tracked.square().mean().backward()
        legacy.step()
        expected_step = legacy.state[tracked]["step"].clone()

        training_config = TrainingConfig(learning_rate=1e-3, weight_decay=0.01)
        named = build_named_optimizer(model, training_config)
        load_optimizer_state_compat(
            named,
            model,
            legacy.state_dict(),
            training_config,
            optimizer_format_version=1,
        )

        self.assertTrue(torch.equal(named.state[tracked]["step"], expected_step))

    def test_warmup_cosine_schedule_boundaries(self) -> None:
        values = [
            warmup_cosine_multiplier(step, total_steps=10, warmup_steps=2, min_lr_ratio=0.1)
            for step in range(11)
        ]

        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(values[index] >= values[index + 1] for index in range(1, 10)))

    def test_training_config_and_amp_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmup_ratio"):
            TrainingConfig(warmup_ratio=1.1)
        with self.assertRaisesRegex(ValueError, "amp_mode"):
            TrainingConfig(amp_mode="invalid")
        self.assertFalse(resolve_amp_settings("auto", torch.device("cpu")).enabled)
        self.assertEqual(resolve_amp_settings("bf16", torch.device("cpu")).dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "only on CUDA"):
            resolve_amp_settings("fp16", torch.device("cpu"))

    def test_early_stopping_is_deterministic(self) -> None:
        stopping = EarlyStopping(patience=2, min_delta=0.1, min_epochs=2)

        self.assertEqual(stopping.update(1.0, epoch=1), (True, False))
        self.assertEqual(stopping.update(0.95, epoch=2), (False, False))
        self.assertEqual(stopping.update(0.94, epoch=3), (False, True))

    def test_early_stopping_does_not_count_metrics_during_warmup(self) -> None:
        stopping = EarlyStopping(patience=1)

        self.assertEqual(stopping.update(1.0, epoch=1, active=False), (True, False))
        self.assertEqual(stopping.update(2.0, epoch=2, active=False), (False, False))
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertEqual(stopping.update(2.0, epoch=3, active=True), (False, True))

    def test_run_epoch_logs_tasks_gradients_and_learning_rates(self) -> None:
        config = TinyTraceConfig(image_size=16, max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        dataset = SyntheticTinyTraceDataset(config, size=1, seed=7)
        loader = DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn)
        training_config = TrainingConfig(learning_rate=1e-3, warmup_ratio=0.0)
        optimizer = build_named_optimizer(model, training_config)
        scheduler = build_warmup_cosine_scheduler(optimizer, 2, 0, 0.1)

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "training.jsonl"
            metrics, global_step = run_epoch(
                model,
                loader,
                torch.device("cpu"),
                optimizer,
                gradient_clip=1.0,
                epoch=1,
                log_every=0,
                split="train",
                scheduler=scheduler,
                amp_settings=resolve_amp_settings("off", torch.device("cpu")),
                event_logger=JsonlLogger(log_path),
            )
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(global_step, 1)
        self.assertIn("time", metrics["loss_components"])
        self.assertIn("score", metrics["loss_components"])
        self.assertIn("text", metrics["loss_components"])
        self.assertIsNotNone(metrics["mean_gradient_norm"])
        self.assertTrue(metrics["learning_rates"])
        self.assertEqual([record["record_type"] for record in records], ["step", "epoch"])

    def test_priority2_checkpoint_restores_optimizer_and_scheduler_layout(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        training_config = TrainingConfig(learning_rate=1e-3)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        optimizer = build_named_optimizer(model, training_config)
        scheduler = build_warmup_cosine_scheduler(optimizer, 10, 2, 0.1)
        early_stopping = EarlyStopping(patience=2)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "priority2.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                epoch=1,
                best_loss=1.0,
                history=[],
                scheduler=scheduler,
                training_config=training_config,
                early_stopping=early_stopping,
                global_step=1,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["optimizer_format_version"], OPTIMIZER_FORMAT_VERSION)
        self.assertIsNotNone(checkpoint["scheduler_state"])
        self.assertEqual(checkpoint["training_config"], training_config.to_dict())
        self.assertEqual(checkpoint["global_step"], 1)
        self.assertTrue(optimizer_group_summary(optimizer))


if __name__ == "__main__":
    unittest.main()
