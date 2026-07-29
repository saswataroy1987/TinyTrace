import tempfile
import types
import unittest
from pathlib import Path

import torch

from tinytrace import (
    EventParseError,
    TinyTraceConfig,
    TinyTraceModel,
    decode_event_sequence,
    serialize_example,
)
from tinytrace.model import TinyTraceOutput
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer

from test_vision import FakeMobileCLIPBackbone


def tokenizers(config: TinyTraceConfig):
    return (
        CharTokenizer(config.text_vocab_size),
        NumericTokenizer(config.time_vocab, width=6),
        NumericTokenizer(config.score_vocab, width=3),
    )


class CorePipelineTests(unittest.TestCase):
    def test_event_serialization_and_parser_round_trip(self) -> None:
        config = TinyTraceConfig()
        text, time, score = tokenizers(config)
        events = [{"timestamp": [1.2, 3.4], "score": [4.5], "caption": "person runs"}]

        token_ids, label_types, prompt_length = serialize_example(
            events,
            "localize the event",
            config,
            text,
            time,
            score,
        )
        decoded = decode_event_sequence(
            token_ids[prompt_length:],
            config,
            text,
            time,
            score,
            strict=True,
        )

        self.assertEqual(decoded, events)
        self.assertEqual(len(token_ids), len(label_types))
        self.assertTrue(all(label_type == -1 for label_type in label_types[:prompt_length]))

    def test_serialization_rejects_more_than_configured_events(self) -> None:
        config = TinyTraceConfig(max_events=1)
        text, time, score = tokenizers(config)
        event = {"timestamp": [0.0, 1.0], "score": [3.0], "caption": "action"}

        with self.assertRaisesRegex(ValueError, "max_events=1"):
            serialize_example([event, event], "localize", config, text, time, score)

    def test_malformed_parser_is_safe_by_default_and_strict_on_request(self) -> None:
        config = TinyTraceConfig()
        text, time, score = tokenizers(config)
        malformed = [config.total_token_vocab + 10, config.sync_token_id, config.eos_token_id]

        self.assertEqual(
            decode_event_sequence(malformed, config, text, time, score),
            [],
        )
        with self.assertRaises(EventParseError):
            decode_event_sequence(malformed, config, text, time, score, strict=True)

    def test_lcem_forward_shape_and_loss(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        text, time, score = tokenizers(config)
        token_ids, label_types, _ = serialize_example(
            [{"timestamp": [0.0, 1.0], "score": [3.0], "caption": "action"}],
            "localize",
            config,
            text,
            time,
            score,
        )
        tokens = torch.tensor([token_ids])
        types_tensor = torch.tensor([label_types])

        output = model(
            torch.rand(1, 1, 3, 32, 32),
            torch.zeros(1, 1),
            tokens,
            labels=tokens,
            label_types=types_tensor,
        )

        expected_length = config.compressed_visual_tokens + config.time_tokens_per_frame + len(token_ids)
        self.assertEqual(output.logits.shape, (1, expected_length, config.total_token_vocab))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(
            set(output.loss_components),
            {"text", "caption_sync", "time", "time_sync", "score", "score_sync", "boundary"},
        )
        self.assertTrue(all(count > 0 for count in output.target_counts.values()))
        self.assertTrue(
            torch.allclose(
                output.loss,
                sum(output.weighted_loss_components.values()),
            )
        )

    def test_multi_task_loss_weights_change_only_aggregation(self) -> None:
        config = TinyTraceConfig(max_frames=1, score_loss_weight=2.0)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        components = {
            "time": torch.tensor(1.0, requires_grad=True),
            "score": torch.tensor(2.0, requires_grad=True),
            "text": torch.tensor(3.0, requires_grad=True),
        }

        total, weighted = model._combine_loss_components(components)

        self.assertIsNotNone(total)
        self.assertEqual(total.item(), 2.0)
        self.assertEqual(weighted["time"].item(), 0.25)
        self.assertEqual(weighted["score"].item(), 1.0)
        self.assertEqual(weighted["text"].item(), 0.75)
        self.assertEqual(components["score"].item(), 2.0)

    def test_ignored_padding_does_not_change_loss_components(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone()).eval()
        text, time, score = tokenizers(config)
        token_ids, label_types, _ = serialize_example(
            [{"timestamp": [0.0, 1.0], "score": [3.0], "caption": "action"}],
            "localize",
            config,
            text,
            time,
            score,
        )
        tokens = torch.tensor([token_ids])
        types_tensor = torch.tensor([label_types])
        padded_tokens = torch.cat([tokens, torch.zeros(1, 2, dtype=torch.long)], dim=1)
        padded_types = torch.cat([types_tensor, torch.full((1, 2), -1, dtype=torch.long)], dim=1)
        frames = torch.rand(1, 1, 3, 32, 32)
        frame_times = torch.zeros(1, 1)

        direct = model(frames, frame_times, tokens, labels=tokens, label_types=types_tensor)
        padded = model(
            frames,
            frame_times,
            padded_tokens,
            labels=padded_tokens,
            label_types=padded_types,
        )

        self.assertEqual(set(direct.loss_components), set(padded.loss_components))
        for name in direct.loss_components:
            self.assertTrue(torch.allclose(direct.loss_components[name], padded.loss_components[name]))

    def test_multi_task_loss_reaches_all_task_heads(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        text, time, score = tokenizers(config)
        token_ids, label_types, _ = serialize_example(
            [{"timestamp": [0.0, 1.0], "score": [3.0], "caption": "action"}],
            "localize",
            config,
            text,
            time,
            score,
        )
        tokens = torch.tensor([token_ids])
        output = model(
            torch.rand(1, 1, 3, 32, 32),
            torch.zeros(1, 1),
            tokens,
            labels=tokens,
            label_types=torch.tensor([label_types]),
        )

        output.loss.backward()

        self.assertIsNotNone(model.text_head.weight.grad)
        self.assertIsNotNone(model.time_head.weight.grad)
        self.assertIsNotNone(model.score_head.weight.grad)

    def test_forward_requires_paired_and_shape_aligned_labels(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        frames = torch.rand(1, 1, 3, 16, 16)
        times = torch.zeros(1, 1)
        tokens = torch.tensor([[config.bos_token_id, config.video_token_id]])

        with self.assertRaisesRegex(ValueError, "both be provided"):
            model(frames, times, tokens, labels=tokens)
        with self.assertRaisesRegex(ValueError, "same shape"):
            model(
                frames,
                times,
                tokens,
                labels=torch.ones(1, 1, dtype=torch.long),
                label_types=torch.ones(1, 1, dtype=torch.long),
            )

    def test_generation_switches_time_score_caption(self) -> None:
        config = TinyTraceConfig(
            timestamp_value_count=0,
            score_value_count=0,
            min_caption_tokens=0,
            max_caption_tokens=0,
            max_events=1,
        )
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        def scripted_forward(self, frames, frame_times, token_ids, **kwargs):
            batch = token_ids.size(0)
            text_logits = torch.full((batch, 1, config.text_vocab_size + 1), float("-inf"))
            time_logits = torch.full((batch, 1, len(config.time_vocab)), float("-inf"))
            score_logits = torch.full((batch, 1, len(config.score_vocab)), float("-inf"))
            text_logits[:, :, config.text_vocab_size] = 0.0
            time_logits[:, :, 0] = 0.0
            score_logits[:, :, 0] = 0.0
            return TinyTraceOutput(
                loss=None,
                logits=torch.empty(batch, 1, config.total_token_vocab),
                text_logits=text_logits,
                time_logits=time_logits,
                score_logits=score_logits,
            )

        model.forward = types.MethodType(scripted_forward, model)
        prompt = torch.tensor([[config.bos_token_id, config.video_token_id]])
        generated = model.generate(
            torch.rand(1, 1, 3, 16, 16),
            torch.zeros(1, 1),
            prompt,
            max_new_tokens=4,
        )

        self.assertEqual(
            generated[0, -4:].tolist(),
            [
                config.sync_token_id,
                config.sync_token_id,
                config.sync_token_id,
                config.eos_token_id,
            ],
        )

    def test_numeric_generation_forces_sync_at_fixed_protocol_lengths(self) -> None:
        config = TinyTraceConfig(
            max_frames=1,
            max_caption_tokens=0,
            min_caption_tokens=0,
            max_events=1,
        )
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        def scripted_forward(self, frames, frame_times, token_ids, **kwargs):
            batch = token_ids.size(0)
            text_logits = torch.zeros(batch, 1, config.text_vocab_size + 1)
            time_logits = torch.zeros(batch, 1, len(config.time_vocab))
            score_logits = torch.zeros(batch, 1, len(config.score_vocab))
            boundary_logits = torch.tensor([[[1.0, 0.0]]]).expand(batch, -1, -1)
            return TinyTraceOutput(
                loss=None,
                logits=torch.empty(batch, 1, config.total_token_vocab),
                text_logits=text_logits,
                time_logits=time_logits,
                score_logits=score_logits,
                boundary_logits=boundary_logits,
            )

        model.forward = types.MethodType(scripted_forward, model)
        prompt = torch.tensor([[config.bos_token_id, config.video_token_id]])
        generated = model.generate(
            torch.rand(1, 1, 3, 16, 16),
            torch.tensor([[9.0]]),
            prompt,
            max_new_tokens=19,
            task_mode="highlight",
        )
        new_tokens = generated[0, prompt.size(1) :].tolist()

        self.assertEqual(len(new_tokens), 19)
        self.assertEqual(new_tokens[13], config.sync_token_id)
        self.assertEqual(new_tokens[17], config.sync_token_id)
        self.assertEqual(new_tokens[18], config.eos_token_id)
        self.assertNotIn(config.sync_token_id, new_tokens[:13])
        self.assertNotIn(config.sync_token_id, new_tokens[14:17])

    def test_checkpoint_config_and_state_round_trip(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "tinytrace.pt"
            torch.save(
                {"model_state": model.state_dict(), "config": config.to_dict()},
                checkpoint_path,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            restored_config = TinyTraceConfig.from_dict(checkpoint["config"])
            restored = TinyTraceModel(
                restored_config,
                mobileclip_backbone=FakeMobileCLIPBackbone(),
            )
            restored.load_state_dict(checkpoint["model_state"])

        self.assertEqual(restored_config, config)
        self.assertTrue(torch.equal(restored.text_head.weight, model.text_head.weight))


if __name__ == "__main__":
    unittest.main()
