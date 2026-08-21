from __future__ import annotations

import unittest

from tinytrace.phase_b_final_causal import FinalCausalConfig, parse_event_sequence


class FinalCausalContractsTests(unittest.TestCase):
    def test_structured_events_parse_in_order(self) -> None:
        sequence = "<EVENT> <START> <T010> <END> <T025> <CAPTION> first event </EVENT> <EVENT> <START> <T030> <END> <T055> <CAPTION> second event </EVENT> <END_EVENTS>"
        events = parse_event_sequence(sequence)
        self.assertEqual([event["caption"] for event in events], ["first event", "second event"])
        self.assertLess(events[0]["start_normalized"], events[1]["start_normalized"])

    def test_context_is_bounded_without_patch_average_pooling(self) -> None:
        config = FinalCausalConfig()
        self.assertEqual(config.max_video_frames * (config.visual_slots_per_frame + config.time_tokens_per_frame), 448)
        self.assertEqual(config.patch_tokens, 64)


if __name__ == "__main__":
    unittest.main()
