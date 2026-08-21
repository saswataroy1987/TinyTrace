from __future__ import annotations

import unittest

import torch

from tinytrace.phase_b_v3 import event_frame_indices, select_event_patch_features


class DirectMobileCLIPSelectionTests(unittest.TestCase):
    def test_preserves_patch_features_for_ordered_event_frames(self) -> None:
        features = torch.arange(1 * 5 * 64 * 2, dtype=torch.float32).reshape(1, 5, 64, 2)
        times = torch.tensor([[0.0, 2.0, 4.0, 6.0, 8.0]])
        mask = torch.tensor([[True, True, True, True, True]])
        segments = torch.tensor([[[0.20, 0.80]]])
        indices, selected_mask = event_frame_indices(times, mask, segments, 4)
        selected, selected_again = select_event_patch_features(features, times, mask, segments, 4)
        self.assertTrue(torch.equal(selected_mask, selected_again))
        self.assertEqual(indices[0, 0, :3].tolist(), [1, 2, 3])
        self.assertTrue(torch.equal(selected[0, 0, :3], features[0, 1:4]))

    def test_narrow_event_uses_nearest_cached_frame(self) -> None:
        features = torch.zeros((1, 3, 64, 1024))
        times = torch.tensor([[0.0, 5.0, 10.0]])
        mask = torch.tensor([[True, True, True]])
        segments = torch.tensor([[[0.44, 0.46]]])
        indices, selected = event_frame_indices(times, mask, segments, 8)
        self.assertTrue(bool(selected[0, 0, 0]))
        self.assertEqual(int(indices[0, 0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
