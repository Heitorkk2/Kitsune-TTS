import unittest

import torch

from kitsune.model.commons import rand_slice_segments, slice_segments


def _reference_segments(x, starts, segment_size):
    result = x.new_zeros((x.size(0), x.size(1), segment_size))
    for batch_index, start in enumerate(starts.tolist()):
        source_start = max(start, 0)
        source_end = min(start + segment_size, x.size(2))
        if source_end > source_start:
            target_start = source_start - start
            target_end = target_start + source_end - source_start
            result[batch_index, :, target_start:target_end] = x[
                batch_index, :, source_start:source_end
            ]
    return result


class SegmentSlicingTests(unittest.TestCase):
    def test_slice_segments_matches_reference_and_zero_pads(self):
        x = torch.arange(2 * 3 * 7, dtype=torch.float32).view(2, 3, 7)
        starts = torch.tensor([2, 5])

        actual = slice_segments(x, starts, segment_size=4)

        self.assertTrue(torch.equal(actual, _reference_segments(x, starts, 4)))

    def test_rand_slice_segments_supports_missing_lengths(self):
        torch.manual_seed(7)
        x = torch.arange(2 * 2 * 3, dtype=torch.float32).view(2, 2, 3)

        segments, starts = rand_slice_segments(x, segment_size=5)

        self.assertEqual(tuple(segments.shape), (2, 2, 5))
        self.assertTrue(torch.equal(segments, _reference_segments(x, starts, 5)))

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile is unavailable")
    def test_slice_segments_compiles_as_a_full_graph(self):
        compiled = torch.compile(slice_segments, backend="eager", fullgraph=True)
        x = torch.randn(3, 4, 12)
        starts = torch.tensor([0, 4, 10])

        actual = compiled(x, starts, 5)

        self.assertTrue(torch.equal(actual, _reference_segments(x, starts, 5)))


if __name__ == "__main__":
    unittest.main()
