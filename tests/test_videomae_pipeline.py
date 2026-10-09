import unittest
import numpy as np
from videomae_pipeline import clip_indices, training_examples, letterbox


class VideoMAETest(unittest.TestCase):
    def test_clip_edge_replication(self):
        np.testing.assert_array_equal(clip_indices(0, 3), [0]*9 + [1] + [2]*6)
        self.assertEqual(len(clip_indices(20, 40)), 16)
        np.testing.assert_array_equal(clip_indices(20, 40), np.arange(12, 28))

    def test_no_held_out_or_ambiguous_training_clips(self):
        labels = np.r_[np.zeros(100), np.full(40, -100), np.ones(100)].astype(int)
        sequences = [{'video': v, 'labels': labels} for v in ['0028_vid', '0031_vid']]
        examples = training_examples(sequences, 16, 42)
        self.assertEqual(len(examples), 16)
        self.assertEqual(sum(label == 1 for _, _, label in examples), 8)
        for sequence, center, label in examples:
            self.assertEqual(sequence, 0)
            self.assertTrue(np.all(labels[clip_indices(center, len(labels))] == label))
        self.assertEqual(examples, training_examples(sequences, 16, 42))

    def test_whole_scene_and_rgb(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        transformed = letterbox(image)
        self.assertEqual(transformed.shape, (224, 224, 3))
        self.assertEqual(transformed[112, 0, 0], 255)
        self.assertEqual(transformed[112, -1, 0], 255)
        self.assertEqual(transformed[0].sum(), 0)


if __name__ == '__main__':
    unittest.main()
