import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("numpy is required for these tests; install requirements.txt first") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Classification"))

try:
    from dataset import gen_stratified_split, get_generation, parse_folder_id
    from prediction import predict_from_probs
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"missing project dependency: {exc.name}") from exc


class DatasetUtilityTests(unittest.TestCase):
    def test_parse_folder_id(self):
        self.assertEqual(parse_folder_id("25"), 25)
        self.assertEqual(parse_folder_id("6-mega-x"), 6)
        self.assertIsNone(parse_folder_id("not-a-pokemon"))

    def test_get_generation(self):
        self.assertEqual(get_generation(1), 1)
        self.assertEqual(get_generation(906), 9)
        self.assertEqual(get_generation(52, "52-alola"), 7)
        self.assertEqual(get_generation(157, "157-hisui"), 8)

    def test_stratified_split_keeps_frames_for_one_id_together(self):
        index = []
        for pokemon_id in range(1, 41):
            label = np.zeros(18, dtype=np.float32)
            label[0] = 1.0
            if pokemon_id % 2 == 0:
                label[1] = 1.0
            for frame in ("front.png", "back.png"):
                index.append((Path("sprites") / str(pokemon_id) / frame, label))

        train_idx, val_idx, test_idx = gen_stratified_split(
            index,
            val_frac=0.2,
            test_frac=0.2,
            seed=123,
        )

        split_by_id = {}
        for name, indices in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
            for idx in indices:
                pokemon_id = parse_folder_id(index[idx][0].parent.name)
                split_by_id.setdefault(pokemon_id, set()).add(name)

        self.assertTrue(train_idx)
        self.assertTrue(val_idx)
        self.assertTrue(test_idx)
        self.assertTrue(all(len(splits) == 1 for splits in split_by_id.values()))


class PredictionTests(unittest.TestCase):
    def test_gap_threshold_prediction(self):
        probs = np.array(
            [
                [0.90, 0.70, 0.10],
                [0.90, 0.50, 0.40],
                [0.90, 0.65, 0.10],
            ]
        )

        preds = predict_from_probs(probs, gap_threshold=0.25)

        np.testing.assert_array_equal(
            preds,
            np.array(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            ),
        )

    def test_prediction_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            predict_from_probs(np.array([0.2, 0.8]))


if __name__ == "__main__":
    unittest.main()
