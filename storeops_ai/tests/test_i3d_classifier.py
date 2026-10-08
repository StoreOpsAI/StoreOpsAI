import unittest

from config.config import CATEGORIES, CATEGORY_THRESHOLDS
from models.i3d_classifier import I3DClassifier


class I3DClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = I3DClassifier.__new__(I3DClassifier)

    def test_scores_use_trained_categories(self):
        scores = {category: 0.0 for category in CATEGORIES}
        self.assertEqual(CATEGORIES, ["정상", "쓰러짐", "쓰레기 투기", "절도"])
        self.assertEqual(len(I3DClassifier.validate_scores(scores)), 4)

    def test_untrained_categories_are_rejected(self):
        scores = {category: 0.0 for category in CATEGORIES}
        scores["싸움"] = 0.0
        with self.assertRaises(ValueError):
            I3DClassifier.validate_scores(scores)

    def test_anomaly_threshold_does_not_depend_on_normal_score(self):
        above = min(CATEGORY_THRESHOLDS["절도"] + 0.01, 1.0)
        scores = {"정상": 0.99, "쓰러짐": 0.01, "쓰레기 투기": 0.01, "절도": above}

        category, confidence, _ = self.classifier.decide(scores)

        self.assertEqual(category, "절도")
        self.assertEqual(confidence, above)

    def test_score_equal_to_threshold_does_not_create_candidate(self):
        scores = {"정상": 0.1, "쓰러짐": CATEGORY_THRESHOLDS["쓰러짐"], "쓰레기 투기": 0.1, "절도": 0.1}

        category, _, _ = self.classifier.decide(scores)

        self.assertEqual(category, "판정 보류")


if __name__ == "__main__":
    unittest.main()
