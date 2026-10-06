import unittest

from config.config import CATEGORIES
from models.i3d_classifier import I3DClassifier


class I3DClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = I3DClassifier.__new__(I3DClassifier)

    def test_scores_use_five_requested_categories(self):
        scores = {category: 0.0 for category in CATEGORIES}
        self.assertEqual(
            CATEGORIES,
            ["정상", "쓰러짐", "싸움", "파손", "쓰레기 투기"],
        )
        self.assertEqual(len(I3DClassifier.validate_scores(scores)), 5)

    def test_anomaly_threshold_does_not_depend_on_normal_score(self):
        scores = {
            "정상": 0.72,
            "쓰러짐": 0.01,
            "싸움": 0.0,
            "파손": 0.61,
            "쓰레기 투기": 0.01,
        }

        category, confidence, _ = self.classifier.decide(scores)

        self.assertEqual(category, "파손")
        self.assertEqual(confidence, 0.61)

    def test_fight_score_is_zero_until_model_is_retrained(self):
        scores = {category: 0.0 for category in CATEGORIES}
        scores["정상"] = 0.2
        scores["쓰러짐"] = 0.2
        scores["파손"] = 0.3
        scores["쓰레기 투기"] = 0.3

        result = I3DClassifier.validate_scores(scores)

        self.assertEqual(result["싸움"], 0.0)

    def test_score_equal_to_threshold_does_not_create_candidate(self):
        scores = {
            "정상": 0.1,
            "쓰러짐": 0.6,
            "싸움": 0.0,
            "파손": 0.1,
            "쓰레기 투기": 0.1,
        }

        category, _, _ = self.classifier.decide(scores)

        self.assertEqual(category, "판정 보류")


if __name__ == "__main__":
    unittest.main()