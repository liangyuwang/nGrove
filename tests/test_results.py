import math
import unittest

from autotrain_bench.results import perplexity_from_nll, validated_nll


class ResultMetricTests(unittest.TestCase):
    def test_nll_validation_does_not_coerce_non_numeric_json_values(self):
        invalid = (True, "1.25", None, float("nan"), float("inf"), -0.1, 10**1000)
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    validated_nll(value)

    def test_perplexity_is_derived_from_mean_nll(self):
        self.assertEqual(validated_nll(0), 0.0)
        self.assertAlmostEqual(perplexity_from_nll(1.5), math.exp(1.5))


if __name__ == "__main__":
    unittest.main()
