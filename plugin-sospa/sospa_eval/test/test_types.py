import unittest
import numpy as np
from sospa_eval.types import MetricsNormType, normalize_cost


class TestMetricsNormType(unittest.TestCase):
    """Test fixture for MetricsNormType.normalize_cost method."""

    def test_no_normalization(self):
        """Test NoNorm - should return cost unchanged."""
        norm_type = MetricsNormType.NoNorm
        cost = 10.0
        len_x = 5
        len_y = 3
        gap_penalty = 2.0
        
        result = normalize_cost(norm_type, cost, len_x, len_y, gap_penalty)
        self.assertEqual(result, 10.0)

    def test_sum_instance_normalization(self):
        """Test SumInstance normalization - divides by scaled_gap * (len_x + len_y)."""
        norm_type = MetricsNormType.SumInstance
        cost = 12.0
        len_x = 5
        len_y = 3
        gap_penalty = 2.0
        
        result = normalize_cost(norm_type, cost, len_x, len_y, gap_penalty)
        expected = 12.0 / (gap_penalty * (5 + 3))  # 12.0 / 16 = 0.75
        self.assertEqual(result, expected)

    def test_max_cost_based_normalization(self):
        """Test MaxCostBased normalization - uses formula (gap_penalty * (len_x+len_y) + cost) / 2."""
        norm_type = MetricsNormType.MaxCostBased
        cost = 10.0
        len_x = 5
        len_y = 3
        gap_penalty = 2.0
        
        result = normalize_cost(norm_type, cost, len_x, len_y, gap_penalty)
        # norm_factor = (2.0 * (5 + 3) + 10.0) / 2 = (16 + 10) / 2 = 13.0
        # result = 10.0 / 13.0
        expected = 10.0 / ((gap_penalty * (len_x + len_y) + cost) / 2.0)
        self.assertAlmostEqual(result, expected)

    def test_zero_norm_factor_raises_error(self):
        """Test that zero normalization factor does not raise error, returns cost."""
        norm_type = MetricsNormType.SumInstance
        cost = 10.0
        len_x = 0
        len_y = 0
        gap_penalty = 2.0
        
        # When norm_factor is 0, the function returns cost unchanged
        result = normalize_cost(norm_type, cost, len_x, len_y, gap_penalty)
        self.assertEqual(result, cost/np.finfo(np.float32).eps)

    def test_zero_cost_normalization(self):
        """Test normalization with zero cost."""
        norm_type = MetricsNormType.SumInstance
        cost = 0.0
        len_x = 5
        len_y = 3
        gap_penalty = 2.0
        
        result = normalize_cost(norm_type, cost, len_x, len_y, gap_penalty)
        self.assertEqual(result, 0.0)

    def test_float_lengths(self):
        """Test normalization with float lengths."""
        norm_type = MetricsNormType.SumInstance
        cost = 10.0
        len_x = 5.5
        len_y = 3.2
        gap_penalty = 2.0
        
        result = normalize_cost(norm_type, cost, len_x, len_y, gap_penalty)
        expected = 10.0 / (gap_penalty * (5.5 + 3.2))
        self.assertAlmostEqual(result, expected)

    def test_all_norm_types_produce_different_results(self):
        """Test that all normalization types produce different results (except NoNorm)."""
        cost = 12.0
        len_x = 5
        len_y = 3
        gap_penalty = 2.0
        
        no_norm = normalize_cost(MetricsNormType.NoNorm, cost, len_x, len_y, gap_penalty)
        sum_inst = normalize_cost(MetricsNormType.SumInstance, cost, len_x, len_y, gap_penalty)
        max_cost = normalize_cost(MetricsNormType.MaxCostBased, cost, len_x, len_y, gap_penalty)
        
        # NoNorm should equal the original cost
        self.assertEqual(no_norm, cost)
        
        # The bounded norms should differ from each other
        self.assertNotEqual(sum_inst, max_cost)

    def test_gap_penalty_effect_on_max_cost_based(self):
        """Test that gap penalty affects MaxCostBased normalization."""
        norm_type = MetricsNormType.MaxCostBased
        cost = 10.0
        len_x = 5
        len_y = 3
        
        result_low_gap = normalize_cost(norm_type, cost, len_x, len_y, scaled_gap_penalty=1.0)
        result_high_gap = normalize_cost(norm_type, cost, len_x, len_y, scaled_gap_penalty=5.0)
        
        # Higher gap penalty should result in larger norm factor, thus smaller normalized cost
        self.assertLess(result_high_gap, result_low_gap)

    def test_negative_cost_handling(self):
        """Test behavior with negative cost (edge case)."""
        norm_type = MetricsNormType.SumInstance
        cost = -10.0
        len_x = 5
        len_y = 3
        scaled_gap_penalty = 2.0
        
        result = normalize_cost(norm_type, cost, len_x, len_y, scaled_gap_penalty)
        expected = -10.0 / (scaled_gap_penalty * (5 + 3))
        self.assertEqual(result, expected)


class TestUnnormalizedCostForMaxCostNorm(unittest.TestCase):
    """Test fixture for unnormalized_cost_for_max_cost_norm function."""

    def test_round_trip_normalization(self):
        """Test that normalizing and then unnormalizing returns the original cost."""
        from sospa_eval.types import unnormalized_cost_for_max_cost_norm
        
        cost = 10.0
        len_x = 5.0
        len_y = 3.0
        scaled_gap_penalty = 2.0
        
        # Normalize using MaxCostBased
        norm_cost = normalize_cost(MetricsNormType.MaxCostBased, cost, len_x, len_y, scaled_gap_penalty)
        
        # Unnormalize
        recovered_cost = unnormalized_cost_for_max_cost_norm(norm_cost, len_x, len_y, scaled_gap_penalty)
        
        # Should recover the original cost
        self.assertAlmostEqual(recovered_cost, cost, places=10)

    def test_round_trip_with_different_parameters(self):
        """Test round trip with various parameter combinations."""
        from sospa_eval.types import unnormalized_cost_for_max_cost_norm
        
        test_cases = [
            (15.0, 7.0, 3.0, 1.5),
            (5.0, 2.0, 8.0, 3.0),
            (100.0, 10.0, 10.0, 5.0),
            (1.0, 1.0, 1.0, 1.0),
            (50.0, 20.0, 5.0, 0.5),
        ]
        
        for cost, len_x, len_y, scaled_gap_penalty in test_cases:
            with self.subTest(cost=cost, len_x=len_x, len_y=len_y, gap=scaled_gap_penalty):
                norm_cost = normalize_cost(MetricsNormType.MaxCostBased, cost, len_x, len_y, scaled_gap_penalty)
                recovered_cost = unnormalized_cost_for_max_cost_norm(norm_cost, len_x, len_y, scaled_gap_penalty)
                self.assertAlmostEqual(recovered_cost, cost, places=10)

    def test_zero_normalized_cost(self):
        """Test unnormalization of zero normalized cost."""
        from sospa_eval.types import unnormalized_cost_for_max_cost_norm
        
        norm_cost = 0.0
        len_x = 5.0
        len_y = 3.0
        scaled_gap_penalty = 2.0
        
        result = unnormalized_cost_for_max_cost_norm(norm_cost, len_x, len_y, scaled_gap_penalty)
        self.assertAlmostEqual(result, 0.0)

    def test_small_cost_values(self):
        """Test with very small cost values."""
        from sospa_eval.types import unnormalized_cost_for_max_cost_norm
        
        cost = 0.001
        len_x = 5.0
        len_y = 3.0
        scaled_gap_penalty = 2.0
        
        norm_cost = normalize_cost(MetricsNormType.MaxCostBased, cost, len_x, len_y, scaled_gap_penalty)
        recovered_cost = unnormalized_cost_for_max_cost_norm(norm_cost, len_x, len_y, scaled_gap_penalty)
        
        self.assertAlmostEqual(recovered_cost, cost, places=10)

    def test_large_cost_values(self):
        """Test with large cost values."""
        from sospa_eval.types import unnormalized_cost_for_max_cost_norm
        
        cost = 1000.0
        len_x = 5.0
        len_y = 3.0
        scaled_gap_penalty = 2.0
        
        norm_cost = normalize_cost(MetricsNormType.MaxCostBased, cost, len_x, len_y, scaled_gap_penalty)
        recovered_cost = unnormalized_cost_for_max_cost_norm(norm_cost, len_x, len_y, scaled_gap_penalty)
        
        self.assertAlmostEqual(recovered_cost, cost, places=8)

    def test_normalized_cost_bounds(self):
        """Test that normalized cost is always less than 2.0 for valid inputs."""
        from sospa_eval.types import unnormalized_cost_for_max_cost_norm
        
        # For any positive cost, the normalized cost should be < 2.0
        # This ensures (2.0 - norm_cost) is always positive
        test_cases = [
            (10.0, 5.0, 3.0, 2.0),
            (100.0, 10.0, 10.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        ]
        
        for cost, len_x, len_y, scaled_gap_penalty in test_cases:
            norm_cost = normalize_cost(MetricsNormType.MaxCostBased, cost, len_x, len_y, scaled_gap_penalty)
            # Normalized cost should be less than 2.0
            self.assertLess(norm_cost, 2.0)

    def test_max_cost_based_bounded_by_one(self):
        """Test that MaxCostBased normalization produces costs in range [0, 1].
        
        Note: Valid costs must satisfy: cost <= scaled_gap_penalty * (len_x + len_y)
        """
        norm_type = MetricsNormType.MaxCostBased
        
        # Test various parameter combinations - all must satisfy cost <= gap * (len_x + len_y)
        test_cases = [
            (10.0, 5.0, 3.0, 2.0),      # 10.0 <= 2.0 * 8 = 16.0 ✓
            (20.0, 10.0, 10.0, 1.0),    # 20.0 <= 1.0 * 20 = 20.0 ✓ (edge case)
            (1.0, 1.0, 1.0, 1.0),       # 1.0 <= 1.0 * 2 = 2.0 ✓
            (12.0, 20.0, 5.0, 0.5),     # 12.0 <= 0.5 * 25 = 12.5 ✓
            (5.0, 2.0, 8.0, 3.0),       # 5.0 <= 3.0 * 10 = 30.0 ✓
            (0.001, 5.0, 3.0, 2.0),     # 0.001 <= 2.0 * 8 = 16.0 ✓
            (16.0, 5.0, 3.0, 2.0),      # 16.0 <= 2.0 * 8 = 16.0 ✓ (edge case)
            (15.0, 7.0, 3.0, 1.5),      # 15.0 <= 1.5 * 10 = 15.0 ✓ (edge case)
        ]
        
        for cost, len_x, len_y, scaled_gap_penalty in test_cases:
            with self.subTest(cost=cost, len_x=len_x, len_y=len_y, gap=scaled_gap_penalty):
                norm_cost = normalize_cost(norm_type, cost, len_x, len_y, scaled_gap_penalty)
                
                # Normalized cost should be >= 0
                self.assertGreaterEqual(norm_cost, 0.0, 
                    f"Normalized cost {norm_cost} is less than 0 for cost={cost}, len_x={len_x}, len_y={len_y}, gap={scaled_gap_penalty}")
                
                # Normalized cost should be <= 1
                self.assertLessEqual(norm_cost, 1.0,
                    f"Normalized cost {norm_cost} exceeds 1.0 for cost={cost}, len_x={len_x}, len_y={len_y}, gap={scaled_gap_penalty}")

    def test_max_cost_based_zero_cost(self):
        """Test that zero cost gives zero normalized cost with MaxCostBased."""
        norm_type = MetricsNormType.MaxCostBased
        cost = 0.0
        len_x = 5.0
        len_y = 3.0
        scaled_gap_penalty = 2.0
        
        norm_cost = normalize_cost(norm_type, cost, len_x, len_y, scaled_gap_penalty)
        self.assertEqual(norm_cost, 0.0)
        self.assertGreaterEqual(norm_cost, 0.0)
        self.assertLessEqual(norm_cost, 1.0)


if __name__ == '__main__':
    unittest.main()
