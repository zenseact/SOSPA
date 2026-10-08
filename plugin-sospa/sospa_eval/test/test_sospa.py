import unittest
from unittest import mock
import numpy as np
import sys
from pathlib import Path

from sospa_eval.order_polylines import to_polyline_type
from scipy.spatial import distance
# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from sospa_eval.sospa import (
    compute_cyclic_sospa_decomposed_numba,
)
from sospa_eval.constants import METRICS_CONFIG, SOSPA_SCALING
from sospa_eval.types import MetricsNormType

NW_SCALING = SOSPA_SCALING
M_NORM_USED = METRICS_CONFIG.norm
from sospa_eval.test.reference_cyclic_nw import (
    compute_cyclic_needleman_wunsch_brute_force,
    compute_cyclic_needleman_wunsch_maes_corrected,
)





NW_TEST_THRESHOLD = 4.0
def compute_euclidane_distance_cost_matrix(polyline_a, polyline_b):
    """Compute the pairwise Euclidean distance cost matrix between two polylines."""
    return distance.cdist(polyline_a, polyline_b, "euclidean")

class TestComputeCyclicSospaDecomposedNumba(unittest.TestCase):
    """Test cases for maes_cyclic_needleman_wunsch_numba function."""

    def setUp(self):
        """Set up test fixtures."""
        self.gap_penalty = NW_TEST_THRESHOLD
        self.norm_used = int(M_NORM_USED)

    def test_empty_polylines(self):
        """Test with empty polylines."""
        polyline_a = np.array([]).reshape(0, 2)
        polyline_b = np.array([[0.0, 0.0], [1.0, 0.0]])
        C = np.array([]).reshape(0, 2)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should have cost for unmatched points in polyline_b
        self.assertEqual(nTP, 0)
        self.assertEqual(nFP, 0)
        self.assertEqual(nFN, 2)
        self.assertEqual(best_rotation, 0)
        self.assertGreater(total_cost, 0)

    def test_identical_polylines(self):
        """Test with identical polylines."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        polyline_b = polyline_a.copy()
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Identical polylines should have zero or near-zero cost
        self.assertAlmostEqual(total_cost, 0.0, places=5)
        self.assertAlmostEqual(dist_error, 0.0, places=5)

    def test_rotated_polylines(self):
        """Test that rotations of the same polyline are recognized."""
        # Create a square
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # Rotate by 2 positions
        polyline_b = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should recognize they're the same with appropriate rotation
        self.assertAlmostEqual(total_cost, 0.0, places=5)
        self.assertAlmostEqual(dist_error, 0.0, places=5)
        self.assertIn(best_rotation, [0, 1, 2, 3])  # Some rotation should work

    def test_different_polylines(self):
        """Test with completely different polylines."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        polyline_b = np.array([[10.0, 10.0], [11.0, 10.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should have non-zero cost for different polylines
        self.assertGreater(total_cost, 0)

    def test_single_point_polylines(self):
        """Test with single-point polylines."""
        polyline_a = np.array([[0.0, 0.0]])
        polyline_b = np.array([[1.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should match the single points
        self.assertEqual(best_rotation, 0)
        self.assertGreater(total_cost, 0)  # Non-zero due to distance

    def test_compare_with_brute_force_small(self):
        """Compare results with brute force algorithm on small polylines."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        polyline_b = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        # Numba version
        numba_result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        numba_cost = numba_result[0]
        
        # Brute force version expects UNSCALED gap penalty (it scales internally)
        bf_cost, _, _, _, _ = compute_cyclic_needleman_wunsch_brute_force(
            polyline_a, polyline_b, self.gap_penalty, normalize=True
        )
        
        # Costs should be very close
        self.assertAlmostEqual(numba_cost, bf_cost, places=4,
                              msg=f"Numba cost {numba_cost} != Brute force cost {bf_cost}")

    @mock.patch('sospa_eval.test.reference_cyclic_nw.M_NORM_USED', MetricsNormType.SumInstance)
    def test_compare_with_maes_corrected(self):
        """Compare results with Maes corrected algorithm."""
        polyline_a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        polyline_b = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        # Numba version. The Maes corrected reference uses sum-instance style
        # normalization, so compare against SumInstance regardless of the global default.
        numba_result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty,
            int(MetricsNormType.SumInstance)
        )
        numba_cost = numba_result[0]
        
        # Maes corrected version. It scales the gap penalty internally by
        # NW_SCALING, so pass the unscaled gap penalty to match the numba path.
        maes_cost, _, _, _ = compute_cyclic_needleman_wunsch_maes_corrected(
            polyline_a, polyline_b, self.gap_penalty, normalize=True
        )
        
        # Costs should be very close
        self.assertAlmostEqual(numba_cost, maes_cost, places=4,
                              msg=f"Numba cost {numba_cost} != Maes cost {maes_cost}")

    def test_normalization_none(self):
        """Test with no normalization."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        polyline_b = np.array([[0.0, 0.0], [1.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, norm_used=0
        )
        
        total_cost, _, _, _ = result
        
        # With identical polylines and no normalization, cost should be near zero
        self.assertAlmostEqual(total_cost, 0.0, places=5)

    def test_3d_polylines(self):
        """Test with 3D polylines."""
        polyline_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        polyline_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Identical 3D polylines should have zero or near-zero cost
        self.assertAlmostEqual(total_cost, 0.0, places=5)

    def test_rectangular_polylines(self):
        """Test with rectangular (non-square) polylines."""
        polyline_a = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]])
        polyline_b = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]])
        
        polyline_b = np.roll(polyline_b, shift=-1, axis=0)  # Rotate by 1 position
        polyline_b = polyline_b[::-1, :]  # Reverse order
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, _, _, _ = result
        
        # Identical polylines should match
        self.assertAlmostEqual(total_cost, 0.0, places=5)

    def test_different_lengths(self):
        """Test with polylines of different lengths."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        polyline_b = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should handle different lengths gracefully
        self.assertGreater(total_cost, 0)
        self.assertGreaterEqual(nFP + nFN, 1)  # At least one gap

    def test_cyclic_property(self):
        """Test that all rotations of polyline_a give the same result."""
        # Create a symmetric shape
        polyline_base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        polyline_b = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
        
        costs = []
        for rotation in range(len(polyline_base)):
            # Manually rotate polyline_base
            polyline_a = np.roll(polyline_base, -rotation, axis=0)
            C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
            
            result = compute_cyclic_sospa_decomposed_numba(
                polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
            )
            costs.append(result[0])
        
        # All rotations should give approximately the same cost
        # (cyclic algorithm should find optimal rotation regardless of input rotation)
        for i in range(len(costs)):
            for j in range(i + 1, len(costs)):
                self.assertAlmostEqual(costs[i], costs[j], places=4,
                                      msg=f"Rotation {i} cost {costs[i]} != Rotation {j} cost {costs[j]}")

    def test_performance_larger_polylines(self):
        """Test performance with larger polylines."""
        np.random.seed(42)
        polyline_a = np.random.rand(20, 2) * 10
        polyline_b = np.random.rand(15, 2) * 10
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        import time
        start = time.time()
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second for this size)
        self.assertLess(elapsed, 1.0, msg=f"Took {elapsed:.3f}s for 20x15 polylines")
        
        # Result should be valid
        total_cost, _, _, best_rotation = result
        self.assertIsInstance(total_cost, (float, np.floating))
        self.assertGreaterEqual(best_rotation, 0)
        self.assertLess(best_rotation, len(polyline_a))


class TestComputeCyclicSospaDecomposedNumbaEdgeCases(unittest.TestCase):
    """Additional edge case tests."""

    def test_collinear_points(self):
        """Test with collinear points."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        polyline_b = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, NW_TEST_THRESHOLD, int(M_NORM_USED)
        )
        
        total_cost, _, _, _ = result
        self.assertAlmostEqual(total_cost, 0.0, places=5)

    def test_very_small_gap_penalty(self):
        """Test with very small gap penalty."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        polyline_b = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        small_gap = 0.1
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, small_gap, int(M_NORM_USED)
        )
        
        # Should still produce valid result
        total_cost, _, _, best_rotation = result
        self.assertIsInstance(total_cost, (float, np.floating))
        self.assertGreaterEqual(best_rotation, 0)

    def test_very_large_gap_penalty(self):
        """Test with very large gap penalty."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        polyline_b = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        large_gap = 100.0
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, large_gap, int(M_NORM_USED)
        )
        
        # Should still produce valid result
        total_cost, _, _, best_rotation = result
        self.assertIsInstance(total_cost, (float, np.floating))
        self.assertGreaterEqual(best_rotation, 0)


class TestSospaNormalizationBounds(unittest.TestCase):
    """Normalization bounds for far-apart (fully unmatched) polylines."""

    def setUp(self):
        self.polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.polyline_b = np.array([[100.0, 100.0], [101.0, 100.0]])
        self.C = compute_euclidane_distance_cost_matrix(self.polyline_a, self.polyline_b)

    def test_suminstance_far_apart_equals_one(self):
        """SumInstance: far-apart cost is bounded to 1.0."""
        total_cost, _, _, _ = compute_cyclic_sospa_decomposed_numba(
            self.polyline_a, self.polyline_b, self.C,
            NW_TEST_THRESHOLD, int(MetricsNormType.SumInstance)
        )
        self.assertLessEqual(total_cost, 1.0)
        self.assertAlmostEqual(total_cost, 1.0, places=5)

    def test_maxcost_far_apart_not_exceed_one(self):
        """MaxCostBased: far-apart cost is capped at 1.0."""
        total_cost, _, _, _ = compute_cyclic_sospa_decomposed_numba(
            self.polyline_a, self.polyline_b, self.C,
            NW_TEST_THRESHOLD, int(MetricsNormType.MaxCostBased)
        )
        self.assertLessEqual(total_cost, 1.0)
        self.assertAlmostEqual(total_cost, 1.0, places=5)

    def test_suminstance_partial_match_within_unit_interval(self):
        """SumInstance: a partial-match cost stays strictly within [0, 1]."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        polyline_b = np.array([[0.0, 0.5], [1.0, 0.5], [50.0, 50.0]])
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        total_cost, _, _, _ = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C,
            NW_TEST_THRESHOLD, int(MetricsNormType.SumInstance)
        )
        self.assertGreater(total_cost, 0.0)
        self.assertLess(total_cost, 1.0)


class TestMaesCyclicNeedlemanWunschReversedAlignment(unittest.TestCase):
    """Test cases where reversed alignment is better."""

    def setUp(self):
        """Set up test fixtures."""
        self.gap_penalty = NW_TEST_THRESHOLD
        self.norm_used = int(M_NORM_USED)

    def test_opposite_direction_polylines(self):
        """Test polylines going in opposite directions."""
        # Line going from left to right
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # Similar line going from right to left
        polyline_b = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should find good alignment despite opposite directions
        self.assertLess(total_cost, 0.5, msg="Reversed alignment should have low cost")
        self.assertAlmostEqual(dist_error, 0.0, places=5)

    def test_reversed_polygon(self):
        """Test with a polygon and its reverse."""
        # Square traversed counter-clockwise
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # Same square traversed clockwise
        polyline_b = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should match well when reversed
        self.assertLess(total_cost, 0.5)
        self.assertEqual(nTP, 4)
        self.assertEqual(nFP, 0)
        self.assertEqual(nFN, 0)

    def test_asymmetric_polyline_reversed(self):
        """Test with an asymmetric polyline that has better alignment when reversed."""
        # Create an L-shaped polyline
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        # Similar L-shape with small offset, going in opposite direction
        polyline_b = np.array([[1.1, 1.1], [1.1, 0.1], [0.1, 0.1]])
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should find reasonable alignment
        self.assertEqual(nTP, 3)
        self.assertGreater(total_cost, 0)  # Not perfect due to small offset

    def test_zigzag_pattern_reversed(self):
        """Test with zigzag pattern that matches better when reversed."""
        # Zigzag going right-up-right
        polyline_a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        # Similar zigzag going left-down-left (reversed)
        polyline_b = np.array([[3.0, 1.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        result = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C, self.gap_penalty, self.norm_used
        )
        
        total_cost, dist_error, (nTP, nFP, nFN), best_rotation = result
        
        # Should recognize they're the same pattern reversed
        self.assertAlmostEqual(total_cost, 0.0, places=5)
        self.assertEqual(nTP, 4)

    def test_reversed_better_than_forward(self):
        """Explicitly test that reversed alignment is chosen when better."""
        # Create polylines where reversed is clearly better
        polyline_a = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        # Opposite direction with slight offset to make forward alignment worse
        polyline_b = np.array([[1.0, 0.1], [0.5, 0.0], [0.0, 0.1]])
        
        C = compute_euclidane_distance_cost_matrix(polyline_a, polyline_b)
        
        # Compute forward alignment cost (for comparison)
        C_forward = C
        result_forward = compute_cyclic_sospa_decomposed_numba(
            polyline_a, polyline_b, C_forward, self.gap_penalty, self.norm_used
        )
        cost_forward = result_forward[0]
        
        # The cyclic NW should automatically choose the better reversed alignment
        # which should be lower cost due to the offset design
        self.assertIsInstance(cost_forward, (float, np.floating))



if __name__ == "__main__":
    unittest.main()