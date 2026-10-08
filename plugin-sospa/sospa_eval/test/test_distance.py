
import unittest
from unittest import mock
import numpy as np
from scipy.spatial import distance

from sospa_eval.sospa import compute_sospa_decomposed_numba
from sospa_eval.distance import (
    double_direction_open_and_closed_frechet_distance,
    frechet_distance,
)
from sospa_eval.types import Polyline
import sospa_eval.distance as distance_module
from sospa_eval.constants import METRICS_CONFIG
from sospa_eval.types import MetricsNormType

M_NORM_USED = METRICS_CONFIG.norm

class TestComputeSospaDecomposedNumba(unittest.TestCase):
    """Test fixture for compute_needleman_wunsch_distance_decmposed_numba function."""

    def test_identical_polylines(self):
        """Test with identical polylines."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")
        
        for normalization in [MetricsNormType.SumInstance, MetricsNormType.MaxCostBased]:
            total_cost, tp_sum_dist_error, ntpfp = compute_sospa_decomposed_numba(
                C, gap_penalty=2.0, norm_used=int(normalization)
            )
            
            # For identical polylines, distance error should be 0
            self.assertEqual(tp_sum_dist_error, 0.0)
            # All points should be true positives
            self.assertEqual(ntpfp[0], 3)  # nTP
            self.assertEqual(ntpfp[1], 0)  # nFP
            self.assertEqual(ntpfp[2], 0)  # nFN
            # Total cost should be 0 for identical polylines
            self.assertEqual(total_cost, 0.0)

    def test_different_length_polylines(self):
        """Test with polylines of different lengths."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")
        
        
        for normalization in [MetricsNormType.SumInstance, MetricsNormType.MaxCostBased]:
            total_cost, _, ntpfp = compute_sospa_decomposed_numba(
                C, gap_penalty=2.0, norm_used=int(normalization)
            )
            # Check that TP + FP = len(polyline_a)
            self.assertEqual(ntpfp[0] + ntpfp[1], len(polyline_a))
            # Check that TP + FN = len(polyline_b)
            self.assertEqual(ntpfp[0] + ntpfp[2], len(polyline_b))
            # Total cost should be greater than 0 for different length polylines
            self.assertGreater(total_cost, 0.0)
     

    def test_completely_different_polylines(self):
        """Test with completely different polylines."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[1.0, 0.0], [1.5, 0.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")

        for normalization in [MetricsNormType.SumInstance, MetricsNormType.MaxCostBased]:
            total_cost, tp_sum_dist_error, ntpfp = compute_sospa_decomposed_numba(
                C, gap_penalty=10.0, norm_used=int(normalization)
            )
            # Distance error should be positive for different polylines
            self.assertGreater(tp_sum_dist_error, 0.0)
            # Check that TP + FP = len(polyline_a)
            self.assertEqual(ntpfp[0] + ntpfp[1], len(polyline_a))
            # Check that TP + FN = len(polyline_b)
            self.assertEqual(ntpfp[0] + ntpfp[2], len(polyline_b))
            # Total cost should be positive
            self.assertGreater(total_cost, 0.0)

    def test_single_point_polylines(self):
        """Test with single point polylines."""
        polyline_a = np.array([[0.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[1.0, 1.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")
        
        _, _, ntpfp = compute_sospa_decomposed_numba(
            C, gap_penalty=2.0, norm_used=int(MetricsNormType.SumInstance)
        )
        
        # Check that TP + FP = len(polyline_a) = 1
        self.assertEqual(ntpfp[0] + ntpfp[1], 1)
        # Check that TP + FN = len(polyline_b) = 1
        self.assertEqual(ntpfp[0] + ntpfp[2], 1)
        # For single points, should have 1 TP, 0 FP, 0 FN
        self.assertEqual(ntpfp[0], 1)
        self.assertEqual(ntpfp[1], 0)
        self.assertEqual(ntpfp[2], 0)

    def test_gap_penalty_effect(self):
        """Test that different gap penalties affect the result."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[0.0, 0.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")

        # SumInstance is normalized to [0, 1]; for a perfect match plus a forced
        # gap the penalty cancels, giving the fraction of unmatched points.
        sum_low, _, _ = compute_sospa_decomposed_numba(
            C, gap_penalty=1.0, norm_used=int(MetricsNormType.SumInstance)
        )
        sum_high, _, _ = compute_sospa_decomposed_numba(
            C, gap_penalty=10.0, norm_used=int(MetricsNormType.SumInstance)
        )
        self.assertAlmostEqual(sum_low, sum_high)
        self.assertLessEqual(sum_high, 1.0)


    def test_no_matches_cost_sum_normalization(self):
        """Test that no matches result in a specific cost with sum normalization."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[10.0, 10.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")
        
        gap_penalty = 2.0
        total_cost_sum_normalization, _, _ = compute_sospa_decomposed_numba(
            C, gap_penalty=gap_penalty, norm_used=int(MetricsNormType.SumInstance)
        )
        # SumInstance is normalized to [0, 1]; a fully unmatched pair gives 1.0.
        self.assertAlmostEqual(total_cost_sum_normalization, 1.0)
        
    def test_return_types(self):
        """Test that the function returns the correct types."""
        polyline_a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        polyline_b = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        C = distance.cdist(polyline_a, polyline_b, "euclidean")
        
        total_cost, tp_sum_dist_error, ntpfp = compute_sospa_decomposed_numba(
            C, gap_penalty=2.0, norm_used=int(MetricsNormType.SumInstance)
        )
        
        # Check return types
        self.assertIsInstance(total_cost, (float, np.floating))
        self.assertIsInstance(tp_sum_dist_error, (float, np.floating))
        self.assertIsInstance(ntpfp, tuple)
        self.assertEqual(len(ntpfp), 3)
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in ntpfp))

class TestComputeNeedlemanWunschBatchDecomposed(unittest.TestCase):
    """Test fixture for compute_needleman_wunsch_batch_decomposed function."""
    
    @mock.patch.object(__import__('sospa_eval.constants', fromlist=['METRICS_CONFIG']).METRICS_CONFIG, 'norm', MetricsNormType.SumInstance)
    def test_all_points_unmatched_single_pair(self):
        """Test batch function with all points unmatched for a single pair."""
        from sospa_eval import pld as pld_module
        from sospa_eval.sospa import compute_sospa_batch_decomposed as compute_needleman_wunsch_batch_decomposed
        # Ensure the unmatched fast-path uses the mocked normalization value.
        # Restore the original defaults afterwards to avoid leaking global state
        # into other tests.
        original_defaults = pld_module.get_decomposed_cost_all_unmatched.__defaults__
        self.addCleanup(
            setattr, pld_module.get_decomposed_cost_all_unmatched, "__defaults__", original_defaults
        )
        pld_module.get_decomposed_cost_all_unmatched.__defaults__ = (
            int(MetricsNormType.SumInstance),
        )
        # Create polylines that are very far apart (distance > gap_penalty)
        pred_lines = [
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        ]
        
        gt_lines = [
            np.array([[100.0, 100.0], [101.0, 100.0]], dtype=np.float64),
        ]
        
        gap_penalty = 2.0
        result, dist_error, nTP, nFP = compute_needleman_wunsch_batch_decomposed(
            pred_lines, gt_lines, gap_penalty=gap_penalty
        )
        
        # All points should be unmatched
        self.assertEqual(nTP[0, 0], 0, "Expected 0 true positives when all points are unmatched")
        self.assertEqual(nFP[0, 0], 3, "Expected 3 false positives (all pred points unmatched)")
        
        # Distance error should be 0 when no matches
        self.assertEqual(dist_error[0, 0], 0.0, "Expected 0 distance error when no points matched")
        
        # Check cost calculation for all unmatched
        len_pred = len(pred_lines[0])
        len_gt = len(gt_lines[0])
        # SumInstance is normalized to [0, 1]; a fully unmatched pair gives 1.0.
        expected_cost = 1.0
        self.assertAlmostEqual(result[0, 0], expected_cost, places=5)

    def test_mixed_matched_and_unmatched(self):
        """Test batch with some pairs matched and some unmatched."""
        from sospa_eval.sospa import compute_sospa_batch_decomposed as compute_needleman_wunsch_batch_decomposed
        
        pred_lines = [
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),  # Close to gt_lines[0]
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),  # Far from gt_lines[1]
        ]
        
        gt_lines = [
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),  # Close to pred_lines[0]
            np.array([[100.0, 100.0], [101.0, 100.0]], dtype=np.float64),  # Far from pred_lines[1]
        ]
        
        gap_penalty = 2.0
        result, dist_error, nTP, nFP = compute_needleman_wunsch_batch_decomposed(
            pred_lines, gt_lines, gap_penalty=gap_penalty
        )
        
        # pred_lines[0] vs gt_lines[0]: should have matches
        self.assertGreater(nTP[0, 0], 0, "Expected matches for identical polylines")
        self.assertEqual(dist_error[0, 0], 0.0)
        
        # pred_lines[1] vs gt_lines[1]: should have no matches
        self.assertEqual(nTP[1, 1], 0, "Expected no matches for distant polylines")
        self.assertEqual(dist_error[1, 1], 0.0)
        
        # pred_lines[0] vs gt_lines[1]: should have no matches
        self.assertEqual(nTP[0, 1], 0)
        self.assertEqual(dist_error[0, 1], 0.0)

    def test_boundary_case_distance_equals_gap_penalty(self):
        """Test behavior when distance exactly equals gap penalty."""
        from sospa_eval.sospa import compute_sospa_batch_decomposed as compute_needleman_wunsch_batch_decomposed
        
        gap_penalty = 2.0
        error = gap_penalty + 0.1
        # Create polylines where distance is exactly gap_penalty
        pred_lines = [
            np.array([[0.0, 0.0]], dtype=np.float64),
        ]
        
        gt_lines = [
            np.array([[error, 0.0]], dtype=np.float64),
        ]
        
        result, dist_error, nTP, nFP = compute_needleman_wunsch_batch_decomposed(
            pred_lines, gt_lines, gap_penalty=gap_penalty
        )
        
        # At boundary, points should be unmatched (distance >= gap_penalty)
        self.assertEqual(nTP[0, 0], 0)
        self.assertEqual(nFP[0, 0], 1)
        self.assertEqual(dist_error[0, 0], 0.0)
        
        error = gap_penalty - 0.1
        # Create polylines where distance is just below gap_penalty
        pred_lines = [
            np.array([[0.0, 0.0]], dtype=np.float64),
        ]
        gt_lines = [
            np.array([[error, 0.0]], dtype=np.float64),
        ]
        result, dist_error, nTP, nFP = compute_needleman_wunsch_batch_decomposed(
            pred_lines, gt_lines, gap_penalty=gap_penalty
        )
        # Just below boundary, points should be matched
        self.assertEqual(nTP[0, 0], 1)
        self.assertEqual(nFP[0, 0], 0)
        self.assertAlmostEqual(dist_error[0, 0], error, places=5)
        
        
    def test_triangle_vs_square(self):
        """Test behavior when distance exactly equals gap penalty."""
        from sospa_eval.sospa import compute_sospa_batch_decomposed as compute_needleman_wunsch_batch_decomposed
        
        gap_penalty = 2.0
        # Triangle anit clockwise
        pred_lines = [
            np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]], dtype=np.float64),
        ]
        # Square clockwise
        gt_lines = [
            np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        ]
        _, dist_error, nTP, nFP = compute_needleman_wunsch_batch_decomposed(
            pred_lines, gt_lines, gap_penalty=gap_penalty
        )
        # Some points should be matched
        self.assertGreater(nTP[0, 0], 0)
        self.assertGreater(dist_error[0, 0], 0.0)
        self.assertEqual(nFP[0, 0], 0)





class TestDoubleDirectionOpenAndClosedFrechetDistance(unittest.TestCase):

    def _open(self, coords):
        return Polyline(geometry=np.array(coords, dtype=np.float64), is_closed=False)

    def _closed(self, coords):
        return Polyline(geometry=np.array(coords, dtype=np.float64), is_closed=True)

    def test_identical_open_polylines(self):
        """Identical open polylines should give distance 0."""
        coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        line = self._open(coords)
        self.assertAlmostEqual(double_direction_open_and_closed_frechet_distance(line, line), 0.0)

    def test_reversed_open_polyline_gives_zero(self):
        """Reversing an open polyline should still give distance 0 (bidirectional)."""
        coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        fwd = self._open(coords)
        rev = self._open(coords[::-1])
        self.assertAlmostEqual(double_direction_open_and_closed_frechet_distance(fwd, rev), 0.0)

    def test_open_bidirectional_less_than_unidirectional(self):
        """Bidirectional result should be <= forward-only frechet for a reversed pair."""
        coords_a = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        coords_b = [[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        line_a = self._open(coords_a)
        line_b = self._open(coords_b)
        bidir = double_direction_open_and_closed_frechet_distance(line_a, line_b)
        forward_only = frechet_distance(np.array(coords_a), np.array(coords_b))
        self.assertLessEqual(bidir, forward_only)

    def test_identical_closed_polylines(self):
        """Identical closed polylines should give distance 0."""
        coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        line = self._closed(coords)
        self.assertAlmostEqual(double_direction_open_and_closed_frechet_distance(line, line), 0.0)

    def test_rotated_closed_polyline_gives_zero(self):
        """A cyclic rotation of a closed polyline should give distance 0."""
        coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        rotated = coords[2:] + coords[:2]  # shift by 2
        line_a = self._closed(coords)
        line_b = self._closed(rotated)
        self.assertAlmostEqual(double_direction_open_and_closed_frechet_distance(line_a, line_b), 0.0)

    def test_closed_cyclic_less_than_open(self):
        """Cyclic frechet on closed polyline should be <= non-cyclic open frechet when rotated."""
        coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        rotated = coords[2:] + coords[:2]
        closed_a = self._closed(coords)
        closed_b = self._closed(rotated)
        open_a = self._open(coords)
        open_b = self._open(rotated)
        cyclic_dist = double_direction_open_and_closed_frechet_distance(closed_a, closed_b)
        open_dist = double_direction_open_and_closed_frechet_distance(open_a, open_b)
        self.assertLessEqual(cyclic_dist, open_dist + 1e-9)

    def test_distant_lines_nonzero(self):
        """Lines far apart should have a large non-zero distance."""
        line_a = self._open([[0.0, 0.0], [1.0, 0.0]])
        line_b = self._open([[100.0, 0.0], [101.0, 0.0]])
        dist = double_direction_open_and_closed_frechet_distance(line_a, line_b)
        self.assertGreater(dist, 50.0)

    def test_closed_flag_triggers_cyclic(self):
        """If either line is closed, cyclic search is used (result <= open-only frechet)."""
        coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        rotated = coords[1:] + coords[:1]
        closed = self._closed(coords)
        open_rot = self._open(rotated)
        result = double_direction_open_and_closed_frechet_distance(closed, open_rot)
        naive = frechet_distance(np.array(coords, dtype=np.float64), np.array(rotated, dtype=np.float64))
        self.assertLessEqual(result, naive + 1e-9)


if __name__ == "__main__":
    unittest.main()

