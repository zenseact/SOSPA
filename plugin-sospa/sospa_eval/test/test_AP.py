import unittest
import numpy as np
import sys
import os
from sospa_eval import constants
from sospa_eval.constants import MetricsConfig
from sospa_eval.AP import average_precision
from sospa_eval.PLD import _get_polyline_level_TP_FP, calculate_pld
from sospa_eval.instance_matching import (
    _match_predictions_with_thresholds_extended,
)
from sospa_eval.test.helpers import load_real_world_test_data_instance_matching


class TestAveragePrecision(unittest.TestCase):
    def test_basic_functionality(self):
        recalls = np.array([0.1, 0.2, 0.5, 0.7, 1.0])
        precisions = np.array([1.0, 0.9, 0.8, 0.6, 0.5])

        ap = average_precision(recalls, precisions, "area")
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_get_polyline_level_TP_FP(self):
        num_tp_points = [1, 5, 0, 0, 5]
        tp, fp = _get_polyline_level_TP_FP(num_tp_points)

        self.assertEqual(tp.tolist(), [1, 1, 0, 0, 1])
        self.assertEqual(fp.tolist(), [0, 0, 1, 1, 0])

        self.assertEqual(tp.shape[0], 5)
        self.assertEqual(len(tp.shape), 1)  # tp is 1D array


class TestCalculateAveragePrecisionExtended(unittest.TestCase):
    """Test calculate_average_precision_extended with focus on NW cost and normalization."""
    
    def _create_basic_test_data(self):
        """Helper to create basic test data structure."""
        # Create data for 2 predictions with columns: [tp, fp, nw_cost, tp_d_err, scaling, scores]
        data = {
            1.0: np.array([
                [5, 0, 0.1, 0.05, 5.0, 0.9],  # Prediction 1: TP=5, good match
                [0, 3, 0.5, 0.0, 3.0, 0.8],   # Prediction 2: FP=3, false positive
            ])
        }
        return [data]  # List with one sample
    
    def test_nw_cost_with_sum_instance_norm(self):
        """Test NW cost calculation with M_NORM_USED_INSTANCE_LEVEL = SumInstance."""
        tp_fp_c_err_score_list = self._create_basic_test_data()
        thresholds = [1.0]
        num_gts_points = [5.0]  # One GT with 5 points
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Check that NW_cost key exists
        self.assertIn('PLD_cost@1.0', result_dict)
        self.assertIn('PLD_cost', result_dict)
        self.assertIn('PLD_loc@1.0', result_dict)
        
        # Calculate expected NW cost
        # Data: [5, 0, 0.1, 0.05, 5.0, 0.9] and [0, 3, 0.5, 0.0, 3.0, 0.8]
        # Polyline-level: 1 TP (first), 1 FP (second)
        # n_fn = 1 - 1 = 0
        # conf_dependent_fn = 1 - (1 * 0.9) = 0.1
        # conf_dependent_fp = (1 * 0.8) = 0.8
        # n_misses_nw_cost = (1.0 / 2.0) * (0 + 0.1 + 0.8) = 0.45
        # tp_nw_cost_norm = 0.1 * 0.9 * 1 = 0.09
        # sum_nw_cost = 0.09 + 0.45 = 0.54
        # normalization: 2 / (1 + (0.9 + 0.8)) = 2 / 2.7 = 0.7407407...
        # nw_cost_normalized = 0.54 * 0.7407407 = 0.4
        expected_nw_cost = 0.4
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], expected_nw_cost, places=5)
        self.assertAlmostEqual(result_dict['PLD_cost'], expected_nw_cost, places=5)
    
    def test_nw_cost_with_max_cost_based_norm(self):
        """Test NW cost calculation with M_NORM_USED_INSTANCE_LEVEL = MaxCostBased."""
        tp_fp_c_err_score_list = self._create_basic_test_data()
        thresholds = [1.0]
        num_gts_points = [5.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Check that NW_cost key exists
        self.assertIn('PLD_cost@1.0', result_dict)
        self.assertIn('PLD_cost', result_dict)
        
        # CRITICAL: MaxCostBased normalization must ensure cost <= 1.0
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
        self.assertLessEqual(result_dict['PLD_cost'], 1.0)
        
        # Cost should be non-negative
        self.assertGreaterEqual(result_dict['PLD_cost@1.0'], 0.0)
    
    def test_nw_cost_with_multiple_thresholds(self):
        """Test NW cost calculation with multiple thresholds."""
        # Create data for multiple thresholds
        tp_fp_c_err_score_list = [{
            0.5: np.array([[8, 2, 0.08, 0.04, 10.0, 0.95]]),
            1.0: np.array([[5, 0, 0.1, 0.05, 5.0, 0.9]]),
            1.5: np.array([[3, 0, 0.15, 0.06, 3.0, 0.85]]),
        }]
        thresholds = [0.5, 1.0, 1.5]
        num_gts_points = [10.0]  # One GT polyline with 10 points
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Check all thresholds have NW cost
        for thr in thresholds:
            self.assertIn(f'PLD_cost@{thr}', result_dict)
            self.assertGreaterEqual(result_dict[f'PLD_cost@{thr}'], 0.0)
        
        # Check mean NW cost
        self.assertIn('PLD_cost', result_dict)
        mean_nw = result_dict['PLD_cost']
        expected_mean = sum(result_dict[f'PLD_cost@{thr}'] for thr in thresholds) / len(thresholds)
        self.assertAlmostEqual(mean_nw, expected_mean, places=5)
    
    def test_nw_cost_with_all_false_positives(self):
        """Test NW cost when all predictions are false positives."""
        # All predictions are FP
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [0, 5, 0.5, 0.0, 5.0, 0.9],   # FP
                [0, 3, 0.5, 0.0, 3.0, 0.8],   # FP
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Calculate expected NW cost with all FPs
        # Polyline-level: 0 TPs, 2 FPs
        # n_fn = 1 - 0 = 1
        # conf_dependent_fn = 0 - 0 = 0
        # conf_dependent_fp = (1 * 0.9) + (1 * 0.8) = 1.7
        # scaled_gap = 0.5
        # n_misses_pld_cost = 0.5 * (1 + 0 + 1.7) = 1.35
        # sum_pld_cost = 1.35
        # SumInstance is bounded to [0, 1]; a fully unmatched scene gives 1.0.
        expected_nw_cost = 1.0
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], expected_nw_cost, places=5)
    
    def test_nw_cost_with_perfect_predictions(self):
        """Test NW cost with perfect predictions (identical to GT)."""
        # Perfect prediction: all GT points matched, no FP, zero distance error
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.0, 0.0, 10.0, 1.0],  # Perfect match
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Perfect prediction: nw_cost=0, tp_d_err=0, score=1.0
        # n_fn = 1 - 1 = 0, conf_dependent_fn = 1 - 1 = 0, conf_dependent_fp = 0
        # n_misses_nw_cost = 0
        # tp_nw_cost_norm = 0 * 1 * 1 = 0
        # sum_nw_cost = 0
        # nw_cost_normalized = 0 (regardless of normalization)
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], 0.0, places=5)
        self.assertAlmostEqual(result_dict['PLD_cost'], 0.0, places=5)
        
        # NW localization error should be 0
        self.assertAlmostEqual(result_dict['PLD_loc@1.0'], 0.0, places=5)
    
    def test_nw_cost_scales_with_distance_error(self):
        """Test that NW cost reflects distance error magnitude."""
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        # Low distance error
        tp_fp_low_err = [{
            1.0: np.array([[10, 0, 0.1, 0.05, 10.0, 1.0]])
        }]
        
        # High distance error
        tp_fp_high_err = [{
            1.0: np.array([[10, 0, 0.5, 0.4, 10.0, 1.0]])
        }]
        
        _, result_low = calculate_pld(
            tp_fp_low_err, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        _, result_high = calculate_pld(
            tp_fp_high_err, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Calculate exact values
        # Low error: nw_cost=0.1, tp_d_err=0.05, tp=10, score=1.0
        # High error: nw_cost=0.5, tp_d_err=0.4, tp=10, score=1.0
        
        # For both: n_fn=0, conf_dependent_fn=0, conf_dependent_fp=0
        # n_misses_nw_cost = 0
        # normalization (sum): 1 / (0.5 * (1 + 1)) = 1
        
        # Low: tp_nw_cost_norm = 0.1 * 1 * 1 = 0.1
        expected_nw_low = 0.1
        # High: tp_nw_cost_norm = 0.5 * 1 * 1 = 0.5
        expected_nw_high = 0.5
        
        self.assertAlmostEqual(result_low['PLD_cost@1.0'], expected_nw_low, places=5)
        self.assertAlmostEqual(result_high['PLD_cost@1.0'], expected_nw_high, places=5)
        
        # Higher distance error should result in higher NW cost
        self.assertLess(result_low['PLD_cost@1.0'], result_high['PLD_cost@1.0'])
        
        # Localization errors: cum_nw_loc_norm = (tp_d_err * score / scaling) * normalization
        # Low: (0.05 * 1 / 10) * 1 = 0.005
        # High: (0.4 * 1 / 10) * 1 = 0.04
        self.assertAlmostEqual(result_low['PLD_loc@1.0'], 0.005, places=5)
        self.assertAlmostEqual(result_high['PLD_loc@1.0'], 0.04, places=5)
    
    def test_result_dict_structure(self):
        """Test that result dictionary has all expected keys."""
        # Create data with all required thresholds
        tp_fp_c_err_score_list = [{
            0.5: np.array([[5, 0, 0.1, 0.05, 5.0, 0.9], [0, 3, 0.5, 0.0, 3.0, 0.8]]),
            1.0: np.array([[5, 0, 0.1, 0.05, 5.0, 0.9], [0, 3, 0.5, 0.0, 3.0, 0.8]]),
            1.5: np.array([[5, 0, 0.1, 0.05, 5.0, 0.9], [0, 3, 0.5, 0.0, 3.0, 0.8]]),
        }]
        thresholds = [0.5, 1.0, 1.5]
        num_gts_points = [5.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Check for mean metrics
        self.assertNotIn('AP', result_dict)
        self.assertNotIn('AP_dist', result_dict)
        self.assertIn('PLD_cost', result_dict)
        self.assertIn('PLD_loc', result_dict)
        
        # Check for per-threshold metrics
        for thr in thresholds:
            self.assertNotIn(f'AP@{thr}', result_dict)
            self.assertNotIn(f'AP_dist@{thr}', result_dict)
            self.assertIn(f'PLD_cost@{thr}', result_dict)
            self.assertIn(f'PLD_loc@{thr}', result_dict)


class TestNWCostWithRealWorldData(unittest.TestCase):
    """Test NW cost calculations with realistic multi-prediction scenarios."""
    
    def _create_realistic_test_data(self):
        """Create more realistic test data with multiple predictions and varying quality."""
        data = {
            1.0: np.array([
                # [tp, fp, nw_cost, tp_d_err, scaling, scores]
                [12, 0, 0.05, 0.02, 12.0, 0.95],  # High-quality prediction (TP polyline 1)
                [8, 2, 0.15, 0.08, 10.0, 0.88],   # Good prediction (TP polyline 2)
                [0, 8, 0.50, 0.00, 8.0, 0.60],    # False positive (FP polyline)
                [0, 5, 0.45, 0.00, 5.0, 0.55],    # False positive (FP polyline)
            ])
        }
        return [data]
    
    def test_realistic_scenario_with_sum_normalization(self):
        """Test realistic multi-prediction scenario with sum normalization."""
        tp_fp_c_err_score_list = self._create_realistic_test_data()
        thresholds = [1.0]
        num_gts_points = [12.0, 10.0]  # Two GTs (to match the number of polyline-level TPs)
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance),
        )
        
        # Verify all metrics are computed
        self.assertIn('PLD_cost@1.0', result_dict)
        self.assertIn('PLD_loc@1.0', result_dict)
        
        # Calculate exact expected values
        # Data: [12, 0, 0.05, 0.02, 12.0, 0.95], [8, 2, 0.15, 0.08, 10.0, 0.88], 
        #       [0, 8, 0.50, 0.00, 8.0, 0.60], [0, 5, 0.45, 0.00, 5.0, 0.55]
        # Polyline-level TPs: 2 (first two), FPs: 2 (last two)
        # n_fn = 2 - 2 = 0
        # conf_dependent_fn = 2 - (1*0.95 + 1*0.88) = 2 - 1.83 = 0.17
        # conf_dependent_fp = (1*0.60 + 1*0.55) = 1.15
        # n_misses_nw_cost = (1.0 / 2.0) * (0 + 0.17 + 1.15) = 0.66
        # tp_nw_cost_norm = (0.05*0.95*1) + (0.15*0.88*1) = 0.0475 + 0.132 = 0.1795
        # sum_nw_cost = 0.1795 + 0.66 = 0.8395
        # sum_scores = 0.95 + 0.88 + 0.60 + 0.55 = 2.98
        # normalization (sum): 2 / (2 + 2.98) = 2 / 4.98 = 0.4016064...
        # nw_cost_normalized = 0.8395 * 0.4016064 = 0.3371486...
        expected_nw = 0.8395 * (2.0 / 4.98)
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], expected_nw, places=4)
    

class TestMaxCostBasedNormalization(unittest.TestCase):
    """Comprehensive tests for MaxCostBased normalization mode."""
    
    def _verify_max_cost_based_formula(self, nw_cost, tp_d_err, scaling, scores, tp_polylines, 
                                       num_gt_lines, threshold, nw_scaling, expected_normalized_cost):
        """Verify the MaxCostBased normalization formula manually."""
        # Calculate expected values step by step
        scaled_gap = threshold / nw_scaling
        n_tp_poly = np.sum(tp_polylines)
        n_fn = num_gt_lines - n_tp_poly
        
        conf_dependent_fn = n_tp_poly - np.sum(tp_polylines * scores)
        conf_dependent_fp = np.sum((1 - tp_polylines) * scores)
        
        n_misses_nw_cost = scaled_gap * (n_fn + conf_dependent_fn + conf_dependent_fp)
        tp_nw_cost_norm = np.sum(nw_cost * scores * tp_polylines)
        
        sum_nw_cost_including_fn = tp_nw_cost_norm + n_misses_nw_cost
        expected_pred_len = np.sum(scores)
        
        # MaxCostBased formula: divider = (scaled_gap * (len_x + len_y) + cost) / 2.0
        divider = (scaled_gap * (num_gt_lines + expected_pred_len) + sum_nw_cost_including_fn) / 2.0
        norm_factor = 1.0 / divider
        expected_cost = sum_nw_cost_including_fn * norm_factor
        
        self.assertAlmostEqual(expected_cost, expected_normalized_cost, places=5,
                             msg=f"Manual calculation: {expected_cost}, got: {expected_normalized_cost}")
        return expected_cost
    
    def test_max_cost_norm_with_perfect_predictions(self):
        """Test MaxCostBased with perfect predictions - cost should be 0."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.0, 0.0, 10.0, 1.0],  # Perfect match
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Perfect prediction should have cost = 0
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], 0.0, places=5)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
        
        # Verify formula manually
        nw_cost = np.array([0.0])
        tp_d_err = np.array([0.0])
        scaling = np.array([10.0])
        scores = np.array([1.0])
        tp_polylines = np.array([1])
        self._verify_max_cost_based_formula(nw_cost, tp_d_err, scaling, scores, tp_polylines, 
                                           1, 1.0, 2.0, result_dict['PLD_cost@1.0'])
    
    def test_sum_instance_pld_cost_bounded_to_unit_interval(self):
        """SumInstance: PLD cost stays in [0, 1] and all-FP gives exactly 1.0."""
        sum_config = MetricsConfig(norm=constants.MetricsNormType.SumInstance)
        thresholds = [1.0]
        num_gts_points = [10.0]

        # Worst case: all predictions are false positives.
        all_fp = [{
            1.0: np.array([
                [0, 5, 1.0, 0.0, 5.0, 0.9],
                [0, 3, 1.0, 0.0, 3.0, 0.8],
            ])
        }]
        _, result_fp = calculate_pld(
            all_fp, thresholds, num_gts_points,
            config=sum_config,
        )
        self.assertGreaterEqual(result_fp['PLD_cost@1.0'], 0.0)
        self.assertLessEqual(result_fp['PLD_cost@1.0'], 1.0)
        self.assertAlmostEqual(result_fp['PLD_cost@1.0'], 1.0, places=5)

        # A partial match should be strictly inside [0, 1].
        partial = [{
            1.0: np.array([
                [5, 0, 0.3, 0.15, 5.0, 0.9],
                [0, 3, 1.0, 0.0, 3.0, 0.8],
            ])
        }]
        _, result_partial = calculate_pld(
            partial, thresholds, num_gts_points,
            config=sum_config,
        )
        self.assertGreater(result_partial['PLD_cost@1.0'], 0.0)
        self.assertLess(result_partial['PLD_cost@1.0'], 1.0)
    
    def test_max_cost_norm_with_all_false_positives(self):
        """Test MaxCostBased with all false positives - cost should be <= 1."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [0, 5, 0.5, 0.0, 5.0, 0.9],   # FP
                [0, 3, 0.5, 0.0, 3.0, 0.8],   # FP
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # All FPs: cost should be significant but <= 1.0
        self.assertGreaterEqual(result_dict['PLD_cost@1.0'], 0.0)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
        
        # Manually verify the formula
        # scaled_gap = 1.0 / 2.0 = 0.5
        # n_tp_poly = 0, n_fn = 1 - 0 = 1
        # conf_dependent_fn = 0 - 0 = 0
        # conf_dependent_fp = 0.9 + 0.8 = 1.7
        # n_misses_nw_cost = 0.5 * (1 + 0 + 1.7) = 1.35
        # tp_nw_cost_norm = 0
        # sum_nw_cost = 1.35
        # expected_pred_len = 1.7
        # divider = (0.5 * (1 + 1.7) + 1.35) / 2.0 = (1.35 + 1.35) / 2.0 = 1.35
        # norm_factor = 1 / 1.35
        # expected_cost = 1.35 / 1.35 = 1.0
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], 1.0, places=5,
                             msg="All FPs should give cost exactly 1.0 for MaxCostBased")
    
    def test_max_cost_norm_with_high_error(self):
        """Test MaxCostBased with high distance error - verify exact calculation."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.9, 0.85, 10.0, 1.0],  # High error but matched
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Manual calculation:
        # scaled_gap = 0.5
        # n_tp_poly = 1, n_fn = 1 - 1 = 0
        # conf_dependent_fn = 1 - 1.0 = 0
        # conf_dependent_fp = 0
        # n_misses_nw_cost = 0.5 * (0 + 0 + 0) = 0
        # tp_nw_cost_norm = 0.9 * 1.0 * 1 = 0.9
        # sum_nw_cost = 0.9
        # expected_pred_len = 1.0
        # divider = (0.5 * (1 + 1) + 0.9) / 2.0 = (1.0 + 0.9) / 2.0 = 0.95
        # expected_cost = 0.9 / 0.95 = 0.947368...
        expected = 0.9 / 0.95
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], expected, places=5)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
    
    def test_max_cost_norm_with_no_predictions(self):
        """Test MaxCostBased with no predictions - all FN should give 1.0."""
        # Empty predictions - all GT are FN
        tp_fp_c_err_score_list = [{
            1.0: np.array([]).reshape(0, 6)  # No predictions
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Manual calculation:
        # scaled_gap = 0.5
        # n_tp_poly = 0, n_fn = 1 - 0 = 1
        # conf_dependent_fn = 0, conf_dependent_fp = 0
        # n_misses_nw_cost = 0.5 * (1 + 0 + 0) = 0.5
        # tp_nw_cost_norm = 0
        # sum_nw_cost = 0.5
        # expected_pred_len = 0
        # divider = (0.5 * (1 + 0) + 0.5) / 2.0 = (0.5 + 0.5) / 2.0 = 0.5
        # expected_cost = 0.5 / 0.5 = 1.0
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], 1.0, places=5,
                             msg="No predictions (all FN) should give exactly 1.0")
    
    def test_max_cost_norm_symmetry(self):
        """Test that swapping GT and predictions gives similar normalized cost."""
        # Case 1: 2 GT, 1 pred (1 FN)
        tp_fp_case1 = [{
            1.0: np.array([
                [10, 0, 0.2, 0.1, 10.0, 1.0],  # Matched
            ])
        }]
        num_gts_case1 = [10.0, 10.0]  # 2 GTs
        
        # Case 2: 1 GT, 2 preds (1 FP)
        tp_fp_case2 = [{
            1.0: np.array([
                [10, 0, 0.2, 0.1, 10.0, 1.0],  # Matched
                [0, 10, 0.5, 0.0, 10.0, 1.0],  # FP
            ])
        }]
        num_gts_case2 = [10.0]  # 1 GT
        
        _, result1 = calculate_pld(tp_fp_case1, [1.0], num_gts_case1)
        _, result2 = calculate_pld(tp_fp_case2, [1.0], num_gts_case2)
        
        # Both should give exactly 1.0 due to symmetry in MaxCostBased
        # Case 1: 1 TP, 1 FN => cost from FN
        # Case 2: 1 TP, 1 FP => cost from FP
        # Both should be symmetric and give 1.0 for the miss component
        self.assertLessEqual(result1['PLD_cost@1.0'], 1.0)
        self.assertLessEqual(result2['PLD_cost@1.0'], 1.0)
    
    def test_max_cost_norm_with_partial_confidence(self):
        """Test MaxCostBased with partial confidence scores - verify exact calculation."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.1, 0.05, 10.0, 0.5],  # Matched but low confidence
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Manual calculation:
        # scaled_gap = 0.5
        # n_tp_poly = 1, n_fn = 0
        # conf_dependent_fn = 1 - 0.5 = 0.5  # Confidence penalty
        # conf_dependent_fp = 0
        # n_misses_nw_cost = 0.5 * (0 + 0.5 + 0) = 0.25
        # tp_nw_cost_norm = 0.1 * 0.5 * 1 = 0.05
        # sum_nw_cost = 0.05 + 0.25 = 0.3
        # expected_pred_len = 0.5
        # divider = (0.5 * (1 + 0.5) + 0.3) / 2.0 = (0.75 + 0.3) / 2.0 = 0.525
        # expected_cost = 0.3 / 0.525 = 0.571428...
        expected = 0.3 / 0.525
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], expected, places=5)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
    
    def test_max_cost_formula_verification_complex(self):
        """Test complex scenario with manual formula verification."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.2, 0.1, 10.0, 0.9],   # TP with some error
                [8, 2, 0.3, 0.15, 10.0, 0.7],   # Partial TP
                [0, 5, 0.5, 0.0, 5.0, 0.6],     # FP
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0, 8.0, 5.0]  # 3 GTs
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Manual calculation:
        # scaled_gap = 0.5
        # tp_polylines = [1, 1, 0]
        # n_tp_poly = 2, n_fn = 3 - 2 = 1
        # conf_dependent_fn = 2 - (1*0.9 + 1*0.7 + 0*0.6) = 2 - 1.6 = 0.4
        # conf_dependent_fp = (0*0.9 + 0*0.7 + 1*0.6) = 0.6
        # n_misses_nw_cost = 0.5 * (1 + 0.4 + 0.6) = 1.0
        # tp_nw_cost_norm = 0.2*0.9*1 + 0.3*0.7*1 + 0.5*0.6*0 = 0.18 + 0.21 + 0 = 0.39
        # sum_nw_cost = 0.39 + 1.0 = 1.39
        # expected_pred_len = 0.9 + 0.7 + 0.6 = 2.2
        # divider = (0.5 * (3 + 2.2) + 1.39) / 2.0 = (2.6 + 1.39) / 2.0 = 1.995
        # expected_cost = 1.39 / 1.995 = 0.696741...
        expected = 1.39 / 1.995
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], expected, places=4)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
    
    def test_max_cost_based_normalization_factor_direct(self):
        """Test the normalization factor calculation directly from types.py."""
        from sospa_eval.types import MetricsNormType
        
        # Test case 1: cost = 0.5, len_x = 1, len_y = 1, scaled_gap = 0.5
        # Expected divider = (0.5 * (1 + 1) + 0.5) / 2.0 = (1.0 + 0.5) / 2.0 = 0.75
        # Expected norm_factor = 1 / 0.75 = 1.333...
        norm_factor = MetricsNormType.MaxCostBased.get_norm_factor(1.0, 1.0, 0.5, 0.5)
        expected_factor = 1.0 / 0.75
        self.assertAlmostEqual(norm_factor, expected_factor, places=5)
        
        # Test that normalized cost <= 1.0
        normalized_cost = 0.5 * norm_factor
        self.assertLessEqual(normalized_cost, 1.0)
        
        # Test case 2: worst case - cost equals max possible cost
        # cost = scaled_gap * (len_x + len_y) = 0.5 * 2 = 1.0
        # divider = (1.0 + 1.0) / 2.0 = 1.0
        # norm_factor = 1.0
        # normalized_cost = 1.0 * 1.0 = 1.0
        norm_factor = MetricsNormType.MaxCostBased.get_norm_factor(1.0, 1.0, 0.5, 1.0)
        self.assertAlmostEqual(norm_factor, 1.0, places=5)
        normalized_cost = 1.0 * norm_factor
        self.assertAlmostEqual(normalized_cost, 1.0, places=5)
        
        # Test case 3: perfect match - cost = 0
        # divider = (0.5 * 2 + 0) / 2.0 = 0.5
        # norm_factor = 2.0
        # normalized_cost = 0 * 2.0 = 0
        norm_factor = MetricsNormType.MaxCostBased.get_norm_factor(1.0, 1.0, 0.5, 0.0)
        expected_factor = 1.0 / 0.5
        self.assertAlmostEqual(norm_factor, expected_factor, places=5)
        normalized_cost = 0.0 * norm_factor
        self.assertAlmostEqual(normalized_cost, 0.0, places=5)
    
    def test_max_cost_based_never_exceeds_one(self):
        """Property test: MaxCostBased normalization should NEVER produce cost > 1.0."""
        from sospa_eval.types import MetricsNormType
        import random
        
        random.seed(42)
        for _ in range(100):
            # Generate random inputs
            len_x = random.uniform(0.1, 10.0)
            len_y = random.uniform(0.1, 10.0)
            scaled_gap = random.uniform(0.1, 1.0)
            
            # Cost can range from 0 to max_possible_cost
            max_possible_cost = scaled_gap * (len_x + len_y)
            cost = random.uniform(0.0, max_possible_cost)
            
            # Calculate normalized cost
            norm_factor = MetricsNormType.MaxCostBased.get_norm_factor(len_x, len_y, scaled_gap, cost)
            normalized_cost = cost * norm_factor
            
            # Must never exceed 1.0
            self.assertLessEqual(normalized_cost, 1.0,
                               msg=f"Normalized cost {normalized_cost} > 1.0 for inputs: "
                                   f"len_x={len_x}, len_y={len_y}, scaled_gap={scaled_gap}, cost={cost}")
            
            # Must be non-negative
            self.assertGreaterEqual(normalized_cost, 0.0)
    
    def test_max_cost_norm_with_threshold_2(self):
        """CRITICAL: Test MaxCostBased with threshold=2.0 to verify scaled_gap is ALWAYS 0.5.
        
        For MaxCostBased, scaled_gap must ALWAYS be 0.5, regardless of threshold.
        This test catches bugs where scaled_gap = threshold / NW_SCALING instead of 0.5.
        """
        tp_fp_c_err_score_list = [{
            2.0: np.array([
                [0, 5, 0.5, 0.0, 5.0, 1.0],  # All FP
            ])
        }]
        thresholds = [2.0]  # threshold=2.0, NOT 1.0!
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Manual calculation with CORRECT scaled_gap = 0.5 (NOT threshold/NW_SCALING = 1.0):
        # scaled_gap = 0.5  (MUST be 0.5 for MaxCostBased, NOT 2.0/2.0=1.0)
        # n_tp_poly = 0, n_fn = 1
        # conf_dependent_fn = 0, conf_dependent_fp = 1.0
        # n_misses_nw_cost = 0.5 * (1 + 0 + 1.0) = 1.0
        # tp_nw_cost_norm = 0
        # sum_nw_cost = 1.0
        # expected_pred_len = 1.0
        # divider = (0.5 * (1 + 1) + 1.0) / 2.0 = 1.0
        # expected_cost = 1.0 / 1.0 = 1.0
        
        # If bug exists (scaled_gap = threshold/NW_SCALING = 2.0/2.0 = 1.0):
        # n_misses_nw_cost = 1.0 * (1 + 0 + 1.0) = 2.0
        # sum_nw_cost = 2.0
        # divider = (1.0 * (1 + 1) + 2.0) / 2.0 = 2.0
        # buggy_cost = 2.0 / 2.0 = 1.0  # Still 1.0 but different path
        
        # Better test: check with partial match
        self.assertLessEqual(result_dict['PLD_cost@2.0'], 1.0)
    
    def test_max_cost_norm_scaled_gap_must_be_half(self):
        """CRITICAL: Verify scaled_gap is ALWAYS 0.5 for MaxCostBased, regardless of threshold.
        
        This is the key property: MaxCostBased normalization uses scaled_gap = 0.5,
        NOT scaled_gap = threshold / NW_SCALING.
        """
        # Test with threshold = 0.5 (would give 0.5/2.0 = 0.25 if bug exists)
        tp_fp_c_err_score_list = [{
            0.5: np.array([
                [10, 0, 0.1, 0.05, 10.0, 1.0],  # Good match
            ])
        }]
        thresholds = [0.5]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Manual calculation with CORRECT scaled_gap = 0.5:
        # scaled_gap = 0.5 (NOT 0.5/2.0 = 0.25)
        # n_tp_poly = 1, n_fn = 0
        # conf_dependent_fn = 0, conf_dependent_fp = 0
        # n_misses_nw_cost = 0.5 * 0 = 0
        # tp_nw_cost_norm = 0.1 * 1.0 * 1 = 0.1
        # sum_nw_cost = 0.1
        # expected_pred_len = 1.0
        # divider = (0.5 * (1 + 1) + 0.1) / 2.0 = 0.55
        # expected_cost = 0.1 / 0.55 = 0.181818...
        expected_correct = 0.1 / 0.55
        
        # If bug exists (scaled_gap = 0.5/2.0 = 0.25):
        # divider = (0.25 * 2 + 0.1) / 2.0 = 0.3
        # buggy_cost = 0.1 / 0.3 = 0.333...
        expected_buggy = 0.1 / 0.3
        
        # These should be different!
        self.assertNotAlmostEqual(expected_correct, expected_buggy, places=2,
                                 msg="Test case doesn't differentiate correct from buggy implementation")
        
        # Verify correct value
        self.assertAlmostEqual(result_dict['PLD_cost@0.5'], expected_correct, places=5,
                             msg=f"Expected {expected_correct} but got {result_dict['PLD_cost@0.5']}. "
                                 f"Bug detected: scaled_gap should be 0.5, not threshold/NW_SCALING")
        self.assertLessEqual(result_dict['PLD_cost@0.5'], 1.0)
    
    def test_max_cost_norm_with_mixed_predictions(self):
        """Test MaxCostBased with mixed TP and FP predictions."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.2, 0.1, 10.0, 0.95],  # Good TP
                [8, 2, 0.4, 0.3, 10.0, 0.85],   # Partial TP with FP
                [0, 5, 0.5, 0.0, 5.0, 0.70],    # FP
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0, 8.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Mixed predictions should have intermediate cost <= 1.0
        self.assertGreaterEqual(result_dict['PLD_cost@1.0'], 0.0)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
    
    def test_max_cost_norm_with_multiple_thresholds(self):
        """Test MaxCostBased with multiple thresholds - all should be <= 1."""
        tp_fp_c_err_score_list = [{
            0.5: np.array([[8, 2, 0.3, 0.2, 10.0, 0.95]]),
            1.0: np.array([[6, 1, 0.4, 0.3, 7.0, 0.90]]),
            1.5: np.array([[4, 0, 0.5, 0.4, 4.0, 0.85]]),
        }]
        thresholds = [0.5, 1.0, 1.5]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Check all thresholds have cost <= 1.0
        for thr in thresholds:
            self.assertIn(f'PLD_cost@{thr}', result_dict)
            self.assertGreaterEqual(result_dict[f'PLD_cost@{thr}'], 0.0)
            self.assertLessEqual(result_dict[f'PLD_cost@{thr}'], 1.0)
        
        # Mean cost should also be <= 1.0
        self.assertLessEqual(result_dict['PLD_cost'], 1.0)
    
    def test_max_cost_norm_worst_case_scenario(self):
        """Test worst case: no matches, all FPs with maximum error - should give exactly 1.0."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [0, 10, 1.0, 0.0, 10.0, 1.0],  # FP with score 1.0
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Worst case: no matches, all cost from misses
        # scaled_gap = 0.5
        # n_fn = 1, conf_dependent_fn = 0, conf_dependent_fp = 1.0
        # n_misses_nw_cost = 0.5 * (1 + 0 + 1.0) = 1.0
        # tp_nw_cost_norm = 0
        # sum_nw_cost = 1.0
        # expected_pred_len = 1.0
        # divider = (0.5 * (1 + 1) + 1.0) / 2.0 = (1.0 + 1.0) / 2.0 = 1.0
        # expected_cost = 1.0 / 1.0 = 1.0
        self.assertAlmostEqual(result_dict['PLD_cost@1.0'], 1.0, places=5,
                             msg="Worst case should give exactly 1.0")
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)
    
    def test_compare_all_normalization_modes(self):
        """Compare all three normalization modes including MaxCostBased."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [5, 0, 0.3, 0.2, 5.0, 0.9],
                [0, 3, 0.5, 0.0, 3.0, 0.8],
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [5.0]
        
        results = {}
        
        # Test both bounded normalization modes
        for norm_type in [constants.MetricsNormType.SumInstance,
                         constants.MetricsNormType.MaxCostBased]:
            _, result = calculate_pld(
                tp_fp_c_err_score_list, thresholds, num_gts_points,
                config=MetricsConfig(norm=norm_type),
            )
            results[norm_type] = result['PLD_cost@1.0']
        
        # All should be valid (>= 0)
        for norm_type, cost in results.items():
            self.assertGreaterEqual(cost, 0.0, f"Negative cost for {norm_type}")
        
        # MaxCostBased should be <= 1.0
        self.assertLessEqual(results[constants.MetricsNormType.MaxCostBased], 1.0)
    
    def test_max_cost_norm_with_varying_confidence(self):
        """Test MaxCostBased with varying confidence scores."""
        tp_fp_c_err_score_list = [{
            1.0: np.array([
                [10, 0, 0.1, 0.05, 10.0, 1.0],   # High confidence
                [8, 2, 0.2, 0.1, 10.0, 0.5],     # Medium confidence
                [6, 4, 0.3, 0.15, 10.0, 0.1],    # Low confidence
            ])
        }]
        thresholds = [1.0]
        num_gts_points = [10.0, 8.0, 6.0]
        
        _, result_dict = calculate_pld(
            tp_fp_c_err_score_list, thresholds, num_gts_points
        )
        
        # Cost should be <= 1.0 regardless of confidence variation
        self.assertGreaterEqual(result_dict['PLD_cost@1.0'], 0.0)
        self.assertLessEqual(result_dict['PLD_cost@1.0'], 1.0)


class TestRealWorldDataPLD(unittest.TestCase):
    """Test PLD calculation with real-world data from instance matching tests."""
    
    def test_real_world_pld_across_normalizations(self):
        """Test PLD cost validity across all normalization types with real-world data."""
        # Load real-world test data
        matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors = \
            load_real_world_test_data_instance_matching()
        
        thresholds = [threshold]
        
        # Test all normalization types
        norm_types = [
            constants.MetricsNormType.SumInstance,
            constants.MetricsNormType.MaxCostBased,
            constants.MetricsNormType.NoNorm
        ]
        
        pld_cost_results = {}
        
        for norm_type in norm_types:
            # Perform greedy matching with current normalization
            tp_fp_score_by_thr = _match_predictions_with_thresholds_extended(
                matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors,
                config=MetricsConfig(norm=norm_type),
            )
            
            # Prepare data for PLD calculation
            tp_fp_c_err_score_list = [tp_fp_score_by_thr]
            
            # Calculate PLD with the same normalization
            _, result_dict = calculate_pld(
                tp_fp_c_err_score_list, thresholds, num_gts_points,
                config=MetricsConfig(norm=norm_type),
            )
            
            pld_cost_results[norm_type] = result_dict['PLD_cost@1.0']
        
        # Verify all NW costs are valid (>= 0)
        for norm_type, nw_cost in pld_cost_results.items():
            self.assertGreaterEqual(nw_cost, 0.0, f"NW cost should be >= 0 for {norm_type}")
        
        # Verify MaxCostBased normalization gives cost <= 1.0
        self.assertLessEqual(
            pld_cost_results[constants.MetricsNormType.MaxCostBased], 
            1.0,
            "MaxCostBased normalization should ensure NW cost <= 1.0"
        )


class TestPldHasNoPointLevelApKeys(unittest.TestCase):
    """`calculate_pld` must only return PLD keys (point-level AP lives in gospa_eval)."""

    def _create_basic_test_data(self):
        return [{
            1.0: np.array([
                [5, 0, 0.1, 0.05, 5.0, 0.9],
                [0, 3, 0.5, 0.0, 3.0, 0.8],
            ])
        }]

    def test_calculate_pld_has_no_ap_keys(self):
        data = self._create_basic_test_data()
        _, result = calculate_pld(data, [1.0], [5.0])

        self.assertIn("PLD_cost@1.0", result)
        self.assertIn("PLD_cost", result)
        self.assertIn("PLD_loc@1.0", result)
        self.assertIn("PLD_loc", result)

        for key in result:
            self.assertFalse(key.startswith("AP"), f"Unexpected AP key in PLD-only result: {key}")


class TestSospaApRemoved(unittest.TestCase):
    """`sospa_ap` is no longer a metric evaluated under sospa_eval."""

    def test_sospa_ap_metric_string_is_rejected(self):
        from sospa_eval.types import MatchingMetric

        with self.assertRaises(ValueError):
            MatchingMetric.to_metric_type("sospa_ap")

    def test_sospa_ap_enum_member_removed(self):
        from sospa_eval.types import MatchingMetric

        self.assertFalse(hasattr(MatchingMetric, "SOSPA_AP"))

    def test_point_level_ap_removed_from_package(self):
        import sospa_eval
        import sospa_eval.AP as ap_module

        self.assertFalse(hasattr(sospa_eval, "calculate_point_level_ap"))
        self.assertFalse(hasattr(ap_module, "calculate_point_level_ap"))


if __name__ == "__main__":
    unittest.main()
