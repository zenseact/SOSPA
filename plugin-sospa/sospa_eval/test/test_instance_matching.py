import copy
import unittest
from unittest import mock
import numpy as np


from sospa_eval.instance_matching import (
    _match_predictions_with_thresholds,
    instance_match,
    instance_match_pld,
    _match_predictions_with_thresholds_extended,
    _match_predictions_with_thresholds_optimally,
)
from sospa_eval.types import MatchingMetric, MetricsNormType
from sospa_eval.PLD import calculate_pld
from sospa_eval import constants
from sospa_eval.constants import MetricsConfig, SOSPA_SCALING
import sospa_eval.instance_matching as instance_matching_module

from sospa_eval.test.helpers import load_real_world_test_data_instance_matching

class TestMatchPredictionsWithThresholds(unittest.TestCase):

    def test_basic_functionality(self):
        # Simple test with 3 predictions and 2 ground truths
        matrix = np.array(
            [
                [0.1, 0.5],  # First prediction is close to first GT
                [0.6, 0.3],  # Second prediction is closer to second GT
                [0.9, 0.8],  # Third prediction is far from both GTs
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])  # Confidence scores
        thresholds = [0.2, 0.5]  # Two different thresholds
        num_preds = 3
        num_gts = 2

        tp_fp_score_by_thr = _match_predictions_with_thresholds(
            matrix, scores, thresholds, num_preds, num_gts
        )

        # Check tp_fp_score_by_thr structure
        self.assertEqual(len(tp_fp_score_by_thr), 2)

        # For threshold 0.2, only the first prediction should be TP
        tp1 = tp_fp_score_by_thr[0.2][:, 0]  # First column is tp
        fp1 = tp_fp_score_by_thr[0.2][:, 1]  # Second column is fp
        self.assertEqual(tp1.tolist(), [1, 0, 0])
        self.assertEqual(fp1.tolist(), [0, 1, 1])

        # For threshold 0.5, first and second predictions should be TP
        tp2 = tp_fp_score_by_thr[0.5][:, 0]
        fp2 = tp_fp_score_by_thr[0.5][:, 1]
        self.assertEqual(tp2.tolist(), [1, 1, 0])
        self.assertEqual(fp2.tolist(), [0, 0, 1])

        # Check tp_fp_score_by_thr structure
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), [0.2, 0.5])

        # Check format of values in dictionary
        for thr in thresholds:
            expected_shape = (num_preds, 3)  # [tp, fp, score] for each prediction
            self.assertEqual(tp_fp_score_by_thr[thr].shape, expected_shape)

    def test_no_predictions(self):
        matrix = np.array([]).reshape(0, 2)
        scores = np.array([])
        thresholds = [0.2, 0.5]
        num_preds = 0
        num_gts = 2

        tp_fp_score_by_thr = _match_predictions_with_thresholds(
            matrix, scores, thresholds, num_preds, num_gts
        )

        self.assertEqual(len(tp_fp_score_by_thr), 2)
        for thr in thresholds:
            tp = tp_fp_score_by_thr[thr][:, 0]
            fp = tp_fp_score_by_thr[thr][:, 1]
            self.assertEqual(tp.tolist(), [])
            self.assertEqual(fp.tolist(), [])

        # Check dictionary result
        for thr in thresholds:
            self.assertTrue(thr in tp_fp_score_by_thr)
            # Empty array with 3 columns (tp, fp, score)
            self.assertEqual(tp_fp_score_by_thr[thr].shape, (0, 3))

    def test_no_ground_truths(self):
        # The case is handled outside the function
        pass

    def test_tied_scores(self):
        matrix = np.array(
            [
                [0.1, 0.5],
                [0.1, 0.2],
                [0.4, 0.3],
            ]
        )
        # Tied scores - should be processed in order
        scores = np.array([0.8, 0.8, 0.7])
        thresholds = [0.5]
        num_preds = 3
        num_gts = 2

        tp_fp_score_by_thr = _match_predictions_with_thresholds(
            matrix, scores, thresholds, num_preds, num_gts
        )

        tp = tp_fp_score_by_thr[thresholds[0]][:, 0]
        fp = tp_fp_score_by_thr[thresholds[0]][:, 1]
        # Should assign predictions with tied scores in order they appear
        self.assertEqual(tp.tolist(), [1, 0, 1])
        self.assertEqual(fp.tolist(), [0, 1, 0])

        # Check the score dictionary
        expected_shape = (num_preds, 3)  # [tp, fp, score] for each prediction
        self.assertEqual(tp_fp_score_by_thr[0.5].shape, expected_shape)

        # Verify the scores in the dictionary match the input scores
        np.testing.assert_array_equal(tp_fp_score_by_thr[0.5][:, 2], scores)

    def test_multiple_thresholds(self):
        matrix = np.array(
            [
                [0.1, 0.5, 0.7],  # First prediction is close to first GT
                [0.3, 0.3, 0.2],
                [0.6, 0.4, 0.7],  # Second prediction is closer to second GT
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        thresholds = [0.1, 0.3, 0.5, 0.7]
        num_preds = 3
        num_gts = 3

        tp_fp_score_by_thr = _match_predictions_with_thresholds(
            matrix, scores, thresholds, num_preds, num_gts
        )

        self.assertEqual(len(tp_fp_score_by_thr), 4)

        # Check dictionary contains all thresholds
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), sorted(thresholds))

        # For threshold 0.1, only first prediction is below threshold
        tp1 = tp_fp_score_by_thr[0.1][:, 0]
        fp1 = tp_fp_score_by_thr[0.1][:, 1]
        self.assertEqual(tp1.tolist(), [1, 0, 0])
        self.assertEqual(fp1.tolist(), [0, 1, 1])

        # For threshold 0.3, first and second predictions are below threshold
        tp2 = tp_fp_score_by_thr[0.3][:, 0]
        fp2 = tp_fp_score_by_thr[0.3][:, 1]
        self.assertEqual(tp2.tolist(), [1, 1, 0])
        self.assertEqual(fp2.tolist(), [0, 0, 1])

        # For threshold 0.5, all predictions should match with at least one GT
        tp3 = tp_fp_score_by_thr[0.5][:, 0]
        fp3 = tp_fp_score_by_thr[0.5][:, 1]
        self.assertEqual(tp3.tolist(), [1, 1, 1])
        self.assertEqual(fp3.tolist(), [0, 0, 0])

        # For threshold 0.7, all predictions should match with at least one GT
        tp4 = tp_fp_score_by_thr[0.7][:, 0]
        fp4 = tp_fp_score_by_thr[0.7][:, 1]
        self.assertEqual(tp4.tolist(), [1, 1, 1])
        self.assertEqual(fp4.tolist(), [0, 0, 0])

        # Check contents in dictionary for one threshold
        expected_format = np.hstack(
            [
                tp1[:, None],  # True positives column
                fp1[:, None],  # False positives column
                scores[:, None],  # Scores column
            ]
        )
        np.testing.assert_array_equal(tp_fp_score_by_thr[0.1], expected_format)

    def test_all_matches_above_threshold(self):
        # All distances are above threshold - all should be FP
        matrix = np.array(
            [
                [0.8, 0.9],
                [0.7, 0.6],
            ]
        )
        scores = np.array([0.95, 0.85])
        thresholds = [0.5]
        num_preds = 2
        num_gts = 2

        tp_fp_score_by_thr = _match_predictions_with_thresholds(
            matrix, scores, thresholds, num_preds, num_gts
        )

        tp = tp_fp_score_by_thr[thresholds[0]][:, 0]
        fp = tp_fp_score_by_thr[thresholds[0]][:, 1]
        # All FP because all distances > threshold
        self.assertEqual(tp.tolist(), [0, 0])
        self.assertEqual(fp.tolist(), [1, 1])

        # Check dictionary
        self.assertEqual(tp_fp_score_by_thr[0.5].shape, (num_preds, 3))
        np.testing.assert_array_equal(tp_fp_score_by_thr[0.5][:, 0], [0, 0])  # TP
        np.testing.assert_array_equal(tp_fp_score_by_thr[0.5][:, 1], [1, 1])  # FP


class TestMatchPredictionsWithThresholdsExtended(unittest.TestCase):

    def test_basic_functionality(self):
        # Simple test with 3 predictions and 2 ground truths
        matrix = np.array(
            [
                [0.1, 0.5],  # First prediction is close to first GT
                [0.6, 0.3],  # Second prediction is closer to second GT
                [0.9, 0.8],  # Third prediction is far from both GTs
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])  # Confidence scores
        thresholds = [0.5]  # One threshold for simplicity
        num_preds = 3
        num_gts = 2
        num_gts_points = [10, 10]  # Number of points in each GT polyline

        # Mock ntp, nfp, and dist_errors
        ntp = np.array(
            [
                [0.8, 0.4],
                [0.3, 0.7],
                [0.1, 0.1],
            ]
        )
        nfp = np.array(
            [
                [0.2, 0.6],
                [0.7, 0.3],
                [0.9, 0.9],
            ]
        )
        dist_errors = np.array(
            [
                [0.05, 0.25],
                [0.30, 0.15],
                [0.45, 0.40],
            ]
        )

        tp_fp_score_by_thr = _match_predictions_with_thresholds_extended(
            matrix, scores, thresholds[0], num_preds, num_gts_points, ntp, nfp, dist_errors
        )

        # Check output structure
        self.assertEqual(len(tp_fp_score_by_thr), 1)
        result = tp_fp_score_by_thr[thresholds[0]]
        tp = result[:, 0]
        fp = result[:, 1]

        # Check structure of dictionary
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), [0.5])

        # For threshold 0.5, predictions 1 and 2 should be TP
        np.testing.assert_array_almost_equal(tp, [0.8, 0.7, 0])
        np.testing.assert_array_almost_equal(fp, [0.2, 0.3, ntp[2, 0] + nfp[2, 0]])

        # Check format of values in dictionary
        self.assertEqual(
            tp_fp_score_by_thr[0.5].shape, (num_preds, 6)
        )  # [tp, fp, nw_cost, tp_d_err, scaling, score]

        # Check dist_errors are properly included (column 3 is tp_d_err)
        expected_tp_err = np.array([0.05, 0.15, 0.0])
        np.testing.assert_almost_equal(tp_fp_score_by_thr[0.5][:, 3], expected_tp_err)


class TestInstanceMatchNWDecomposed(unittest.TestCase):

    def setUp(self):
        self.config = MetricsConfig(norm=constants.MetricsNormType.SumInstance)

    def test_basic_functionality(self):
        # Create simple pred and gt lines
        thresholds = [0.3, 0.5]

        pred_lines = [np.array([[0, 0], [1, 1]]), np.array([[2, 2], [3, 3]])]
        gt_lines = [
            np.array([[0 + thresholds[0] - 0.1, 0], [1, 1]]),
            np.array([[2, 2 + thresholds[1] - 0.1], [3 + thresholds[1] + 0.1, 3]]),
        ]
        scores = np.array([0.9, 0.8])

        # Call the function
        tp_fp_score_by_thr = instance_match_pld(
            pred_lines, scores, gt_lines, thresholds,
            config=self.config
        )

        # Check output structure
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), sorted(thresholds))

        # Check shape of result for each threshold
        for thr in thresholds:
            self.assertEqual(
                tp_fp_score_by_thr[thr].shape, (2, 6)
            )  # 2 preds, 6 columns (tp, fp, nw_cost, tp_d_err, scaling, score)

        # For threshold 0.3, only first prediction should be TP
        # For threshold 0.5, both predictions should be TP
        # These assertions depend on our mock function behavior
        np.testing.assert_array_almost_equal(tp_fp_score_by_thr[0.3][:, 0], [2, 0])  #
        np.testing.assert_array_almost_equal(
            tp_fp_score_by_thr[0.3][:, 1], [0, 2]
        )  # FP

        self.assertGreater(
            tp_fp_score_by_thr[0.3][0, 2], 0.0
        )  # nw_cost should be > 0 for TPs
        self.assertGreaterEqual(tp_fp_score_by_thr[0.3][1, 2], 0.0)  # nw_cost >= 0

        np.testing.assert_array_almost_equal(
            tp_fp_score_by_thr[0.5][:, 0], [2, 1]
        )  # TP
        np.testing.assert_array_almost_equal(
            tp_fp_score_by_thr[0.5][:, 1], [0, 1]
        )  # FP
        for thr in thresholds:
            np.testing.assert_array_almost_equal(tp_fp_score_by_thr[thr][:, 5], scores)  # Column 5 is scores

    def test_no_predictions(self):
        pred_lines = []
        gt_lines = [np.array([[0, 0], [1, 1]])]
        scores = np.array([])
        thresholds = [0.3, 0.5]

        tp_fp_score_by_thr = instance_match_pld(
            pred_lines, scores, gt_lines, thresholds,
            config=self.config
        )

        # Check structure
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), sorted(thresholds))

        # Check empty results
        for thr in thresholds:
            self.assertEqual(tp_fp_score_by_thr[thr].shape, (0, 6))

    def test_no_ground_truths(self):
        pred_lines = [np.array([[0, 0], [1, 1]])]
        gt_lines = []
        scores = np.array([0.9])
        thresholds = [0.3, 0.5]

        tp_fp_score_by_thr = instance_match_pld(
            pred_lines, scores, gt_lines, thresholds,
            config=self.config
        )

        # Check structure
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), sorted(thresholds))

        # All should be FP
        for thr in thresholds:
            self.assertEqual(tp_fp_score_by_thr[thr].shape, (1, 6))
            np.testing.assert_array_equal(tp_fp_score_by_thr[thr][:, 0], [0.0])  # TP
            self.assertGreater(tp_fp_score_by_thr[thr][0, 1], 0.0)  # FP should be > 0
            # FP should equal the number of points in the prediction
            self.assertEqual(tp_fp_score_by_thr[thr][0, 1], len(pred_lines[0]))
            # Score should match input
            self.assertEqual(tp_fp_score_by_thr[thr][0, 5], scores[0])
            # NW cost should be threshold / NW_SCALING
            expected_nw_cost = thr / SOSPA_SCALING
            if self.config.norm == constants.MetricsNormType.SumInstance:
                expected_nw_cost /= thr/ SOSPA_SCALING
            self.assertAlmostEqual(tp_fp_score_by_thr[thr][0, 2], expected_nw_cost)
            # Scaling should be len(pred_lines[0]) or len(pred_lines[0])/2
            expected_scaling = len(pred_lines[0])
            if self.config.norm == constants.MetricsNormType.SumInstance:
                expected_scaling *= thr/ SOSPA_SCALING
            self.assertAlmostEqual(tp_fp_score_by_thr[thr][0, 4], expected_scaling)

    def test_no_predictions_no_ground_truths(self):
        """Test when both predictions and ground truths are empty."""
        pred_lines = []
        gt_lines = []
        scores = np.array([])
        thresholds = [0.3, 0.5]

        tp_fp_score_by_thr = instance_match_pld(
            pred_lines, scores, gt_lines, thresholds,
            config=self.config
        )

        # Check structure
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), sorted(thresholds))

        # Check empty results for all thresholds
        for thr in thresholds:
            self.assertEqual(tp_fp_score_by_thr[thr].shape, (0, 6))

    def test_multiple_predictions_no_ground_truths(self):
        """Test when there are multiple predictions but no ground truths."""
        pred_lines = [
            np.array([[0, 0], [1, 1], [2, 2]]),  # 3 points
            np.array([[0, 0], [1, 1]]),          # 2 points
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]])  # 4 points
        ]
        gt_lines = []
        scores = np.array([0.9, 0.8, 0.7])
        thresholds = [0.3, 0.5]

        tp_fp_score_by_thr = instance_match_pld(
            pred_lines, scores, gt_lines, thresholds,
            config=self.config
        )

        # Check structure
        self.assertEqual(sorted(tp_fp_score_by_thr.keys()), sorted(thresholds))

        for thr in thresholds:
            result = tp_fp_score_by_thr[thr]
            self.assertEqual(result.shape, (3, 6))
            
            # All TPs should be 0
            np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0])
            
            # FPs should equal the number of points in each prediction
            np.testing.assert_array_equal(result[:, 1], [3.0, 2.0, 4.0])
            
            # Scores should match input
            np.testing.assert_array_equal(result[:, 5], scores)
            
            # TP distance errors should be 0
            np.testing.assert_array_equal(result[:, 3], [0.0, 0.0, 0.0])
            
            # Check NW costs
            expected_nw_cost = thr / SOSPA_SCALING
            if self.config.norm == constants.MetricsNormType.SumInstance:
                expected_nw_cost = 1.0 
            np.testing.assert_array_almost_equal(
                result[:, 2], 
                [expected_nw_cost, expected_nw_cost, expected_nw_cost]
            )
            
            # Check scaling factors
            expected_scalings = np.array([3.0, 2.0, 4.0])
            if self.config.norm == constants.MetricsNormType.SumInstance:
                expected_scalings *= thr / SOSPA_SCALING
            np.testing.assert_array_almost_equal(result[:, 4], expected_scalings)


class TestNormalizationTypesNoGroundTruths(unittest.TestCase):
    """Test different normalization types when there are no ground truths."""
    
    def test_no_ground_truths_sum_instance(self):
        """Test no ground truths with SumInstance normalization."""
        pred_lines = [
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),  # 4 points
            np.array([[0, 0], [1, 1], [2, 2]])           # 3 points
        ]
        gt_lines = []
        scores = np.array([0.9, 0.8])
        threshold = 1.0
        
        result = instance_match_pld(pred_lines, scores, gt_lines, [threshold], config=MetricsConfig(norm=constants.MetricsNormType.SumInstance))
        output = result[threshold]
        
        # FP should equal number of points in each prediction
        self.assertEqual(output[0, 1], 4.0)
        self.assertEqual(output[1, 1], 3.0)
        
        # For SumInstance: nw_cost = 2.0 * threshold / NW_SCALING = 2.0 * 1.0 / 2.0 = 1.0
        # scaling = fp / 2.0
        np.testing.assert_array_almost_equal(output[:, 2], [1.0, 1.0])
        np.testing.assert_array_almost_equal(output[:, 4], [2.0, 1.5])
    
    def test_no_ground_truths_max_cost_based(self):
        """Test no ground truths with MaxCostBased normalization."""
        pred_lines = [
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),  # 4 points
            np.array([[0, 0], [1, 1], [2, 2]])           # 3 points
        ]
        gt_lines = []
        scores = np.array([0.9, 0.8])
        threshold = 1.0
        
        result = instance_match_pld(pred_lines, scores, gt_lines, [threshold])
        output = result[threshold]
        
        # FP should equal number of points in each prediction
        self.assertEqual(output[0, 1], 4.0)
        self.assertEqual(output[1, 1], 3.0)
        
        # For MaxCostBased: nw_cost = 1.0, scaling = threshold * nFP / 2 = 1.0 * 4 / 2 = 2.0 and 1.0 * 3 / 2 = 1.5
        np.testing.assert_array_almost_equal(output[:, 2], [1.0, 1.0])
        np.testing.assert_array_almost_equal(output[:, 4], [2.0, 1.5])
    
    def test_no_ground_truths_no_norm(self):
        """Test no ground truths with NoNorm normalization."""
        pred_lines = [
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),  # 4 points
            np.array([[0, 0], [1, 1], [2, 2]])           # 3 points
        ]
        gt_lines = []
        scores = np.array([0.9, 0.8])
        threshold = 1.0
        
        result = instance_match_pld(pred_lines, scores, gt_lines, [threshold], config=MetricsConfig(norm=constants.MetricsNormType.NoNorm))
        output = result[threshold]
        
        # FP should equal number of points in each prediction
        self.assertEqual(output[0, 1], 4.0)
        self.assertEqual(output[1, 1], 3.0)
        
        # For NoNorm: nw_cost = threshold / NW_SCALING * fp, scaling = 1.0
        # First: 1.0 / 2.0 * 4 = 2.0
        # Second: 1.0 / 2.0 * 3 = 1.5
        np.testing.assert_array_almost_equal(output[:, 2], [2.0, 1.5])
        np.testing.assert_array_almost_equal(output[:, 4], [1.0, 1.0])


class TestInstanceMatchingDecomposed(unittest.TestCase):

    def setUp(self):
        self.config = MetricsConfig(norm=constants.MetricsNormType.SumInstance)

    def test_with_identical_gt_to_pred(self):
        # Create identical prediction and ground truth lines
        pred_lines = [
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),  # 3 points
            np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])  # 4 points
        ]
        gt_lines = copy.deepcopy(pred_lines)
        
        scores = np.array([1.0, 1.0])  # High confidence scores
        thresholds = [1.0]
        
        result = instance_matching_module.instance_match_pld(
            pred_lines, scores, gt_lines, thresholds,
            config=self.config
        )

        self.assertIn(thresholds[0], result)
        output = result[thresholds[0]]
        self.assertEqual(output.shape, (2, 6))  # 2 predictions, 6 columns (tp, fp, nw_cost, tp_d_err, scaling, scores)
        
        # All predictions should be true positives
        tp = output[:, 0]
        fp = output[:, 1]
        
        # Check that we have some true positives and minimal false positives
        self.assertEqual(tp[0], len(gt_lines[0]), "First prediction should have some true positives")
        self.assertEqual(tp[1], len(gt_lines[1]), "Second prediction should have some true positives")
        
        self.assertEqual(fp[0], 0, "First prediction should have zero false positives")
        self.assertEqual(fp[1], 0, "Second prediction should have zero false positives")
        
        # NW cost should be very low (close to 0) for identical lines
        nw_cost = output[:, 2]
        self.assertLess(nw_cost[0], 0.1, "NW cost should be very low for identical lines")
        self.assertLess(nw_cost[1], 0.1, "NW cost should be very low for identical lines")
        
        # With identical lines, distance error should be very small
        tp_d_err = output[:, 3]
        self.assertLess(tp_d_err[0], 0.1, "Distance error should be very small for identical lines")
        self.assertLess(tp_d_err[1], 0.1, "Distance error should be very small for identical lines")
        
        scaling = output[:, 4]

        self.assertEqual(scaling[0], len(gt_lines[0]), "Scaling factor should be 1.0 for both normalizations")
        self.assertEqual(scaling[1], len(gt_lines[1]), "Scaling factor should be 1.0 for both normalizations")

        # Scores should match input
        scores_output = output[:, 5]
        np.testing.assert_array_equal(scores_output, scores)
        
class TestInstanceMatching(unittest.TestCase):
    
    def test_basic_matching(self):
        matrix = np.array([[0.1, 0.5], [0.3, 0.2]])
        scores = np.array([0.9, 0.7])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[4, 1], [2, 5]])
        nfp = np.array([[1, 4], [3, 1]])
        dist_errors = np.array([[0.05, 0.25], [0.15, 0.1]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        self.assertIn(threshold, result)
        output = result[threshold]
        self.assertEqual(output.shape, (2, 6))

    def test_gt_already_covered(self):
        
        # First GT is best match for both predictions
        matrix = np.array([[0.1, 0.5], [0.1, 0.2]])
        scores = np.array([0.9, 0.7])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[4, 1], [3, 2]])
        nfp = np.array([[1, 4], [2, 3]])
        dist_errors = np.array([[0.05, 0.25], [0.06, 0.26]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )

        tp = result[threshold][:, 0]

        self.assertEqual(tp[0], 4)
        self.assertEqual(tp[1], 0)


        # First GT is best match for first pred, second GT for second prediction
        matrix = np.array([[0.1, 0.5], [0.5, 0.2]])
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )

        tp = result[threshold][:, 0]
        self.assertEqual(tp[0], 4)
        self.assertEqual(tp[1], 2)

    def test_no_valid_matches(self):
        matrix = np.array([[0.1, 0.5], [0.3, 0.2]])
        scores = np.array([0.9, 0.7])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[0, 0], [0, 0]])
        nfp = np.array([[5, 4], [3, 6]])
        dist_errors = np.array([[0.05, 0.25], [0.15, 0.1]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        output = result[threshold]
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[1, 0], 0)

    def test_empty_predictions(self):
        matrix = np.array([]).reshape(0, 2)
        scores = np.array([])
        threshold = 0.4
        num_preds = 0
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([]).reshape(0, 2)
        nfp = np.array([]).reshape(0, 2)
        dist_errors = np.array([]).reshape(0, 2)
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        output = result[threshold]
        self.assertEqual(output.shape, (0, 6))
    

class TestNormalizationTypes(unittest.TestCase):
    """Test _match_predictions_with_thresholds_extended with different normalization types."""
    
    def test_with_sum_instance_normalization(self):
        """Test with SumInstance normalization type."""
        matrix = np.array([[0.2, 0.6], [0.3, 0.25]])
        scores = np.array([0.95, 0.85])
        threshold = 0.5
        num_preds = 2
        num_gts_points = np.array([15.0, 20.0])
        ntp = np.array([[12, 5], [8, 15]])
        nfp = np.array([[3, 8], [5, 5]])
        dist_errors = np.array([[0.1, 0.3], [0.15, 0.12]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors,
            config=MetricsConfig(norm=constants.MetricsNormType.SumInstance)
        )
        
        output = result[threshold]
        
        # First prediction matches first GT: tp=12, fp=3, gt=15
        # scaling = threshold * (12+3+15)/2 = 15.0*0.5 = 7.5
        self.assertAlmostEqual(output[0, 4], 7.5, places=5)  # 15.0 * 0.5
        
        # Second prediction matches second GT: tp=15, fp=5, gt=20
        # scaling = threshold * (15+5+20)/2 = 20.0*0.5 = 10.0
        self.assertAlmostEqual(output[1, 4], 10.0, places=5)
    
    def test_with_max_cost_based_normalization(self):
        """Test with MaxCostBased normalization type."""
        matrix = np.array([[0.2, 0.6], [0.3, 0.25]])
        scores = np.array([0.95, 0.85])
        threshold = 0.5
        num_preds = 2
        num_gts_points = np.array([15.0, 20.0])
        ntp = np.array([[12, 5], [8, 15]])
        nfp = np.array([[3, 8], [5, 5]])
        dist_errors = np.array([[0.1, 0.3], [0.15, 0.12]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        output = result[threshold]
        
        # For MaxCostBased:
        # scaled_gap = threshold / NW_SCALING = 0.5 / 2.0 = 0.25
        # First prediction: n_pred=15, n_gt=15, norm_cost=0.2
        # G = 0.25 * (15 + 15) = 7.5
        # unorm_cost = (norm_cost * G) / (2.0 - norm_cost) = (0.2 * 7.5) / (2.0 - 0.2) = 1.5 / 1.8 = 0.8333...
        # scaling = (G + unorm_cost) / 2 = (7.5 + 0.8333...) / 2 = 4.166...
        self.assertAlmostEqual(output[0, 4], 4.166666666666667, places=5)
        
        # Second prediction: n_pred=20, n_gt=20, norm_cost=0.25
        # G = 0.25 * (20 + 20) = 10.0
        # unorm_cost = (0.25 * 10.0) / (2.0 - 0.25) = 2.5 / 1.75 = 1.4285...
        # scaling = (10.0 + 1.4285...) / 2 = 5.7142...
        self.assertAlmostEqual(output[1, 4], 5.714285714285714, places=5)
    
    def test_with_no_normalization(self):
        """Test with NoNorm normalization type."""
        matrix = np.array([[0.2, 0.6], [0.3, 0.25]])
        scores = np.array([0.95, 0.85])
        threshold = 0.5
        num_preds = 2
        num_gts_points = np.array([15.0, 20.0])
        ntp = np.array([[12, 5], [8, 15]])
        nfp = np.array([[3, 8], [5, 5]])
        dist_errors = np.array([[0.1, 0.3], [0.15, 0.12]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors,
            config=MetricsConfig(norm=constants.MetricsNormType.NoNorm)
        )
        
        output = result[threshold]
        
        # For NoNorm, scaling should always be 1.0
        self.assertAlmostEqual(output[0, 4], 1.0, places=5)
        self.assertAlmostEqual(output[1, 4], 1.0, places=5)
    
    def test_unmatched_prediction_no_norm(self):
        """Test unmatched prediction with NoNorm."""
        matrix = np.array([[0.8, 0.9]])
        scores = np.array([0.9])
        threshold = 0.5
        num_preds = 1
        num_gts_points = np.array([10.0])
        ntp = np.array([[0]])
        nfp = np.array([[5]])
        dist_errors = np.array([[0.0]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors,
            config=MetricsConfig(norm=constants.MetricsNormType.NoNorm)
        )
        
        output = result[threshold]
        
        # For unmatched with NoNorm:
        # fp = 0 + 5 = 5
        # nw_cost = threshold / NW_SCALING * fp = 0.5 / 2.0 * 5 = 1.25
        # scaling = 1.0
        self.assertEqual(output[0, 0], 0)  # TP
        self.assertEqual(output[0, 1], 5)  # FP
        self.assertAlmostEqual(output[0, 2], 1.25, places=5)  # nw_cost
        self.assertAlmostEqual(output[0, 4], 1.0, places=5)  # scaling
    
    def test_unmatched_prediction_max_cost_based(self):
        """Test unmatched prediction with MaxCostBased."""
        matrix = np.array([[0.8, 0.9]])
        scores = np.array([0.9])
        threshold = 0.5
        num_preds = 1
        num_gts_points = np.array([10.0])
        ntp = np.array([[0]])
        nfp = np.array([[5]])
        dist_errors = np.array([[0.0]])
        
        result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        output = result[threshold]
        
        # For unmatched with MaxCostBased:
        # fp = 0 + 5 = 5
        # nw_cost = 1.0
        # scaling = threshold * nFP /2
        self.assertEqual(output[0, 0], 0)  # TP
        self.assertEqual(output[0, 1], 5)  # FP
        self.assertAlmostEqual(output[0, 2], 1.0, places=5)  # nw_cost
        self.assertAlmostEqual(output[0, 4], 0.5 * 5 / 2, places=5)  # scaling
    
    def test_compare_normalization_types(self):
        """Compare all normalization types side by side."""
        matrix = np.array([[0.2, 0.6], [0.3, 0.25]])
        scores = np.array([0.95, 0.85])
        threshold = 0.5
        num_preds = 2
        num_gts_points = np.array([15.0, 20.0])
        ntp = np.array([[12, 5], [8, 15]])
        nfp = np.array([[3, 8], [5, 5]])
        dist_errors = np.array([[0.1, 0.3], [0.15, 0.12]])
        
        results = {}
        norm_types = [
            constants.MetricsNormType.SumInstance,
            constants.MetricsNormType.MaxCostBased,
            constants.MetricsNormType.NoNorm
        ]
        
        for norm_type in norm_types:
            result = _match_predictions_with_thresholds_extended(
                matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors,
                config=MetricsConfig(norm=norm_type)
            )
            results[norm_type] = result[threshold]
        
        # TP and FP should be identical across all normalization types
        for norm_type in norm_types[1:]:
            np.testing.assert_array_equal(
                results[norm_types[0]][:, 0], results[norm_type][:, 0], 
                err_msg=f"TP differs for {norm_type}"
            )
            np.testing.assert_array_equal(
                results[norm_types[0]][:, 1], results[norm_type][:, 1],
                err_msg=f"FP differs for {norm_type}"
            )
        
        # Scaling should differ between normalization types
        sum_inst_scaling = results[constants.MetricsNormType.SumInstance][:, 4]
        max_cost_scaling = results[constants.MetricsNormType.MaxCostBased][:, 4]
        no_norm_scaling = results[constants.MetricsNormType.NoNorm][:, 4]
        
        # Verify expected scaling values
        np.testing.assert_array_almost_equal(sum_inst_scaling, [7.5, 10.0])
        np.testing.assert_array_almost_equal(max_cost_scaling, [4.166666666666667, 5.714285714285714])
        np.testing.assert_array_almost_equal(no_norm_scaling, [1.0, 1.0])


class TestNormalizationTypesRealWorld(unittest.TestCase):
    """Test normalization types with real-world data."""
    
    def test_real_world_data_with_all_normalization_types(self):
        """Test all normalization types with real-world data."""
        matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors = load_real_world_test_data_instance_matching()
        
        results = {}
        norm_types = [
            constants.MetricsNormType.SumInstance,
            constants.MetricsNormType.MaxCostBased,
            constants.MetricsNormType.NoNorm
        ]
        
        for norm_type in norm_types:
            result = _match_predictions_with_thresholds_extended(
                matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors,
                config=MetricsConfig(norm=norm_type)
            )
            results[norm_type] = result[threshold]
        
        # TP and FP should be identical across all normalization types
        for norm_type in norm_types[1:]:
            np.testing.assert_array_equal(
                results[norm_types[0]][:, 0], results[norm_type][:, 0],
                err_msg=f"TP differs for {norm_type}"
            )
            np.testing.assert_array_equal(
                results[norm_types[0]][:, 1], results[norm_type][:, 1],
                err_msg=f"FP differs for {norm_type}"
            )
        
        # Verify that at least some predictions are matched (sanity check)
        num_matched = np.sum(results[norm_types[0]][:, 0] > 0)
        self.assertGreater(num_matched, 0, "Expected at least some predictions to be matched")
        
        # Verify that NoNorm scaling is always 1.0
        no_norm_scaling = results[constants.MetricsNormType.NoNorm][:, 4]
        for i in range(num_preds):
            self.assertAlmostEqual(no_norm_scaling[i], 1.0, places=5)
        
        # Verify that MaxCostBased produces different values than SumInstance
        sum_inst_scaling = results[constants.MetricsNormType.SumInstance][:, 4]
        max_cost_scaling = results[constants.MetricsNormType.MaxCostBased][:, 4]
        differences = np.abs(sum_inst_scaling - max_cost_scaling)
        num_different = np.sum(differences > 0.001)
        self.assertGreater(num_different, 0, "Expected MaxCostBased scaling to differ from SumInstance")


class TestOptimalInstanceMatching(unittest.TestCase):

    def test_optimal_matching_basic(self):
        matrix = np.array([[0.1, 0.5], [0.3, 0.2]])
        scores = np.array([0.9, 0.7])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[4, 1], [2, 5]])
        nfp = np.array([[1, 4], [3, 1]])
        dist_errors = np.array([[0.05, 0.25], [0.15, 0.1]])
        
        result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        self.assertIn(threshold, result)
        output = result[threshold]
        self.assertEqual(output.shape, (2, 6))


    def test_gt_already_covered(self):
        
        # First GT is best match for both predictions, yet we assign both
        matrix = np.array([[0.1, 0.5], [0.1, 0.2]])
        scores = np.array([0.9, 0.7])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[4, 1], [3, 2]])
        nfp = np.array([[1, 4], [2, 3]])
        dist_errors = np.array([[0.05, 0.25], [0.06, 0.26]])

        result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )

        tp = result[threshold][:, 0]

        self.assertEqual(tp[0], 4)
        self.assertEqual(tp[1], 2)


    def test_optimal_vs_greedy_matching(self):
        matrix = np.array([[0.1, 0.5], [0.2, 0.1]])
        scores = np.array([0.7, 0.9])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[4, 1], [2, 5]])
        nfp = np.array([[1, 4], [3, 1]])
        dist_errors = np.array([[0.05, 0.25], [0.15, 0.1]])
        
        greedy_result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        optimal_result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        greedy_output = greedy_result[threshold]
        optimal_output = optimal_result[threshold]
        
        self.assertEqual(greedy_output[0, 0], 4)
        self.assertEqual(greedy_output[1, 0], 5)
        self.assertEqual(optimal_output[0, 0], 4)
        self.assertEqual(optimal_output[1, 0], 5)

    def test_optimal_no_valid_matches(self):
        matrix = np.array([[0.1, 0.5], [0.3, 0.2]])
        scores = np.array([0.9, 0.7])
        threshold = 0.4
        num_preds = 2
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([[0, 0], [0, 0]])
        nfp = np.array([[5, 4], [3, 6]])
        dist_errors = np.array([[0.05, 0.25], [0.15, 0.1]])
        
        result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        output = result[threshold]
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[1, 0], 0)

    def test_optimal_empty_predictions(self):
        matrix = np.array([]).reshape(0, 2)
        scores = np.array([])
        threshold = 0.4
        num_preds = 0
        num_gts_points = np.array([5.0, 6.0])
        ntp = np.array([]).reshape(0, 2)
        nfp = np.array([]).reshape(0, 2)
        dist_errors = np.array([]).reshape(0, 2)
        
        result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        output = result[threshold]
        self.assertEqual(output.shape, (0, 6))

    def test_real_world_optimal_vs_greedy(self):
        """Test that optimal matching performs better than or equal to greedy matching on real-world data."""
        matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors = load_real_world_test_data_instance_matching()
        
        # Run both matching algorithms
        greedy_result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        optimal_result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        greedy_output = greedy_result[threshold]
        optimal_output = optimal_result[threshold]
        
        # Extract TP, FP, and error metrics
        greedy_tp = greedy_output[:, 0].sum()
        optimal_tp = optimal_output[:, 0].sum()
        
        greedy_fp = greedy_output[:, 1].sum()
        optimal_fp = optimal_output[:, 1].sum()
        
        # Optimal should have >= TP than greedy (better matching)
        self.assertGreaterEqual(optimal_tp, greedy_tp, 
                               "Optimal matching should have more or equal true positives")
        
        # If TP is the same, optimal should have <= FP (better precision)
        if optimal_tp == greedy_tp:
            self.assertLessEqual(optimal_fp, greedy_fp,
                               "With same TP, optimal should have fewer or equal false positives")
        
        # Verify shapes match
        self.assertEqual(greedy_output.shape, optimal_output.shape)
        self.assertEqual(greedy_output.shape[0], num_preds)

    def test_real_world_optimal_vs_greedy_with_AP(self):
        """Test that optimal matching performs better than or equal to greedy matching on real-world data."""
        matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors = load_real_world_test_data_instance_matching()
        
        # Run both matching algorithms
        greedy_result = _match_predictions_with_thresholds_extended(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        optimal_result = _match_predictions_with_thresholds_optimally(
            matrix, scores, threshold, num_preds, num_gts_points, ntp, nfp, dist_errors
        )
        
        # calculate_pld expects a list of dicts (one per sample)
        nw_cost_greedy, greedy_result_dict = calculate_pld(
            [greedy_result], [threshold], num_gts_points
        )

        nw_cost_optimal, optimal_result_dict = calculate_pld(
            [optimal_result], [threshold], num_gts_points
        )

        self.assertLessEqual(nw_cost_optimal, nw_cost_greedy,
                             "Optimal matching should have less or equal PLD cost than greedy matching")


class TestInstanceMatchFrechet(unittest.TestCase):
    """Test that instance_match uses Fréchet distance when metric_type=FRECHET."""

    def test_frechet_distance_is_used(self):
        """Verify that frechet_distance_variable_size_batch is called when metric_type=FRECHET."""
        pred_lines = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[2.0, 0.0], [3.0, 0.0]])]
        gt_lines = [np.array([[0.0, 0.0], [1.0, 0.0]])]
        scores = np.array([0.9, 0.8])
        thresholds = [0.5]

        with mock.patch(
            'sospa_eval.instance_matching.frechet_distance_variable_size_batch',
            wraps=instance_matching_module.frechet_distance_variable_size_batch,
        ) as mock_frechet:
            instance_match(pred_lines, scores, gt_lines, thresholds, metric_type=MatchingMetric.FRECHET)
            mock_frechet.assert_called_once_with(pred_lines, gt_lines)

    def test_frechet_results_match_close_lines(self):
        """Check that instance_match with FRECHET returns TP for close lines."""
        pred_lines = [np.array([[0.0, 0.0], [1.0, 0.0]])]
        gt_lines = [np.array([[0.0, 0.05], [1.0, 0.05]])]
        scores = np.array([0.9])
        thresholds = [0.5]

        result = instance_match(pred_lines, scores, gt_lines, thresholds, metric_type=MatchingMetric.FRECHET)

        tp = result[0.5][:, 0]
        self.assertEqual(tp.tolist(), [1], "Close lines should be matched as TP with Fréchet distance")


if __name__ == '__main__':
    unittest.main()