import numpy as np

from .constants import METRICS_CONFIG, MetricsConfig, SOSPA_SCALING
from .types import MatchingMetric, MetricsNormType, unnormalized_cost_for_max_cost_norm

from .distance import (
    chamfer_distance_variable_size_batch,
    frechet_distance_variable_size_batch,
)
from .sospa import compute_sospa_batch_decomposed
from .PLD import (
    construct_pld_cost_matrix,
    linear_assignment,
)
from typing import List, Tuple, Union


def instance_match(
    pred_lines: list,
    scores: list,
    gt_lines: list,
    thresholds: Union[Tuple, List],
    metric_type: MatchingMetric = MatchingMetric.Chamfer,
) -> Tuple[List, dict]:
    """Compute whether detected lines are true positive or false positive.

    Args:
        pred_lines (array): Detected lines of a sample, of shape (M, varSize, 2 or 3).
        scores (array): Confidence score of each line, of shape (M, ).
        gt_lines (array): GT lines of a sample, of shape (N, varSize, 2 or 3).
        thresholds (list of tuple): List of thresholds.
        metric_type (MatchingMetric): Distance function for lines matching. Default: MatchingMetric.Chamfer.

    Returns:
        list_of_tp_fp (list): tp-fp matching result at all thresholds
    """
    assert metric_type in [MatchingMetric.Chamfer, MatchingMetric.FRECHET], "Only Chamfer and FRECHET are supported in this function. For other metrics, please use instance_match_pld."
    
    num_preds, num_gts = len(pred_lines), len(gt_lines)

    if num_gts == 0 or num_preds == 0:
        return _handle_special_cases_tp_fp(thresholds, num_preds, num_gts, scores)
    
    # distance matrix: M x N
    batch_matching_func = _from_metric_type_to_matching_function(metric_type)
    matrix = batch_matching_func(pred_lines, gt_lines)

    return _match_predictions_with_thresholds(
        matrix, scores, thresholds, num_preds, num_gts
    )


def _match_predictions_with_thresholds(
    matrix, scores, thresholds, num_preds, num_gts
) -> dict:
    """Match predictions with ground truths under different thresholds.
    Limitations:
      1. Only best match for each prediction is considered. If the best match has lready been assigned, then the prediction is considered a FP.
    """
    tp_fp_score_by_thr = {thr: [] for thr in thresholds}
    matrix_min = matrix.min(axis=1)
    matrix_argmin = matrix.argmin(axis=1)
    sort_inds = np.argsort(-scores)

    for thr in thresholds:
        tp = np.zeros((num_preds), dtype=np.float32)
        fp = np.zeros((num_preds), dtype=np.float32)
        gt_covered = np.zeros(num_gts, dtype=bool)

        for i in sort_inds:
            if matrix_min[i] <= thr:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp_fp_score_by_thr[thr] = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
    return tp_fp_score_by_thr


def _handle_special_cases_tp_fp(thresholds, num_preds, num_gts, scores) -> Tuple:

    tp_fp_score_by_thr = {thr: [] for thr in thresholds}
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)

    # if there is no gt lines in this sample, then all pred lines are false positives
    if num_gts == 0:
        fp[...] = 1
        for thr in thresholds:
            tp_fp_score_by_thr[thr] = np.hstack(
                [tp[:, None], fp[:, None], scores[:, None]]
            )
        return tp_fp_score_by_thr

    if num_preds == 0:
        for thr in thresholds:
            tp_fp_score_by_thr[thr] = np.hstack(
                [tp[:, None], fp[:, None], scores[:, None]]
            )
        return tp_fp_score_by_thr

    return tp_fp_score_by_thr


def _stack(tp, fp, base_cost, tp_d_err, scaling, scores):
    return np.hstack(
        [
            tp[:, None],
            fp[:, None],
            base_cost[:, None],
            tp_d_err[:, None],
            scaling[:, None],
            scores[:, None],
        ]
    )



def instance_match_pld(
    pred_lines: list,
    scores: list,
    gt_lines: list,
    thresholds: Union[Tuple, List],
    metric_type: MatchingMetric = MatchingMetric.SOSPA,
    config: MetricsConfig = None,
) -> dict:
    """Compute whether detected lines are true positive or false positive.

    Args:
        pred_lines (array): Detected lines of a sample, of shape (M, varSize, 2 or 3).
        scores (array): Confidence score of each line, of shape (M, ).
        gt_lines (array): GT lines of a sample, of shape (N, varSize, 2 or 3).
        thresholds (list of tuple): List of thresholds.
        metric (str): Distance function for lines matching. Default: 'chamfer'.

    Returns:
        list_of_tp_fp (list): tp-fp matching result at all thresholds
    """

    matching_fn = _from_metric_type_to_matching_function(metric_type)
    
    num_preds, num_gts = len(pred_lines), len(gt_lines)

    if config is None:
        config = METRICS_CONFIG

    if num_gts == 0 or num_preds == 0:
        return _handle_special_cases_tp_fp_pld(
            thresholds, pred_lines, num_preds, num_gts, scores, config
        )

    # distance matrix: M x N
    tp_fp_score_by_thr = {}
    for thr in thresholds:

        if metric_type == MatchingMetric.SOSPA:
            (matrix, dist_errors, ntp, nfp) = matching_fn(
                pred_lines, gt_lines, thr, config=config
            )
        else:
            (matrix, dist_errors, ntp, nfp) = matching_fn(pred_lines, gt_lines, thr)
            
        num_gt_points = np.array([len(gt) for gt in gt_lines], dtype=np.float32)

        if config.use_optimal_assignment:
            result_dict = _match_predictions_with_thresholds_optimally(
                matrix, scores, thr, num_preds, num_gt_points, ntp, nfp, dist_errors,
                config
            )
        else:
            result_dict = _match_predictions_with_thresholds_extended(
                matrix, scores, thr, num_preds, num_gt_points, ntp, nfp, dist_errors,
                config
            )
        
        tp_fp_score_by_thr.update(result_dict)

    return tp_fp_score_by_thr

def _from_metric_type_to_matching_function(metric_type: MatchingMetric):
    if metric_type == MatchingMetric.Chamfer:
        return chamfer_distance_variable_size_batch
    elif metric_type == MatchingMetric.SOSPA:
        return compute_sospa_batch_decomposed
    elif metric_type == MatchingMetric.FRECHET:
        return frechet_distance_variable_size_batch
    else:
        raise ValueError(f"Unsupported metric type for point-level matching: {metric_type}")

def _match_predictions_with_thresholds_extended(
    matrix, scores, threshold, num_preds, num_gts_points: list, ntp, nfp, dist_errors,
    config: MetricsConfig = None
) -> dict:
    """Match predictions with ground truths under different thresholds.
    Limitations:
        1. Only best match for each prediction is considered. If the best match has already been assigned, then the prediction is considered a FP.
    """
    if config is None:
        config = METRICS_CONFIG

    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)
    tp_d_err = np.zeros((num_preds), dtype=np.float32)
    base_cost = np.zeros((num_preds), dtype=np.float32)
    gt_covered = np.zeros(len(num_gts_points), dtype=bool)
    scaling = np.zeros((num_preds), dtype=np.float32)
    
    matrix_argmin = matrix.argmin(axis=1)
    sort_inds = np.argsort(-scores)
    for i in sort_inds:
        matched_gt = matrix_argmin[i]
        # Adding ntp> 0 affect the final result
        if not gt_covered[matched_gt] and ntp[i, matched_gt] > 0:
            gt_covered[matched_gt] = True
            tp[i] = ntp[i, matched_gt]
            fp[i] = nfp[i, matched_gt]

            # TPs metrics
            tp_d_err[i] = dist_errors[i, matched_gt]
            base_cost[i] = matrix[i, matched_gt]

            scaling[i] = _compute_scaling_for_tp(tp[i], fp[i], num_gts_points[matched_gt], threshold, base_cost[i], config)
            
        else:
            fp[i], base_cost[i], scaling[i] = _fill_in_fp_pld_scaling_for_unmatched_polylines(
                ntp[i, 0], nfp[i, 0], threshold, config
            )
            
    tp_fp_score_by_thr = {threshold: _stack(tp, fp, base_cost, tp_d_err, scaling, scores)}
    return tp_fp_score_by_thr


def _match_predictions_with_thresholds_optimally(
    matrix, scores, threshold, num_preds, num_gts_points: list, ntp, nfp, dist_errors,
    config: MetricsConfig = None
) -> dict:
    
    if config is None:
        config = METRICS_CONFIG

    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)
    tp_d_err = np.zeros((num_preds), dtype=np.float32)
    base_cost = np.zeros((num_preds), dtype=np.float32)
    scaling = np.zeros((num_preds), dtype=np.float32)
    
    num_gts = num_gts_points.shape[0]

    normalized_threshold = 1.0 if config.norm.is_unit_bounded() else threshold
    scaled_matrix = construct_pld_cost_matrix(matrix, scores, normalized_threshold, num_preds, num_gts)
    matches = linear_assignment(scaled_matrix)
    
    matched_predictions = set()
    for i, matched_gt in matches:
        if matched_gt < num_gts and ntp[i, matched_gt] > 0:
            matched_predictions.add(i)
            tp[i] = ntp[i, matched_gt]
            fp[i] = nfp[i, matched_gt]
            # TPs metrics
            tp_d_err[i] = dist_errors[i, matched_gt]
            base_cost[i] = matrix[i, matched_gt]
            scaling[i] = _compute_scaling_for_tp(tp[i], fp[i], num_gts_points[matched_gt], threshold, base_cost[i], config)
    
    unmatched_predictions = set(range(num_preds)) - matched_predictions 
    
    for i in unmatched_predictions:
        fp[i], base_cost[i], scaling[i] = _fill_in_fp_pld_scaling_for_unmatched_polylines(
            ntp[i, 0], nfp[i, 0], threshold, config
        )
        
    tp_fp_score_by_thr = {threshold: _stack(tp, fp, base_cost, tp_d_err, scaling, scores)}
    return tp_fp_score_by_thr


def _handle_special_cases_tp_fp_pld(
    thresholds, pred_lines, num_preds, num_gts, scores, config: MetricsConfig
) -> dict:
    """Assumes Gt or Pred lines are empty."""

    tp_fp_score_by_thr = {thr: [] for thr in thresholds}
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)
    tp_d_err = np.zeros((num_preds), dtype=np.float32)
    base_cost = np.zeros((num_preds), dtype=np.float32)
    scaling = np.zeros((num_preds), dtype=np.float32)
    # if there is no gt lines in this sample, then all pred lines points are false positives
    if num_gts == 0:
        
        fp = np.array([len(pred) for pred in pred_lines], dtype=np.float32)
        
        for thr in thresholds:
            base_cost, scaling = _get_fp_vals(thr, fp, config)
            tp_fp_score_by_thr[thr] = _stack(tp, fp, base_cost, tp_d_err, scaling, scores)

        return tp_fp_score_by_thr

    if num_preds == 0:
        for thr in thresholds:
            tp_fp_score_by_thr[thr] = _stack(tp, fp, base_cost, tp_d_err, scaling, scores)

        return tp_fp_score_by_thr

    return tp_fp_score_by_thr


def _compute_scaling_for_tp(tp: int, fp:int, n_gt_points:int, threshold: float, norm_cost:float, config: MetricsConfig) -> float:
    
    norm_type = config.norm
    n_pred_points = tp + fp
    
    scaled_gap = threshold / SOSPA_SCALING
    
    max_cost = scaled_gap * (n_pred_points + n_gt_points)
    if norm_type == MetricsNormType.NoNorm:
        return 1.0

    elif norm_type == MetricsNormType.SumInstance:
        
        return max_cost
    
    elif norm_type == MetricsNormType.MaxCostBased:
        
        unorm_cost = unnormalized_cost_for_max_cost_norm(norm_cost, n_pred_points, n_gt_points, scaled_gap)
        
        return (max_cost + unorm_cost) / 2.0
    
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
    
    
def _fill_in_fp_pld_scaling_for_unmatched_polylines(n_tp, n_fp, pld_thr, config: MetricsConfig):
    """Scaling is used only for PLD localization error computation.
    I.e., it is needed only for TPs.
    """
    fp = n_tp + n_fp

    base_cost, scaling = _get_fp_vals(pld_thr, fp, config)

    return fp, base_cost, scaling


def _get_fp_vals(threshold:int, n_fp : Union[np.ndarray, int], config: MetricsConfig):
    
    norm_type = config.norm
    is_array = hasattr(n_fp, "__len__")
    
    if is_array:
        vector_handler = np.ones_like(n_fp, dtype=np.float32)
    else:
        vector_handler = 1.0
    
    scaled_gap = threshold / SOSPA_SCALING
    max_cost_vector = scaled_gap * (n_fp + 0) * vector_handler
        
    if norm_type == MetricsNormType.NoNorm:
        base_cost = max_cost_vector
        scaling = vector_handler
        
    elif norm_type == MetricsNormType.SumInstance:
        base_cost = 1.0 * vector_handler
        scaling = max_cost_vector
        
    elif norm_type == MetricsNormType.MaxCostBased:
        base_cost = 1.0 * vector_handler
        scaling = max_cost_vector
        
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
    
    return base_cost, scaling