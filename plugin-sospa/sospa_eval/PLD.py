from typing import NamedTuple

import numpy as np
import scipy.optimize as so

from .constants import METRICS_CONFIG, MetricsConfig, SOSPA_SCALING
from .types import MetricsNormType, normalize_cost


class PLDDetectionColumns(NamedTuple):
    """Per-threshold detection records, one array per field, sorted by
    descending confidence score.

    Replaces the positional ``(num_dets, 6)`` array whose columns had to be
    remembered by index (``tp_fp_score[:, 2]`` and friends). Each field is a
    1-D ``np.ndarray`` of length ``num_dets``, so downstream code stays fully
    vectorised with no per-detection Python overhead.
    """

    tp: np.ndarray         # matched GT points per prediction
    fp: np.ndarray         # non-matched prediction points
    base_cost: np.ndarray  # per-pair base metric cost (SOSPA / GOSPA)
    tp_d_err: np.ndarray   # true-positive distance error
    scaling: np.ndarray    # PLD scaling factor
    scores: np.ndarray     # confidence score

    @classmethod
    def from_stacked_ordered_by_score(cls, tp_fp_score: np.ndarray) -> "PLDDetectionColumns":
        """Split a stacked ``(num_dets, 6)`` array into named columns, sorted by
        descending confidence score."""
        assert (
            tp_fp_score.shape[1] == 6
        ), "tp_fp_score should have 6 columns: tp, fp, base_cost, tp_d_err, scaling, scores"

        sort_inds = np.argsort(-tp_fp_score[:, -1])

        # (6, num_dets): one row per field; NamedTuple unpacks them by position.
        return cls(*tp_fp_score[sort_inds].T)


def calculate_pld(
    tp_fp_c_err_score_list: list, thresholds: list, num_gts_points: list,
    config: MetricsConfig = None,
) -> tuple:
    """PLD cost and localization metrics over all thresholds.

    Args:
        tp_fp_c_err_score_list: One entry per scene; each entry maps threshold →
            stacked ``(num_dets, 6)`` array (columns: tp, fp, base_cost,
            tp_d_err, scaling, scores).
        thresholds: Evaluation distance thresholds.
        num_gts_points: Number of GT points per GT instance ``(num_gt,)``.
        config: Metrics configuration (defaults to module-level ``METRICS_CONFIG``).

    Returns:
        tuple: (PLD_cost, result_dict) where result_dict has keys
        ``PLD_cost@<thr>``, ``PLD_loc@<thr>`` for every threshold, plus mean
        ``PLD_cost`` and ``PLD_loc``.
    """
    if config is None:
        config = METRICS_CONFIG

    sum_pld_AP = 0
    sum_pld_loc_AP = 0
    result_dict = {}

    num_gt_instances = len(num_gts_points)

    for thr in thresholds:
        stacked = np.vstack([i[thr] for i in tp_fp_c_err_score_list])
        det = PLDDetectionColumns.from_stacked_ordered_by_score(stacked)
        tp_polyline, _ = _get_polyline_level_TP_FP(det.tp)

        pld_cost_AP, pld_loc_AP = _calculate_pld(
            det.base_cost,
            det.tp_d_err,
            det.scaling,
            det.scores,
            tp_polyline,
            num_gt_instances,
            thr,
            config.norm,
        )

        sum_pld_AP += pld_cost_AP
        sum_pld_loc_AP += pld_loc_AP
        result_dict[f"PLD_cost@{thr}"] = pld_cost_AP
        result_dict[f"PLD_loc@{thr}"] = pld_loc_AP

    result_dict["PLD_cost"] = sum_pld_AP / len(thresholds)
    result_dict["PLD_loc"] = sum_pld_loc_AP / len(thresholds)
    return result_dict["PLD_cost"], result_dict


def _calculate_pld(
    base_cost, tp_d_err, scaling, scores, tp_polylines, num_gt_lines, threshold,
    norm: MetricsNormType
):
    """Calculate normalized PLD cost and PLD localization components.
    OBS: assuming optimal assignment of polylines, this should give the same
    PLD measure on polyline level excluding order.
    Args:
        base_cost: Per-pair base metric cost array (SOSPA/GOSPA), one entry per prediction
        tp_d_err: True positive distance error array
        scaling: Scaling factor array
        scores: Confidence scores for each prediction
        tp_polylines: True positive polyline array
        num_gt_lines: Number of ground truth polylines
        threshold: Distance threshold for considering matches
        norm: Normalization type (MetricsNormType)

    Returns:
        tuple: (pld_cost_normalized, pld_loc) - normalized PLD cost and localization cost
    """
    normalized_threshold = 1.0 if norm.is_unit_bounded() else threshold
    scaled_gap = normalized_threshold / SOSPA_SCALING
    
    n_tp_poly = np.sum(tp_polylines, axis=0)
    n_fn = num_gt_lines - n_tp_poly
    assert n_fn >= 0, "Number of false negatives cannot be negative"

    conf_dependent_fn = n_tp_poly - np.sum(
        tp_polylines * scores, axis=0
    )  # = 0 if all TPs has confidence of 1
    conf_dependent_fp = np.sum((1 - tp_polylines) * scores, axis=0)

    n_misses_pld_cost = (
        scaled_gap * (n_fn + conf_dependent_fn + conf_dependent_fp)
    )
    tp_pld_cost_norm = base_cost * scores * tp_polylines

    sum_pld_cost_including_fn = np.sum(tp_pld_cost_norm) + n_misses_pld_cost

    cum_pld_loc_norm = tp_d_err * scores / scaling
    tot_pld_loc = np.sum(cum_pld_loc_norm)

    expected_pred_len = np.sum(scores)

    norm_factor = norm.get_norm_factor(
        num_gt_lines, expected_pred_len, scaled_gap, sum_pld_cost_including_fn
    )

    pld_cost_normalized = sum_pld_cost_including_fn * norm_factor
    pld_loc = tot_pld_loc * norm_factor

    return pld_cost_normalized, pld_loc


def _get_polyline_level_TP_FP(num_tp_points: list):
    tp = [1 if ntp > 0 else 0 for ntp in num_tp_points]
    fp = [1 if ntp == 0 else 0 for ntp in num_tp_points]

    assert np.sum(tp) + np.sum(fp) == len(num_tp_points)
    assert len(tp) == len(num_tp_points)

    return np.array(tp), np.array(fp)


# ---------------------------------------------------------------------------
# Shared helpers (used by both SOSPA-batch and PLD-batch pipelines)
# ---------------------------------------------------------------------------
def are_all_points_unmatched(cost_matrix, gap_penalty) -> bool:
    """True iff no pred-gt point pair lies within the gap-penalty radius."""
    return np.all(cost_matrix >= gap_penalty)


def get_decomposed_cost_all_unmatched(len_a, len_gt, gap_penalty, norm_used=None):
    """Closed-form decomposition when every prediction point is unmatched."""
    if norm_used is None:
        norm_used = METRICS_CONFIG.norm

    nTP = 0
    nFP = len_a
    nFN = len_gt
    tp_sum_dist_error = 0.0

    scaled_gap = gap_penalty / SOSPA_SCALING
    final_cost = (nFP + nFN) * scaled_gap

    total_cost = normalize_cost(norm_used, final_cost, len_a, len_gt, scaled_gap)
    return total_cost, tp_sum_dist_error, (nTP, nFP, nFN)


# ---------------------------------------------------------------------------
# Linear assignment used by P-GOSPA / PLD
# ---------------------------------------------------------------------------
def linear_assignment(cost_matrix) -> np.ndarray:
    x, y = so.linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def construct_pld_cost_matrix(
    matrix, scores, threshold, num_preds: int, num_gts: int
) -> np.ndarray:
    """Construct cost matrix for the PLD / P-GOSPA optimal assignment.

    Reference: https://github.com/yuhsuansia/Probabilistic-GOSPA/blob/main/PGOSPA.m
    """
    # Use (0 - r) since we need to subtract all gts cost from the problem
    scaled_matrix = scores[:, None] * matrix.copy() + (0 - scores[:, None]) * (
        threshold / SOSPA_SCALING
    )
    scaled_matrix = np.concatenate(
        [
            scaled_matrix,
            np.ones((num_preds, num_preds), dtype=scaled_matrix.dtype)
            * threshold
            * 1e4,
        ],
        axis=1,
    )
    for i in range(num_preds):
        scaled_matrix[i, num_gts + i] = scores[i] * threshold / SOSPA_SCALING
    return scaled_matrix
