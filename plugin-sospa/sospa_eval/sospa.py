"""
SOSPA (Sequence Optimal Sub-Pattern Assignment) single-instance distance.

SOSPA solves the minimum-cost edit-trace problem between two ordered point
sequences. This module provides numba-accelerated DP kernels (open, cyclic,
bidirectional) and a single-instance entry point operating on ``Polyline``
objects. It also exposes a batched variant used by the PLD pipeline.

Historically the algorithm was referred to as "Needleman-Wunsch" / OGOSPA;
the paper labels it SOSPA, which is what we use throughout.
"""

import numpy as np
import numba
from scipy.spatial import distance

from .types import MetricsNormType, Polyline
from .constants import METRICS_CONFIG, SOSPA_SCALING
from .order_polylines import to_polyline_type

# Snapshot at import time for numba default-argument baking. Runtime reads of
# the config (in non-numba code) go through ``METRICS_CONFIG`` directly so
# tests can override via ``mock.patch.object``.
_NORM_DEFAULT = int(METRICS_CONFIG.norm)

@numba.jit(nopython=True)
def _normalize_cost_numba(norm_type_int: int, cost: float, len_x: float, len_y: float, scaled_gap_penalty: float) -> float:
    """
    Numba-compatible version of normalize_cost.
    Only for internal use within numba-compiled functions.
    """
    if norm_type_int == MetricsNormType.NoNorm:  # NoNorm
        return cost
    
    if norm_type_int == MetricsNormType.SumInstance:  # SumInstance
        # Divide by the maximum possible cost (all points unmatched) so the
        # normalized cost is bounded to [0, 1].
        norm_factor = scaled_gap_penalty * (len_x + len_y)
    elif norm_type_int == MetricsNormType.MaxCostBased:  # MaxCostBased
        norm_factor = (scaled_gap_penalty * (len_x + len_y) + cost) / 2.0
    else:
        norm_factor = 1.0
    
    if norm_factor > 0.0:
        return cost / norm_factor
    
    return cost


@numba.jit(nopython=True)
def _sospa_cost_numba(C, m, n, scaled_gap):
    """SOSPA alignment cost (no traceback) between two ordered point sequences."""
    """
    Args:
        C: pairwise distance matrix (m, n)
        m, n: lengths of the two sequences
        scaled_gap: gap penalty after dividing by SOSPA scaling factor
    Returns:
        float: SOSPA cost
    """

    # Compute DP table for this rotation (fast path without traceback)
    F = np.zeros((m + 1, n + 1), dtype=np.float64)
    
    for i in range(1, m + 1):
        F[i, 0] = F[i - 1, 0] + scaled_gap
    for j in range(1, n + 1):
        F[0, j] = F[0, j - 1] + scaled_gap
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_cost = F[i - 1, j - 1] + C[i - 1, j - 1] 
            gap_b = F[i - 1, j] + scaled_gap
            gap_a = F[i, j - 1] + scaled_gap
            F[i, j] = min(match_cost, gap_b, gap_a)
    
    return F[m, n]


@numba.jit(nopython=True)
def compute_sospa_decomposed_numba(
    C: np.ndarray,
    gap_penalty: float = 2.0,
    norm_used: int = _NORM_DEFAULT,
) -> tuple:
    """
    SOSPA alignment of two 2D polylines with full traceback and cost decomposition.

    Args:
        len_a: Length of the first polyline
        len_b: Length of the second polyline
        C: Cost matrix (len_a, len_b) containing pairwise distances between points of the two polylines
        gap_penalty: Cost for introducing a gap
        normalize: Whether to normalize costs by polyline lengths

    Returns:
        tuple: (total_cost, gaps_cost, matches_cost)
    """
    len_a, len_b = C.shape
    
    # Initialize DP matrices
    F = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float32)
    traceback = np.zeros((len_a + 1, len_b + 1), dtype=np.int32)

    scaled_gap_penalty = gap_penalty / SOSPA_SCALING
    dist_f = 1.0
    # Initialize first row and column with gap penalties
    F[0, 0] = 0.0
    for i in range(1, len_a + 1):
        F[i, 0] = F[i - 1, 0] + scaled_gap_penalty
        traceback[i, 0] = 1  # Up direction (gap in B)

    for j in range(1, len_b + 1):
        F[0, j] = F[0, j - 1] + scaled_gap_penalty
        traceback[0, j] = 2  # Left direction (gap in A)

    # Fill the DP matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            match_dist = dist_f * C[i - 1, j - 1]

            # Cost options: match, gap in B, gap in A
            match_cost = F[i - 1, j - 1] + match_dist
            delete_cost = F[i - 1, j] + scaled_gap_penalty
            insert_cost = F[i, j - 1] + scaled_gap_penalty

            costs = np.array([match_cost, delete_cost, insert_cost], dtype=np.float32)
            min_idx = np.argmin(costs)
            F[i, j] = costs[min_idx]
            traceback[i, j] = np.int32(min_idx)

    # Traceback to compute gap and match costs
    i, j = len_a, len_b
    tp_sum_dist_error = 0.0
    nTP = 0
    nFP = 0
    nFN = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i, j] == 0:  # Match/mismatch
            tp_sum_dist_error += C[i - 1, j - 1]
            nTP += 1
            i -= 1
            j -= 1
        elif i > 0 and traceback[i, j] == 1:  # Gap in B (skip in A)
            nFP += 1
            i -= 1
        else:  # Gap in A (skip in B)
            nFN += 1
            j -= 1

    total_cost = F[len_a, len_b]

    total_cost = _normalize_cost_numba(
        norm_used,
        total_cost,
        len_a,
        len_b,
        scaled_gap_penalty,
    )

    return total_cost, tp_sum_dist_error, (nTP, nFP, nFN)


@numba.jit(nopython=True)
def compute_cyclic_sospa_decomposed_numba(
    polyline_a: np.ndarray,
    polyline_b: np.ndarray,
    C: np.ndarray,
    gap_penalty: float = 2.0,
    norm_used: int = _NORM_DEFAULT,
) -> tuple:
    """
    Compute cyclic SOSPA distance using the extended edit-graph (Maes) approach.
    
    This implementation tries all rotations of polyline_a and finds the one
    with minimum alignment cost. For the best rotation, it performs full
    traceback to compute accurate TP/FP/FN metrics.
    
    Time complexity: O(m^2 n) - try all m rotations, each costs O(mn)
    Space complexity: O(mn) for the DP table
    
    Args:
        polyline_a: First polyline (m, 2 or 3)
        polyline_b: Second polyline (n, 2 or 3)
        C: Pairwise distance matrix between polyline_a and polyline_b (m, n)
        gap_penalty: Cost for introducing a gap
        norm_used: Normalization type (0=SumBoth, 1=SumInstance, 2=SumGT)
        
    Returns:
        tuple: (total_cost, dist_error, (nTP, nFP, nFN), best_rotation)
            - total_cost: Normalized alignment cost
            - dist_error: Sum of matched point distances
            - nTP: Number of matched points
            - nFP: Number of unmatched points in polyline_a
            - nFN: Number of unmatched points in polyline_b
            - best_rotation: Optimal rotation index
    """
    m = len(polyline_a)
    n = len(polyline_b)
    
    if m == 0 or n == 0:
        scaled_gap = gap_penalty / SOSPA_SCALING
        cost = (m + n) * scaled_gap
        normalized_cost = _normalize_cost_numba(norm_used, cost, m, n, scaled_gap)
        return (normalized_cost, 0.0, (0, m, n), 0)
    
    scaled_gap = gap_penalty / SOSPA_SCALING
    
    # Create extended cost matrix by concatenating C with itself
    C_extended = np.vstack((C, C))  # Shape: (2*m, n)
    
    # Find best rotation by trying all rotations
    best_cost = np.inf
    best_rotation = 0
    is_b_reversed = False
    for rotation in range(m):
        # Extract cost matrix for this rotation
        C_rot = C_extended[rotation:rotation + m, :]
        
        cost = _sospa_cost_numba(C_rot, m, n, scaled_gap)
        
        # Check flipped reversed polyline b order for asymmetry (only if not using optimal assignment)
        cost_reversed = _sospa_cost_numba(C_rot[::-1, :], m, n, scaled_gap)
        
        if cost < best_cost:
            best_cost = cost
            best_rotation = rotation
            is_b_reversed = False
        if cost_reversed < best_cost:
            best_cost = cost_reversed
            best_rotation = rotation
            is_b_reversed = True
    
    # Now compute full traceback for best rotation to get accurate metrics
    C_best = C_extended[best_rotation:best_rotation + m, :]
    if is_b_reversed:
        C_best = C_best[::-1, :]
        
    normalized_cost, dist_error, ntpfpfn = compute_sospa_decomposed_numba(
        C_best, gap_penalty=gap_penalty, norm_used=norm_used
    )
    
    return (normalized_cost, dist_error, ntpfpfn, best_rotation)


def _run_bidirectional_sospa_decomposed(
    C: np.ndarray,
    gap_penalty: float,
    norm_used: int = _NORM_DEFAULT,
) -> tuple:
    """
    Run SOSPA in both forward and reversed direction; return the lower-cost result.

    Args:
        C: Cost matrix (len_a, len_b) containing pairwise distances between points
        gap_penalty: Cost for introducing a gap
        norm_used: Normalization type

    Returns:
        tuple: (total_cost, dist_error, (nTP, nFP, nFN))
    """
    tmp_cost, tmp_dist_err, tmp_npfp = compute_sospa_decomposed_numba(
        C, gap_penalty, norm_used=norm_used
    )
    reversed_tmp_cost, reversed_tmp_dist_err, reversed_tmp_npfp = compute_sospa_decomposed_numba(
        C[::-1, :], gap_penalty, norm_used=norm_used
    )
    if tmp_cost <= reversed_tmp_cost:
        return tmp_cost, tmp_dist_err, tmp_npfp
    return reversed_tmp_cost, reversed_tmp_dist_err, reversed_tmp_npfp


def compute_sospa_single_instance(
    pred_line: Polyline,
    gt_line: Polyline,
    gap_penalty,
    norm_used=None,
):
    """SOSPA distance between a single (pred, gt) ``Polyline`` pair.

    For closed polylines, the cyclic variant is used; for open polylines, both
    directions are tried and the minimum-cost direction is returned.
    """
    norm_used = int(METRICS_CONFIG.norm) if norm_used is None else norm_used

    pred_geometry = pred_line.geometry
    gt_geometry = gt_line.geometry

    C = distance.cdist(pred_geometry, gt_geometry, "euclidean")

    if pred_line.is_closed or gt_line.is_closed:
        cost, dist_err, ntpfp, _ = compute_cyclic_sospa_decomposed_numba(
            pred_geometry, gt_geometry, C, gap_penalty=gap_penalty, norm_used=norm_used
        )
        return cost, dist_err, ntpfp

    return _run_bidirectional_sospa_decomposed(C, gap_penalty, norm_used=norm_used)


# ---------------------------------------------------------------------------
# Batched SOSPA over (pred_lines, gt_lines)
# ---------------------------------------------------------------------------
def compute_sospa_batch_decomposed(pred_lines, gt_lines, gap_penalty=None, config=None):
    """Compute SOSPA distance for every (pred_line, gt_line) pair.

    Args:
        pred_lines: iterable of predicted polylines (raw points or ``Polyline``)
        gt_lines:   iterable of ground-truth polylines
        gap_penalty: gap penalty for unmatched points.
        config: Metrics configuration providing the normalization type.
            Defaults to the module-level ``METRICS_CONFIG``.

    Returns:
        (result_matrix, dist_error_matrix, nTP, nFP), each shape (m, n).
    """
    # Import here to avoid a circular import with the PLD helpers.
    from .PLD import (
        are_all_points_unmatched,
        get_decomposed_cost_all_unmatched,
    )

    if config is None:
        config = METRICS_CONFIG

    norm_used = int(config.norm)

    m, n = len(pred_lines), len(gt_lines)
    result_matrix = np.zeros((m, n), dtype=np.float32)
    nTP = np.zeros((m, n), dtype=np.float32)
    nFP = np.zeros((m, n), dtype=np.float32)
    dist_error_matrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            pred_poly = to_polyline_type(pred_lines[i])
            gt_poly = to_polyline_type(gt_lines[j])

            C = distance.cdist(pred_poly.geometry, gt_poly.geometry, "euclidean")

            if are_all_points_unmatched(C, gap_penalty):
                cost, derr, ntpfp = get_decomposed_cost_all_unmatched(
                    len(pred_poly.geometry), len(gt_poly.geometry), gap_penalty
                )
            else:
                cost, derr, ntpfp = compute_sospa_single_instance(
                    pred_poly, gt_poly, gap_penalty,
                    norm_used=norm_used,
                )

            result_matrix[i, j] = cost
            dist_error_matrix[i, j] = derr
            nTP[i, j] = ntpfp[0]
            nFP[i, j] = ntpfp[1]

    return result_matrix, dist_error_matrix, nTP, nFP
