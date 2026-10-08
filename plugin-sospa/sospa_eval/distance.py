"""
Pairwise geometric distances between polylines.

Only base geometric measures live here: Chamfer and Fréchet (point, batched,
bidirectional + cyclic-aware variants). SOSPA lives in ``sospa.py`` and the
PLD multi-instance aggregation lives in ``pld.py``.
"""

import numpy as np
from scipy.spatial import distance

import torch
import numba

from numpy.typing import NDArray

from .order_polylines import to_polyline_type
from .types import Polyline


def chamfer_distance(line1: NDArray, line2: NDArray) -> float:
    """Chamfer distance between two interpolated polylines."""
    dist_matrix = distance.cdist(line1, line2, "euclidean")
    dist12 = dist_matrix.min(-1).sum() / len(line1)
    dist21 = dist_matrix.min(-2).sum() / len(line2)
    return (dist12 + dist21) / 2


@numba.njit(cache=True)
def _frechet_from_dist_matrix(D):
    n = D.shape[0]
    m = D.shape[1]
    M = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            d = D[i, j]
            if i == 0 and j == 0:
                M[i, j] = d
            elif i > 0 and j == 0:
                M[i, j] = max(M[i - 1, 0], d)
            elif i == 0 and j > 0:
                M[i, j] = max(M[0, j - 1], d)
            else:
                M[i, j] = max(min(M[i - 1, j], M[i - 1, j - 1], M[i, j - 1]), d)

    return M[-1, -1]


def frechet_distance(line1, line2):
    D = distance.cdist(line1, line2, "euclidean")
    return _frechet_from_dist_matrix(D)


def double_direction_open_and_closed_frechet_distance(line1: Polyline, line2: Polyline) -> float:
    """Minimum Fréchet distance accounting for bidirectionality and cyclic shifts.

    Open polylines: try forward and reversed. Closed polylines: try all
    rotations of ``line1`` in both directions. Mirrors the strategy used by
    ``compute_sospa_single_instance`` / ``compute_cyclic_sospa_decomposed_numba``.
    """
    P = line1.geometry
    Q = line2.geometry

    D = distance.cdist(P, Q, "euclidean")

    if line1.is_closed or line2.is_closed:
        n = len(P)
        best = np.inf
        for r in range(n):
            D_rot = np.roll(D, -r, axis=0)
            best = min(best, _frechet_from_dist_matrix(D_rot))
            best = min(best, _frechet_from_dist_matrix(D_rot[::-1]))
        return best

    return min(_frechet_from_dist_matrix(D), _frechet_from_dist_matrix(D[::-1]))


def frechet_distance_variable_size_batch(pred_lines, gt_lines):
    """Per-pair Fréchet distance, accepting variable-length polylines."""
    m, n = len(pred_lines), len(gt_lines)
    result_matrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            pred_poly = to_polyline_type(pred_lines[i])
            gt_poly = to_polyline_type(gt_lines[j])
            result_matrix[i, j] = double_direction_open_and_closed_frechet_distance(pred_poly, gt_poly)
    return result_matrix


def chamfer_distance_variable_size_batch(pred_lines, gt_lines):
    """Per-pair Chamfer distance, accepting variable-length polylines."""
    m, n = len(pred_lines), len(gt_lines)
    result_matrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            result_matrix[i, j] = chamfer_distance(pred_lines[i], gt_lines[j])
    return result_matrix


def chamfer_distance_batch(pred_lines, gt_lines):
    """Per-pair Chamfer distance for fixed-size, pre-interpolated polylines."""
    _, num_pts, coord_dims = pred_lines.shape

    if not isinstance(pred_lines, torch.Tensor):
        pred_lines = torch.tensor(pred_lines)
    if not isinstance(gt_lines, torch.Tensor):
        gt_lines = torch.tensor(gt_lines)
    dist_mat = torch.cdist(
        pred_lines.view(-1, coord_dims), gt_lines.view(-1, coord_dims), p=2
    )
    dist_mat = torch.stack(torch.split(dist_mat, num_pts))
    dist_mat = torch.stack(torch.split(dist_mat, num_pts, dim=-1))

    dist1 = dist_mat.min(-1)[0].sum(-1)
    dist2 = dist_mat.min(-2)[0].sum(-1)

    dist_matrix = (dist1 + dist2).transpose(0, 1) / (2 * num_pts)
    return dist_matrix.numpy()
