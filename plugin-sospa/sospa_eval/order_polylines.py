from typing import List, Tuple
import numpy as np
from .types import Polyline


def to_polyline_type(polyline: np.ndarray) -> Polyline:
    is_closed = _is_shape_closed(polyline)
    
    if is_closed:
        polyline = polyline[:-1]
        
    return Polyline(geometry=polyline, is_closed=is_closed)

def get_all_permutation_pairs(poly_ref: Polyline, poly: Polyline) -> List[Tuple[np.ndarray, np.ndarray]]:    
    
    if poly_ref.is_closed:
        permutations_ref = _get_all_cyclicly_shifted_polygons(poly_ref.geometry)
    else:
        permutations_ref = [poly_ref.geometry, poly_ref.geometry[::-1]]

    if poly.is_closed and not poly_ref.is_closed:
        permutations = _get_all_cyclicly_shifted_polygons(poly.geometry)
    else:
        if poly_ref.is_closed:
            permutations = [poly.geometry, poly.geometry[::-1]]
        else:
            permutations = [poly.geometry]

    perms = []
    for p1 in permutations_ref:
        for p2 in permutations:
            perms.append((p1, p2))
    return perms

# Needed when polygon's closing point is shifted for different permutations
def get_all_permutation_pairs_brute_force(poly_ref, poly):    
    if _is_shape_closed(poly_ref):
        permutations_ref = _get_all_cyclicly_shifted_polygons(poly_ref)
    else:
        permutations_ref = [poly_ref, poly_ref[::-1]]

    if _is_shape_closed(poly):
        permutations = _get_all_cyclicly_shifted_polygons(poly)
    else:
        permutations = [poly, poly[::-1]]

    perms = []
    for p1 in permutations_ref:
        for p2 in permutations:
            perms.append((p1, p2))
    return perms


# ____________Polygons helpers_________________#
def _is_shape_closed(polyline):
    """
    Determine if a polyline represents a closed shape.

    Args:
        polyline: List of points [(x1,y1), (x2,y2), ...]
        tolerance_factor: Multiplier for average segment length to determine closure threshold

    Returns:
        Boolean indicating if the shape is likely closed
    """
    if len(polyline) <= 2:
        return False

    # Calculate distance between start and end points
    start_end_distance = np.linalg.norm(np.array(polyline[-1]) - np.array(polyline[0]))

    #NOTE: It can be beneficial to make this dependent on sampling distance
    threshold_start_end_point_close = 0.1
    if start_end_distance <= threshold_start_end_point_close:
        return True

    return False


def _get_all_cyclicly_shifted_polygons(poly):
    """Generate all cyclic permutations of a closed polygon, ensuring they are in clockwise order. Do not introduce any strange operation (like removing teh closure point -> shift -> insert closure point back) that may change the structure of the polygon and cause issues with interpolation. Instead, just shift the polygon as is, and let the ordering function handle the closure if needed."""
    if _is_polygon_counter_clockwise(poly):
        poly = poly[::-1]

    n = len(poly)
    shifted_polygons = []

    for start_idx in range(n):
        current_perm = [poly[(start_idx + i) % n] for i in range(n)]

        shifted_polygons.append(np.array(current_perm))

    return shifted_polygons

def _is_polygon_counter_clockwise(polyline):
    """
    Determine if a polyline should be flipped based on its signed area.

    Args:
        polyline: List of points [(x1,y1), (x2,y2), ...]

    Returns:
        Boolean indicating if the polyline should be flipped
    """
    signed_area = 0
    for i in range(len(polyline)):
        x1, y1 = polyline[i]
        x2, y2 = polyline[(i + 1) % len(polyline)]
        signed_area += (x1 * y2) - (x2 * y1)

    signed_area = signed_area / 2

    return signed_area > 0