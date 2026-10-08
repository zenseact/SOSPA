import numpy as np
from shapely.geometry import LineString, Polygon
from numpy.typing import NDArray
from typing import List
from .constants import INTERP_NUM, SAMPLE_DIST


def interpolate_polylines(
    vectors: List[NDArray], num_pts: int = INTERP_NUM, sample_dist=SAMPLE_DIST
) -> List[NDArray]:

    if sample_dist is not None:
        return [interp_fixed_dist(vector, sample_dist) for vector in vectors]
    if num_pts is not None:
        return [interp_fixed_num(vector, num_pts) for vector in vectors]
    
    return vectors


def interp_fixed_num(vector: NDArray, num_pts: int) -> NDArray:
    """Interpolate a polyline.

    Args:
        vector (array): line coordinates, shape (M, 2)
        num_pts (int):

    Returns:
        sampled_points (array): interpolated coordinates
    """
    line = LineString(vector)
    distances = np.linspace(0, line.length, num_pts)
    sampled_points = np.array(
        [list(line.interpolate(distance).coords) for distance in distances]
    ).squeeze()

    return sampled_points


def interp_fixed_dist(vector: NDArray, sample_dist: float) -> NDArray:
    """Interpolate a line at fixed interval.

    Args:
        vector (LineString): vector
        sample_dist (float): sample interval

    Returns:
        points (array): interpolated points, shape (N, 2)
    """
    line = LineString(vector)
    distances = list(np.arange(sample_dist, line.length, sample_dist))
    # make sure to sample at least two points when sample_dist > line.length
    distances = (
        [0]
        + distances
        + [
            line.length,
        ]
    )

    sampled_points = np.array(
        [list(line.interpolate(distance).coords) for distance in distances]
    ).squeeze()

    return sampled_points


def cap_polylines_to_fov(
    vectors: List[NDArray], x_range: List[float], y_range: List[float]
) -> List[NDArray]:
    """Cap polylines to field of view boundaries."""
    min_x, max_x = x_range
    min_y, max_y = y_range

    # Create FOV polygon directly
    fov_polygon = Polygon(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    )

    capped_vectors = []
    indices = []
    for i, vector in enumerate(vectors):
        # Skip vectors with less than 2 points
        if len(vector) < 2:
            continue
            
        line = LineString(vector)
        intersection = line.intersection(fov_polygon)

        if intersection.is_empty:
            continue

        # Handle both single LineString and MultiLineString
        if intersection.geom_type == "LineString":
            capped_vectors.append(np.array(intersection.coords))
            indices.append(i)
        elif intersection.geom_type == "MultiLineString":
            for line_geom in intersection.geoms:
                capped_vectors.append(np.array(line_geom.coords))
                indices.append(i)

    return capped_vectors, indices