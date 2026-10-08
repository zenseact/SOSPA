from dataclasses import dataclass

from enum import IntEnum
import numpy as np


class MetricsNormType(IntEnum):
    """Enum for different types of normalization methods used in metrics."""

    NoNorm = 0  # No normalization applied.
    SumInstance = 2  # Normalization based on the sum of two instances, bounded to [0, 1].
    MaxCostBased = 3  # Normalization based on cost, calculated as 2 * c / (max_c + c).


    def is_unit_bounded(self) -> bool:
        """Whether normalized costs are bounded to [0, 1] (1.0 == fully unmatched)."""
        return self in (MetricsNormType.SumInstance, MetricsNormType.MaxCostBased)

    def get_norm_factor(self, len_x: float, len_y: float, scaled_gap_penalty: float, cost: float) -> float:
        """Get the normalization factor based on the selected normalization type."""
        if self == MetricsNormType.SumInstance:
            # Divide by the maximum possible cost (all points unmatched) so the
            # normalized cost is bounded to [0, 1].
            divider = scaled_gap_penalty * (len_x + len_y)
        elif self == MetricsNormType.MaxCostBased:
            divider = (scaled_gap_penalty * (len_x + len_y) + cost) / 2.0
        else:  # NoNorm
            divider = 1.0

        divider = np.maximum(divider, float(np.finfo(np.float32).eps))
        return 1.0 / divider

def normalize_cost(norm_type_int: int, cost: float, len_x: float, len_y: float, scaled_gap_penalty: float) -> float:
    """
    Numba-compatible version of normalize_cost.
    Only for internal use within numba-compiled functions.
    """
    
    norm_factor = MetricsNormType(norm_type_int).get_norm_factor(len_x, len_y, scaled_gap_penalty, cost)
    cost = cost * norm_factor
    return cost


def unnormalized_cost_for_max_cost_norm(norm_cost, len_x: float, len_y: float, scaled_gap_penalty: float) -> float:
    """Get the unnormalized cost for MaxCostBased normalization."""
    G = scaled_gap_penalty * (len_x + len_y)
    return (norm_cost * G) / (2.0 - norm_cost)

class MatchingMetric(IntEnum):
    """Enum for different evaluation metric types."""

    Chamfer = 0
    SOSPA = 1     # Sequence Optimal Sub-Pattern Assignment (order-aware per-pair instance cost); computes PLD only
    FRECHET = 5

    @staticmethod
    def to_metric_type(metric_str: str):
        """Convert string to MatchingMetric enum."""
        metric_map = {
            "chamfer": MatchingMetric.Chamfer,
            "sospa": MatchingMetric.SOSPA,
            "frechet": MatchingMetric.FRECHET,
        }
        try:
            return metric_map[metric_str]
        except KeyError:
            raise ValueError(f"Unknown metric type: {metric_str}")
        
    def to_mean_aggregate_score_key(self) -> str:
        """Get the corresponding instance-level metric key for this metric type."""
        if self == MatchingMetric.Chamfer:
            return "mAP"
        elif self == MatchingMetric.SOSPA:
            return "mPLD"
        elif self == MatchingMetric.FRECHET:
            return "mAP"
        else:
            raise ValueError(f"Unknown metric type: {self}")


@dataclass
class Polyline:
    geometry: np.ndarray
    is_closed: bool