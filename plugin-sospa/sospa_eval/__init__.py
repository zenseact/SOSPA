"""sospa_eval: framework-free polyline matching evaluation metrics."""

import importlib
import sys

from .types import MatchingMetric, MetricsNormType, Polyline
from .constants import MetricsConfig, METRICS_CONFIG
from .AP import calculate_average_precision
from .PLD import calculate_pld
from .instance_matching import instance_match, instance_match_pld
from .interpolation import interpolate_polylines
from .light_evaluator import LightVectorEvaluate

_pld_module = importlib.import_module(".PLD", __name__)
sys.modules[f"{__name__}.pld"] = _pld_module
pld = _pld_module

__all__ = [
    "MatchingMetric",
    "MetricsNormType",
    "Polyline",
    "MetricsConfig",
    "METRICS_CONFIG",
    "calculate_average_precision",
    "calculate_pld",
    "pld",
    "instance_match",
    "instance_match_pld",
    "interpolate_polylines",
    "LightVectorEvaluate",
]
