import os
import sys
from dataclasses import dataclass

from .types import MetricsNormType


# Eval constants
THRESHOLDS_60x30 = [0.5, 1.0, 1.5]  # AP thresholds
THRESHOLDS_100x50 = [1.0, 1.5, 2.0]  # AP thresholds
# Set SAMPLE_DIST to None to run fixed points interpolation
SAMPLE_DIST = 0.5 # 0.70  # sample distance for interpolation
INTERP_NUM = None  # number of points to interpolate during evaluation

# limit evaluation 
EVAL_SAMPLES_LIMIT = None


# Fixed gap-penalty scaling factor used by the SOSPA base distance. Not
# user-tunable; use the value as-is.
SOSPA_SCALING: float = 2.0


# Metrics configuration
#
# Governs both the SOSPA (Sequence Optimal Sub-Pattern Assignment, formerly
# "Needleman-Wunsch") base distance and the PLD (Polyline Localisation and
# Detection, formerly "_pgospa") multi-instance metric. A single ``norm`` is
# shared between the two.
#
# Implemented as a dataclass so that callers (e.g. tests) can build a variant
# config and pass it explicitly instead of mocking global state.
@dataclass
class MetricsConfig:
    norm: MetricsNormType = MetricsNormType.MaxCostBased  # normalisation, shared by SOSPA and PLD
    use_optimal_assignment: bool = True                   # PLD assignment strategy


# Module-level default instance (used when callers don't pass an explicit config).
METRICS_CONFIG = MetricsConfig()


def get_num_workers():
    if 'N_WORKERS' in os.environ:
        n_workers = int(os.environ['N_WORKERS']) 
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_workers = 20
    print(f"Using {n_workers} workers for parallel processing.")
    sys.stdout.flush()
    return n_workers

N_WORKERS =  get_num_workers() # num workers to parallel



DEBUG_MODE = False  # set to True to enable debug messages