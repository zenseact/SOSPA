"""
Lightweight map evaluation script.

Evaluates pre-computed predictions against pre-converted ground-truth files
without requiring any mmdet3d / mmcv config or model infrastructure.

Usage
-----
python tools/evaluate_light.py \
    --gts    path/to/gts.pkl \
    --preds  path/to/submission.json \
    --roi-size 60x30 \
    --metric chamfer

Ground-truth format  (pickle)
------------------------------
A dict mapping scene token → per-class polyline lists:

    {
        "<token>": {
            0: [np.ndarray, ...],   # class 0 polylines
            1: [np.ndarray, ...],
            2: [np.ndarray, ...],
        },
        ...
    }

Prediction format  (json or pickle)
-------------------------------------
Submission dict with the same schema used by the training pipeline:

    {
        "meta": { ... },
        "results": {
            "<token>": {
                "vectors": [[x, y, x, y, ...], ...],
                "scores":  [0.9, 0.8, ...],
                "labels":  [0, 1, 2, ...],
            },
            ...
        }
    }

Range config / roi-size
-----------------------
Pass as WxH string, e.g. ``60x30`` or ``100x50``.
This selects the matching AP thresholds automatically.

Categories
----------
The default NuScenes categories are used unless --categories is provided as
a JSON mapping of  name → label-int, e.g.:
    '{"divider": 0, "ped_crossing": 1, "boundary": 2}'
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Put plugin-sospa/ on the path so the ``sospa_eval`` package can be imported
# straight from the repo (no pip install needed, no mmdet3d / mmcv).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PLUGIN_DIR = os.path.join(_REPO_ROOT, "plugin-sospa")
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from sospa_eval.light_evaluator import LightVectorEvaluate, _load_file
from sospa_eval.types import MatchingMetric

# ---------------------------------------------------------------------------
# Default categories (NuScenes 3-class)
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES: Dict[str, int] = {
    "divider": 0,
    "ped_crossing": 1,
    "boundary": 2,
}

# ---------------------------------------------------------------------------
# JSON output helper
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars/arrays as plain Python types for JSON serialisation."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_results_json(
    output_dir: str,
    roi_size: Tuple[int, int],
    metric: str,
    gts_path: str,
    preds_path: str,
    categories: Dict[str, int],
    result_dict: Dict,
    num_scenes: int,
) -> str:
    """Serialise evaluation results to a JSON file and return its path.

    Filename format: ``{W}x{H}_{metric}_{YYYYMMDD_HHMMSS}.json``
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    roi_str = f"{roi_size[0]}x{roi_size[1]}"
    mScore_key = MatchingMetric.to_metric_type(metric).to_mean_aggregate_score_key()
    filename = f"{roi_str}_{metric}_{timestamp}.json"
    out_path = os.path.join(output_dir, filename)

    payload = {
        "gts_path": os.path.abspath(gts_path),
        "preds_path": os.path.abspath(preds_path),
        "roi_size": roi_str,
        "metric": metric,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_scenes": num_scenes,
        "categories": categories,
        mScore_key: result_dict.get(mScore_key),
        "results": {
            cat: {k: v for k, v in vals.items()}
            for cat, vals in result_dict.items()
            if cat != mScore_key
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_roi_size(s: str) -> Tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"roi-size must be WxH, e.g. '60x30'. Got: '{s}'"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight HD-map evaluation (no training framework required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gts",
        required=True,
        metavar="FILE",
        help="Ground-truth pickle file ({token: {class_id: [polylines]}}).",
    )
    parser.add_argument(
        "--preds",
        required=True,
        metavar="FILE",
        help="Prediction file (.json or .pkl) in submission format.",
    )
    parser.add_argument(
        "--roi-size",
        required=True,
        type=_parse_roi_size,
        metavar="WxH",
        help="Perception range, e.g. '60x30' or '100x50'. Determines AP thresholds.",
    )
    parser.add_argument(
        "--metric",
        default="chamfer",
        choices=["chamfer", "frechet", "sospa"],
        help="Distance metric for evaluation (default: chamfer).",
    )
    parser.add_argument(
        "--categories",
        default=None,
        metavar="JSON",
        help=(
            "Category name→label mapping as a JSON string. "
            'Default: {"divider":0,"ped_crossing":1,"boundary":2}. '
            'Example: \'{"divider":0,"ped_crossing":1,"boundary":2}\''
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        metavar="N",
        help="Number of parallel workers (default: 0 = single process).",
    )
    parser.add_argument(
        "--output-dir",
        default="test_light",
        metavar="DIR",
        help=(
            "Directory to save the JSON results file "
            "(default: test_light). Created if it does not exist. "
            "Filename: {roi}_{metric}_{YYYYMMDD_HHMMSS}.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Categories
    if args.categories:
        categories = json.loads(args.categories)
        categories = {k: int(v) for k, v in categories.items()}
    else:
        categories = deepcopy(DEFAULT_CATEGORIES)

    print(f"Loading GTs from  : {args.gts}")
    print(f"Loading preds from: {args.preds}")
    gts = _load_file(args.gts)
    print(f"  {len(gts)} scenes loaded.")

    evaluator = LightVectorEvaluate(
        gts=gts,
        roi_size=args.roi_size,
        categories=categories,
        n_workers=args.n_workers,
    )

    result_dict = evaluator.evaluate(pred_file=args.preds, metric=args.metric)

    out_path = _save_results_json(
        output_dir=args.output_dir,
        roi_size=args.roi_size,
        metric=args.metric,
        gts_path=args.gts,
        preds_path=args.preds,
        categories=categories,
        result_dict=result_dict,
        num_scenes=len(gts),
    )
    print(f"\nResults saved to  : {out_path}")


if __name__ == "__main__":
    main()
