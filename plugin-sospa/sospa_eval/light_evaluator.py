"""
LightVectorEvaluate — standalone evaluator for vectorised HD-map predictions.

This module contains the evaluator class and all helper functions it depends on.
It can be used independently of the mmdet3d/mmcv training framework.
"""

import json
import os
import pickle
from functools import partial
from multiprocessing import Pool
from time import time
from typing import Dict, List, Tuple

import numpy as np
import prettytable

from .AP import calculate_average_precision
from .PLD import calculate_pld
from .constants import (
    EVAL_SAMPLES_LIMIT,
    INTERP_NUM,
    METRICS_CONFIG,
    N_WORKERS,
    SAMPLE_DIST,
    THRESHOLDS_100x50,
    THRESHOLDS_60x30,
)
from .instance_matching import instance_match, instance_match_pld
from .interpolation import interpolate_polylines
from .types import MatchingMetric

# ---------------------------------------------------------------------------
# Threshold presets keyed by roi_size tuple
# ---------------------------------------------------------------------------
THRESHOLD_PRESETS: Dict[Tuple[int, int], List[float]] = {
    (60, 30): THRESHOLDS_60x30,
    (100, 50): THRESHOLDS_100x50,
}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_file(path: str):
    """Load a .pkl or .json file and return its contents."""
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .pkl or .json.")


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_single(pred_vectors, scores, groundtruth, thresholds, metric_type):
    """Per-scene, per-class matching."""
    pred_lines = interpolate_polylines(pred_vectors, INTERP_NUM, SAMPLE_DIST)
    gt_lines = interpolate_polylines(groundtruth, INTERP_NUM, SAMPLE_DIST)
    scores = np.array(scores)

    if metric_type in (MatchingMetric.Chamfer, MatchingMetric.FRECHET):
        tp_fp = instance_match(pred_lines, scores, gt_lines, thresholds, metric_type)
    else:
        tp_fp = instance_match_pld(
            pred_lines, scores, gt_lines, thresholds, metric_type
        )

    num_gt = [len(gt) for gt in gt_lines]
    return tp_fp, num_gt


def _evaluate_scenes(samples, thresholds, metric_type, n_workers=0):
    """Run matching for all scenes and return (summary_scalar, per-threshold results)."""
    fn = partial(_evaluate_single, thresholds=thresholds, metric_type=metric_type)

    if n_workers > 0:
        with Pool(n_workers) as pool:
            results = pool.starmap(fn, samples)
    else:
        results = [fn(*s) for s in samples]

    n_gt = np.array([item for _, n_gt_list in results for item in n_gt_list])
    tpfp_list = [tp for tp, _ in results]

    if metric_type in (MatchingMetric.Chamfer, MatchingMetric.FRECHET):
        return calculate_average_precision(tpfp_list, thresholds, len(n_gt))

    return calculate_pld(tpfp_list, thresholds, n_gt)


def _get_metric_label(metric_type: MatchingMetric) -> str:
    interp = (
        f"Equidistant {SAMPLE_DIST}m" if SAMPLE_DIST is not None else f"{INTERP_NUM} pts"
    )
    optimiser = "Optimal" if METRICS_CONFIG.use_optimal_assignment else "Greedy"
    norm = METRICS_CONFIG.norm.name

    labels = {
        MatchingMetric.Chamfer: f"Chamfer Distance — {interp}",
        MatchingMetric.FRECHET: f"Fréchet Distance — {interp}",
        MatchingMetric.SOSPA: f"SOSPA (PLD) — {interp} | norm={norm} | order=Optimal | {optimiser}",
    }
    return labels.get(metric_type, f"Unknown({metric_type})")


def _build_ap_table(id2cat, result_dict, thresholds):
    """Build the AP prettytable."""
    cols = ["category", "num_preds", "num_gts"]
    cols += [f"AP@{t}" for t in thresholds]
    cols += ["AP"]

    table = prettytable.PrettyTable(cols)

    ap_sums = {f"AP@{t}": 0.0 for t in thresholds}
    ap_sums["AP"] = 0.0

    total_preds = total_gts = 0
    n = len(id2cat)

    for label in id2cat:
        d = result_dict[id2cat[label]]
        row = [id2cat[label], d["num_preds"], d["num_gts"]]
        row += [round(d[f"AP@{t}"], 4) for t in thresholds]
        row += [round(d["AP"], 4)]
        table.add_row(row)
        total_preds += d["num_preds"]
        total_gts += d["num_gts"]
        for t in thresholds:
            ap_sums[f"AP@{t}"] += d[f"AP@{t}"]
        ap_sums["AP"] += d["AP"]

    table.add_row(["-" * len(str(c)) for c in table.field_names])
    avg_row = ["average", int(total_preds / n), int(total_gts / n)]
    avg_row += [round(ap_sums[f"AP@{t}"] / n, 4) for t in thresholds]
    avg_row += [round(ap_sums["AP"] / n, 4)]
    table.add_row(avg_row)

    mAP = ap_sums["AP"] / n
    return table, mAP


def _build_pld_table(id2cat, result_dict, thresholds):
    """Build the PLD cost/loc prettytable for SOSPA-based evaluation."""
    cols = ["category", "num_preds", "num_gts"]
    cols += [f"PLD_cost@{t}" for t in thresholds] + ["PLD_cost"]
    cols += [f"PLD_loc@{t}" for t in thresholds] + ["PLD_loc"]

    table = prettytable.PrettyTable(cols)

    pld_cost_sums = {f"PLD_cost@{t}": 0.0 for t in thresholds}
    pld_cost_sums["PLD_cost"] = 0.0
    pld_loc_sums = {f"PLD_loc@{t}": 0.0 for t in thresholds}
    pld_loc_sums["PLD_loc"] = 0.0

    total_preds = total_gts = 0
    n = len(id2cat)

    for label in id2cat:
        d = result_dict[id2cat[label]]
        row = [id2cat[label], d["num_preds"], d["num_gts"]]
        row += [round(d[f"PLD_cost@{t}"], 4) for t in thresholds]
        row += [round(d["PLD_cost"], 4)]
        row += [round(d[f"PLD_loc@{t}"], 4) for t in thresholds]
        row += [round(d["PLD_loc"], 4)]
        table.add_row(row)
        total_preds += d["num_preds"]
        total_gts += d["num_gts"]
        for t in thresholds:
            pld_cost_sums[f"PLD_cost@{t}"] += d[f"PLD_cost@{t}"]
            pld_loc_sums[f"PLD_loc@{t}"] += d[f"PLD_loc@{t}"]
        pld_cost_sums["PLD_cost"] += d["PLD_cost"]
        pld_loc_sums["PLD_loc"] += d["PLD_loc"]

    table.add_row(["-" * len(str(c)) for c in table.field_names])
    avg_row = ["average", int(total_preds / n), int(total_gts / n)]
    avg_row += [round(pld_cost_sums[f"PLD_cost@{t}"] / n, 4) for t in thresholds]
    avg_row += [round(pld_cost_sums["PLD_cost"] / n, 4)]
    avg_row += [round(pld_loc_sums[f"PLD_loc@{t}"] / n, 4) for t in thresholds]
    avg_row += [round(pld_loc_sums["PLD_loc"] / n, 4)]
    table.add_row(avg_row)

    pld_cost = pld_cost_sums["PLD_cost"] / n
    pld_loc = pld_loc_sums["PLD_loc"] / n
    pld_cardi = pld_cost - pld_loc
    return table, pld_cost, pld_loc, pld_cardi


def _print_results(id2cat, result_dict, thresholds, metric_type):
    """Pretty-print evaluation results (no mmcv dependency).

    The AP table is printed for metrics that compute AP (Chamfer / Fréchet); the
    PLD cost/loc table is printed for metrics that compute PLD
    (SOSPA). Which table to print is decided directly from
    `metric_type`, rather than sniffing the result dict's keys.
    """
    is_ap_metric = metric_type in (MatchingMetric.Chamfer, MatchingMetric.FRECHET)

    if is_ap_metric:
        ap_table, mAP = _build_ap_table(id2cat, result_dict, thresholds)
        print(ap_table)
        print(f"\nmAP = {mAP:.4f}")
        return mAP

    pld_table, pld_cost, pld_loc, pld_cardi = _build_pld_table(
        id2cat, result_dict, thresholds
    )
    print(pld_table)
    print(
        f"PLD (SOSPA-based) cost,loc,card = "
        f"{pld_cost:.3f}, {pld_loc:.3f}, {pld_cardi:.3f}"
    )
    return pld_cost


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class LightVectorEvaluate:
    """Evaluate vectorised HD-map predictions without any training-framework deps.

    Parameters
    ----------
    gts : dict
        Ground-truth dict ``{token: {class_id: [polyline_ndarray, ...]}}``.
    roi_size : tuple[int, int]
        (width, height) of the perception range, e.g. ``(60, 30)``.
        Determines the AP distance thresholds.
    categories : dict[str, int]
        Mapping from class name to integer label, e.g.
        ``{"divider": 0, "ped_crossing": 1, "boundary": 2}``.
    n_workers : int
        Number of parallel workers (0 = single-process).
    """

    def __init__(
        self,
        gts: Dict,
        roi_size: Tuple[int, int],
        categories: Dict[str, int],
        n_workers: int = N_WORKERS,
    ):
        self.gts = gts
        self.roi_size = roi_size
        self.cat2id = categories
        self.id2cat = {v: k for k, v in categories.items()}
        self.n_workers = n_workers
        self.thresholds = THRESHOLD_PRESETS.get(roi_size, THRESHOLDS_60x30)
        if roi_size not in THRESHOLD_PRESETS:
            print(
                f"[warn] roi_size {roi_size} not in preset table — "
                f"using THRESHOLDS_60x30 = {THRESHOLDS_60x30}"
            )

    def evaluate(self, pred_file: str, metric: str = "chamfer") -> Dict:
        """Run evaluation and return the full result dict.

        Parameters
        ----------
        pred_file : str
            Path to prediction file (.json or .pkl).
        metric : str
            One of ``chamfer``, ``frechet``, ``sospa``.
            ``chamfer``/``frechet`` compute detection AP; ``sospa``
            computes PLD cost/localisation only.

        Returns
        -------
        dict
            Full result dict with per-category entries (each containing
            ``num_gts``, ``num_preds``, plus either AP-related keys or
            PLD-related keys depending on the metric) and a top-level
            mean-score key: ``"mAP"`` (AP mean) for ``chamfer``/``frechet``,
            ``"mPLD"`` (PLD cost mean) for ``sospa``.
        """
        submission = _load_file(pred_file)
        results = submission["results"]

        metric_type = MatchingMetric.to_metric_type(metric)

        samples_by_cls = {lbl: [] for lbl in self.id2cat}
        num_gts = {lbl: 0 for lbl in self.id2cat}
        num_preds = {lbl: 0 for lbl in self.id2cat}

        counter = 0
        for token, gt in self.gts.items():
            counter += 1
            if EVAL_SAMPLES_LIMIT is not None and counter > EVAL_SAMPLES_LIMIT:
                break

            pred = results.get(token, {"vectors": [], "scores": [], "labels": []})

            vecs_by_cls = {lbl: [] for lbl in self.id2cat}
            scores_by_cls = {lbl: [] for lbl in self.id2cat}

            for i in range(len(pred["labels"])):
                lbl = pred["labels"][i]
                vecs_by_cls[lbl].append(pred["vectors"][i])
                scores_by_cls[lbl].append(pred["scores"][i])

            for lbl in self.id2cat:
                samples_by_cls[lbl].append(
                    (vecs_by_cls[lbl], scores_by_cls[lbl], gt[lbl])
                )
                num_gts[lbl] += len(gt[lbl])
                num_preds[lbl] += len(scores_by_cls[lbl])

        print(f"\nROI size : {self.roi_size[0]}x{self.roi_size[1]}")
        print(f"Metric   : {_get_metric_label(metric_type)}")
        print(f"Workers  : {self.n_workers}")
        print(f"Scenes   : {counter}")
        print("=" * 60)

        result_dict: Dict = {}
        mScore_key = metric_type.to_mean_aggregate_score_key()
        sum_mScore = 0.0
        t0 = time()

        for lbl in self.id2cat:
            cat_name = self.id2cat[lbl]
            print(f"  Evaluating '{cat_name}' ...")
            class_score, class_res = _evaluate_scenes(
                samples_by_cls[lbl], self.thresholds, metric_type, self.n_workers
            )
            result_dict[cat_name] = {
                "num_gts": num_gts[lbl],
                "num_preds": num_preds[lbl],
                **class_res,
            }
            sum_mScore += class_score

        mScore = sum_mScore / len(self.id2cat)
        result_dict[mScore_key] = mScore

        print(f"\nCompute time: {time() - t0:.2f}s")
        _print_results(self.id2cat, result_dict, self.thresholds, metric_type)

        return result_dict
