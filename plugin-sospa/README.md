# sospa_eval

Framework-free evaluation metrics for polyline/trajectory map prediction:
**SOSPA** (Sequence Optimal Sub-Pattern Assignment), **PLD**
(Polyline Localisation & Detection), Chamfer and Fréchet distances.

This package is the extracted, standalone core of the map-tracking evaluation
suite. It has **no dependency on `mmdet3d`/`mmcv`** — the framework glue
(`raster_eval`, `vector_eval`) lives in the consuming repository.

## Setup

The package is not installed; it is used directly from the repo. From the repo root:

```bash
python -m pip install -r requirements.txt
```

To import `sospa_eval` from your own code, run from `plugin-sospa/` or add it to
`PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/plugin-sospa:$PYTHONPATH
```

## Example Data

This repository includes trimmed evaluation fixtures under `plugin-sospa/examples/`:

- `nusc_60x30_gts_100.pkl`
- `nusc_60x30_preds_100.json`

They are extracted from the larger workspace fixtures and contain the first 100
scene tokens from:

- `work_dirs/streamapnet_gts/tmp_gts_nusc_60x30_newsplit.pkl`
- `work_dirs/streammapnet_master/submission_vector_60x30.json`

These files are intended for documentation examples and quick smoke tests, so
you do not need to change `EVAL_SAMPLES_LIMIT` to reproduce the commands below.

## Minimal Working Example

```python
import json
import pickle

from sospa_eval import LightVectorEvaluate

with open("plugin-sospa/examples/nusc_60x30_gts_100.pkl", "rb") as f:
    gts = pickle.load(f)

evaluator = LightVectorEvaluate(
    gts=gts,
    roi_size=(60, 30),
    categories={"divider": 0, "ped_crossing": 1, "boundary": 2},
    n_workers=0,
)

result = evaluator.evaluate(
    pred_file="plugin-sospa/examples/nusc_60x30_preds_100.json",
    metric="sospa",
)

print(result["mAP"])  # For SOSPA this is the mean PLD_cost across classes.
print(result["divider"]["PLD_cost"])
print(result["divider"]["PLD_loc"])
```

## Running LightVectorEvaluate With SOSPA

If you want a Python API, use `LightVectorEvaluate` as shown above.

There is also a CLI helper in `tools/evaluate_light.py`:

```bash
python tools/evaluate_light.py \
  --gts tools/gts_pred_examples/nusc_60x30_gts_100.pkl \
  --preds tools/gts_pred_examples/nusc_60x30_preds_100.json \
  --roi-size 60x30 \
  --metric sospa \
  --n-workers 0
```

That command prints the per-class PLD table and writes a JSON result file under
`test_light/`.

## Input / Output Contracts

### Ground-truth input

Ground-truth files must be pickle files with this structure:

```python
{
    "<token>": {
        0: [np.ndarray, ...],
        1: [np.ndarray, ...],
        2: [np.ndarray, ...],
    },
    ...
}
```

- The outer key is a scene token.
- Each class id maps to a list of polylines.
- Each polyline is expected to be an `ndarray` of shape `(N, 2)` or `(N, 3)`.

### Prediction input

Prediction files may be JSON or pickle, and must follow the submission layout:

```python
{
    "meta": {...},
    "results": {
        "<token>": {
            "vectors": [polyline, ...],
            "scores": [0.9, 0.8, ...],
            "labels": [0, 1, 2, ...],
        },
        ...
    }
}
```

- `vectors`, `scores`, and `labels` must be aligned by index.
- Missing scene tokens are treated as empty predictions by the evaluator.

## Public API

```python
from sospa_eval import (
    MatchingMetric, MetricsNormType, Polyline,
    MetricsConfig, METRICS_CONFIG,
    calculate_average_precision, calculate_pld,
    instance_match, instance_match_pld,
    interpolate_polylines,
    LightVectorEvaluate,
)
```

## Package Boundaries

- `plugin-sospa` / `sospa_eval` owns SOSPA, PLD, Chamfer, Fréchet, interpolation,
    instance matching, and the framework-free evaluator class.
- The framework-specific glue for the main training/evaluation stack remains in
    the parent repository, not in this package.

## Tests

```bash
cd plugin-sospa && python -m unittest discover -s sospa_eval/test -t .
```
