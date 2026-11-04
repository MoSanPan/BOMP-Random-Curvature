"""Check packaged inputs, aggregation, and numerical reproduction."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import calibration
import geometry
from real_data import load_public_bank


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def typed(rows):
    result = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if key == "case_id":
                continue
            try:
                item[key] = float(value)
            except ValueError:
                pass
        result.append(item)
    return result


def compare_rows(expected, actual, keys):
    def index(rows):
        mapping = {tuple(str(r[k]) for k in keys): r for r in typed(rows)}
        if len(mapping) != len(rows):
            raise AssertionError("Duplicate result keys")
        return mapping
    old, new = index(expected), index(actual)
    if old.keys() != new.keys():
        raise AssertionError(f"Result keys differ: {old.keys() ^ new.keys()}")
    checked = 0
    maximum = 0.0
    for key, left in old.items():
        right = new[key]
        for field, value in left.items():
            # Timing varies by machine; historical sweeps are routing metadata.
            # Tiny linear-solve residuals also vary across BLAS implementations.
            if field == "sweeps" or "seconds" in field or "solve_residual" in field:
                continue
            # The historical n=32768 extension did not record sampler metadata.
            if value == "" and field in {"kappa", "lambda_tilt", "proposal_count", "accepted_count", "sample_norm"}:
                continue
            other = right[field]
            if isinstance(value, float):
                if not (math.isfinite(value) and math.isfinite(other)):
                    raise AssertionError((key, field, "nonfinite"))
                if not math.isclose(value, other, rel_tol=1e-9, abs_tol=1e-12):
                    raise AssertionError((key, field, value, other))
                maximum = max(maximum, abs(value-other))
                checked += 1
            elif value != other:
                raise AssertionError((key, field, value, other))
    return {"rows": len(old), "numeric_values_checked": checked,
            "max_absolute_difference": maximum}


FILES = {
    "geometry/response_trials.csv": ["sweep", "q", "n", "model", "trial"],
    "geometry/response_summary.csv": ["sweep", "q", "n", "model"],
    "geometry/block_trials.csv": ["ell", "method", "trial"],
    "geometry/block_summary.csv": ["ell", "method"],
    "geometry/centered_moments.csv": ["ell"],
    **{f"calibration/{stem}_{kind}.csv": ["case_id", "method"] +
       (["trial"] if kind == "trials" else [])
       for stem in ("synthetic", "wine_red", "em") for kind in ("trials", "summary")},
}


def compare_results(reference, actual):
    return {name: compare_rows(read(reference/name), read(actual/name), keys)
            for name, keys in FILES.items()}


def check_saved(root):
    hashes = json.loads((root/"checksums.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        observed = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if observed != expected:
            raise AssertionError(f"Packaged input changed: {name}")
    bank, metadata = load_public_bank(root/"data")
    results = root/"results"
    checks = {}
    for stem in ("synthetic", "wine_red", "em"):
        rows = typed(read(results/f"calibration/{stem}_trials.csv"))
        saved = read(results/f"calibration/{stem}_summary.csv")
        checks[stem] = compare_rows(saved, calibration.summarize(rows), ["case_id", "method"])
        assert all(int(r["trials"]) == 100 for r in saved)
    for stem, keys, metrics in [
        ("response", ["sweep", "q", "n", "model"], ["response_error", "scaled_response"]),
        ("block", ["r", "ell", "d", "n", "epsilon", "eta", "beta", "method"],
         ["excess_risk", "normalized_risk", "ungated_risk", "gated", "mean_error", "noise_norm", "data_error"]),
    ]:
        rows = typed(read(results/f"geometry/{stem}_trials.csv"))
        saved = read(results/f"geometry/{stem}_summary.csv")
        checks[stem] = compare_rows(saved, geometry.summary(rows, keys, metrics), keys)
        assert all(int(r["repetitions"]) == 100 for r in saved)
    return {"packaged_hashes_checked": len(hashes), "wine_reference_rows": len(bank),
            "quality_labels_used": False, "aggregation": checks}


def check_figure_values(root, figure_dir):
    expected = json.loads((root/"docs/paper_figure_values.json").read_text(encoding="utf-8"))
    actual = json.loads((figure_dir/"restyle_manifest.json").read_text(encoding="utf-8"))["figures"]
    def visit(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                visit(a[k], b[k])
        else:
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    visit(expected, actual)
    theoretical = compare_rows(read(root/"docs/theoretical_bounds.csv"),
                               read(figure_dir/"theoretical_bounds.csv"), ["family", "q"])
    return {"figures": len(expected) + 1, "empirical_figures": len(expected),
            "means_and_intervals_match": True, "theoretical_bounds": theoretical}
