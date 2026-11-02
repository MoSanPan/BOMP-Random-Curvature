"""Public Wine Quality feature bank for the fixed calibration benchmark.

This is data preparation, not an estimator.  The complete UCI table is public
reference material used to define a finite population.  An estimator receives
only the rows sampled by the experiment driver, never this population's moment.
The wine quality response is not used: this is a covariance calibration
task using real features, not wine quality prediction.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
DATASET_URL = "https://archive.ics.uci.edu/dataset/186/wine+quality"
ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
EXPECTED_SHA256 = {
    "wine_quality_uci.zip":
        "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4",
    "winequality-red.csv":
        "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
    "winequality.names":
        "15d215e73b39105952380fd487e4ce1bf5a6b83a425abb285c2ec55193d3ca92",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify(name: str, value: bytes) -> None:
    observed = _sha256(value)
    if observed != EXPECTED_SHA256[name]:
        raise ValueError(
            f"Public data checksum changed for {name}: {observed}; "
            "refusing to replace the pinned benchmark silently."
        )


def fetchraw(data_dir: str | Path | None = None) -> Path:
    """Fetch once, then verify and reuse the pinned official CSV cache.

    A changed or damaged cache raises an error.  No automatic redownload hides
    the mismatch.  The archive contains both wine tables, but only the red CSV
    and its original documentation are extracted.
    """
    target = Path(data_dir) if data_dir is not None else HERE.parent / "data"
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "winequality-red.csv"
    names_path = target / "winequality.names"
    archive_path = target / "wine_quality_uci.zip"

    if csv_path.exists() and names_path.exists():
        _verify(csv_path.name, csv_path.read_bytes())
        _verify(names_path.name, names_path.read_bytes())
        if archive_path.exists():
            _verify(archive_path.name, archive_path.read_bytes())
        return csv_path

    if archive_path.exists():
        archive = archive_path.read_bytes()
    else:
        with urllib.request.urlopen(ZIP_URL, timeout=60) as response:
            archive = response.read()
        _verify(archive_path.name, archive)
        archive_path.write_bytes(archive)
    _verify(archive_path.name, archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in (csv_path.name, names_path.name):
            value = zf.read(name)
            _verify(name, value)
            (target / name).write_bytes(value)
    return csv_path


def load_public_bank(
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Return unit-norm public reference rows and transparent provenance.

    The fixed transformation is fit to the public reference table *once*, not
    to any simulated private sample.  Its per-column scale is the public RMS,
    without centering; retaining a nonzero intercept-feature moment is needed
    to make the fixed public calibration task nontrivial.  Prepend 1 and divide
    each row by its own Euclidean norm.  The driver samples with replacement,
    multiplies by sqrt(h), and draws an independent fair global sign.  Signs
    make the feature population mean zero without changing any second moment.
    """
    csv_path = fetchraw(data_dir)
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(reader)
        table = np.asarray([[float(x) for x in row] for row in reader],
                           dtype=np.float64)
    if table.shape != (1599, 12) or len(header) != 12 or header[-1] != "quality":
        raise ValueError(f"Unexpected red Wine Quality layout: {table.shape}")
    if not np.isfinite(table).all():
        raise ValueError("The pinned benchmark unexpectedly contains nonfinite data")

    raw_features = table[:, :-1]
    public_rms = np.sqrt(np.mean(raw_features ** 2, axis=0))
    if np.any(public_rms <= 0):
        raise ValueError("Cannot RMS-scale a zero public feature column")
    augmented = np.column_stack((np.ones(len(table)), raw_features / public_rms))
    bank = np.ascontiguousarray(
        augmented / np.linalg.norm(augmented, axis=1, keepdims=True)
    )
    norms = np.linalg.norm(bank, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=5e-15):
        raise AssertionError("Public feature bank must have unit-norm rows")

    metadata = {
        "name": "UCI Wine Quality (red), real-data covariance calibration",
        "source_page": DATASET_URL,
        "source_download": ZIP_URL,
        "dataset_doi": "10.24432/C56S3T",
        "creators": ["Paulo Cortez", "A. Cerdeira", "F. Almeida",
                     "T. Matos", "J. Reis"],
        "citation": ("Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. "
                     "(2009). Wine Quality [Dataset]. UCI Machine Learning "
                     "Repository. https://doi.org/10.24432/C56S3T."),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "official_license_verified_date": "2026-10-28",
        "downloaded_date": "2026-10-28",
        "sha256": dict(EXPECTED_SHA256),
        "raw_shape": list(table.shape),
        "reference_rows": len(table),
        "raw_feature_count": raw_features.shape[1],
        "feature_names": header[:-1],
        "ignored_response": "quality",
        "unit_bank_shape": list(bank.shape),
        "lower_dimension_q": bank.shape[1],
        "decision_dimension_r": bank.shape[1] - 1,
        "public_rms": public_rms.tolist(),
        "preprocessing": [
            "Exclude the wine quality response; do not fit or score a predictor.",
            "Compute each feature RMS on the complete PUBLIC reference table.",
            "Divide public reference features by their public RMS; do not center.",
            "Prepend the public intercept coordinate 1.",
            "Normalize each augmented row to unit Euclidean norm.",
        ],
        "population_definition": (
            "Choose one row uniformly with replacement from the public bank, "
            "multiply by sqrt(h), then by an independent fair sign. "
            "Duplicate source rows retain their empirical multiplicities."
        ),
        "second_moment_for_evaluation_only":
            "Sigma_P = h * unit_bank.T @ unit_bank / reference_rows",
        "mean_zero": "Exact in the population after the independent global sign.",
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "privacy_interpretation": (
            "The experiment SIMULATES private samples from a population defined "
            "using public material. Public-reference preprocessing is fixed before "
            "each simulated sample; it is not free preprocessing of genuinely "
            "private data. The estimator must receive only its sampled rows and "
            "public parameters, never the full bank or Sigma_P."
        ),
        "scope": (
            "Fixed public calibration on real physicochemical "
            "features; not wine quality prediction, classification, or a "
            "real-world application performance claim."
        ),
    }
    return bank, metadata


