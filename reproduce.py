"""One entry point for checking, redrawing, and reproducing the paper artifact."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "plots", "smoke", "full", "diagnostics"))
    parser.add_argument("--out", type=Path, help="New output folder; existing folders are refused")
    args = parser.parse_args()
    import verification
    started = time.perf_counter()
    report = {"mode": args.mode, "python": platform.python_version()}
    if args.mode == "check":
        if args.out is not None:
            parser.error("check is read-only and does not use --out")
        report.update(verification.check_saved(ROOT))
        print(json.dumps(report, indent=2))
        return
    out = (args.out or ROOT/"outputs"/args.mode).resolve()
    if out.exists():
        parser.error("Output directory already exists; select a new --out path")
    out.mkdir(parents=True)
    import numpy, scipy, matplotlib
    report["versions"] = {"numpy": numpy.__version__, "scipy": scipy.__version__,
                          "matplotlib": matplotlib.__version__}
    if args.mode == "diagnostics":
        import geometry, sampler_diagnostics
        report["geometry_checks"] = geometry.checks(42)
        sampler_diagnostics.run(out)
        report["sampler_diagnostics_passed"] = True
    else:
        data = ROOT/"results"
        if args.mode in ("smoke", "full"):
            import geometry, calibration_experiments
            repetitions = 2 if args.mode == "smoke" else 100
            data = out/"results"
            geometry.run(data/"geometry", seed=42, repetitions=repetitions)
            calibration_experiments.run(data/"calibration", seed=42, trials=repetitions)
            report.update(seed=42, repetitions=repetitions)
            if args.mode == "full":
                report["comparison_with_saved_results"] = verification.compare_results(ROOT/"results", data)
        subprocess.run([sys.executable, str(ROOT/"src/plot_figures.py"),
                        "--results", str(data), "--out", str(out/"figures")], check=True)
        if args.mode != "smoke":
            report["figure_values"] = verification.check_figure_values(ROOT, out/"figures")
    report["elapsed_seconds"] = time.perf_counter()-started
    report["passed"] = True
    (out/"verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
