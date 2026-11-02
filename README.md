# Random-Curvature Bilevel Optimization: Experimental Artifact

Reproducibility materials for **What Is the Cost of Privacy in Bilevel Optimization with Random Curvature?**

This repository contains the code, real-world and synthetic datasets, saved trial results, and plotting scripts for the current manuscript's experimental section. The three figures and the Wine Quality results are covered by the commands below. No external project directory, private dataset, API key, GPU, or network download is needed with the included data files.

## Quick start

Use Python 3.12 and run these commands from the repository root:

```console
python -m pip install -r requirements.txt
python reproduce.py check
python reproduce.py plots
python reproduce.py full
```

`check` verifies the packaged checksums, the Wine data, and the aggregation of saved trial results. `plots` redraws all three paper figures from the saved CSVs. `full` reruns every included paper setting with seed 42 and 100 repetitions per setting, compares the numerical results against the saved reference, and redraws the figures.

For a quick execution check or the separate sampler diagnostics:

```console
python reproduce.py smoke
python reproduce.py diagnostics
```

The smoke run uses two repetitions and is not a reproduction of the reported 100-run statistics. The diagnostics command checks the block risk identity, sufficient-statistic construction, radial noise moment, and exponential sampler moments and acceptance rates against independent quadrature.

Commands write into `outputs/<mode>/`. Existing output directories are refused to prevent accidental replacement; use a new path on a second run:

```console
python reproduce.py full --out outputs/full_second_run
```

The saved `results/`, `data/`, and `figures/` directories are not overwritten. The Wine loader can fetch the pinned UCI archive if its cached data files are removed, but the supplied package runs offline. A changed cached file raises a checksum error.

## Included experiments

| Paper result | Settings | Saved results |
| --- | --- | --- |
| Actual response error, Figure 1 | Diagonal and two-basis models; q = 8, 16, 32, 64, 128; n = 1024; mu = h = 1 | `results/geometry/response_*.csv` |
| Additional response scaling | q = 64; n = 128, 512, 2048, 8192, 32768; both models | Same response files |
| Block geometry, Figure 2 | r = 256; n = 8192; epsilon = 0.25; eta = 0.3; mu = 1; ell = 1, 4, 16, 64, 256; d = r/ell | `results/geometry/block_*.csv` |
| Centered-law directional moment | The same block geometries, with fair signs | `results/geometry/centered_moments.csv` |
| Synthetic calibration, Figure 3 | r = 16; eta = 0.3; mu = epsilon = 1; n = 128, 512, 2048, 8192, 32768 | `results/calibration/synthetic_*.csv` and `em_*.csv` |
| Wine Quality results in the text | 1599 source rows, 11 features; sampled n = 2048; r = 11; eta = 0.3; mu = 1; epsilon = 1 and 0.005 | `results/calibration/wine_red_*.csv` |

All utility and response settings have 100 repetitions. Trial randomness is derived deterministically from the master seed and the setting/trial identifiers. Samples are shared across methods; the radial noise draws are coupled for paired comparisons. The centered moment check is separate from the biased-sign block utility experiment.

## Files

```text
reproduce.py                   Command-line entry point
requirements.txt               Tested Python package versions
checksums.json                 Hashes of distributed code, data and reference results
src/
  calibration.py               Calibration fits, mechanisms, risk, summaries
  calibration_experiments.py   Paper calibration settings and exponential adapter
  geometry.py                  Response and block experiments
  real_data.py                 Wine preprocessing, download fallback, checksums
  tilted_ball_sampler.py       Exponential-mechanism rejection sampler
  sampler_diagnostics.py       Moment, acceptance and envelope diagnostics
  plot_figures.py              Current paper figure styling
  verification.py              Integrity, aggregation and reproduction comparisons
data/                          Original UCI red-wine CSV and documentation
results/                       Saved trial-level and aggregated results
figures/                       Paper figures in PDF format
docs/                          Experiment details and reference figure values
```

`docs/paper_figure_values.json` records the exact x coordinates, means, and intervals used in the current manuscript figures. Replotting checks these values. Figure font selection prefers Times New Roman, then Liberation Serif, then DejaVu Serif. Font availability and PDF metadata can change the rendered files' bytes without changing the plotted numbers.

## Data and interpretation

Wine Quality is a real-world dataset. Its public red-wine feature table is used to define the calibration population; this experiment does not predict wine quality. Quality labels are unused. The complete transformation and sampling procedure are described in [data/README.md](data/README.md), including RMS scaling, a constant coordinate, row normalization, resampling with replacement, and independent global signs. Public-population moments are used only for evaluation; the fitted estimator receives its sampled records and public parameters.

The primary calibration comparison is coefficient perturbation (Algorithm 1), tight output perturbation, and a tight normalized exponential adapter. These baselines are independently implemented. Additional saved rows retain the nonprivate/public-zero references and diagnostic variants; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for exact method identifiers.

The mechanisms implement continuous-noise constructions in floating-point arithmetic. The numerical checks are implementation diagnostics, not a machine-level differential-privacy certificate. These experiments measure pointwise finite-sample utility on the specified populations; they do not establish minimax lower bounds or uniform superiority over all baselines.

## Verification and environment

The pinned environment is Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.8.4. The runner sets BLAS/OpenMP thread counts to one. Only a CPU is required.

Full reproduction compares trial-level values and summaries using relative tolerance 1e-9 and absolute tolerance 1e-12. Wall-clock timings, tiny solver residuals, and historical sweep-routing tags are excluded from cross-run comparisons. Timing fields are diagnostic bookkeeping, not an end-to-end system benchmark; exponential-mechanism timing fields can be zero placeholders. Utility, noise, gate, and aggregation fields are checked.

Running `full` writes a verification report to `outputs/full/verification.json`; `diagnostics` writes its reports to `outputs/diagnostics/`. Replotting generates PDF figures and PNG previews in the selected output directory. Generated reports and previews are not included in this upload package. Runtime depends on the machine and installed numerical libraries.

## Dataset attribution

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). **Wine Quality** [Dataset]. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C56S3T).

The UCI dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original CSV and its documentation are included unchanged. This data license applies to the dataset; it does not assign a license to the experimental code.
