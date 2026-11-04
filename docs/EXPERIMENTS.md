# Experiment definitions and result fields

## Response error

`response_trials.csv` stores the actual response norm error, its sqrt(n)-scaled value, directional and Frobenius fluctuations, and the directional bound for each trial. `sweep=q` supplies Figure 1; `sweep=n` supplies the additional fixed-dimension scaling check. Exact binomial sufficient statistics generate each empirical Hessian; they are not a Gaussian approximation.

## Theoretical upper bounds (Figure 2)

This deterministic plot evaluates the full statistical-plus-privacy upper bounds in Section IV.C, using the respective Gibbs calibrations. It does not measure empirical excess risk. Set n=1024, mu=h=epsilon=1, decision dimension r=1, and lower dimension q in {8, 16, 32, 64, 128}.

- Directional bound, both families: h/[mu*(mu+h/2)*sqrt(n)] + 2*h/(mu^2*n*epsilon).
- Optimized Frobenius bound, diagonal: h/(mu^2*sqrt(n)) + 2*h/(mu^2*n*epsilon).
- Optimized Frobenius bound, two-basis: sqrt(q) times the diagonal Frobenius bound.
- Operator-norm bound, both families: h/[mu*(mu+h/2)] * min(1, sqrt(2*log(2*q)/n)) + 2*h/(mu^2*n*epsilon).

The operator-norm benchmark uses the sharp global stability constant 1/mu^2 for privacy and the population-centered constant 1/[mu*(mu+h/2)] for utility. Its privacy term matches the directional bound. For independent fair-bit records, the centered matrix has scalar moment generating function cosh(theta*h/2)*I. Iterated conditional Golden-Thompson bounds give the displayed logarithmic statistical term; the minimum also uses the deterministic fluctuation bound h/2. See Tropp (2012), *User-Friendly Tail Bounds for Sums of Random Matrices*, Foundations of Computational Mathematics 12:389-434, https://doi.org/10.1007/s10208-011-9099-z.

The directional bound is 0.02278645833 and the diagonal Frobenius bound is 0.033203125. The two-basis Frobenius-to-directional ratio ranges from 4.1214 to 16.4857. The operator-norm bound ranges from 0.05101187594 to 0.07133267593 in both families, with operator-to-directional ratios 2.2387 to 3.1305. For diagonal curvature, the optimized Frobenius bound is tighter than this operator-norm bound. These are ratios of theoretical bounds, not measured utility improvements. Both axes are logarithmic; the panels have independent vertical ranges. The plotting script checks the expressions against the primitive variance, diameter, and stability constants.

## Block geometry

`block` adds independent radial noise per block; `ambient` adds ambient radial noise. Both project to the same product of balls and apply their own public-bound zero fallback. `nonprivate` uses the empirical mean with the same projection and scaling, and `public_zero` is the zero reference. The exact population loss is evaluated through the population inverse matrix.

The utility law has a uniform block index and independent signs with positive probability 3/4 within the selected block. Figure 3 divides excess risk by s/sqrt(ell), with s = h/(2*mu+h). The `ungated_risk` column is a diagnostic before the public zero fallback. The separate centered-moment calculation uses fair signs and should not be interpreted as a moment calculation for the biased utility law.

## Calibration methods

| CSV method | Meaning |
| --- | --- |
| `coefficient` | Algorithm 1: perturb the r-dimensional fitted coefficient, normalize, and apply the public gate |
| `output_tight` | Add tight radial noise to the full r+1 fitted response, then normalize its final r coordinates and apply the gate |
| `em_tight_normalized` | Sample the tight exponential density on the r-ball, normalize, and apply the gate; the exponential curve in Figure 4 |
| `em_tight` | The same exponential sample before normalization, with the same gate; diagnostic |
| `output_cms2011` | Output perturbation with twice the tight noise scale; diagnostic, not the plotted output curve |
| `nonprivate` | Normalize the fitted coefficient without adding noise |
| `public_zero` | The zero decision |

For calibration the gate is n^(-1/2) + r/(n*epsilon) > 1. The exact population excess is norm(q_P) - q_P dot x. The synthetic population uses independent biased coordinate signs and an independent fair global sign. `em_*.csv` contains exponential results for both the synthetic and Wine populations; filter `family` or `dataset` when analyzing these files.

All methods use the same sampled data for a given setting and trial. Paired radial draws preserve the required individual noise distributions. The exponential adapter uses its own deterministic noise seed. The original broader sweeps are not included in this artifact; the included settings cover the current experimental section and the additional response scaling check.

## Summaries and verification

`*_trials.csv` contains one row per setting, method/model and trial. `*_summary.csv` contains the mean, sample standard error and pointwise 95% Monte Carlo interval. Calibration interval lower endpoints are truncated at zero; response and block summaries use mean +/- 1.96 standard errors without truncation. These are pointwise intervals, not simultaneous confidence bands.

The saved reference records retain the original numerical results. Repository preparation merged the n=32768 calibration point with the other four n settings and selected the reported Wine settings. The original `sweeps` tags may list other sweeps passing through the same setting; the numerical configuration is identified by its parameter columns and `case_id`.

The historical n=32768 exponential extension did not save optional sampler metadata (tilt, proposal/acceptance counts, and sample norm), so these fields are blank in its reference rows. New runs record these fields for every setting. Verification checks the historical diagnostic values where present and compares all utility fields for every setting.

Cross-run verification compares all included trial and summary files, excluding timings, solver residuals, and sweep tags. Figure verification separately checks the x coordinates, means, and interval endpoints against those extracted from the current manuscript's plotting manifest. PDF byte equality is not required because rendering metadata and fonts can differ.
