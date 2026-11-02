"""Numerical calibration routines used by the paper experiments."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
METHODS = ("coefficient", "output_tight", "output_cms2011", "nonprivate", "public_zero")
BASE = {"n": 2048, "epsilon": 1.0, "r": 16, "eta": 0.3, "mu": 1.0}

def seed_for(seed, *parts):
    b = json.dumps([int(seed), *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little")


def normalize(v):
    length = float(np.linalg.norm(v))
    return v / length if length > 0 else np.zeros_like(v)


def population(family, r, h):
    """Evaluator-only exact moments for two fixed, non-degenerate laws."""
    v = np.ones(r) / math.sqrt(r)
    if family == "dense_signs":
        beta = 0.5
        m = beta * v
        second = (1-beta**2) * np.eye(r) / r + beta**2 * np.outer(v, v)
    elif family == "spike_isotropic":
        p = 0.25
        m = p * v
        second = (1-p) * np.eye(r) / r + p * np.outer(v, v)
    else:
        raise ValueError(family)
    sigma = np.empty((r+1, r+1))
    sigma[0, 0] = 1
    sigma[0, 1:] = m
    sigma[1:, 0] = m
    sigma[1:, 1:] = second
    return (h/2) * sigma


def sample_synthetic(family, n, r, h, rng):
    if family == "dense_signs":
        z = (2 * (rng.random((n, r)) < 0.75) - 1) / math.sqrt(r)
    elif family == "spike_isotropic":
        is_spike = rng.random(n) < 0.25
        z = np.zeros((n, r))
        z[is_spike] = 1 / math.sqrt(r)
        rest = np.flatnonzero(~is_spike)
        idx = rng.integers(r, size=len(rest))
        z[rest, idx] = 2 * rng.integers(2, size=len(rest)) - 1
    else:
        raise ValueError(family)
    a = np.empty((n, r+1))
    a[:, 0] = 1
    a[:, 1:] = z
    a *= math.sqrt(h/2)
    a *= (2*rng.integers(2, size=n)-1)[:, None]
    return a


def fit_private_table(a, mu):
    """Estimator only sees sampled records and public mu, not population data."""
    qmat = a.T @ a / len(a)
    qmat.flat[::qmat.shape[0]+1] += mu
    rhs = np.zeros(qmat.shape[0]); rhs[0] = mu
    fitted = np.linalg.solve(qmat, rhs)
    residual = float(np.linalg.norm(qmat @ fitted-rhs))
    return fitted, residual


def coupled_noises(r, eta, n, epsilon, rng):
    """Exact continuous radial laws, coupled only for paired MC comparisons."""
    g = rng.normal(size=r+1)
    direction_r = normalize(g[1:])
    direction_q = normalize(g)
    radius_r = rng.gamma(shape=r, scale=1.0)
    radius_q = radius_r + rng.exponential()
    scale = eta / (n*epsilon)
    return scale*radius_r*direction_r, scale*radius_q*direction_q


def mechanisms(fitted, n, epsilon, eta, noise_rng):
    r = len(fitted)-1
    gate = n**(-0.5)+r/(n*epsilon) > 1
    t0 = time.perf_counter()
    wr, wq = coupled_noises(r, eta, n, epsilon, noise_rng)
    noise_shared = time.perf_counter()-t0
    vectors = {
        "coefficient": fitted[1:]+wr,
        "output_tight": (fitted+wq)[1:],
        "output_cms2011": (fitted+2*wq)[1:],
        "nonprivate": fitted[1:],
        "public_zero": np.zeros(r),
    }
    answer = {}
    for method in METHODS:
        t0 = time.perf_counter()
        x_raw = normalize(vectors[method]) if method != "public_zero" else np.zeros(r)
        is_gated = gate and method in METHODS[:3]
        x = np.zeros(r) if is_gated else x_raw
        post_seconds = time.perf_counter()-t0
        # Shared coupling draw is charged in full to every private arm, so this
        # is a conservative kernel timing, not a standalone speed comparison.
        if method in METHODS[:3]:
            post_seconds += noise_shared
        answer[method] = (x, x_raw, is_gated, post_seconds)
    return answer


def model_q(sigma, mu):
    rhs = np.zeros(len(sigma)); rhs[0] = mu
    return np.linalg.solve(mu*np.eye(len(sigma))+sigma, rhs)[1:]


def risk(qp, x):
    value = float(np.linalg.norm(qp) - qp @ x)
    if value < -1e-12:
        raise AssertionError((value, qp, x))
    return max(0.0, value)


def csv_write(path, rows):
    if not rows:
        raise ValueError("empty output")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    se = float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0.0
    return mean, se, max(0.0, mean-1.96*se), mean+1.96*se


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["case_id"], row["method"]), []).append(row)
    summaries = []
    for (_, method), group in groups.items():
        row = {k: group[0][k] for k in ("case_id", "dataset", "family", "n", "r", "mu", "eta", "epsilon", "sweeps")}
        row.update(method=method, trials=len(group))
        for key in ("excess_risk", "ungated_risk", "coefficient_error", "fit_seconds", "total_seconds"):
            mean, se, lo, hi = mean_ci([g[key] for g in group])
            row.update({key+"_mean": mean, key+"_se": se, key+"_lo95": lo, key+"_hi95": hi})
        row["gate_fraction"] = sum(int(g["gated"]) for g in group)/len(group)
        row["population_zero_risk"] = group[0]["population_zero_risk"]
        row["max_solve_residual"] = max(g["solve_residual"] for g in group)
        summaries.append(row)
    return summaries


def paired_summaries(rows):
    trial_map = {}
    for row in rows:
        trial_map.setdefault((row["case_id"], row["trial"]), {})[row["method"]] = row
    groups = {}
    for (case_id, _), methods in trial_map.items():
        for other in ("output_tight", "output_cms2011", "public_zero"):
            groups.setdefault((case_id, other), []).append(methods["coefficient"]["excess_risk"]-methods[other]["excess_risk"])
    out = []
    for (case_id, other), values in groups.items():
        a = np.asarray(values)
        se = float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0.0
        out.append(dict(case_id=case_id, difference="coefficient_minus_"+other,
                        trials=len(a), mean=float(a.mean()), se=se,
                        lo95=float(a.mean()-1.96*se), hi95=float(a.mean()+1.96*se)))
    return out


def run_cases(configs, outdir, seed, trials, bank=None):
    rows = []
    for ci, c in enumerate(configs):
        n, r, eta, eps, mu = c["n"], c["r"], c["eta"], c["epsilon"], c["mu"]
        h = mu*eta
        source = "wine_red" if bank is not None else "synthetic"
        case_id = hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!="sweeps"},sort_keys=True).encode()).hexdigest()[:12]
        sigma = h*(bank.T @ bank)/len(bank) if bank is not None else population(c["family"], r, h)
        qp = model_q(sigma, mu)
        assert np.linalg.norm(qp) <= eta/(2*(1+eta))+1e-12
        if bank is None:
            # Independent Schur-complement formula, separate from the solve.
            beta = 0.5 if c["family"]=="dense_signs" else 0.25
            eig = ((1-beta**2)/r+beta**2) if c["family"]=="dense_signs" else ((1-beta)/r+beta)
            cc = h*beta/2
            denominator = (mu+h/2)*(mu+h*eig/2)-cc**2
            exact = -mu*cc/denominator * np.ones(r)/math.sqrt(r)
            assert np.max(np.abs(qp-exact)) < 1e-12
        for trial in range(trials):
            sample_rng = np.random.default_rng(seed_for(seed, "sample", case_id, trial))
            noise_rng = np.random.default_rng(seed_for(seed, "noise", case_id, trial))
            if bank is None:
                a = sample_synthetic(c["family"], n, r, h, sample_rng)
            else:
                a = math.sqrt(h)*bank[sample_rng.integers(len(bank), size=n)]
                a *= (2*sample_rng.integers(2,size=n)-1)[:,None]
            assert np.max(np.einsum("ij,ij->i",a,a)) <= h*(1+1e-12)
            t0 = time.perf_counter()
            fitted, residual = fit_private_table(a, mu)
            fit_seconds = time.perf_counter()-t0
            assert residual < 1e-10 and np.linalg.norm(fitted) <= 1+1e-10
            coef_error = float(np.linalg.norm(fitted[1:]-qp))
            outputs = mechanisms(fitted, n, eps, eta, noise_rng)
            for method, (x, xraw, gated, post) in outputs.items():
                assert np.linalg.norm(x) <= 1+1e-12
                fit_cost = 0.0 if gated or method=="public_zero" else fit_seconds
                row = dict(case_id=case_id, dataset=source, family=c["family"], n=n, r=r,
                    mu=mu, eta=eta, epsilon=eps, sweeps=";".join(c["sweeps"]), trial=trial,
                    method=method, excess_risk=risk(qp,x), ungated_risk=risk(qp,xraw),
                    coefficient_error=coef_error, population_zero_risk=float(np.linalg.norm(qp)),
                    gated=int(gated), solve_residual=residual, fit_seconds=fit_cost,
                    total_seconds=fit_cost+(0.0 if gated else post))
                rows.append(row)
        print(f"{source} {ci+1}/{len(configs)} {c['family']} n={n} r={r} eps={eps} eta={eta}", flush=True)
    csv_write(outdir/f"{source}_trials.csv",rows)
    summary = summarize(rows)
    csv_write(outdir/f"{source}_summary.csv",summary)
    csv_write(outdir/f"{source}_paired.csv",paired_summaries(rows))
    return rows, summary
