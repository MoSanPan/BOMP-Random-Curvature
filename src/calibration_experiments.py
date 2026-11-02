"""Run the calibration settings reported in the current paper."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import calibration as base
from real_data import load_public_bank
from tilted_ball_sampler import sample_tilted_ball
N_GRID = (128, 512, 2048, 8192, 32768)

def run(out, seed=42, trials=100):
    data = out
    data.mkdir(parents=True, exist_ok=True)
    synthetic = [dict(base.BASE, n=n, family="dense_signs", sweeps=["n"]) for n in N_GRID]
    bank, _ = load_public_bank(Path(__file__).resolve().parent.parent / "data")
    wine = [dict(base.BASE, r=11, epsilon=e, family="wine_reference",
                 sweeps=["n" if e == 1 else "epsilon"]) for e in (1.0, .005)]
    base.run_cases(synthetic, data, seed, trials)
    base.run_cases(wine, data, seed, trials, bank=bank)
    rows = []
    for config, reference in [(c, None) for c in synthetic] + [(c, bank) for c in wine]:
        n, r, mu, eta, eps = (config[k] for k in ("n", "r", "mu", "eta", "epsilon"))
        cid = hashlib.sha256(json.dumps(
            {k: v for k, v in config.items() if k != "sweeps"}, sort_keys=True
        ).encode()).hexdigest()[:12]
        sigma = mu * eta * (reference.T @ reference) / len(reference) if reference is not None else base.population(config["family"], r, mu * eta)
        qp = base.model_q(sigma, mu)
        gated = n ** (-.5) + r / (n * eps) > 1
        for trial in range(trials):
            rng = np.random.default_rng(base.seed_for(seed, "sample", cid, trial))
            if reference is None:
                a = base.sample_synthetic(config["family"], n, r, mu * eta, rng)
            else:
                a = math.sqrt(mu * eta) * reference[rng.integers(len(reference), size=n)]
                a *= (2 * rng.integers(2, size=n) - 1)[:, None]
            fitted, residual = base.fit_private_table(a, mu)
            rng = np.random.default_rng(base.seed_for(seed, "recent_em", "em_tight", cid, trial))
            kappa = n * eps / (2 * eta)
            raw, diagnostic = sample_tilted_ball(rng, fitted[1:], kappa, return_diagnostics=True)
            for method, xraw in (("em_tight", raw), ("em_tight_normalized", base.normalize(raw))):
                x = np.zeros(r) if gated else xraw
                rows.append(dict(
                    case_id=cid, dataset="wine_red" if reference is not None else "synthetic",
                    family=config["family"], n=n, r=r, mu=mu, eta=eta, epsilon=eps,
                    sweeps=";".join(config["sweeps"]), trial=trial, method=method,
                    excess_risk=base.risk(qp, x), ungated_risk=base.risk(qp, xraw),
                    coefficient_error=float(np.linalg.norm(fitted[1:] - qp)),
                    population_zero_risk=float(np.linalg.norm(qp)), gated=int(gated),
                    solve_residual=residual, fit_seconds=0.0, total_seconds=0.0,
                    kappa=kappa, lambda_tilt=float(kappa * np.linalg.norm(fitted[1:])),
                    ungated_sampler_seconds=0.0,
                    proposal_count=diagnostic.get("proposal_count", diagnostic.get("proposals", -1)),
                    accepted_count=diagnostic.get("accepted_count", 1),
                    sample_norm=float(np.linalg.norm(xraw)),
                ))
    base.csv_write(data / "em_trials.csv", rows)
    base.csv_write(data / "em_summary.csv", base.summarize(rows))
    return data
