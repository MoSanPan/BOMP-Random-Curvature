"""Rejection sampling for exp(kappa * q.T @ x) on the Euclidean unit ball.

The rejection construction is exact in real arithmetic. This NumPy implementation
is a numerical experiment implementation, not a finite-arithmetic DP certificate.
No dataset operations, privacy calibrations, or baseline names are defined here.
"""
from __future__ import annotations
import math
import numpy as np


def _parameters(r: int, lam: float):
    """Return a, beta, v0, gamma scale. Valid for integer r >= 2."""
    a = (r + 1.0) / 2.0
    beta = a - 1.0
    # The quadratic discriminant is (2*lam-1)^2 + r^2-1.
    # The reciprocal expression avoids cancellation in the smaller root.
    if lam > 1.0:
        inv = 0.5 / lam
        scaled_disc = math.hypot(1.0 - inv, math.sqrt(r*r-1.0)*inv)
        v0 = ((r+1.0)*inv) / (1.0 + r*inv + scaled_disc)
        scale = inv / (1.0 + beta*inv/(1.0-v0))
    else:
        disc = math.hypot(2.0*lam-1.0, math.sqrt(r*r-1.0))
        v0 = (r+1.0)/(2.0*lam+r+disc)
        scale = 1.0/(2.0*lam+beta/(1.0-v0))
    return a, beta, v0, scale


def _log_acceptance(v, beta: float, v0: float):
    # log(1-v) - log(1-v0) = log(1-d), d=(v-v0)/(1-v0).
    # log(1-d)+d <= 0, including negative d.
    d = (v-v0)/(1.0-v0)
    value = beta*(np.log1p(-d)+d)
    return np.minimum(value, 0.0)


def _uniform_ball(rng: np.random.Generator, size: int, r: int):
    z = rng.normal(size=(size,r))
    norms = np.linalg.norm(z, axis=1)
    # A Gaussian zero vector has probability zero in the ideal construction.
    # Retrying handles that event in the finite RNG representation.
    bad = norms == 0
    while np.any(bad):
        z[bad] = rng.normal(size=(int(np.sum(bad)),r))
        norms = np.linalg.norm(z, axis=1)
        bad = norms == 0
    return z/norms[:,None] * rng.random(size)[:,None]**(1.0/r)


def sample_tilted_ball(
    rng: np.random.Generator,
    qhat,
    kappa: float,
    size: int | None = None,
    return_diagnostics: bool = False,
):
    """Sample X with Lebesgue density proportional to exp(kappa*qhat.T X).

    qhat: one-dimensional nonempty coefficient vector.
    kappa: finite nonnegative temperature.
    size=None: return one vector; integer size: return shape (size,r).
    return_diagnostics=True: return (X, dict) instead of X.

    Each Gamma rejection batch may produce more accepted values than needed;
    unused values are discarded. accepted_count counts all acceptances in the
    batches, while output_count is the number of returned vectors.
    """
    q = np.asarray(qhat,dtype=float)
    if q.ndim != 1 or q.size < 1 or not np.all(np.isfinite(q)):
        raise ValueError("qhat must be a finite, nonempty one-dimensional vector")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    one = size is None
    count = 1 if one else int(size)
    if count < 1 or (not one and count != size):
        raise ValueError("size must be a positive integer or None")
    r = q.size
    qnorm = float(np.linalg.norm(q))
    lam = float(kappa*qnorm)
    if not np.isfinite(lam):
        raise ValueError("kappa*norm(qhat) overflowed")
    info = {"dimension":int(r),"lambda":lam,"output_count":count,
            "proposal_count":0,"accepted_count":0,"construction":"gamma-envelope"}

    if lam == 0.0:
        x = _uniform_ball(rng,count,r)
        info.update(construction="uniform-ball",proposal_count=count,
                    accepted_count=count,acceptance_rate=1.0)
    else:
        if r == 1:
            # V=(1-T)/2 has a truncated exponential density on [0,1].
            mass = -math.expm1(-2.0*lam) if lam < 350.0 else 1.0
            v = -np.log1p(-rng.random(count)*mass)*(0.5/lam)
            info.update(construction="truncated-exponential",
                        proposal_count=count,accepted_count=count,
                        acceptance_rate=1.0)
        else:
            a,beta,v0,scale = _parameters(r,lam)
            values = []
            left = count
            proposals = 0
            accepted = 0
            while left:
                batch = max(left,32)
                candidate = rng.gamma(a,scale,size=batch)
                inside = (candidate > 0.0)&(candidate < 1.0)
                logacc = np.full(batch,-np.inf)
                logacc[inside] = _log_acceptance(candidate[inside],beta,v0)
                keep = inside & (np.log(rng.random(batch)) <= logacc)
                good = candidate[keep]
                proposals += batch
                accepted += good.size
                take = min(left,good.size)
                if take:
                    values.append(good[:take])
                    left -= take
            v = np.concatenate(values)
            info.update(proposal_count=proposals,accepted_count=accepted,
                        acceptance_rate=accepted/proposals,a=a,
                        tangent_point=v0,gamma_scale=scale)
        t = 1.0-2.0*v
        coordinates = np.zeros((count,r))
        coordinates[:,0] = t
        if r > 1:
            # Conditional cross-section: a uniform (r-1)-ball. Computing
            # 1-t^2 as 4*v*(1-v) is stable when lambda is large.
            perpendicular = _uniform_ball(rng,count,r-1)
            coordinates[:,1:] = perpendicular*np.sqrt(4.0*v*(1.0-v))[:,None]
        direction = q/qnorm
        # Stable orthogonal transformation mapping e0 to direction.
        sign = 1.0 if direction[0] >= 0 else -1.0
        householder = sign*direction.copy()
        householder[0] += 1.0
        denom = float(householder@householder)
        x = -sign*(coordinates-2.0*np.outer(
            coordinates@householder,householder)/denom)
    if one:
        x=x[0]
    return (x,info) if return_diagnostics else x


def projection_reference(r: int, lam: float, quadrature_order: int = 512):
    """Independent deterministic quadrature for T and T^2.

    Integrates the Gamma-proposal coordinate with Gauss-Legendre quadrature.
    The discarded upper tail is far beyond a unit-rate Gamma mean; it is
    bounded in the diagnostics by a standard Chernoff bound.
    """
    if lam == 0:
        return {"mean_t":0.0,"mean_t2":1.0/(r+2.0),
                "mean_norm2":r/(r+2.0),"acceptance":1.0,
                "gamma_tail_bound":0.0}
    if r < 2:
        raise ValueError("quadrature reference uses r>=2")
    a,beta,v0,scale = _parameters(r,lam)
    limit=min(1.0/scale,max(100.0,a+30.0*math.sqrt(a)))
    z,weights=np.polynomial.legendre.leggauss(quadrature_order)
    # W=s^2 removes the fractional Gamma density endpoint at W=0:
    # W^(a-1) dW = 2*s^(2a-1) ds, and 2a-1=r is an integer.
    s=(z+1.0)*math.sqrt(limit)/2.0
    weights=weights*math.sqrt(limit)/2.0
    w=s*s
    v=w*scale
    logbase=r*np.log(s)-w+math.log(2.0)-math.lgamma(a)
    logdensity=logbase+_log_acceptance(v,beta,v0)
    mass=np.exp(logdensity)
    total=float(weights@mass)
    t=1.0-2.0*v
    et=float(weights@(mass*t))/total
    et2=float(weights@(mass*t*t))/total
    # This tail is for proposal W~Gamma(a,1), so also bounds the accepted
    # (unnormalized) integral. It vanishes when the finite support is used.
    tail=0.0 if limit==1.0/scale else math.exp(-limit+a+a*math.log(limit/a))
    return {"mean_t":et,"mean_t2":et2,
            "mean_norm2":(r-1.0)/(r+1.0)+2.0*et2/(r+1.0),
            "acceptance":total,"gamma_tail_bound":tail}


if __name__ == "__main__":
    print("Import sample_tilted_ball; run validate_tilted_ball.py for diagnostics.")

