"""Numerical diagnostics for the exact-real-arithmetic rejection construction."""
from pathlib import Path
import math,json,hashlib,time
from functools import lru_cache
import numpy as np
from tilted_ball_sampler import sample_tilted_ball,projection_reference,_parameters,_log_acceptance
HERE=Path(__file__).resolve().parent

@lru_cache(None)
def nodes(order):
    return np.polynomial.legendre.leggauss(order)

def direct_t_reference(r,lam,order=512):
    t,w=nodes(order)
    beta=(r-1)/2
    logweight=lam*(t-1)+beta*np.log1p(-t*t)
    mass=np.exp(logweight-np.max(logweight))
    total=w@mass
    et=float(w@(mass*t)/total)
    et2=float(w@(mass*t*t)/total)
    return dict(mean_t=et,mean_t2=et2,
                mean_norm2=(r-1)/(r+1)+2*et2/(r+1))

def run(out):
    out.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    rng=np.random.default_rng(44)
    rows=[];checks=[]
    max_z=0.;max_norm_violation=0.;max_reference_difference=0.
    for r in [1,2,4,11,16,64]:
        for lam in [0.,.01,.1,1.,10.,100.,1000.,10000.]:
            q=rng.normal(size=r);q=q/np.linalg.norm(q)
            # Direct T quadrature has a different density formula from the
            # implementation's tangent-Gamma envelope.
            direct=direct_t_reference(r,lam,1024)
            if r>1:
                ref=projection_reference(r,lam,512)
                if lam<=1000:
                    difference=max(abs(ref[k]-direct[k]) for k in
                                   ["mean_t","mean_t2","mean_norm2"])
                    max_reference_difference=max(max_reference_difference,difference)
                    # r=2 has an endpoint square root; the direct quadrature
                    # endpoint error is larger than in r>=4 but remains small.
                    checks.append({"kind":"independent_quadrature","r":r,"lambda":lam,
                                   "max_difference":difference,
                                   "passed":difference < (3e-6 if r==2 else 3e-8)})
            else:
                ref=direct
            x,info=sample_tilted_ball(rng,q,lam,size=30000,return_diagnostics=True)
            t=x@q
            norm2=np.sum(x*x,axis=1)
            max_norm_violation=max(max_norm_violation,float(np.max(norm2)-1))
            moments={"mean_t":t,"mean_t2":t*t,"mean_norm2":norm2}
            stats={}
            passed=True
            for name,v in moments.items():
                mean=float(np.mean(v));se=float(np.std(v,ddof=1)/np.sqrt(len(v)))
                error=mean-ref[name]
                z=abs(error)/max(se,1e-15)
                max_z=max(max_z,z)
                ok=abs(error)<=6.5*se+3e-9
                passed &= ok
                stats[name]={"observed":mean,"reference":ref[name],
                             "mcse":se,"z":z,"passed":bool(ok)}
            if r>1 and lam>0:
                observed=info["acceptance_rate"]
                # Rejection batch acceptance is an iid Bernoulli diagnostic.
                p=ref["acceptance"]
                se=math.sqrt(max(p*(1-p),1e-16)/info["proposal_count"])
                ok=abs(observed-p)<=6.5*se+2e-7
                passed &= ok
                stats["acceptance"]={"observed":observed,"reference":p,
                                      "mcse":se,"passed":bool(ok)}
            ok=float(np.max(norm2))<=1+1e-12
            passed &= ok
            rows.append({"r":r,"lambda":lam,"samples":len(x),"moments":stats,
                         "max_norm2":float(np.max(norm2)),
                         "sampler":info,"passed":bool(passed)})
    # Numerical envelope diagnostic, including extremely small and large
    # tilts; this supplements, rather than replaces, the concavity proof.
    envelope=[]
    for r in [2,4,11,16,64]:
        for lam in [1e-12,1e-6,.01,.1,1.,10.,100.,1e4,1e12,1e100]:
            a,beta,v0,scale=_parameters(r,lam)
            grid=np.r_[np.linspace(1e-12,1-1e-12,5000),v0]
            raw=beta*(np.log1p(-(grid-v0)/(1-v0))+(grid-v0)/(1-v0))
            mx=float(np.max(raw))
            valid=0<v0<1 and scale>0 and mx<1e-11
            envelope.append(dict(r=r,lambda_=lam,tangent=v0,scale=scale,
                                 maximum_log_acceptance=mx,passed=bool(valid)))
    # Explicit orthogonal-direction edge cases.
    rotation=[]
    for q in [np.array([1.,0.,0.,0.]),np.array([-1.,0.,0.,0.]),
              np.array([1e-12,1.,0.,0.])]:
        q=q/np.linalg.norm(q)
        x=sample_tilted_ball(rng,q,25.,size=30000)
        perpendicular=x-(x@q)[:,None]*q
        means=np.mean(perpendicular,axis=0)
        se=np.std(perpendicular,axis=0,ddof=1)/np.sqrt(len(x))
        ok=bool(np.all(np.abs(means)<=7*se+1e-12))
        rotation.append(dict(q=q.tolist(),perpendicular_mean=means.tolist(),
                             mcse=se.tolist(),passed=ok))
    passed=all(r["passed"] for r in rows+checks+envelope+rotation)
    report={"passed":passed,"scope":"Numerical diagnostics, not machine-level DP certification",
            "seed":44,"samples_per_case":30000,"cases":len(rows),
            "max_standardized_moment_error":max_z,
            "max_norm_squared_violation":max_norm_violation,
            "max_independent_quadrature_difference":max_reference_difference,
            "moments":rows,"quadrature_checks":checks,
            "envelope_checks":envelope,"rotation_checks":rotation,
            "sampler_sha256":hashlib.sha256((HERE/"tilted_ball_sampler.py").read_bytes()).hexdigest(),
            "elapsed_seconds":time.perf_counter()-started}
    (out/"sampler_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in
          ["moments","quadrature_checks","envelope_checks","rotation_checks"]}),flush=True)
    if not passed:
        failed=[r for r in rows+checks+envelope+rotation if not r["passed"]]
        print(json.dumps({"failed":failed},indent=2),flush=True)
        raise SystemExit(1)



