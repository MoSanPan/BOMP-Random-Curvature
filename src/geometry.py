"""Actual response error and block geometry experiments."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
import scipy.linalg as la
import matplotlib

def rng_for(seed, *parts):
    b = json.dumps([seed, *parts]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(b).digest()[:8], 'little'))

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def summary(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, values in groups.items():
        out = dict(zip(keys, key)); out['repetitions'] = len(values)
        for metric in metrics:
            x = np.array([row[metric] for row in values])
            mean = float(x.mean()); se = float(x.std(ddof=1)/np.sqrt(len(x)))
            out.update({metric+'_mean': mean, metric+'_se': se,
                        metric+'_lo95': mean-1.96*se, metric+'_hi95': mean+1.96*se})
        result.append(out)
    return result

def orthogonal(q):
    v = np.ones(q)/np.sqrt(q)
    w = np.eye(q)[0]-v; w /= la.norm(w)
    return np.eye(q)-2*np.outer(w, w)

def response_trial(q, n, model, rng):
    # Sufficient statistics exactly reproduce independent fair record bits.
    v = np.ones(q)/np.sqrt(q)
    if model == 'diagonal':
        diag = rng.binomial(n, .5, q)/n
        response = v/(1+diag)
        directional = la.norm((diag-.5)*v/1.5)
        matrix = la.norm(diag-.5)
    else:
        n1 = int(rng.binomial(n, .5))
        d0 = rng.binomial(n-n1, .5, q)/n
        d1 = rng.binomial(n1, .5, q)/n
        U = orthogonal(q)
        H = np.diag(d0)+(U*d1)@U.T
        response = la.solve(np.eye(q)+H, v, assume_a='pos')
        directional = la.norm((H-.5*np.eye(q))@v/1.5)
        matrix = la.norm(H-.5*np.eye(q), 'fro')
    # c=0, B=v, X=[-1,1]: the supremum is attained at |x|=1.
    error = float(la.norm(response-v/1.5))
    return dict(response_error=error, scaled_response=error*np.sqrt(n),
                directional_fluctuation=float(directional), matrix_fluctuation=float(matrix),
                directional_bound=1/(3*np.sqrt(n)))

def project_blocks(v, ell, d):
    w = v.reshape(ell, d)
    return (w/np.maximum(1., ell*la.norm(w, axis=1))[:, None]).ravel()

def radial_noise(rng, count, dim, scale):
    g = rng.normal(size=(count, dim))
    g /= la.norm(g, axis=1)[:, None]
    return g*rng.gamma(dim, scale, size=(count, 1))

def block_population(ell, d, beta, mu=1., eta=.3):
    r = ell*d; b = eta*mu/2; s = b/(mu+b)
    v = np.full(r, beta/(ell*np.sqrt(d)))
    conditional = ((1-beta**2)*np.eye(d)+beta**2*np.ones((d,d)))/d
    moment = np.kron(np.eye(ell), conditional)/ell
    H = b*np.block([[np.ones((1,1)),v[None,:]], [v[:,None],moment]])
    Q = mu*np.eye(r+1)+H
    # Evaluator matrix maps mean estimation error to exact population risk.
    transform = la.solve(Q, np.eye(r+1)[:,1:], assume_a='pos')*(mu*s)
    return v, Q, transform, s

def block_sample_mean(rng, ell, d, n, beta):
    counts = rng.multinomial(n, np.full(ell, 1/ell))
    # Conditional coordinate signs are independent. This is the exact law
    # of the sample mean, with no Gaussian or fixed-count approximation.
    plus = rng.binomial(counts[:,None], (1+beta)/2, size=(ell,d))
    return ((2*plus-counts[:,None])/(n*np.sqrt(d))).ravel()

def centered_check(ell, d, seed):
    r=ell*d; mu=1.; h=.3; b=h/2
    alpha=mu/(mu+b); beta=mu/(mu+b/r)
    rng=rng_for(seed,'centered',ell,d)
    xis=[]; js=[]
    for _ in range(100):
        u=np.zeros(r); j=rng.integers(ell)
        u[j*d:(j+1)*d]=(2*rng.integers(2,size=d)-1)/np.sqrt(d)
        # Direct moment matrices, independently of the simplified exact norms.
        a=np.r_[1.,u]; H=b*np.outer(a,a)
        D=H-b*np.diag(np.r_[1.,np.full(r,1/r)])
        xi=-alpha*D[:,0]; J=beta*D[:,1:]
        xis.append(float(xi@xi)); js.append(float(np.sum(J*J)))
    measured=(np.sqrt(np.mean(xis))+np.sqrt(np.mean(js)))/mu
    exact=b*(alpha+beta*np.sqrt(2-1/r))/mu
    assert abs(measured-exact)<1e-12
    return dict(r=r,ell=ell,d=d,mu=mu,eta=h/mu,sigma_xi=np.sqrt(np.mean(xis)),
                sigma_J=np.sqrt(np.mean(js)),VQ_measured=measured,VQ_exact=exact)

def checks(seed):
    rng=rng_for(seed,'independent_checks')
    worst=0.; sensitivity=0.
    for ell,d in [(1,4),(2,2),(4,1)]:
        v,Q,T,s=block_population(ell,d,.5)
        # Independent direct lower optimization and population upper loss.
        for _ in range(30):
            est=project_blocks(rng.normal(size=ell*d),ell,d)
            x=s*est; rhs=np.r_[1.,x]
            actual=la.norm(la.solve(Q,rhs)-np.r_[1/1.15,np.zeros(ell*d)])
            fast=la.norm(T@(est-v))
            worst=max(worst,abs(actual-fast))
            assert abs(actual-fast)<1e-12
            assert la.norm(est.reshape(ell,d),axis=1).max()<=1/ell+1e-12
            # Full-domain replacement sensitivity, including different blocks.
            u=np.zeros((ell,d)); u2=np.zeros((ell,d))
            for target in [u,u2]:
                j=rng.integers(ell); z=rng.normal(size=d); z/=max(1.,la.norm(z))
                target[j]=z
            sensitivity=max(sensitivity,float(la.norm(u-u2,axis=1).sum()))
            assert la.norm(u-u2,axis=1).sum()<=2+1e-12
    for q in [8,16]:
        U=orthogonal(q)
        assert np.allclose(U.T@U,np.eye(q))
        assert np.allclose(U[:,0],np.ones(q)/np.sqrt(q))
        n=13; basis=rng.integers(2,size=n); bits=rng.integers(2,size=(n,q))
        direct=sum((np.diag(z) if k==0 else (U*z)@U.T) for k,z in zip(basis,bits))/n
        aggregated=np.diag(bits[basis==0].sum(axis=0)/n)+(U*(bits[basis==1].sum(axis=0)/n))@U.T
        assert np.allclose(direct,aggregated)
    W=radial_noise(rng,100000,4,2/4096)
    empirical=float(np.mean(np.sum(W*W,axis=1)))
    expected=4*4*5/4096**2
    assert abs(empirical/expected-1)<.02
    return dict(max_population_risk_identity_error=worst,max_checked_replacement_norm=sensitivity,
                radial_second_moment=empirical,radial_second_moment_exact=expected,
                sufficient_statistics_matrix_check=True,
                note='Numerical implementation checks, not a proof or empirical certification of DP.')


def run(out, seed=42, repetitions=100):
    out.mkdir(parents=True, exist_ok=True)
    check=checks(seed)
    response=[]
    for sweep,configs in [('q',[(q,1024) for q in [8,16,32,64,128]]),
                           ('n',[(64,n) for n in [128,512,2048,8192,32768]])]:
        for q,n in configs:
            for model in ['diagonal','two_basis']:
                for trial in range(repetitions):
                    result=response_trial(q,n,model,rng_for(seed,'response',sweep,q,n,model,trial))
                    response.append(dict(sweep=sweep,q=q,n=n,model=model,trial=trial,**result))
    print('Response trials complete:',len(response),flush=True)
    block=[]; centered=[]
    r=256; n=8192; eps=.25; eta=.3; beta=.5
    for ell in [1,4,16,64,256]:
        d=r//ell; v,Q,T,s=block_population(ell,d,beta,eta=eta)
        centered.append(centered_check(ell,d,seed))
        for trial in range(repetitions):
            mean=block_sample_mean(rng_for(seed,'data',ell,trial),ell,d,n,beta)
            # Valid paired coupling; at ell=1 the mechanisms coincide exactly.
            nrng=rng_for(seed,'paired_noise',ell,trial)
            g=nrng.normal(size=(ell,d)); lengths=la.norm(g,axis=1)
            radii=nrng.gamma(d,2/(n*eps),size=ell)
            noise=(g/lengths[:,None]*radii[:,None]).ravel()
            ambient=g.ravel()/la.norm(g)*radii.sum()
            for method,W in [('block',noise),('ambient',ambient),('nonprivate',np.zeros(r)),('public_zero',None)]:
                estimate=np.zeros(r) if W is None else project_blocks(mean+W,ell,d)
                ungated=float(la.norm(T@(estimate-v)))
                factor=ell*d*(d+1) if method=='block' else r*(r+1)
                gate=(method in ['block','ambient'] and n**(-.5)+2*np.sqrt(factor)/(n*eps)>1/np.sqrt(ell))
                if gate: estimate=np.zeros(r)
                risk=float(la.norm(T@(estimate-v)))
                block.append(dict(r=r,ell=ell,d=d,n=n,epsilon=eps,eta=eta,beta=beta,trial=trial,
                                  method=method,excess_risk=risk,normalized_risk=risk/(s/np.sqrt(ell)),
                                  ungated_risk=ungated,gated=int(gate),
                                  mean_error=float(la.norm(estimate-v)),
                                  noise_norm=0. if W is None else float(la.norm(W)),
                                  data_error=float(la.norm(mean-v)),
                                  private_term=d*np.sqrt(ell)/(n*eps),saturation=1/np.sqrt(ell)))
        print('Block geometry complete:',ell,d,flush=True)
    rs=summary(response,['sweep','q','n','model'],['response_error','scaled_response'])
    bs=summary(block,['r','ell','d','n','epsilon','eta','beta','method'],
               ['excess_risk','normalized_risk','ungated_risk','gated','mean_error','noise_norm','data_error'])
    write_csv(out/'response_trials.csv',response); write_csv(out/'response_summary.csv',rs)
    write_csv(out/'block_trials.csv',block); write_csv(out/'block_summary.csv',bs)
    write_csv(out/'centered_moments.csv',centered)
    manifest=dict(seed=seed,repetitions=repetitions,response_trials=len(response),
                  block_method_rows=len(block),checks=check,python=platform.python_version(),
                  numpy=np.__version__,scipy=__import__('scipy').__version__,matplotlib=matplotlib.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Pointwise Monte Carlo utility on specified laws; no empirical minimax or DP certification.',
                  mechanism='Both private comparisons use projection and their public-bound zero fallback; continuous laws give pure DP in real arithmetic.',
                  noise_density='Block: exp(-n*epsilon*sum_j ||w_j||/2); ambient: exp(-n*epsilon*||w||/2).')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
