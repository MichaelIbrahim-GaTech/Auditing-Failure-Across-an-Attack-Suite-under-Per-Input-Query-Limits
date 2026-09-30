"""Finite-suite audit confidence projection. No fitted mixing family."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import hypergeom, beta
from scipy.optimize import linprog


def sampling_matrix(d, k):
    return hypergeom.pmf(np.arange(k+1)[:,None], d, np.arange(d+1)[None,:], k)


def cp_interval(successes, m, alpha):
    successes=np.asarray(successes)
    lo=np.where(successes==0, 0., beta.ppf(alpha/2, successes, m-successes+1))
    hi=np.where(successes==m, 1., beta.ppf(1-alpha/2, successes+1, m-successes))
    return np.nan_to_num(lo,nan=0), np.nan_to_num(hi,nan=1)


def dual_lower_bound(c,A,b,res):
    """Repair LP dual feasibility; this does not certify beta inverse arithmetic."""
    lam=np.minimum(res.ineqlin.marginals, 0.)
    z=np.min(c-A.T@lam)
    # Recompute in extended precision and subtract a rounding allowance.
    t=np.longdouble
    residual=np.min(c.astype(t)-A.astype(t).T@lam.astype(t)-t(z))
    z=t(z)+min(t(0),residual)
    ans=b.astype(t)@lam.astype(t)+z
    allowance=1e-10*(1+float(np.sum(np.abs(b*lam)))+abs(float(z)))
    return float(ans)-allowance


def confidence_interval(hist,d,k,delta=.05):
    hist=np.asarray(hist,dtype=int); m=int(hist.sum())
    if len(hist)!=k+1 or m<1: raise ValueError('Invalid histogram')
    if k==d:
        lo,hi=cp_interval(m-int(hist[0]),m,delta)
        return dict(lower=float(lo),upper=float(hi),status='full_suite',max_primal_residual=0.)
    H=sampling_matrix(d,k)
    # Half the error budget for all categories, half for detection.
    lo,hi=cp_interval(hist,m,delta/(2*(k+1)))
    dl,du=cp_interval(m-int(hist[0]),m,delta/2)
    det=1-H[0]
    A=np.vstack([H,-H,det,-det])
    b=np.r_[hi,-lo,float(du),-float(dl)]
    # Outward numerical padding of probability constraints.
    b=b+1e-12
    h=np.r_[0.,np.ones(d)]
    values=[]; residuals=[]
    for c in (h,-h):
        res=linprog(c,A_ub=A,b_ub=b,A_eq=np.ones((1,d+1)),b_eq=[1.],bounds=(0,None),method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        if not res.success:
            return dict(lower=0.,upper=1.,status='uninformative_solver_failure',message=res.message)
        values.append(dual_lower_bound(c,A,b,res))
        residuals.append(max(0.,float(np.max(A@res.x-b)),abs(float(res.x.sum()-1)),float(-np.min(res.x))))
    return dict(lower=max(0.,values[0]),upper=min(1.,-values[1]),status='ok',max_primal_residual=max(residuals))


def detection_interval(hist,d,k,delta=.05):
    m=int(np.sum(hist)); lo,hi=cp_interval(m-int(hist[0]),m,delta)
    return float(lo),min(1.,float(hi)*d/k)


def smoke():
    H=sampling_matrix(8,3)
    assert np.max(abs(H.sum(axis=0)-1))<1e-12
    full=confidence_interval([80,0,0,0,0,0,0,0,20],8,8)
    expected=cp_interval(20,100,.05)
    assert np.allclose([full['lower'],full['upper']],expected)
    # Population exactly one vulnerable coordinate vs all-coordinate/zero mixture:
    # at k=1, same count law and union risks 1 and 1/d.
    H=sampling_matrix(64,1)
    a=np.zeros(65); a[1]=1
    b=np.zeros(65); b[0]=63/64; b[64]=1/64
    assert np.allclose(H@a,H@b)
    assert abs((a[1:].sum()-b[1:].sum())-63/64)<1e-12
    return {'sampling_matrix':'passed','full_suite_reduction':'passed','exact_k1_pair':'passed'}

if __name__=='__main__':
    print(json.dumps(smoke(),indent=2))
