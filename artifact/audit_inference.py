"""Population oracle and bias/range-regularized linear audit inference.

The linear fit uses only d,k,m, never the population mu or the sampled data.
Its bias bound and range are recomputed from the returned coefficients.
"""
from functools import lru_cache
import warnings
import numpy as np
from scipy.optimize import linprog
from confidence import sampling_matrix

@lru_cache(None)
def fit_linear_estimator(d,k,m):
    d,k,m=int(d),int(k),int(m)
    if not 1<=k<=d or m<1: raise ValueError('Require 1<=k<=d and m>=1')
    H=sampling_matrix(d,k);h=np.r_[0.,np.ones(d)]
    n=k+1
    # Variables g_0,...,g_k,b,l,u; |H^T g-h|<=b and l<=g_j<=u.
    A=np.zeros((2*(d+1)+2*n,n+3));b=np.r_[h,-h,np.zeros(2*n)]
    A[:d+1,:n]=H.T;A[:d+1,n]=-1
    A[d+1:2*(d+1),:n]=-H.T;A[d+1:2*(d+1),n]=-1
    offset=2*(d+1)
    A[offset:offset+n,:n]=np.eye(n);A[offset:offset+n,n+2]=-1
    A[offset+n:,:n]=-np.eye(n);A[offset+n:,n+1]=1
    lam=1/(2*np.sqrt(m));c=np.r_[np.zeros(n),1.,-lam,lam]
    res=linprog(c,A_ub=A,b_ub=b,bounds=[(None,None)]*n+[(0,None),(None,None),(None,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not res.success: raise RuntimeError(res.message)
    g=res.x[:n]
    # Long-double recomputation plus conservative arithmetic allowance.
    bias=float(np.max(np.abs(H.astype(np.longdouble).T@g.astype(np.longdouble)-h)))
    allowance=2e-12*(1+float(np.max(np.abs(g))))
    bias+=allowance
    ran=float(np.max(g)-np.min(g))
    return dict(d=d,k=k,m=m,g=g,bias_bound=bias,range=ran,
        objective=bias+lam*ran,squared_error_bound=bias*bias+ran*ran/(4*m),
        lp_objective=float(res.fun),max_constraint_residual=max(0.,float(np.max(A@res.x-b))))

def linear_interval(hist,fit,delta=.05):
    hist=np.asarray(hist,dtype=int);m=int(hist.sum())
    if len(hist)!=fit['k']+1 or m!=fit['m']: raise ValueError('Histogram does not match fit')
    raw=float(hist@fit['g']/m)
    radius=float(fit['bias_bound']+fit['range']*np.sqrt(np.log(2/delta)/(2*m)))
    point=float(np.clip(raw,0,1))
    lo=float(np.clip(raw-radius,0,1));hi=float(np.clip(raw+radius,0,1))
    return dict(lower=lo,upper=hi,point=point,raw_point=raw,radius=radius,width=hi-lo)

def linear_population_risk(mu,fit):
    q=sampling_matrix(fit['d'],fit['k'])@np.asarray(mu)
    g=fit['g'];theta=float(1-np.asarray(mu)[0]);mean=float(q@g)
    var=float(max(0.,q@(g-mean)**2)/fit['m'])
    return dict(raw_expectation=mean,raw_bias=mean-theta,raw_variance=var,
                raw_mse=(mean-theta)**2+var,clipped_mse_upper_bound=(mean-theta)**2+var,
                uniform_mse_upper_bound=fit['squared_error_bound'])

def discrete_polynomial_basis(d,k):
    x=np.linspace(-1.,1.,d+1);Q=np.empty((d+1,k+1));Q[:,0]=1/np.sqrt(d+1)
    for j in range(1,k+1):
        z=x*Q[:,j-1]
        for _ in range(2): z-=Q[:,:j]@(Q[:,:j].T@z)
        Q[:,j]=z/np.linalg.norm(z)
    return Q

# Exact finite-grid certificates: floating LP discovers coefficients/supports;
# rational arithmetic verifies the polynomial bound against every integer S.
from fractions import Fraction as F
from math import comb

@lru_cache(None)
def exact_polynomial_rows(d,k):
    rows=[]
    for degree in range(k+1):
        row=[]
        for s in range(d+1):
            value=sum((F((-1)**j*comb(degree,j)*comb(degree+j,j)*comb(s,j),comb(d,j))
                       for j in range(min(degree,s)+1)),F(0))
            row.append(value)
        scale=max(map(abs,row));rows.append(tuple(v/scale for v in row))
    return tuple(rows)

def _exact_dual_bound(c,rows,mu,res):
    n=len(rows)-1
    coeff=[F(float(a))-F(float(b)) for a,b in zip(res.ineqlin.marginals[:n],res.ineqlin.marginals[n:])]
    polynomial=[sum((a*rows[j+1][s] for j,a in enumerate(coeff)),F(0)) for s in range(len(mu))]
    intercept=min(F(int(c[s]))-polynomial[s] for s in range(len(mu)))
    values=[v+intercept for v in polynomial]
    assert all(v<=int(cc) for v,cc in zip(values,c))
    return sum((w*v for w,v in zip(mu,values)),F(0)),coeff,intercept

def _exact_primal_value(res,mu,k):
    support=[int(s) for s in np.flatnonzero(res.x>1e-11)]
    if len(support)>k+1: return None
    ww=[]
    for s in support:
        others=[t for t in support if t!=s]
        den=math_product(s-t for t in others)
        num=sum((w*math_product(t-r for r in others) for t,w in enumerate(mu) if w),F(0))
        ww.append(num/den)
    if any(w<0 for w in ww): return None
    for j in range(k+1):
        if sum((w*s**j for s,w in zip(support,ww)),F(0)) != sum((w*s**j for s,w in enumerate(mu)),F(0)):
            return None
    return dict(value=sum((w for s,w in zip(support,ww) if s>0),F(0)),
                support=support,weights_exact=list(map(str,ww)))

def math_product(values):
    ans=1
    for value in values: ans*=value
    return ans

def oracle_interval(mu,d,k):
    mu=np.asarray(mu,dtype=float);H=sampling_matrix(d,k)
    if len(mu)!=d+1 or min(mu)<0 or abs(mu.sum()-1)>1e-10: raise ValueError('Invalid law')
    # Frozen empirical frequencies admit exact rational reconstruction.
    rm=tuple(F(float(v)).limit_denominator(100000000) for v in mu)
    if sum(rm)!=1 or max(abs(float(v)-w) for v,w in zip(rm,mu))>1e-14:
        raise ValueError('Oracle certificate expects recoverable rational probabilities')
    theta=1-rm[0];max_s=int(np.flatnonzero(mu)[-1]);truth=float(theta)
    if k==d or max_s<k:
        reason='full_suite' if k==d else 'support_max_below_k'
        return dict(d=d,k=k,lower=truth,upper=truth,width=0.,theta=truth,
            lower_exact=str(theta),upper_exact=str(theta),width_upper_exact='0',
            endpoint_certificate_gap=0.,width_lower=0.,certification='exact_singleton_'+reason,
            max_S=max_s,max_audit_law_residual=0.,truth_feasible=True,lp_attempts=0)
    rows=exact_polynomial_rows(d,k);A=np.array([[float(v) for v in row] for row in rows[1:]])
    bb=np.array([float(sum((a*w for a,w in zip(row,rm)),F(0))) for row in rows[1:]])
    h=np.r_[0.,np.ones(d)]
    lower=F(0);upper=F(1);min_primal=theta;max_primal=theta
    diagnostics=[]
    for eps in [1e-8,1e-10,1e-12]:
        for sign in [1,-1]:
            c=sign*h
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r=linprog(c,A_ub=np.r_[A,-A],b_ub=np.r_[bb+eps,-bb+eps],
                    A_eq=np.ones((1,d+1)),b_eq=[1.],bounds=(0,None),method='highs',
                    options={'small_matrix_value':1e-12,'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
            if not r.success:
                diagnostics.append(dict(epsilon=eps,sign=sign,success=False,message=r.message));continue
            bound,coeff,intercept=_exact_dual_bound(c,rows,rm,r)
            primal=_exact_primal_value(r,rm,k)
            if sign==1:
                lower=max(lower,bound)
                if primal is not None: min_primal=min(min_primal,primal['value'])
            else:
                upper=min(upper,-bound)
                if primal is not None: max_primal=max(max_primal,primal['value'])
            diagnostics.append(dict(epsilon=eps,sign=sign,success=True,lp_objective=float(r.fun),
                exact_dual_bound=str(bound),exact_primal_value=None if primal is None else str(primal['value']),
                exact_primal_witness=None if primal is None else {key:val for key,val in primal.items() if key!='value'},
                audit_law_residual=float(np.max(np.abs(H@r.x-H@mu))),
                polynomial_coefficients_exact=list(map(str,coeff)),intercept_exact=str(intercept)))
        if float(min_primal-lower+upper-max_primal)<1e-9:break
    assert lower<=theta<=upper and lower<=min_primal<=max_primal<=upper
    gap=float(min_primal-lower+upper-max_primal)
    if not any(r['success'] for r in diagnostics):raise RuntimeError('No oracle LP solved')
    return dict(d=d,k=k,lower=float(lower),upper=float(upper),width=float(upper-lower),theta=truth,
        lower_exact=str(lower),upper_exact=str(upper),width_upper_exact=str(upper-lower),
        width_lower=float(max_primal-min_primal),endpoint_certificate_gap=gap,
        certification='exact_rational_endpoint_certificates' if gap<1e-9 else 'exact_rational_outer_bounds',
        max_S=max_s,max_audit_law_residual=max(r.get('audit_law_residual',0) for r in diagnostics),
        truth_feasible=True,lp_attempts=len(diagnostics),diagnostics=diagnostics)
