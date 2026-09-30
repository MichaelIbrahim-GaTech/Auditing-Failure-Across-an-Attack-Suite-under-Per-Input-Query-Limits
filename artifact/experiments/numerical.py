"""Exact finite-suite identification widths, with no external data.
Run: python numerical.py ; dependencies: numpy, scipy, matplotlib.

S is the number of successful attacks among d deterministic attacks.
Querying k uniformly sampled distinct attacks gives J ~ Hypergeom(d,S,k).
LP discovers supports for laws of S maximizing their difference in Pr(S>0)
while having identical audit laws. Exact Fraction arithmetic then verifies
all moment identities, all audit probabilities, and a degree-k polynomial
error bound at EVERY integer s=0,...,d. LP tolerances do not certify results;
exact rational primal and dual certificates do.
"""
from pathlib import Path
from fractions import Fraction as F
import json, math, platform, time
import numpy as np
import scipy
from numpy.polynomial import chebyshev as C
from scipy.optimize import linprog
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
DS=[16,32,64,128]
KS=[1,2,4,8,16,32]

def divided_differences(x,y):
    a=list(y)
    for j in range(1,len(x)):
        for i in range(len(x)-1,j-1,-1):
            a[i]=(a[i]-a[i-1])/F(x[i]-x[i-j])
    return a

def newton_evaluate(x,a,t):
    value=a[-1]
    for i in reversed(range(len(a)-1)): value=value*(t-x[i])+a[i]
    return value

def choose(n,r):
    return math.comb(n,r) if 0<=r<=n else 0

def hg_fraction(d,s,k,j):
    return F(choose(s,j)*choose(d-s,k-j),choose(d,k))

def exact_certificate(d,k,support):
    assert len(support)==k+2 and support[0]==0
    raw=[F(1,math.prod(s-t for t in support if t!=s)) for s in support]
    mass=sum(w for w in raw if w>0)
    signed=[w/mass for w in raw]
    sep=sum(w for s,w in zip(support,signed) if s>0)
    if sep<0: signed=[-w for w in signed]; sep=-sep
    wa=[max(w,F(0)) for w in signed]
    wb=[max(-w,F(0)) for w in signed]
    assert sum(wa)==sum(wb)==1
    residuals=[sum(w*s**j for s,w in zip(support,signed)) for j in range(k+1)]
    assert all(r==0 for r in residuals)
    e=sep/2
    vals=[F(int(s>0))-e*(1 if w>0 else -1) for s,w in zip(support,signed)]
    coeff=divided_differences(support[:k+1],vals[:k+1])
    errors=[F(int(s>0))-newton_evaluate(support,coeff,s) for s in range(d+1)]
    assert max(abs(v) for v in errors)==e
    assert all(abs(v)<=e for v in errors)
    ca=[sum(w*hg_fraction(d,s,k,j) for s,w in zip(support,wa)) for j in range(k+1)]
    cb=[sum(w*hg_fraction(d,s,k,j) for s,w in zip(support,wb)) for j in range(k+1)]
    assert ca==cb and sum(ca)==1
    h=np.array([int(s>0) for s in support])
    aa,bb=np.array(list(map(float,wa))),np.array(list(map(float,wb)))
    H=np.array([[hypergeom.pmf(j,d,s,k) for s in support] for j in range(k+1)])
    return dict(identification_width=float(sep),identification_width_exact=str(sep),
        minimax_uniform_error_exact=str(e),support=support,
        weights_A=list(map(float,wa)),weights_B=list(map(float,wb)),
        weights_A_exact=list(map(str,wa)),weights_B_exact=list(map(str,wb)),
        target_A=float(aa@h),target_B=float(bb@h),
        target_A_exact=str(sum(w for s,w in zip(support,wa) if s>0)),
        target_B_exact=str(sum(w for s,w in zip(support,wb) if s>0)),
        common_count_distribution=list(map(float,ca)),common_count_distribution_exact=list(map(str,ca)),
        exact_count_probability_discrepancy='0',
        floating_count_probability_discrepancy=float(np.max(np.abs(H@aa-H@bb))),
        exact_moment_discrepancy='0',exact_polynomial_max_error=str(max(abs(v) for v in errors)),
        exact_certificate_passed=True,newton_interpolation_nodes=support[:k+1],
        newton_coefficients_exact=list(map(str,coeff)),
        polynomial_errors_at_integer_grid=list(map(float,errors)),
        polynomial_errors_at_integer_grid_exact=list(map(str,errors)))

def one_case(d,k):
    started=time.time()
    if k==d:
        return dict(d=d,k=k,identification_width=0.,identification_width_exact='0',
            minimax_uniform_error_exact='0',exact_certificate_passed=True,
            certificate_type='Full observation J=S; exact degree-d interpolation',
            exact_count_probability_discrepancy='0',floating_count_probability_discrepancy=0.,seconds=time.time()-started)
    n=d+1; h=np.r_[0.,np.ones(d)]
    Q=np.linalg.qr(C.chebvander(np.linspace(-1,1,n),k),mode='reduced')[0]
    eq=np.vstack([np.hstack([Q.T,-Q.T]),np.r_[np.ones(n),np.zeros(n)]])
    res=linprog(np.r_[-h,h],A_eq=eq,b_eq=np.r_[np.zeros(k+1),1.],bounds=(0,None),method='highs',
        options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not res.success: raise RuntimeError(res.message)
    support=[int(s) for s in np.flatnonzero(np.abs(res.x[:n]-res.x[n:])>1e-12)]
    cert=exact_certificate(d,k,support)
    return dict(d=d,k=k,**cert,certificate_type='Exact rational primal and dual certificates',
        lp_width=float(-res.fun),lp_minus_exact_width=float(-res.fun-cert['identification_width']),seconds=time.time()-started)

def figures(cases):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    colors=['#4477AA','#228833','#CC9944','#AA3377']; markers=['o','s','^','D']
    fig,axes=plt.subplots(1,2,figsize=(7.,3.05))
    for d,col,mark in zip(DS,colors,markers):
        sub=[r for r in cases if r['d']==d]; kk=np.array([r['k'] for r in sub]); yy=[r['identification_width'] for r in sub]
        axes[0].plot(kk,yy,marker=mark,color=col,label=f'd = {d}',lw=1.6,ms=4)
        axes[1].plot(kk/np.sqrt(d),yy,marker=mark,color=col,label=f'd = {d}',lw=1.6,ms=4)
    axes[0].set_xscale('log',base=2); axes[0].set_xticks(KS,list(map(str,KS)))
    axes[0].set(xlabel='Attacks queried per input, k',ylabel='Worst-case identification width',ylim=(-.025,1.025))
    axes[1].set(xlabel=r'Normalized audit budget, $k/\sqrt{d}$',ylim=(-.025,1.025))
    axes[1].set_xscale('log',base=2)
    axes[1].set_xticks([.125,.25,.5,1,2,4,8],['1/8','1/4','1/2','1','2','4','8'])
    for ax in axes: ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,ncol=2,loc='lower left',fontsize=8)
    fig.tight_layout(w_pad=1.4)
    for ext in ['pdf','png']: fig.savefig(OUT/f'finite_suite_width.{ext}',dpi=220)
    plt.close(fig)
    row=next(r for r in cases if r['d']==64 and r['k']==4)
    fig,axes=plt.subplots(1,2,figsize=(7.,3.05)); support=np.array(row['support'])
    positions=np.arange(len(support))
    axes[0].bar(positions-.18,row['weights_A'],width=.36,color=colors[0],label='Law A')
    axes[0].bar(positions+.18,row['weights_B'],width=.36,color=colors[-1],label='Law B')
    axes[0].set(xlabel='Successful attacks per input, S',ylabel='Probability mass',ylim=(0,1.03))
    axes[0].set_xticks(positions,list(map(str,support))); axes[0].legend(frameon=False)
    xx=np.arange(5); axes[1].bar(xx,row['common_count_distribution'],color='#777777',width=.65)
    axes[1].set(xlabel='Successes in four audit queries, J',ylabel='Probability (both laws)',ylim=(0,1.03)); axes[1].set_xticks(xx)
    axes[1].text(.98,.94,f"Worst-case failure\nLaw A: {row['target_A']:.3f}\nLaw B: {row['target_B']:.3f}",transform=axes[1].transAxes,ha='right',va='top',fontsize=9)
    fig.tight_layout(w_pad=1.4)
    for ext in ['pdf','png']: fig.savefig(OUT/f'finite_suite_matching_pair.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.,2.8)); err=np.array(row['polynomial_errors_at_integer_grid']); e=float(F(row['minimax_uniform_error_exact']))
    ax.axhline(e,color='#999999',linestyle='--',lw=1); ax.axhline(-e,color='#999999',linestyle='--',lw=1)
    ax.plot(np.arange(65),err,'o-',color=colors[0],ms=2.7,lw=1)
    ax.scatter(support,err[support],s=28,color=colors[-1],zorder=4,label='Extremal support')
    ax.set(xlabel='Successful attacks per input, S',ylabel='Indicator approximation error',xlim=(-1,65))
    ax.legend(frameon=False,loc='upper right'); ax.grid(axis='y',alpha=.15); fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(OUT/f'finite_suite_dual_certificate.{ext}',dpi=220)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True); cases=[]
    for d in DS:
        for k in KS:
            if k>d: continue
            row=one_case(d,k); cases.append(row)
            print(f"d={d:3d} k={k:2d} width={row['identification_width']:.12g} exact_certificate={row['exact_certificate_passed']} count_float_residual={row['floating_count_probability_discrepancy']:.2g}",flush=True)
            (OUT/'finite_suite_results.json').write_text(json.dumps(cases,indent=2)+'\n')
    figures(cases)
    meta=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
        case_count=len(cases),all_exact_certificates_passed=all(r['exact_certificate_passed'] for r in cases),
        method='LP support discovery followed by exact Fraction arithmetic: all moments, all hypergeometric audit probabilities, and polynomial error on every integer grid point',
        max_floating_count_probability_discrepancy=max(r['floating_count_probability_discrepancy'] for r in cases),
        max_lp_width_error=max(abs(r.get('lp_minus_exact_width',0)) for r in cases),total_solver_seconds=sum(r['seconds'] for r in cases))
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    lines=['# Finite-suite exact numerical results','','Every nonzero width has an exact rational matching-law certificate and an exact rational degree-k polynomial certificate verified at every S=0,...,d. Full-audit entries are zero because J=S.','','| d | k | Width | Exact width |','|---:|---:|---:|:---|']
    for r in cases: lines.append(f"| {r['d']} | {r['k']} | {r['identification_width']:.12g} | {r['identification_width_exact']} |")
    (OUT/'numerical_summary.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
