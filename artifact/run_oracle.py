"""Execute oracle versus finite-sample audit comparisons on frozen populations."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_data import load_populations
from audit_inference import oracle_interval,fit_linear_estimator,linear_interval,linear_population_risk
from confidence import confidence_interval,sampling_matrix
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'oracle_results';OUT.mkdir(exist_ok=True)
REPS=100

def wilson(successes,n):
    z=norm.ppf(.975);p=successes/n;den=1+z*z/n
    ctr=(p+z*z/(2*n))/den;rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(ctr-rad),float(ctr+rad)

def main():
    started=time.time();populations=load_populations();oracles=[]
    for p in populations:
        for k in ([4,16,36] if p['d']==36 else [1,4,16,64,98]):
            r=oracle_interval(p['mu'],p['d'],k)
            r.update(name=p['name'],identifier=p['identifier'],family=p['family'])
            oracles.append(r)
    (OUT/'oracle_certificates.json').write_text(json.dumps(oracles,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k!='diagnostics'} for r in oracles]).to_csv(OUT/'oracle.csv',index=False)
    lookup={(r['name'],r['k']):r for r in oracles}
    old_strong=pd.read_csv(ROOT/'strongreject_audit_results/all_intervals.csv')
    old_digits=pd.read_csv(ROOT/'confidence_results/all_intervals.csv')
    a=np.load(ROOT/'empirical_results/audit_histograms.npz',allow_pickle=False)
    digit_models=list(map(str,a['model_names']));m_values=list(a['m_values']);k_values=list(a['k_values'])
    rows=[];analytic=[]
    for pi,p in enumerate(populations):
        d=p['d'];theta=p['theta']
        for k in ([4,16,36] if d==36 else [1,4,16,64,98]):
            q=sampling_matrix(d,k)@p['mu']
            for m in [100,500,2000]:
                fit=fit_linear_estimator(d,k,m)
                ar=linear_population_risk(p['mu'],fit)
                analytic.append(dict(name=p['name'],family=p['family'],d=d,k=k,m=m,theta=theta,
                    bias_bound=fit['bias_bound'],coefficient_range=fit['range'],**ar))
                for rep in range(REPS):
                    if d==98:
                        hist=a['histograms'][digit_models.index(p['identifier']),m_values.index(m),k_values.index(k),rep,:k+1]
                        previous=old_digits[(old_digits.model==p['identifier'])&(old_digits.m==m)&(old_digits.k==k)&(old_digits.rep==rep)].iloc[0]
                        assert abs(1-hist[0]/m-previous.observed_detected)<1e-12
                        ci=dict(lower=previous.lower,upper=previous.upper,status='reused_existing_histogram',max_primal_residual=previous.primal_residual)
                    elif m==500:
                        # Precisely reconstruct the original histogram and reuse its projection.
                        rng=np.random.default_rng(np.random.SeedSequence([20260920,375,pi,k,rep]))
                        latent=rng.choice(p['S'],size=m,replace=True)
                        counts=rng.hypergeometric(latent,d-latent,k);hist=np.bincount(counts,minlength=k+1)
                        previous=old_strong[(old_strong.model==p['identifier'])&(old_strong.k==k)&(old_strong.rep==rep)].iloc[0]
                        assert abs(1-hist[0]/m-previous.mean_detected)<1e-12
                        ci=dict(lower=previous.partial_lower,upper=previous.partial_upper,status='reused_original_seed',max_primal_residual=previous.residual)
                    else:
                        rng=np.random.default_rng(np.random.SeedSequence([20260922,610,pi,k,m,rep]))
                        hist=rng.multinomial(m,q)
                        ci=confidence_interval(hist,d,k)
                    lc=linear_interval(hist,fit)
                    rows.append(dict(name=p['name'],identifier=p['identifier'],family=p['family'],d=d,k=k,m=m,rep=rep,theta=theta,
                        oracle_width_upper=lookup[(p['name'],k)]['width'],oracle_certification=lookup[(p['name'],k)]['certification'],
                        projection_lower=ci['lower'],projection_upper=ci['upper'],projection_width=ci['upper']-ci['lower'],
                        projection_coverage=ci['lower']<=theta<=ci['upper'],projection_status=ci['status'],
                        projection_primal_residual=ci.get('max_primal_residual',0.),
                        linear_lower=lc['lower'],linear_upper=lc['upper'],linear_width=lc['width'],linear_coverage=lc['lower']<=theta<=lc['upper'],
                        linear_point=lc['point'],linear_raw_point=lc['raw_point'],linear_error=lc['point']-theta,
                        linear_squared_error=(lc['point']-theta)**2))
                print(p['name'],k,m,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'finite_intervals.csv',index=False)
    analytic=pd.DataFrame(analytic);analytic.to_csv(OUT/'linear_analytic_risk.csv',index=False)
    summary=[]
    for (name,family,d,k,m),g in df.groupby(['name','family','d','k','m'],sort=False):
        row=dict(name=name,family=family,d=d,k=k,m=m,theta=g.theta.iloc[0],repetitions=len(g),
            oracle_width_upper=g.oracle_width_upper.iloc[0],oracle_certification=g.oracle_certification.iloc[0])
        for method in ['projection','linear']:
            widths=g[method+'_width'];covers=g[method+'_coverage']
            lo,hi=wilson(int(covers.sum()),len(g))
            row.update({method+'_mean_width':float(widths.mean()),method+'_width_se':float(widths.std(ddof=1)/np.sqrt(len(g))),
                method+'_coverage':float(covers.mean()),method+'_coverage_wilson_lower':lo,method+'_coverage_wilson_upper':hi})
        row.update(linear_clipped_bias=float(g.linear_error.mean()),linear_clipped_mse=float(g.linear_squared_error.mean()),
            linear_clipped_mse_se=float(g.linear_squared_error.std(ddof=1)/np.sqrt(len(g))))
        summary.append(row)
    s=pd.DataFrame(summary);s.to_csv(OUT/'finite_summary.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.65),sharex=True)
    for ax,(name,k) in zip(axes.flat,[('GPT-4o-mini',4),('GPT-4o-mini',16),('Forest',4),('Forest',16)]):
        sub=s[(s.name==name)&(s.k==k)].sort_values('m');oracle=lookup[(name,k)]
        for method,label,color,marker in [('projection','Confidence projection','#176b8f','o'),('linear','Bias/range linear interval','#b16a1c','s')]:
            ax.errorbar(sub.m,sub[method+'_mean_width'],yerr=1.96*sub[method+'_width_se'],label=label,color=color,marker=marker,ms=3.5,capsize=2,lw=1.3)
        oracle_label='Sharp oracle width' if oracle['certification']=='exact_rational_endpoint_certificates' else 'Certified oracle upper bound'
        ax.axhline(oracle['width'],color='#76458d',ls='--',lw=1.3)
        ax.annotate(oracle_label,xy=(.98,oracle['width']),xycoords=('axes fraction','data'),
            xytext=(0,5),textcoords='offset points',ha='right',va='bottom',fontsize=7.5,color='#76458d')
        ax.set_title(f'{name}, k = {k}');ax.set_xscale('log');ax.set_xticks([100,500,2000],['100','500','2000']);ax.set_xlim(80,2500)
        # Keep a near-zero reference visible above the bottom spine.
        upper=ax.get_ylim()[1]
        ax.set_ylim(bottom=-.04*upper if oracle['width']<.01*upper else 0);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Mean 95% interval width')
    for ax in axes[1]:ax.set_xlabel('Inputs audited, m')
    handles,labels=axes[0,0].get_legend_handles_labels()
    # The oracle certification status is stated directly in every panel.
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=1,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(OUT/'oracle_vs_finite.pdf',bbox_inches='tight');fig.savefig(OUT/'oracle_vs_finite.png',dpi=220,bbox_inches='tight');plt.close(fig)
    meta=dict(repetitions=REPS,settings=len(s),intervals=len(df),elapsed_seconds=time.time()-started,
        reused_digits_histograms=True,reconstructed_original_StrongREJECT_m500_seeds=True,
        linear_fit_population_independent=True,oracle_exact_singletons=sum(r['certification'].startswith('exact_singleton') for r in oracles),
        oracle_sharp_to_1e9=sum(r['certification']=='exact_rational_endpoint_certificates' for r in oracles),
        oracle_outer_only=sum(r['certification']=='exact_rational_outer_bounds' for r in oracles),
        statuses=df.projection_status.value_counts().to_dict(),max_projection_residual=float(df.projection_primal_residual.max()))
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
