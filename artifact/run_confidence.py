from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from confidence import confidence_interval, detection_interval, cp_interval
ROOT=Path(__file__).resolve().parent
start=time.time()
a=np.load(ROOT/'empirical_results/audit_histograms.npz',allow_pickle=False)
print('keys:',a.files,flush=True)
for k in a.files: print(k,a[k].shape,a[k].dtype,flush=True)
summary=json.loads((ROOT/'empirical_results/empirical_summary.json').read_text())
rows=[]
for mi,model in enumerate(a['model_names']):
    theta=summary['models'][str(model)]['suite_failure_theta']
    for ii,m in enumerate(a['m_values']):
      for jj,k in enumerate(a['k_values']):
        r=int(m*k//98)
        for rep in range(int(a['repetitions'])):
          hist=a['histograms'][mi,ii,jj,rep,:k+1]
          ci=confidence_interval(hist,98,int(k))
          dl,du=detection_interval(hist,98,int(k))
          rng=np.random.default_rng(np.random.SeedSequence([20260920,99,mi,int(m),int(k),rep]))
          success=int(rng.binomial(r,theta))
          fl,fu=cp_interval(success,r,.05) if r else (0.,1.)
          rows.append(dict(model=str(model),m=int(m),k=int(k),rep=rep,theta=theta,full_inputs=r,lower=ci['lower'],upper=ci['upper'],coverage=ci['lower']<=theta<=ci['upper'],width=ci['upper']-ci['lower'],status=ci['status'],primal_residual=ci.get('max_primal_residual',0),detect_lower=dl,detect_upper=du,detect_width=du-dl,detect_coverage=dl<=theta<=du,observed_detected=1-hist[0]/m,full_lower=float(fl),full_upper=float(fu),full_width=float(fu-fl),full_coverage=fl<=theta<=fu))
        print(str(model),int(m),int(k),'done',flush=True)
out=ROOT/'confidence_results';out.mkdir(exist_ok=True)
df=pd.DataFrame(rows);df.to_csv(out/'all_intervals.csv',index=False)
s=df.groupby(['model','m','k']).agg(theta=('theta','first'),full_inputs=('full_inputs','first'),mean_width=('width','mean'),coverage=('coverage','mean'),detection_mean_width=('detect_width','mean'),detection_coverage=('detect_coverage','mean'),mean_detected=('observed_detected','mean'),full_mean_width=('full_width','mean'),full_coverage=('full_coverage','mean'),max_primal_residual=('primal_residual','max')).reset_index()
s.to_csv(out/'summary.csv',index=False)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.65),sharey=True)
for ax,model,title in zip(axs,a['model_names'],['Logistic regression','Random forest']):
    sub=s[(s.model==model)&(s.m==500)]
    for col,label,marker,color in [('mean_width','Partial audit projection','o','#145c85'),('detection_mean_width','Detection-only bound','s','#bf7a25'),('full_mean_width','Full suite on fewer inputs','^','#35805c')]:
      ax.plot(sub.k,sub[col],marker=marker,label=label,color=color,lw=1.5,ms=4)
    ax.set_xscale('log');ax.set_xlabel('Attacks per input, k');ax.set_title(title);ax.grid(alpha=.2);ax.set_xticks([1,4,16,64,98],['1','4','16','64','98'])
axs[0].set_ylabel('Mean 95% interval width')
axs[1].legend(fontsize=7,loc='upper right')
fig.tight_layout();fig.savefig(out/'confidence_widths.pdf',bbox_inches='tight');fig.savefig(out/'confidence_widths.png',dpi=180,bbox_inches='tight')
meta={'repetitions_per_setting':100,'settings':len(s),'intervals':len(df),'elapsed_seconds':time.time()-start,'all_statuses':df.status.value_counts().to_dict(),'maximum_primal_residual':float(df.primal_residual.max()),'min_coverage':float(s.coverage.min()),'max_coverage':float(s.coverage.max()),'full_min_coverage':float(s.full_coverage.min()),'full_max_coverage':float(s.full_coverage.max()),'coverage_monte_carlo_se_at_point95':float(np.sqrt(.95*.05/100))}
(out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
