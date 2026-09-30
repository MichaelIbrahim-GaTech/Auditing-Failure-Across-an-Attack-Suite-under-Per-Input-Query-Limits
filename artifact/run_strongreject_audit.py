from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from confidence import sampling_matrix, confidence_interval, cp_interval
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'strongreject_audit_results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'strongreject_results/score_metadata.csv')
df=df[~df.jailbreak.isin(['none','evil_system_prompt'])]
rows=[]; sens=[]; names=[]; scores=[]; contexts=[]
start=time.time()
for mi,(model,part) in enumerate(df.groupby('model')):
 matrix=part.pivot(index='context_id',columns='jailbreak',values='score').sort_index().sort_index(axis=1)
 assert matrix.shape==(313,36) and matrix.notna().all().all()
 names.append(model);scores.append(matrix.to_numpy());contexts.append(matrix.index.to_numpy(dtype=str))
 for threshold in [.25,.5,.75]:
  S=(matrix.to_numpy()>=threshold).sum(axis=1)
  sens.append(dict(model=model,threshold=threshold,contexts=313,suite_size=36,union_count=int((S>0).sum()),theta=float((S>0).mean()),mean_S=float(S.mean())))
 S=(matrix.to_numpy()>=.5).sum(axis=1);theta=float((S>0).mean())
 for k in [4,16,36]:
  m=500;r=m*k//36
  for rep in range(100):
   rng=np.random.default_rng(np.random.SeedSequence([20260920,375,mi,k,rep]))
   latent=rng.choice(S,size=m,replace=True)
   C=rng.hypergeometric(latent,36-latent,k)
   hist=np.bincount(C,minlength=k+1)
   ci=confidence_interval(hist,36,k)
   full_x=int(rng.binomial(r,theta));fl,fu=cp_interval(full_x,r,.05)
   rows.append(dict(model=model,m=m,k=k,rep=rep,theta=theta,full_inputs=r,partial_lower=ci['lower'],partial_upper=ci['upper'],partial_width=ci['upper']-ci['lower'],partial_coverage=ci['lower']<=theta<=ci['upper'],full_x=full_x,full_lower=float(fl),full_upper=float(fu),full_width=float(fu-fl),full_coverage=fl<=theta<=fu,mean_detected=float((C>0).mean()),status=ci['status'],residual=ci.get('max_primal_residual',0.)))
  print(model,k,'done',flush=True)
np.savez_compressed(OUT/'primary_score_matrices.npz',models=np.array(names),scores=np.array(scores),context_ids=np.array(contexts),methods=matrix.columns.to_numpy(dtype=str))
res=pd.DataFrame(rows);res.to_csv(OUT/'all_intervals.csv',index=False)
summary=res.groupby(['model','k']).agg(theta=('theta','first'),full_inputs=('full_inputs','first'),partial_mean_width=('partial_width','mean'),full_mean_width=('full_width','mean'),partial_coverage=('partial_coverage','mean'),full_coverage=('full_coverage','mean'),mean_detected=('mean_detected','mean'),max_residual=('residual','max')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(sens).to_csv(OUT/'threshold_sensitivity.csv',index=False)
meta={'settings':len(summary),'intervals':len(res),'partial_coverage_range':[float(summary.partial_coverage.min()),float(summary.partial_coverage.max())],'full_coverage_range':[float(summary.full_coverage.min()),float(summary.full_coverage.max())],'max_residual':float(res.residual.max()),'elapsed_seconds':time.time()-start}
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
print(summary.to_string(index=False),flush=True);print(json.dumps(meta),flush=True)
