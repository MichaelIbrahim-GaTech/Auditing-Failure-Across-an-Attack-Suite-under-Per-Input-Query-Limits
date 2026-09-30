from pathlib import Path
import json
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from confidence import sampling_matrix
root=Path(__file__).resolve().parent
info=json.loads((root/'jbb_results/summary.json').read_text())
rows=[]
for r in info['reports']:
 if r['suite']!='common_methods': continue
 for k in (1,2,3,4):
  H=sampling_matrix(4,k); q=np.array([float(Fraction(v)) for v in r['oracle_hypergeom_counts'][str(k)]])
  h=np.r_[0.,np.ones(4)]
  fits=[linprog(c,A_eq=H,b_eq=q,bounds=(0,None),method='highs') for c in (h,-h)]
  assert all(x.success for x in fits)
  rows.append({'model':r['model'],'k':k,'theta':r['union_rate'],'lower':fits[0].fun,'upper':-fits[1].fun,'detected':1-q[0]})
(root/'jbb_results/identified_intervals.json').write_text(json.dumps(rows,indent=2))
for r in rows: print(r)
