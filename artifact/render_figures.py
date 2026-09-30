from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'experiments/results/finite_suite_results.json').read_text())
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(3.45,2.6))
for d,color,marker in zip([16,32,64,128],['#407bad','#1d893a','#ce983e','#a3347a'],['o','s','^','D']):
 r=[c for c in cases if c['d']==d]
 ax.plot([c['k'] for c in r],[c['identification_width'] for c in r],marker=marker,label=f'd = {d}',color=color,ms=3.5,lw=1.3)
ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.set_ylim(-.015,1.025)
ax.set_ylabel('Worst-case identification width');ax.set_xlabel('Attacks queried per input, k')
ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=7,frameon=False,loc='lower left')
fig.tight_layout();fig.savefig(ROOT/'figures/diameter.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/diameter.png',dpi=200,bbox_inches='tight')
