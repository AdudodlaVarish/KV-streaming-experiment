import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/(name+'.json')).read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
colors={'baseline':'#637083','adaptive':'#327bb4','adaptive_resume':'#138879','fixed':'#c17b26','fixed_resume':'#a4475d'}
labels={'baseline':'Default','adaptive':'Adaptive','adaptive_resume':'Adaptive + resume','fixed':'Fixed 30%','fixed_resume':'Fixed + resume'}
old=load('previous_summary')['policies']['baseline_pressure']
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
policies=['baseline','adaptive','fixed']; x=list(range(3))
covered=[old[p]['wait_covered_gap_s'] for p in policies]
remaining=[old[p]['total_long_gap_s']-old[p]['wait_covered_gap_s'] for p in policies]
axes[0].bar(x,covered,color='#327bb4',label='Overlaps eviction-to-replay waiting')
axes[0].bar(x,remaining,bottom=covered,color='#e6b052',label='Remainder of client gap')
for i,p in enumerate(policies): axes[0].text(i,old[p]['total_long_gap_s']+2,f"{old[p]['wait_coverage_fraction']:.1%}",ha='center')
axes[0].set(xticks=x,xticklabels=[labels[p] for p in policies],ylabel='Total duration of client gaps >1 s (s)',title='Waiting covers 96–97% of long gaps')
axes[0].legend(fontsize=8,loc='upper right'); axes[0].set_ylim(0,170)
ee=load('previous_episodes')
for p in policies:
    chosen=[e for e in ee if e['policy']==p]
    axes[1].scatter([e['wait_s'] for e in chosen],[e['replay_to_delivery_s'] for e in chosen],s=22,alpha=.7,color=colors[p],label=labels[p])
axes[1].set(xlabel='Eviction → first replay scheduling step (s)',ylabel='First replay step → next client delivery (s)',title='The long component precedes replay')
axes[1].legend(fontsize=8); axes[1].set_xlim(left=0); axes[1].set_ylim(bottom=0)
fig.suptitle('Reanalysis of the earlier 168-trial study · 16 default-eviction workloads',fontsize=12)
for suffix in ('png','pdf'):fig.savefig(ROOT/f'waiting_mechanism.{suffix}')
plt.close(fig)
summary=load('summary'); agg=summary['policies']['baseline_pressure']; policies=[p for p in colors if p in agg]
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained'); a,b,c,d=axes.flat; x=list(range(len(policies)))
a.bar(x,[agg[p]['excess_s'] for p in policies],color=[colors[p] for p in policies])
for i,p in enumerate(policies):a.text(i,agg[p]['excess_s'],f"{agg[p]['stalled_requests']} streams",ha='center',va='bottom',fontsize=8)
a.set(xticks=x,xticklabels=[labels[p].replace(' + ','\n+ ') for p in policies],ylabel='Sum of max(0, gap − 1 s) (s)',title='Long interruptions across pressure workloads');a.margins(y=.2)
pairs=[p for p in load('pairs') if p['policy']=='adaptive_resume' and p['reference']=='adaptive' and p['slots']==3072]
for model,marker in [('llama1b','o'),('qwen1.5b','^')]:
    pp=[p for p in pairs if p['model']==model]
    b.scatter([p['reference_gap'] for p in pp],[p['policy_gap'] for p in pp],marker=marker,s=55,label=model)
lim=max([p[k] for p in pairs for k in ('reference_gap','policy_gap')]+[1])*1.1
b.plot([0,lim],[0,lim],color='#888',lw=1,ls='--');b.set(xlim=(0,lim),ylim=(0,lim),xlabel='Adaptive: worst client gap (s)',ylabel='Adaptive + resume: worst gap (s)',title='Each point is the same workload');b.legend(fontsize=9)
base=[0]*len(policies)
for key,label,color in [('capacity_decision_s','Physical capacity','#637083'),('watermark_decision_s','Reserve only','#e6b052'),('no_attempt_decision_s','No attempt yet','#97bcb6')]:
    values=[agg[p][key] for p in policies];c.barh(x,values,left=base,label=label,color=color);base=[v+w for v,w in zip(base,values)]
c.set(yticks=x,yticklabels=[labels[p] for p in policies],xlabel='Sampled eviction-to-replay waiting (s)',title='Which allocation check blocks resumption?');c.legend(fontsize=8)
for policy,offset in [('adaptive',-.10),('adaptive_resume',0),('fixed',.10),('fixed_resume',.20)]:
    pp=[p for p in load('pairs') if p['policy']==policy and p['reference']=='baseline' and p['slots']==3072]
    d.scatter([i+offset for i in range(len(pp))],[100*p['throughput_ratio'] for p in pp],s=25,color=colors[policy],label=labels[policy],alpha=.8)
d.axhline(100,color='#888',ls='--',lw=1);d.set(xlabel='Paired pressure workload index',ylabel='Throughput / default (%)',title='Throughput cost varies by workload');d.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.18))
fig.suptitle(f"Resume reserve waiver · {summary['completed_trials']} trials / {summary['completed_requests']} requests",fontsize=13)
for suffix in ('png','pdf'):fig.savefig(ROOT/f'resume_results.{suffix}')
plt.close(fig)
print('Saved waiting_mechanism and resume_results (PNG/PDF).')

