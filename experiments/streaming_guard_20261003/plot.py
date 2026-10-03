import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/(name+'.json')).read_text())
colors={'baseline':'#637083','adaptive_resume':'#327bb4','slai_port':'#d89832','growth_only':'#795ba7','guard128':'#138879','guard256':'#a4475d','guard_fast':'#b23432','guard_relaxed':'#4599a7'}
labels={'baseline':'Default','adaptive_resume':'Adaptive resume','slai_port':'SLAI core port','growth_only':'Growth only','guard128':'Service + growth128','guard256':'Service + growth256','guard_fast':'Service target .25 s','guard_relaxed':'Service target 1 s'}
plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
s=load('summary');main=s['policies']['main_pressure'];policies=[p for p in colors if p in main];pairs=load('pairs')
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');a,b,c,d=axes.flat;x=list(range(len(policies)))
a.bar(x,[main[p]['stalled'] for p in policies],color=[colors[p] for p in policies])
for i,p in enumerate(policies):a.text(i,main[p]['stalled']+.2,f"{main[p]['stalled']}/{main[p]['requests']}",ha='center',fontsize=9)
a.set(xticks=x,xticklabels=[labels[p].replace('Service + ','Service\n+ ') for p in policies],ylabel='Requests with a client gap >1 s',title='Do already-started streams keep getting service?');a.tick_params(axis='x',labelrotation=20);a.margins(y=.2)
for i,p in enumerate(policies):
    pp=[q for q in pairs if q['reference']=='baseline' and q['policy']==p and q['seed']!=381 and q['slots']==3072 and q['cap']!=3]
    if not pp:continue
    b.scatter([i]*len(pp),[100*q['throughput_ratio'] for q in pp],color=colors[p],s=25,alpha=.7)
    summary=s['paired']['main_pressure'][p+'_vs_baseline'];b.scatter([i],[100*summary['throughput_ratio_geo']],color='black',marker='_',s=160)
b.axhline(100,color='#888',ls='--');b.set(xticks=x,xticklabels=[labels[p].replace('Service + ','Service\n+ ') for p in policies],ylabel='Throughput / matched default (%)',title='Dots: workloads · black mark: geometric mean');b.tick_params(axis='x',labelrotation=20)
for p in policies:
    rows=[r for r in load('analysis') if r['policy']==p and r['seed']!=381 and r['slots']==3072 and r['cap']!=3]
    c.scatter([r['trace_preemptions'] for r in rows],[r['max_silence'] for r in rows],s=35,color=colors[p],label=labels[p],alpha=.75)
c.axhline(.5,color='#aaa',ls='--',lw=1);c.set(xlabel='Evictions per workload',ylabel='Worst client delivery gap (s)',title='Eviction counts and interruption can diverge');c.set_yscale('log');c.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.18))
for i,p in enumerate(policies):
    pp=[q for q in pairs if q['reference']=='baseline' and q['policy']==p and q['seed']!=381 and q['slots']==3072 and q['cap']!=3]
    if pp:d.scatter([i]*len(pp),[q['ttft_ratio'] for q in pp],color=colors[p],s=25,alpha=.7)
d.axhline(1,color='#888',ls='--');d.set(xticks=x,xticklabels=[labels[p].replace('Service + ','Service\n+ ') for p in policies],ylabel='Median TTFT / matched default',title='The first-token waiting cost');d.tick_params(axis='x',labelrotation=20)
fig.suptitle("Main pressure comparison · 12 matched workloads / 144 streams per policy",fontsize=13)
for suffix in ('png','pdf'):fig.savefig(ROOT/f'streaming_results.{suffix}')
plt.close(fig)
long=s['policies']['long_tail']
if long:
    policies=[p for p in colors if p in long];fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained');x=list(range(len(policies)))
    for i,p in enumerate(policies):
        axes[0].bar(i,long[p]['worst'],color=colors[p]);rows=[r for r in load('analysis') if r['policy']==p and r['seed']==381];budget=rows[0]['budget'];axes[0].scatter(i,budget,color='black',marker='_',s=90) if p!='baseline' else None
        pp=[q for q in pairs if q['reference']=='baseline' and q['policy']==p and q['seed']==381]
        ratio=100*s['paired']['long_tail'][p+'_vs_baseline']['throughput_ratio_geo'] if pp else 100
        axes[1].bar(i,ratio,color=colors[p]);axes[2].bar(i,long[p]['replay']+1,color=colors[p]);axes[2].text(i,long[p]['replay']+1,f"{long[p]['evictions']:,}",rotation=0,ha='center',va='top',fontsize=8,color='white')
    for ax in axes:ax.set_xticks(x,[labels[p].replace('Service + ','Service\n+ ').replace('Service target ','Target\n') for p in policies],rotation=30,ha='right')
    axes[0].set(ylabel='Worst client gap (s)',title='Bars: observed · marks: configured targets');axes[0].set_yscale('log')
    axes[1].set(ylabel='Throughput / default (%)',title='Price of tighter service');axes[1].axhline(100,color='#aaa',ls='--')
    axes[2].set(ylabel='Scheduled replay positions +1',title='Replay cost · white labels: evictions');axes[2].set_yscale('log');axes[2].margins(y=.3)
    fig.suptitle('Held-out long tail · seed 381 · outputs up to 1,536 tokens · clustered and staggered arrivals',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(ROOT/f'long_tail_cost.{suffix}')
    plt.close(fig)
print('Saved verified-results figures.')



