import json, pathlib, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
checks=json.loads((ROOT/'checks.json').read_text())
assert checks['diagnostics_excluded'] and checks['engine_arrival_order_verified']
assert checks['completed_trials']==168 and not checks['unfinished_cells'],checks
rows=json.loads((ROOT/'analysis.json').read_text())
pairs=json.loads((ROOT/'paired.json').read_text())
summary=json.loads((ROOT/'summary.json').read_text())
configs=sorted({(r['slots'],r['cap']) for r in rows if r['model']=='llama1b'})
workloads=[('balanced','burst'),('balanced','stagger'),('generation','burst'),('generation','stagger')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(15,11),gridspec_kw={'height_ratios':[.9,1.1]})
heat=[]
for policy in ('baseline','adaptive'):
    values=np.zeros((4,len(configs)))
    for j,(mix,arrival) in enumerate(workloads):
        for i,(slots,cap) in enumerate(configs):
            group=[r for r in rows if r['model']=='llama1b' and r['policy']==policy and r['slots']==slots and r['cap']==cap and r['mix']==mix and r['arrival']==arrival]
            assert len(group)==2
            values[j,i]=sum(r['stalled_requests'] for r in group)
    heat.append(values)
maximum=max(1.,max(float(v.max()) for v in heat))
for ax,values,title in zip(axes[0],heat,['Default admission','Adaptive headroom']):
    ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=maximum,aspect='auto')
    for (j,i),v in np.ndenumerate(values):
        ax.text(i,j,f'{int(v)}/24',ha='center',va='center',color='white' if v>maximum*.65 else '#172d3d',fontweight='bold')
    ax.set_xticks(range(len(configs)),[f'{s:,}\ncap {c}' for s,c in configs],fontsize=10)
    ax.set_yticks(range(4),['Balanced / burst','Balanced / stagger','Generation / burst','Generation / stagger'],fontsize=10)
    ax.set_xlabel('Nominal KV token slots / concurrency cap')
    ax.set_title(title,loc='left',fontweight='bold',pad=12)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
ax=axes[1,0]
colors={3072:'#D17B3D',5120:'#3980A5',12288:'#478650'}
adaptive=[p for p in pairs if p['policy']=='adaptive']
for p in adaptive:
    ax.scatter(p['throughput_ratio'],100*(p['policy_stalled']-p['baseline_stalled'])/12,
        color=colors[p['slots']],marker='o' if p['model']=='llama1b' else '^',s=45,alpha=.8,edgecolors='white',linewidths=.4)
xs=[p['throughput_ratio'] for p in adaptive]
ys=[100*(p['policy_stalled']-p['baseline_stalled'])/12 for p in adaptive]
xmin=min(.85,min(xs)-.04)
xmax=max(1.08,max(xs)+.04)
ymin=min(-5,min(ys)-5)
ymax=max(5,max(ys)+5)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.fill_between([.9,xmax],[ymin,ymin],[0,0],color='#e9f4ec',zorder=-1)
ax.axhline(0,color='#7c8790',linewidth=.8)
ax.axvline(1,color='#7c8790',linewidth=.8)
ax.axvline(.9,color='#7c8790',linewidth=.8,linestyle='--')
ax.set_xlabel('Adaptive throughput / paired default throughput')
ax.set_ylabel('Change in stalled requests (percentage points)')
ax.set_title('Each point is a matched workload / seed / configuration',loc='left',fontsize=12,fontweight='bold',pad=12)
ax.text(xmax-.015,ymin+2,'Fewer stalls; retains at least 90% throughput',ha='right',va='bottom',fontsize=9,color='#42664b')
ax.grid(alpha=.12)
legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[s],markeredgecolor='none',label=f'{s:,} slots') for s in colors]
legend += [Line2D([0],[0],marker=m,color='#52616b',linestyle='none',label=l) for m,l in [('o','Llama 1B'),('^','Qwen 1.5B')]]
fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.115,.072),fontsize=9,ncol=5,frameon=False)
ax=axes[1,1]
pressure=summary['pressure']
metrics=[('evictions','baseline_evictions','policy_evictions'),('replayed positions','baseline_replay','policy_replay'),('requests with >1 s gaps','baseline_stalled_requests','policy_stalled_requests')]
for offset,policy,color in [(-.18,'fixed','#3980A5'),(.18,'adaptive','#D17B3D')]:
    p=pressure[policy]
    vals=[p[pol]/p[base] if p[base] else 0 for _,base,pol in metrics]
    positions=np.arange(3)+offset
    ax.bar(positions,vals,width=.34,color=color,label='Fixed 30%' if policy=='fixed' else 'Adaptive')
    for x,v in zip(positions,vals):
        ax.text(x,v+.025,f'{v:.0%}',ha='center',fontsize=10,fontweight='bold')
ax.axhline(1,color='#7c8790',linestyle='--',linewidth=.8)
ax.set_xticks(range(3),['Evictions','Replayed\npositions','Requests with\n>1 s gaps'],fontsize=10)
ax.set_ylim(0,max(1.15,ax.get_ylim()[1]))
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylabel('Burden remaining relative to paired default')
ax.set_title('Cases where default admission evicted requests',loc='left',fontsize=12,fontweight='bold',pad=12)
ax.legend(frameon=False)
ax.grid(axis='y',alpha=.12)
ax.set_axisbelow(True)
fig.suptitle('Admission policies across workloads and cache budgets',x=.08,ha='left',y=.98,fontsize=20,fontweight='bold')
fig.text(.08,.938,'168 trials · 2,016 requests · two randomized seeds/orders · Llama 3.2 1B + Qwen 2.5 1.5B · RTX 4060 8 GB',fontsize=11,color='#4b5965')
fig.text(.08,.900,'Top: Llama requests with a post-first-token gap >1 second, pooled over two seeds. Darker cells mean more stalled streams.',fontsize=10,color='#4b5965')
fig.text(.08,.025,'Client token-delivery timing; eager execution; fixed output lengths; synthetic arrivals.\nPaired comparisons share workload specifications and verified server arrival order. This is a controlled local study, not a production benchmark.',fontsize=10,color='#4b5965')
fig.subplots_adjust(left=.12,right=.975,top=.855,bottom=.18,hspace=.44,wspace=.35)
fig.savefig(ROOT/'robustness.png',dpi=180)
fig.savefig(ROOT/'robustness.svg')
# Prospective illustrative case, selected before looking at evaluation results.
chosen=[r for r in rows if r['model']=='llama1b' and r['slots']==3072 and r['cap']==12 and r['seed']==77 and r['mix']=='generation' and r['arrival']=='burst']
assert len(chosen)==3
chosen.sort(key=lambda r:['baseline','fixed','adaptive'].index(r['policy']))
fig,axes=plt.subplots(4,1,figsize=(14,13),gridspec_kw={'height_ratios':[1,1,1,.7]})
max_end=0
for ax,row in zip(axes[:3],chosen):
    folder=ROOT/row['run']/row['folder']
    requests=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    origin=min(r['start'] for r in requests)
    for r in requests:
        i=r['index']
        start=r['start']-origin
        first=start+r['ttft_s']
        end=start+r['elapsed_s']
        max_end=max(max_end,end)
        ax.broken_barh([(start,first-start)],(i-.3,.6),facecolors='#dce2e7')
        ax.broken_barh([(first,end-first)],(i-.3,.6),facecolors='#4283ae')
        times=[start+t for t,_ in r['token_events']]
        ax.broken_barh([(a,b-a) for a,b in zip(times,times[1:]) if b-a>1],(i-.3,.6),facecolors='#d75445')
    ax.set_ylim(-.8,11.8)
    ax.invert_yaxis()
    ax.set_yticks(range(12),[str(i+1) for i in range(12)],fontsize=8)
    ax.set_ylabel('Request arrival order')
    label={'baseline':'Default','fixed':'Fixed 30% headroom','adaptive':'Adaptive headroom'}[row['policy']]
    ax.set_title(f'{label} — {row["throughput"]:.0f} output tok/s; {row["stalled_requests"]}/12 streams stalled >1 s',loc='left',fontweight='bold',fontsize=12,pad=10)
    ax.grid(axis='x',alpha=.12)
    events=[]
    for line in (ROOT/row['run']/'scheduler_trace.jsonl').open():
        e=json.loads(line)
        if e['kind']=='admission' and e['trial']==row['trial'] and row['start_perf_s']<=e['t']<=row['end_perf_s']:
            events.append(e)
    axes[3].plot([e['t']-origin for e in events],[e['reserve'] for e in events],label=label,color={'baseline':'#52616b','fixed':'#3980A5','adaptive':'#D17B3D'}[row['policy']])
for ax in axes:
    ax.set_xlim(0,max_end+1)
axes[3].set_ylim(-.02,.50)
axes[3].set_ylabel('Reserved headroom')
axes[3].set_xlabel('Seconds from first arrival')
axes[3].yaxis.set_major_formatter(PercentFormatter(1))
axes[3].legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
axes[3].grid(alpha=.12)
fig.suptitle('One controlled pressure case: queueing versus post-start silence',x=.08,ha='left',y=.985,fontsize=18,fontweight='bold')
fig.text(.08,.95,'Llama 1B · 3,072 KV slots · concurrency cap 12 · mixed generation lengths · clustered arrivals · seed 77',fontsize=11,color='#4b5965')
fig.legend(handles=[Patch(color='#dce2e7',label='Before first token'),Patch(color='#4283ae',label='Streaming interval'),Patch(color='#d75445',label='Token-delivery gap >1 s')],loc='lower right',bbox_to_anchor=(.98,.028),ncol=3,frameon=False,fontsize=10)
fig.subplots_adjust(left=.08,right=.98,top=.905,bottom=.10,hspace=.40)
fig.savefig(ROOT/'pressure_timeline.png',dpi=180)
fig.savefig(ROOT/'pressure_timeline.svg')
print('Figures saved')

