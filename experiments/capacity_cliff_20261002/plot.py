import json, pathlib, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=pathlib.Path(__file__).resolve().parent
rows=json.loads((ROOT/'analysis.json').read_text())
configs=['confirm_kv128_cap3','confirm_kv128_cap8','headroom_kv128_cap8','confirm_kv384_cap8']
labels=['128 MiB\ncap 3','128 MiB\ncap 8','128 MiB\ncap 8 + 30% headroom','384 MiB\ncap 8']
colors=['#315975','#DB7430','#267E8D','#478650']
groups=[[r for r in rows if r['config']==c] for c in configs]
assert all(len(g)==3 for g in groups), [len(g) for g in groups]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(14,10),gridspec_kw={'height_ratios':[.8,1.2]})
for ax,key,title,ylabel in [(axes[0,0],'output_tok_s','Throughput can reward aggressive admission','Output tokens / second'),(axes[0,1],'max_stream_gap_s','Streaming pauses tell a different story','Longest pause in each burst (seconds)')]:
    medians=[statistics.median([r[key] for r in g]) for g in groups]
    ax.bar(range(4),medians,color=colors,width=.65,alpha=.85)
    for i,(m,g) in enumerate(zip(medians,groups)):
        ax.scatter([i-.08,i,i+.08],[r[key] for r in g],color='#172d3d',s=22,zorder=3)
        ax.text(i,m+max(medians)*.045,f'{m:.2f}' if key=='max_stream_gap_s' else f'{m:.0f}',ha='center',fontweight='bold')
    ax.set_xticks(range(4),labels,fontsize=10)
    ax.set_ylim(0,max(medians)*1.23)
    ax.set_ylabel(ylabel)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=14)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
for ax,config,title in [(axes[1,0],configs[0],'Cap 3: wait first, then stream smoothly'),(axes[1,1],configs[1],'Cap 8: start earlier, then some streams stall')]:
    records=json.loads((ROOT/config/'rep0'/'requests.json').read_text())
    origin=min(r['start'] for r in records)
    records.sort(key=lambda r:r['start']+r['ttft_s'])
    for i,r in enumerate(records):
        start=r['start']-origin
        ttft=start+r['ttft_s']
        end=start+r['elapsed_s']
        ax.broken_barh([(start,ttft-start)],(i-.3,.6),facecolors='#dce2e7')
        ax.broken_barh([(ttft,end-ttft)],(i-.3,.6),facecolors='#4283ae')
        times=[start+t for t in r['stream_event_times_s']]
        gaps=[(a,b-a) for a,b in zip(times,times[1:]) if b-a>1]
        ax.broken_barh(gaps,(i-.3,.6),facecolors='#d75445')
    ax.set_xlim(0,24)
    ax.set_ylim(-.8,11.8)
    ax.invert_yaxis()
    ax.set_yticks(range(12),[str(i+1) for i in range(12)])
    ax.set_ylabel('Requests ordered by first text')
    ax.set_xlabel('Seconds from burst launch')
    ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=12)
    ax.grid(axis='x',alpha=.12)
fig.legend(handles=[Patch(color='#dce2e7',label='Before first text'),Patch(color='#4283ae',label='Streaming interval'),Patch(color='#d75445',label='Pause >1 second')],loc='center right',bbox_to_anchor=(.98,.558),ncol=3,frameon=False,fontsize=9)
fig.suptitle('A KV-cache tradeoff hidden by aggregate latency',x=.075,ha='left',fontsize=20,fontweight='bold',y=.98)
fig.text(.075,.935,'Llama 3.2 1B · RTX 4060 8 GB · vLLM 0.30 + FlashInfer · 12 requests × (768 input + 384 output tokens)',fontsize=11,color='#4b5965')
fig.text(.075,.025,'Bars: median of three bursts. Dots: individual bursts. Timelines: first measured burst.\nPauses are between nonempty text chunks at the client; they are not exact per-token kernel timings.',fontsize=10,color='#4b5965')
fig.subplots_adjust(left=.075,right=.98,top=.875,bottom=.12,hspace=.46,wspace=.25)
fig.savefig(ROOT/'findings.png',dpi=180)
fig.savefig(ROOT/'findings.svg')
print(ROOT/'findings.png')


