import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent;LAB=ROOT.parent.parent
s=json.loads((ROOT/'summary.json').read_text());assert s['completed_trials']==96
batch=json.loads((ROOT/'batching.json').read_text())
main=s['policies']['main_pressure'];long=s['policies']['long_tail']
names={'baseline':'Default','adaptive_resume':'Adaptive + resume waiver','slai_port':'SLAI core port','growth_only':'Growth only (128)','guard128':'Service + growth128','guard256':'Service + growth256'}
def ratio(group,p):return 1 if p=='baseline' else s['paired'][group][p+'_vs_baseline']['throughput_ratio_geo']
def gain_range(policy):
    vals=[b['gain_retained'] for b in batch if b['policy']==policy and b['gain_retained'] is not None]
    return f'{min(vals):.1%}–{max(vals):.1%}' if vals else 'not interpretable'
text=f'''# KV-cache pressure and streaming service

A focused empirical project asking whether already-started responses can keep receiving service when physical KV capacity is tight. Native vLLM experiments on an 8GB RTX4060 Laptop GPU; `dash.py` remains the live aggregate dashboard.

## Latest findings

**96 new trials / 1,152 completed requests** compare proactive growth admission, deadline service, the prior resume waiver, and a same-engine SLAI core-policy port. Two models, two main workload/order seeds, mixed lengths, clustered/staggered arrivals, larger-cache and lower-concurrency controls, plus a held-out long-output seed.

The **12 prespecified pressure workloads /144 requests per policy** show:

| Policy | Throughput / default | Streams with >1 s gap | Worst gap (s) | Evictions |
|---|---:|---:|---:|---:|
'''
for p,name in names.items():
    g=main[p];text+=f"| {name} | {ratio('main_pressure',p):.1%} | {g['stalled']} | {g['worst']:.3f} | {g['evictions']} |\n"
text+=f'''
Throughput retention is the geometric mean of matched ratios. Gaps measure actual client token-ID delivery **after the first delivery**. Main group selection uses configuration, not default eviction outcomes.

- **Proactive protection helps on the tested main workloads.** Reserve rolling room for active streams to grow, and pause fresh admission while a started stream is waiting. Adding early deadline service changes stalled streams {main['growth_only']['stalled']} → {main['guard128']['stalled']} but also changes replay {main['growth_only']['replay']:,} → {main['guard128']['replay']:,} positions.
- **Smoothness has admission and replay costs.** Median first-token latency for service128/256 is {s['paired']['main_pressure']['guard128_vs_baseline']['ttft_ratio_geo']:.2f}x / {s['paired']['main_pressure']['guard256_vs_baseline']['ttft_ratio_geo']:.2f}x default. Their 0.5 s client target misses are {main['guard128']['budget_misses']}/144 and {main['guard256']['budget_misses']}/144. Only{gain_range('guard128')} /{gain_range('guard256')} of incremental batching gain over cap3 remains in the three qualifying controls; retaining most total throughput does not meet the stronger batching-benefit objective.
- **The long tail exposes the limit.** On held-out outputs up to1,536 tokens, the strict0.25 s version retains {ratio('long_tail','guard_fast'):.1%} throughput, causes {long['guard_fast']['evictions']:,} evictions and {long['guard_fast']['replay']:,} replayed positions, and still reaches a {long['guard_fast']['worst']:.3f} s gap. No tested policy establishes a hard service bound.

The earlier resumption study found about88% of adaptive resumption waiting followed physical-capacity refusals. A reserve waiver helped partly but could not create cache. The broader workload study found the batching/smoothness tradeoff in7/8 tight-cache comparisons. This study tests proactive admission and service directly; the growth-only ablation and long-tail failure keep the attribution honest.

TBT scheduling and proactive KV reservation already have close prior art. SLAI is a core-policy port in our pinned engine, **not the original system or a reproduction of its published performance**. One GPU, synthetic forced outputs, short contexts and few seeds limit generalization.

## Reports and data

- [Streaming-service study](experiments/streaming_guard_20261003/REPORT.md) · [Related work](experiments/streaming_guard_20261003/RELATED_WORK.md) · [Figure](experiments/streaming_guard_20261003/streaming_results.png) · [Data/code bundle](experiments/streaming_guard_20261003/results.zip)
- [Resumption mechanism](experiments/resume_admission_20261003/REPORT.md)
- [Broader workloads](experiments/admission_robustness_20261002/REPORT.md)
- [Initial capacity cliff](experiments/capacity_cliff_20261002/REPORT.md)

Keep the four study directories together; later studies reuse earlier helpers. `experiments/` holds frozen designs, code, client records, native traces, audits and figures. `archive/setup/` preserves setup history. `activate.sh`, `requirements.lock.txt`, `.venv/` and `cuda-libs/` retain the runtime.

## Run

```bash
cd ~/kv-cache-lab
source activate.sh
python dash.py
```

The dashboard reads local port8017 metrics while an experiment server runs; benchmarks stop their servers afterward. Streaming gaps/replay require client records and traces beyond dashboard aggregates. To repeat the latest matrix:

```bash
python experiments/streaming_guard_20261003/run.py --prefix rerun
python experiments/streaming_guard_20261003/analyze.py --prefix rerun
```

Run folders are never overwritten; analysis updates derived exports. Preserve exports before rerunning. See the report for plotting, audit and bundle scope. Runtime: Python3.12, vLLM0.30.0, Torch2.13.0+cu132, FlashInfer0.6.18.post1.
'''
(LAB/'README.md').write_text(text.replace('about90','about 90').replace('about88','about 88').replace('All1','All 1').replace('are12','are 12').replace('order0','order 0').replace('over1','over 1').replace('above1','above 1').replace('over5','over 5').replace('below0','below 0').replace('cap12','cap 12').replace('cap6','cap 6').replace('cap3','cap 3').replace('seed381','seed 381').replace('seeds23','seeds 23').replace('up to1','up to 1').replace('Only33','Only 33').replace('Only{','Only {').replace('Only'+gain_range('guard128'),'Only '+gain_range('guard128')).replace('strict0.25','strict 0.25').replace('in7/8','in 7/8').replace('complete96','complete 96').replace('RTX4060','RTX 4060').replace('Python3.12','Python 3.12').replace('vLLM0.30','vLLM 0.30').replace('Torch2.13','Torch 2.13').replace('FlashInfer0.6','FlashInfer 0.6').replace('Llama3.2','Llama 3.2').replace('Qwen2.5','Qwen 2.5'))
print('Updated concise lab README; dashboard unchanged.')



