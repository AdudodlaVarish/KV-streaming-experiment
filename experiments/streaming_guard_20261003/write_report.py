import json,pathlib,statistics,collections
ROOT=pathlib.Path(__file__).resolve().parent
load=lambda n:json.loads((ROOT/(n+'.json')).read_text())
s=load('summary');rows=load('analysis');checks=load('checks');batch=load('batching')
assert (s['completed_trials'],s['completed_requests'],s['completed_configs'])==(96,1152,9)
assert not checks['unfinished'] and all(v for v in checks.values() if isinstance(v,bool))
main=s['policies']['main_pressure'];long=s['policies']['long_tail'];relief=s['policies']['relief']
names={'baseline':'Default FCFS','adaptive_resume':'Adaptive + resume waiver','slai_port':'SLAI core port','growth_only':'Growth only (128)','guard128':'Service + growth128','guard256':'Service + growth256','guard_fast':'Service target 0.25 s','guard_relaxed':'Service target 1 s'}
mainorder=['baseline','adaptive_resume','slai_port','growth_only','guard128','guard256']
stressorder=['baseline','slai_port','guard128','guard256','guard_fast','guard_relaxed']
ratio=lambda group,p,field='throughput_ratio_geo':1 if p=='baseline' else s['paired'][group][p+'_vs_baseline'][field]
def table(group,policies):
    out=['| Policy | Throughput / default | Streams >0.5 / >1 s | Worst gap (s) | Evictions | Replay positions | Median TTFT / default |','|---|---:|---:|---:|---:|---:|---:|']
    for p in policies:
        g=s['policies'][group][p]
        out.append(f"| {names[p]} | {ratio(group,p):.1%} | {g['stalled_gt_0.5']} / {g['stalled']} | {g['worst']:.3f} | {g['evictions']:,} | {g['replay']:,} | {ratio(group,p,'ttft_ratio_geo'):.2f}x |")
    return '\n'.join(out)
g=main['guard128'];g256=main['guard256'];growth=main['growth_only'];baseline=main['baseline'];slai=main['slai_port'];waiver=main['adaptive_resume']
ab=s['paired']['main_pressure']['guard128_vs_growth_only'];vs=s['paired']['main_pressure']['guard128_vs_slai_port']
def gain_range(policy):
    vals=[b['gain_retained'] for b in batch if b['policy']==policy and b['gain_retained'] is not None]
    return f'{min(vals):.1%}–{max(vals):.1%}' if vals else 'not interpretable'
text=f'''# Streaming-aware admission and service under KV pressure

**96 retained GPU trials / 1,152 completed requests.** The proactive policy reduced main-workload streams with client gaps over one second from **{baseline['stalled']}/144 to {g['stalled']}/144**, at **{ratio('main_pressure','guard128'):.1%}** of matched default throughput. The larger growth allowance reached **{g256['stalled']}/144** at **{ratio('main_pressure','guard256'):.1%}**. This is a measured tradeoff, not a service guarantee: deadline misses and the held-out long tail matter as much as the ordinary-workload improvement.

In the qualifying low-concurrency comparisons, service128 retains **{gain_range('guard128')}** of the incremental batching gain, and service256 **{gain_range('guard256')}**. Retaining about90% of total throughput therefore does not mean retaining most of the extra batching benefit in every workload. The combined objective is not established. These descriptive ranges apply to the frozen controls, not to an optimized policy.

The useful distinction is **protecting current streams before new admission versus repeatedly repairing an interruption after capacity is exhausted**. A growth-only ablation reached {growth['stalled']}/144 stalled streams at {ratio('main_pressure','growth_only'):.1%} throughput. Adding deadline ordering, early cache handoff and temporary repeat-victim protection changed throughput to {ab['throughput_ratio_geo']:.1%} of that ablation, evictions {growth['evictions']} → {g['evictions']}, replay {growth['replay']:,} → {g['replay']:,}, and worst gap {growth['worst']:.3f} → {g['worst']:.3f} s. The extra machinery should be judged by that ablation rather than assumed to help.

## What was tested

The main service policy gives each resident stream a rolling allowance of 128 additional positions (256 in the second version). Fresh requests enter only if all current growth allowances plus the candidate fit free blocks, and no already-started stream is paused. The allowance includes unfinished prefills; resumed requests still need physically sufficient cache for their known context. An idle escape allows a fitting prompt when there are no resident or paused streams. The policy does not use output targets or remaining lengths.

Already-started requests precede fresh ones, ordered by observed engine-output deadline. A request evicted within the last second receives one target interval of priority credit. After half the target interval of silence, a paused stream may take cache from one less-urgent, safely evictable resident per scheduler pass. Native async completion, stale-output handling and physical allocator checks remain active. The default target is 0.5 s; the held-out study also tests 0.25 and 1 s. This is a scheduling preference and early corrective handoff, not a hard client deadline.

Controls: native FCFS; the prior adaptive reserve with resume waiver; growth admission alone; a **same-engine SLAI core-policy port**; larger-cache relief; and contemporaneous low-concurrency FCFS. The SLAI port implements critical decode, active/fresh prefill, then noncritical decode service, with published urgency and victim rules. It uses the existing vLLM stack rather than the original Sarathi/Torch2.3/CUDA12.1 engine. Native resume feasibility, output history, async completion and scheduling rollback constrain the adaptation; active-prefill ties use prompt length. Its batch-time estimate is dispatch-to-output delay, not kernel profiling. This is not a reproduction of SLAI's published performance. Exact adaptations are in [DESIGN.md](DESIGN.md) and [RELATED_WORK.md](RELATED_WORK.md); reference commit `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380`.

Hardware/runtime: RTX4060 Laptop 8GB, Python3.12.14, vLLM0.30.0, Torch2.13.0+cu132, FlashInfer0.6.18.post1; BF16, eager async execution, block16, context/batch limit2048, prefix caching off. Two model sizes: Llama3.2 1B and Qwen2.5 1.5B. Main seeds23/77 shuffle both configurations and policy/workload order. Twelve requests per trial use mixed prompt/output lengths and clustered or staggered arrivals. The pressure configurations have 3,072 nominal KV positions at cap12 / cap6; relief has 12,288 positions. Llama cap3 controls measure how much batching gain remains. Held-out seed381 uses independent prompt/output mixes with outputs up to1,536 tokens, exceeding both allowances. All requests finish without rejection, timeout cancellation or truncation.

## Main pressure comparison

The prespecified main group has **12 workload cells / 144 requests per policy**: both seeds, models and arrival modes, plus balanced and generation-heavy Llama mixes. It is selected by configuration, not by whether default eviction occurred. Larger-cache, cap3 and long-tail trials are reported separately.

{table('main_pressure',mainorder)}

Throughput and latency ratios are geometric means of matched workload ratios. TTFT ratios compare workload median time to first token; completion ratios below compare workload P95 completion latency. A stalled stream has at least one client token-ID delivery gap over1 s **after its first delivery**. Each gap is a delivery event interval, not necessarily a separate token when events carry multiple IDs.

The service128 policy versus the SLAI core port: **{vs['throughput_ratio_geo']:.1%}** throughput; streams over1 s {slai['stalled']} → {g['stalled']}; streams over0.5 s {slai['stalled_gt_0.5']} → {g['stalled_gt_0.5']}; evictions {slai['evictions']} → {g['evictions']}. The port favors some batching opportunities differently; retain the engine-specific scope when interpreting this comparison.

Target attainment must remain visible. Service128 misses its 0.5 s client target for **{g['budget_misses']}/144** streams; service256 for **{g256['budget_misses']}/144**. Engine callbacks themselves exceed the target for {g['engine_over_budget']} and {g256['engine_over_budget']} streams. Service128's worst gap is {g['worst']:.3f} s; reducing counts does not establish a bound. Relative P95 completion ratios are {ratio('main_pressure','guard128','completion_ratio_geo'):.3f} and {ratio('main_pressure','guard256','completion_ratio_geo'):.3f}. Aggregate excess pause above1 s is {baseline['excess']:.3f} s default, {waiver['excess']:.3f} resume waiver, {slai['excess']:.3f} SLAI port, {growth['excess']:.3f} growth-only, {g['excess']:.3f} service128, {g256['excess']:.3f} service256.

### Does the effect persist across workloads?

| Service128 vs default stratum | Cells | Throughput retained | >1 s streams, default → service | Worst gap, default → service (s) |
|---|---:|---:|---:|---:|
'''
for key,p in s['guard_strata'].items():
    text+=f"| {key} | {p['pairs']} | {p['throughput_ratio_geo']:.1%} | {p['reference_stalled']} → {p['policy_stalled']} | {p['reference_worst']:.3f} → {p['policy_worst']:.3f} |\n"
text+='\nThese overlapping strata describe the observed cells; two main seeds do not support broad statistical generalization. Per-cell ratios, regressions and every request are in the CSV/JSON exports.\n\n'
text+='### Retaining the batching benefit\n\n'
text+='The four paired Llama generation controls compare cap12 with cap3 at identical cache/workload settings. Incremental gain retention is `(policy throughput − cap3 throughput) / (default cap12 throughput − cap3 throughput)`, reported only if default cap12 is over5% faster. It is stricter than total throughput retention; a value below0 loses the observed cap3 benefit.\n\n'
text+='| Policy | Qualifying controls | Incremental gain retained, range |\n|---|---:|---:|\n'
for p in mainorder[1:]:
    vals=[b['gain_retained'] for b in batch if b['policy']==p and b['gain_retained'] is not None]
    text+=f"| {names[p]} | {len(vals)} | {min(vals):.1%}–{max(vals):.1%} |\n" if vals else f"| {names[p]} | 0 | Not interpretable |\n"
low=[r for r in rows if r['cap']==3]
text+=f"\nCap3 controls: {sum(r['stalled_requests'] for r in low)} stalled streams, {sum(r['trace_preemptions'] for r in low)} evictions. Relief controls: default {relief['baseline']['stalled']} and service128 {relief['guard128']['stalled']} stalled streams, worst gaps {relief['baseline']['worst']:.3f} / {relief['guard128']['worst']:.3f} s; service128 throughput {ratio('relief','guard128'):.1%} of default.\n\n"
text+=f'''## Held-out long tail: the failure/cost test

**Two cells / 24 requests per policy**, seed381; long outputs exceed the fixed growth allowance. No parameters were changed after diagnostic observations or after evaluation began.

{table('long_tail',stressorder)}

| Service policy | Its target (s) | Streams missing that target /24 | Early handoffs | Repeat victims | P95 completion / default |
|---|---:|---:|---:|---:|---:|
'''
for p in stressorder[2:]:
    text+=f"| {names[p]} | {next(r['budget'] for r in rows if r['seed']==381 and r['policy']==p):g} | {long[p]['budget_misses']} | {long[p]['handoffs']:,} | {long[p]['repeat_victims']} | {ratio('long_tail',p,'completion_ratio_geo'):.2f}x |\n"
text+=f'''
The strict 0.25 s version reaches a worst gap of **{long['guard_fast']['worst']:.3f} s**, with **{long['guard_fast']['evictions']:,} evictions / {long['guard_fast']['replay']:,} replayed positions**, and **{ratio('long_tail','guard_fast'):.1%}** of default throughput. A one-second target reaches {long['guard_relaxed']['worst']:.3f} s at {ratio('long_tail','guard_relaxed'):.1%}. The tradeoff cannot be summarized by the number of gaps over1 s alone: repeated handoffs can keep streams visibly moving while spending substantial work rebuilding their context and delaying completion. Temporary victim credit does not prevent repeated eviction.

For the default 0.5 s variants, client target misses are {long['guard128']['budget_misses']}/24 (128) and {long['guard256']['budget_misses']}/24 (256). The selected allowance and target are fixed empirical choices; these results do not demonstrate a robust controller for unbounded generation. A next implementation would need to handle long-context replay cost explicitly before tightening service targets, and must retain this stress test and the growth-only ablation.

A **post hoc handoff audit** joins each corrective eviction to the next native scheduling pass. Service128 schedules its intended target in {long['guard128']['handoff_target_next_pass']:,}/{long['guard128']['handoffs']:,} such passes; the strict version in {long['guard_fast']['handoff_target_next_pass']:,}/{long['guard_fast']['handoffs']:,}. One victim does not always free enough cache for the target's full known context. Absence of immediate dispatch does not prove a useless eviction: several victims or a later pass may be needed. These exploratory counts identify a further audit dimension; they were not used to retune or select any trial.

## Measurement and audit

The [prospective design](DESIGN.md), policy/driver hashes and workload/order rules were frozen before eight excluded diagnostic trials. Diagnostics exposed severe strict-target replay cost; settings stayed fixed. One accounting edge case required an explicit measurement correction **before the main study**: an evicted request may finish from a final in-flight output without replaying the lost prefix. [MEASUREMENT_NOTES.md](MEASUREMENT_NOTES.md) documents this. Final accounting conserves **lost positions = scheduled replay + terminal discarded context**; main evaluation has {sum(r['terminal_evictions'] for r in rows)} such terminal episodes and {sum(r['terminal_discard'] for r in rows):,} discarded positions. Scheduled replay is not executed FLOPs or billing.

All1,152 requested outputs, API usage and native output-service counts agree; native success counts are12 per trial. Native arrival order0..11 is verified, preemptions match metrics, replay/terminal context conserves, every admitted growth inequality and native physical check holds, urgency thresholds/victim ordering are audited, and frozen source/runtime helper hashes are retained. Diagnostics, warmups and any order-invalid attempts are excluded and preserved, never selected by performance. Cross-process monotonic clock reads are bounded in provenance. Logical prompt/generation/iteration and extended prefill-computed counters continue to omit replay; replay is audited separately from scheduler traces.

The study measures one laptop GPU, short contexts, forced output lengths and eager async execution. Tracing is enabled for every policy, but its overhead was not separately measured; policies can generate different numbers of trace events. Only one held-out seed covers the long tail. The port retains vLLM constraints and does not establish superiority over complete SLAI, CacheOPT or Chronos systems. Growth prediction, TBT-aware scheduling and KV congestion control already have close prior art; [RELATED_WORK.md](RELATED_WORK.md) records the positioning. The supported contribution is the measured admission/service/replay tradeoff, its ablations and its failure conditions.

## Artifacts and reproduction

- [Main scientific figure](streaming_results.png) / [PDF](streaming_results.pdf)
- [Held-out cost figure](long_tail_cost.png) / [PDF](long_tail_cost.pdf)
- [Matched summaries](summary.json), [trial table](analysis.csv), [requests](requests.csv), [comparisons](pairs.csv), [eviction episodes](episodes.csv), [batching controls](batching.csv)
- [Growth decisions](growth.csv), [early handoffs](handoffs.csv), [handoff effects](handoff_effects.csv), [checks](checks.json), [provenance](provenance.json), [frozen hashes](evaluation_code_hashes.json)
- [Data/code bundle](results.zip) and [bundle manifest](bundle_manifest.json)

Keep the four study directories together because the current client and audit reuse the earlier helpers. From the existing WSL environment:

```bash
cd ~/kv-cache-lab
source activate.sh
python experiments/streaming_guard_20261003/run.py --prefix rerun
python experiments/streaming_guard_20261003/analyze.py --prefix rerun
/home/varish/miniconda3/envs/kvzip/bin/python experiments/streaming_guard_20261003/plot.py
```

Run folders are never overwritten. The analyzer/plotter update derived exports; preserve them before a rerun. `write_report.py` and `finalize.py` apply to a complete96-trial matrix. Model snapshots are local and not bundled. Existing prior-study raw bundles remain separate. `dash.py` remains unchanged.
'''
(ROOT/'REPORT.md').write_text(text.replace('about90','about 90').replace('about88','about 88').replace('All1','All 1').replace('are12','are 12').replace('order0','order 0').replace('over1','over 1').replace('above1','above 1').replace('over5','over 5').replace('below0','below 0').replace('cap12','cap 12').replace('cap6','cap 6').replace('cap3','cap 3').replace('seed381','seed 381').replace('seeds23','seeds 23').replace('up to1','up to 1').replace('Only33','Only 33').replace('Only{','Only {').replace('Only'+gain_range('guard128'),'Only '+gain_range('guard128')).replace('strict0.25','strict 0.25').replace('in7/8','in 7/8').replace('complete96','complete 96').replace('RTX4060','RTX 4060').replace('Python3.12','Python 3.12').replace('vLLM0.30','vLLM 0.30').replace('Torch2.13','Torch 2.13').replace('FlashInfer0.6','FlashInfer 0.6').replace('Llama3.2','Llama 3.2').replace('Qwen2.5','Qwen 2.5'))
print('Wrote complete verified-matrix REPORT.md.')





