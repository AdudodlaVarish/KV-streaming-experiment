import json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
load=lambda n:json.loads((ROOT/(n+'.json')).read_text())
s=load('summary');checks=load('checks');old=load('previous_summary')
assert s['completed_trials']==76 and s['completed_requests']==912 and not checks['unfinished_cells']
assert old['completed_trials']==168 and old['completed_requests']==2016
p=s['policies']['baseline_pressure']; count=s['baseline_pressure_cases']; a=s['paired']['adaptive_resume_vs_adaptive']['baseline_pressure']; f=s['paired']['fixed_resume_vs_fixed']['baseline_pressure']
policies=['baseline','adaptive','adaptive_resume','fixed','fixed_resume']; labels={'baseline':'Default','adaptive':'Adaptive','adaptive_resume':'Adaptive + resume waiver','fixed':'Fixed 30%','fixed_resume':'Fixed + resume waiver'}
rows=[]
for policy in policies:
    r=p[policy]
    paired=s['paired'].get(policy+'_vs_baseline',{}).get('baseline_pressure',{})
    ratio=100*paired.get('throughput_ratio_geo',1)
    rows.append(f"| {labels[policy]} | {ratio:.1f}% | {r['stalled_requests']}/{r['requests']} | {r['evictions']} | {r['replay']:,} | {r['excess_s']:.2f} | {r['worst_gap']:.3f} |")
strata=[]
for key,r in s['paired']['adaptive_resume_vs_adaptive']['strata'].items():
    strata.append(f"| {key} | {r['pairs']} | {100*r['throughput_ratio_geo']:.1f}% | {r['reference_stalled']} → {r['policy_stalled']} | {r['reference_excess_s']:.2f} → {r['policy_excess_s']:.2f} |")
batching=load('batching');batchrows=[]
for r in batching:
    value='Not estimated: default gain ≤5%' if r['gain_retained'] is None else f"{100*r['gain_retained']:.1f}%"
    batchrows.append(f"| {r['seed']} | {r['arrival']} | {labels[r['policy']]} | {100*r['default_gain_ratio']:.1f}% | {value} |")
case=[r for r in load('analysis') if r['model']=='llama1b' and r['slots']==3072 and r['cap']==12 and r['seed']==23 and r['mix']=='generation' and r['arrival']=='stagger']
case={r['policy']:r for r in case}
eps=[e for e in load('episodes') if e['model']=='llama1b' and e['slots']==3072 and e['cap']==12 and e['seed']==23 and e['mix']=='generation' and e['arrival']=='stagger']
example=max([e for e in eps if e['policy']=='adaptive'],key=lambda e:e['wait_s'])
oldp=old['policies']['baseline_pressure']['baseline']; capshare=p['adaptive']['capacity_decision_s']/p['adaptive']['wait_sum_s']; wmshare=p['adaptive']['watermark_decision_s']/p['adaptive']['wait_sum_s']
resume_change=100*(1-a['policy_excess_s']/a['reference_excess_s'])
fixed_change=100*(1-f['policy_excess_s']/f['reference_excess_s']) if f['reference_excess_s'] else 0
report=f'''# Eviction counts versus interruption: measuring resumption delay

October 3, 2026 · RTX 4060 Laptop · vLLM 0.30.0

The targeted result is a **partial improvement, with a clear mechanism and failure boundary**. Waiving the admission reserve for preempted requests reduced excess pause time {resume_change:.1f}% versus the unchanged adaptive policy, at {100*a['throughput_ratio_geo']:.1f}% of its throughput. Stalled streams changed {a['reference_stalled']} → {a['policy_stalled']}; worst pause changed {a['reference_worst_gap']:.3f} → {a['policy_worst_gap']:.3f} seconds. Physical cache capacity remains the dominant resumption obstacle. This simple intervention does not bound silence. Across four Llama generation controls it retains 48–110% of the incremental gain over cap3, so nearly full throughput retention should not be confused with retaining the full batching benefit.

**Completed:** 76 new retained trials / 912 requests across eight server configurations, plus a fresh timing reanalysis of the previous 168 retained trials / 2,016 requests. Diagnostic trials and warmups are excluded. All native arrival orders, token counts, preemption/replay accounting, admission invariants and frozen evaluation hashes passed. See [checks](checks.json), [prospective design](DESIGN.md), and [related work](RELATED_WORK.md).

## 1. Waiting dominates the observed interruption

In the previous study's 16 default-eviction workloads, {oldp['wait_coverage_fraction']:.1%} of the **full duration of client gaps longer than one second** overlaps that same request's eviction-to-first-replay scheduling wait. The next client delivery followed the first replay scheduling step after a median {oldp['clean_replay_to_delivery_median']:.3f} seconds (P95 {oldp['clean_replay_to_delivery_p95']:.3f}). Adaptive and fixed headroom showed 96.9% coverage as well. The new default pressure runs independently show {p["baseline"]["wait_coverage_fraction"]:.1%} coverage, with {p["baseline"]["stalled_requests"]} stalled streams, all themselves evicted. All 29 default streams with >1-second pauses were themselves evicted.

This is temporal attribution, not a GPU-kernel decomposition: the second interval includes dispatch, GPU execution, asynchronous output processing and client transport. Multiple waits are unioned within each request before measuring coverage, so overlapping evictions are not double-counted. Conditional per-victim medians across policies compare different victims and should not be interpreted as paired causal estimates.

The proxy counterexample is broader than one anecdote: the previous adaptive policy had **fewer evictions but a worse maximum gap in 4 of 16 pressure workloads**; fixed headroom did so in 2. A 50 ms difference is used only as a descriptive reporting filter, not statistical significance.

![Waiting and replay timing](waiting_mechanism.png)

## 2. A controlled resumption intervention

Native vLLM already prepends preempted requests to its waiting queue. The intervention waives only the cache watermark when the native allocator considers a `PREEMPTED` request; fresh `WAITING` requests retain the same global reserve. Native physical-capacity and full-input checks, queue order, victim selection, and stale asynchronous-output handling remain active. No output-length predictor or future output oracle is supplied. Requests evicted before first delivery are eligible, too; the traces distinguish their client state.

Policies are default (zero watermark), fixed 30%, and the prior adaptive controller (.05 start, +.08 per eviction, −.02 after 128 calm nonempty steps, .45 maximum), with the latter two repeated with the resume waiver. Every trial resets controller state. Constants, matrix and evaluation code were frozen before diagnostic performance observations and were not retuned.

The matrix uses seeds 23/77 and shuffled configurations/policies: Llama 3.2 1B, 3,072 nominal KV slots/cap12, two heterogeneous length mixes and two arrival modes, all five policies (40 trials); Qwen 2.5 1.5B, 3,072/cap6, generation-heavy mix and both arrival modes, all policies (20); Llama 12,288/cap12 capacity-relief controls (12); Llama 3,072/cap3 batching controls (4). Twelve requests per workload. Clustered arrivals are spaced 50 ms; staggered arrivals add exponential spacing. Exact prompt IDs, output targets and native command lines are saved with each run.

BF16, block16, prefix caching disabled, eager asynchronous FlashInfer, maximum context and batch-token limit 2,048. Nominal cache sizes are 96/384 MiB for Llama and 84 MiB for Qwen; the allocator reserves a null block. Synthetic output lengths are forced with EOS ignored. Runtime and helper/native-source hashes are saved in [provenance](provenance.json).

## 3. Results on all {count} default-eviction workloads

Each policy has the same {12*count} request arrivals and targets in this subset. Throughput is generated tokens divided by workload wall time; retention is the geometric mean of paired ratios. Excess pause sums `max(0, gap − 1 s)` over all post-first-delivery gaps, whereas stalled-stream counts count each request once.

| Policy | Throughput / default | Streams with >1 s gap | Evictions | Replay positions | Excess pause (s) | Worst gap (s) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The adaptive waiver has {a['reference_evictions']} → {a['policy_evictions']} evictions and {a['reference_replay']:,} → {a['policy_replay']:,} replay positions, despite lower aggregate excess pause. There are {a['gap_improved']} paired workloads with a smaller worst gap, {a['gap_worsened']} with a larger one, and the rest within 50 ms. Fixed waiver reduces excess pause {fixed_change:.1f}% at {100*f['throughput_ratio_geo']:.1f}% of fixed-policy throughput, with {f['reference_stalled']} → {f['policy_stalled']} stalled streams and {f['reference_evictions']} → {f['policy_evictions']} evictions. Repeatedly evicted requests change {p['adaptive']['repeatedly_evicted_requests']} → {p['adaptive_resume']['repeatedly_evicted_requests']} for adaptive and {p['fixed']['repeatedly_evicted_requests']} → {p['fixed_resume']['repeatedly_evicted_requests']} for fixed. The balanced-length stratum regresses: excess pause rises from 0.20 to 0.45 s while throughput falls 4.3%; generation-heavy workloads show the aggregate improvement. Seed77 has no reduction in stalled-stream count. These are descriptive workload samples, not confidence bounds.

| Adaptive waiver versus adaptive, stratum | Pairs | Throughput retained | Stalled streams | Excess pause (s) |
|---|---:|---:|---:|---:|
{chr(10).join(strata)}

![Matched outcomes and allocation decisions](resume_results.png)

## 4. What actually prevents resumption?

For adaptive admission, sampled eviction-to-replay wait totals {p['adaptive']['wait_sum_s']:.2f} request-seconds: {p['adaptive']['capacity_decision_s']:.2f} s ({capshare:.1%}) follow physical-capacity refusals, {p['adaptive']['watermark_decision_s']:.2f} s ({wmshare:.1%}) follow reserve-only refusals, and {p['adaptive']['no_attempt_decision_s']:.2f} s precede the first traced attempt. The waiver produces {p['adaptive_resume']['binding_waiver_admissions']} admissions that the current watermark would have refused, removes reserve-only refusals, and leaves {p['adaptive_resume']['capacity_decision_s']:.2f} s of physical-capacity waiting. Fixed waiver has {p['fixed_resume']['binding_waiver_admissions']} binding admissions. An allocation check is classified using exact native full-input block requirements and free blocks.

These durations carry the last observed decision forward until the next attempt or replay step; they are sampled decision-state attribution, not a continuously observed cause or independent counterfactual. Native queue/stale-output handling can delay the first attempt. Policies can change later cache occupancy, victim choice and feedback, so removing reserve waiting does not imply an equally sized net reduction in pauses. The old study has no allocation-reason trace; its corresponding fields are null.

An illustrative continuation of the earlier counterexample—Llama seed23, generation-heavy, staggered, 3,072 slots/cap12—shows default/adaptive/resume worst gaps {case['baseline']['max_silence']:.3f}/{case['adaptive']['max_silence']:.3f}/{case['adaptive_resume']['max_silence']:.3f} s, evictions {case['baseline']['trace_preemptions']}/{case['adaptive']['trace_preemptions']}/{case['adaptive_resume']['trace_preemptions']}, and throughput {case['baseline']['throughput']:.1f}/{case['adaptive']['throughput']:.1f}/{case['adaptive_resume']['throughput']:.1f} tokens/s. Adaptive's longest-wait victim (request {example['index']}) spends {example['capacity_decision_s']:.3f} s after capacity refusals and {example['watermark_decision_s']:.3f} s after watermark refusals. Resume-aware execution changes the victim trajectory; it is not a same-victim replay experiment.

## 5. Throughput retention is not batching-gain retention

Keeping nearly all high-cap throughput can still discard much of the gain over a low cap. The contemporaneous Llama generation controls measure `(policy throughput − cap3 throughput) / (default cap12 throughput − cap3 throughput)`. Fractions are reported only if default high-cap throughput exceeds cap3 by >5%; these are cross-configuration comparisons with laptop clock/run variability.

| Seed | Arrivals | Policy | Default cap12 gain over cap3 | Incremental gain retained |
|---|---|---|---:|---:|
{chr(10).join(batchrows)}

TTFT retention here compares each trial's median first-delivery latency; completion retention compares its P95 completion latency, then geometrically averages paired ratios. Relative to adaptive, its waiver has {a['ttft_ratio_geo']:.3f}× median TTFT and {a['completion_ratio_geo']:.3f}× P95 completion. Relative to default, fixed headroom has {s['paired']['fixed_vs_baseline']['baseline_pressure']['ttft_ratio_geo']:.3f}× median TTFT and {s['paired']['fixed_vs_baseline']['baseline_pressure']['completion_ratio_geo']:.3f}× P95 completion. With just twelve requests per cell, P95 is a descriptive tail summary. The larger-cache controls have no evictions or >1 s client gaps; the files include every control result.

## 6. Replay accounting and research position

The original prompt/generation/iteration token metrics still report logical token volumes despite traced replay, and the additional `request_prefill_kv_computed_tokens_sum` also equals original prompt count in every retained trial with prefix caching disabled. Preemption counts themselves agree with the traces. This distinguishes logical usage from scheduled replay work; it is not evidence of incorrect billing or exact extra FLOPs. The detailed upstream context and narrower novelty limits are in [RELATED_WORK.md](RELATED_WORK.md).

The supported direction is an empirical characterization of **proxy mismatch and admission-to-resumption delay**, with a small ablation demonstrating both reserve-induced waiting and its limited influence when physical capacity dominates. A generic adaptive watermark or generic resume priority would overlap established work. The present waiver provides no interruption guarantee; incremental batching-gain retention varies substantially by workload. A stronger next intervention would need to address the future capacity of already-streaming requests and repeated victimization, with direct comparisons against TBT-aware systems.

## Reproduce and inspect

In the existing WSL lab, run with a fresh prefix (run directories are never overwritten):

```bash
cd ~/kv-cache-lab
source activate.sh
python experiments/resume_admission_20261003/run.py --prefix rerun
python experiments/resume_admission_20261003/analyze.py --prefix rerun
python experiments/resume_admission_20261003/analyze.py --previous
```

Analysis overwrites derived exports; copy them first to preserve this result. Figure generation uses the already installed plotting environment: `/home/varish/miniconda3/envs/kvzip/bin/python experiments/resume_admission_20261003/plot.py`. The report generator requires the complete 76-trial exports. Keep all three sibling experiment directories because helpers are reused. [results.zip](results.zip) contains this study, earlier raw studies/helpers, reports, figures and source manifests; models and runtime packages are excluded.

One GPU, two small models, two workload/order seeds, synthetic forced lengths, no tracing-overhead ablation, eager execution, and short contexts limit generalization. The timing comparison uses the same Linux/Python system-wide `perf_counter` clock, verified with bounded child-process reads in provenance; it is not a join between unrelated wall clocks. [Python documents this clock as system-wide](https://docs.python.org/3.12/library/time.html#time.perf_counter). No parameters were selected on evaluation performance. Performance is compared within this study; cross-day baseline differences are not policy effects.
'''
(ROOT/'REPORT.md').write_text(report)
print('Saved final REPORT.md from verified complete exports.')



