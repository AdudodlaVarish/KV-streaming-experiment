"""Generate the report only after all prospective evaluation trials pass checks."""
import json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parent
checks=json.loads((ROOT/'checks.json').read_text())
assert checks['completed_trials']==168 and checks['completed_requests']==2016 and not checks['unfinished_cells']
assert all(checks[k] for k in ('token_counts_verified','preemption_trace_matches_metrics','replayed_positions_match_evicted','policy_state_verified','diagnostics_excluded','engine_arrival_order_verified'))
rows=json.loads((ROOT/'analysis.json').read_text())
s=json.loads((ROOT/'summary.json').read_text())
x=json.loads((ROOT/'extra_analysis.json').read_text())
assert x['locked_hashes']==x['actual_hashes'],'Evaluation source changed during the run'
batching=json.loads((ROOT/'batching.json').read_text())
base=[r for r in rows if r['policy']=='baseline']
pressure=[r for r in base if r['trace_preemptions']>0]
pressure_keys={tuple(r[k] for k in ('model','slots','cap','seed','mix','arrival')) for r in pressure}
worst_by_policy={policy:max(r['max_silence'] for r in rows if r['policy']==policy and tuple(r[k] for k in ('model','slots','cap','seed','mix','arrival')) in pressure_keys) for policy in ('baseline','fixed','adaptive')}
control=[r for r in base if r['slots']==12288]
low=[r for r in base if r['model']=='llama1b' and r['cap']==3]
a=s['pressure']['adaptive']; f=s['pressure']['fixed']
def pct(v): return f'{100*v:.1f}%'
def remain(p,key): return pct(p['policy_'+key]/p['baseline_'+key]) if p['baseline_'+key] else 'n/a'
def table(header, body):
    return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in body)
policy_table=table(['Policy','Throughput retained*','Requests with >1 s gaps','Evictions','Replayed positions','Excess silence, seconds','Worst gap, seconds','Median TTFT ratio*','p95 completion ratio*'],[
    ['Default','100%',a['baseline_stalled_requests'],a['baseline_evictions'],f"{a['baseline_replay']:,}",f"{a['baseline_excess_s']:.2f}",f"{worst_by_policy['baseline']:.2f}",'1.00','1.00'],
    ['Fixed 30% headroom',pct(f['throughput_ratio_geomean']),f['policy_stalled_requests'],f['policy_evictions'],f"{f['policy_replay']:,}",f"{f['policy_excess_s']:.2f}",f"{worst_by_policy['fixed']:.2f}",f"{f['ttft_p50_ratio_geomean']:.2f}",f"{f['e2e_p95_ratio_geomean']:.2f}"],
    ['Adaptive headroom',pct(a['throughput_ratio_geomean']),a['policy_stalled_requests'],a['policy_evictions'],f"{a['policy_replay']:,}",f"{a['policy_excess_s']:.2f}",f"{worst_by_policy['adaptive']:.2f}",f"{a['ttft_p50_ratio_geomean']:.2f}",f"{a['e2e_p95_ratio_geomean']:.2f}"]])
slice_table=table(['Default-admission slice','Trials','Trials with eviction','Trials with >1 s gaps','Stalled requests / requests','Worst gap, seconds'],[
    [key,val['trials'],val['trials_with_evictions'],val['trials_with_stalls'],f"{val['stalled_requests']} / {val['requests']}",f"{val['max_silence']:.2f}"] for key,val in x['baseline_slices'].items()])
policy_slices=table(['Pressure-case slice','Pairs','Adaptive throughput retained','Default stalled requests','Adaptive stalled requests','Fixed stalled requests'],[
    [key,val['pressure']['adaptive']['pairs'],pct(val['pressure']['adaptive']['throughput_ratio_geomean']),val['pressure']['adaptive']['baseline_stalled_requests'],val['pressure']['adaptive']['policy_stalled_requests'],val['pressure']['fixed']['policy_stalled_requests']]
    for key,val in s['strata'].items() if key.split('=')[0] in ('model','arrival','seed') and 'adaptive' in val['pressure']])
threshold_table=table(['Gap threshold','Default stalled requests','Fixed stalled requests','Adaptive stalled requests'],[
    [key+' s',val['adaptive']['baseline_stalled'],val['fixed']['policy_stalled'],val['adaptive']['policy_stalled']] for key,val in x['thresholds'].items()])
batching_table=table(['KV slots','Higher cap','Paired workloads/seeds','Throughput gain + more stalls'],[
    [slots,cap,len(group),sum(r['tradeoff'] for r in group)]
    for slots,cap in sorted({(r['slots'],r['cap']) for r in batching})
    for group in [[r for r in batching if r['slots']==slots and r['cap']==cap]]])
gain=[r for r in batching if 'adaptive_batching_gain_retained' in r]
pressure_gain=[r for r in gain if r['high_stalled']>0]
gain_median=statistics.median(r['adaptive_batching_gain_retained'] for r in gain)
pressure_gain_median=statistics.median(r['adaptive_batching_gain_retained'] for r in pressure_gain) if pressure_gain else None
case_tables=[]
for model,seed,arrival,cap in [('llama1b',77,'burst',12),('llama1b',23,'stagger',12),('qwen1.5b',77,'burst',6)]:
    selected=[r for r in rows if r['model']==model and r['slots']==3072 and r['cap']==cap and r['seed']==seed and r['mix']=='generation' and r['arrival']==arrival]
    selected.sort(key=lambda r:['baseline','fixed','adaptive'].index(r['policy']))
    case_tables.append(f'**{model}, 3,072 slots, cap {cap}, generation mix, {arrival}, seed {seed}**\n\n'+table(['Policy','Output tok/s','Stalled streams / 12','Worst gap, s','Evictions','Replay positions','Longest eviction-to-resume wait, s'],[
        [r['policy'],f"{r['throughput']:.1f}",r['stalled_requests'],f"{r['max_silence']:.3f}",r['trace_preemptions'],r['rescheduled_token_positions'],f"{r['preempt_to_resume_max_s']:.3f}"] for r in selected]))
provenance=dict(date='2026-10-02 local Chicago start',hardware='NVIDIA RTX 4060 Laptop GPU, 8188 MiB, 55 W',software=dict(vllm='0.30.0',torch='2.13.0+cu132',flashinfer='0.6.18.post1'),models=json.loads((ROOT/'design.json').read_text())['model_paths'],weights_and_kv_dtype='BF16',bytes_per_kv_token=dict(llama1b=32768,qwen15b=28672),evaluation_hashes=x['actual_hashes'],gpu_snapshot_ranges=x['gpu_snapshot_ranges'],checks=checks)
provenance['order_replacement']=json.loads((ROOT/'order_replacement.json').read_text())
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2))
content=f'''# Admission pressure across mixed workloads

Completed controlled local GPU study: **168 retained evaluation trials, 2,016 requests**, with all token counts, arrival orders, eviction counters, replay accounting, and policy states verified. Setup diagnostics and early uncontrolled simultaneous-arrival batches are excluded. Evaluation parameters were frozen before the retained runs. One entire 12-trial configuration was replaced after an engine arrival-order violation; the excluded run is preserved, and the repeat uses unchanged code, workload, settings, warmups, and policy order. The first repeat passing the order check is retained. Its placement at the end differs from the prospective configuration order; this is documented in order_replacement.json.

## What the experiment establishes

Admission pressure remains relevant when prompt lengths, output lengths, arrivals, cache capacity, concurrency, model architecture, and random orders vary. The throughput-versus-streaming tradeoff is conditional: it appears under pressure, and relaxes when sufficient cache capacity or lower concurrency prevents eviction. It is not present in every sampled workload.

Across the {len(batching)} available paired larger-cap versus cap-3 comparisons, **{sum(r['tradeoff'] for r in batching)}** had both more than 3% higher throughput and more streams with a post-first-token gap over one second. This is a descriptive filter, not a significance test. The low-cap baseline controls had {sum(r['trace_preemptions'] for r in low)} evictions and {sum(r['stalled_requests'] for r in low)} stalled requests across {len(low)} trials. The 12,288-slot baseline controls had {sum(r['trace_preemptions'] for r in control)} evictions and {sum(r['stalled_requests'] for r in control)} stalled requests across {len(control)} trials; their worst gap was {max(r['max_silence'] for r in control):.3f} seconds.

{batching_table}

The adaptive controller reduced eviction/replay burden, but its pause reduction must be judged separately. In the **{len(pressure)} workloads where default admission evicted requests**, it retained {pct(a['throughput_ratio_geomean'])} of paired baseline throughput. Stalled requests changed from {a['baseline_stalled_requests']} to {a['policy_stalled_requests']}; evictions from {a['baseline_evictions']} to {a['policy_evictions']}; replayed positions from {a['baseline_replay']:,} to {a['policy_replay']:,}. It improved the number of stalled streams in {a['count_stall_improved']} cases and worsened it in {a['count_stall_worsened']}; {a['fewer_stalls_retains_90pct']} cases improved while retaining at least 90% throughput. The constant-reserve policy provides a useful conservative comparator.

## Policy comparison under pressure

{policy_table}

*Throughput and latency ratios are geometric means of paired, per-trial ratios. Counts and excess silence are summed across the same baseline-pressure cases; each policy has {len(pressure)*12} requests in this subset. TTFT is client first-token delay. Completion p95 is calculated among the twelve requests in each trial. Excess silence sums the portion above one second for every inter-delivery gap after the first token. Counts can improve even if the single worst pause increases.*

Across **all 56 matched workloads**, adaptive retained {pct(s['all']['adaptive']['throughput_ratio_geomean'])} throughput and fixed headroom retained {pct(s['all']['fixed']['throughput_ratio_geomean'])}. Across all cases, default had {s['all']['adaptive']['baseline_stalled_requests']} stalled requests, adaptive had {s['all']['adaptive']['policy_stalled_requests']}, and fixed headroom had {s['all']['fixed']['policy_stalled_requests']}. Adaptive improved the number of stalled streams in {s['all']['adaptive']['count_stall_improved']} cases and worsened it in {s['all']['adaptive']['count_stall_worsened']}. Averaging in many pressure-free controls can conceal the behavior of interest, which is why pressure cases are reported separately. All-case totals also include any failures that a policy introduces when the baseline was smooth.

Keeping 90% of total throughput is different from keeping 90% of the gain from batching. Where default admission beat cap 3 by more than 5%, the adaptive policy retained a median **{pct(gain_median)} of that incremental gain** across {len(gain)} comparisons. This fraction is `(adaptive - cap3) / (default high-cap - cap3)` and can exceed 100% or be negative. For those comparisons that also had default post-start stalls, the median is {pct(pressure_gain_median) if pressure_gain_median is not None else 'n/a'} across {len(pressure_gain)} comparisons. Among these {len(pressure_gain)} initially stalled comparisons with a measurable batching gain, adaptive reduced the number of stalled streams and kept at least half of that incremental gain in {sum(r["adaptive_stalled"]<r["high_stalled"] and r["adaptive_batching_gain_retained"]>=.5 for r in pressure_gain)}. The 5% denominator filter is descriptive, not statistical.

![Workloads, paired throughput retention, and replay burden](robustness.png)

## Where the pattern appeared

{slice_table}

{policy_slices}

These are overlapping slices, not independent samples. Both seeds randomize request pairings, server-configuration order, and policy order. The secondary model differs in architecture as well as parameter count; its two sampled configurations are an architecture check, not an isolated model-size experiment.

{threshold_table}

The threshold sensitivity table uses exactly the baseline-pressure subset at each threshold. It does not choose or drop cases based on the adaptive result. Default stalled requests in trials without an eviction: **{x['baseline_stalled_in_non_eviction_trials']}**. Of {x['baseline_stalled_requests_total']} default requests stalled beyond one second, {x['baseline_stalled_requests_ever_evicted']} were themselves evicted at some point. An engine eviction can precede delivery of an already in-flight first token; client delivery and scheduler events are not perfectly aligned.

## Concrete cases and the controller's limitation

{case_tables[0]}

This burst case was selected for the timeline before observing the retained evaluation outcomes.

![Arrival, first-token queueing, streaming gaps, and controller reserve](pressure_timeline.png)

{case_tables[1]}

The staggered seed-23 case is a counterexample to assuming that fewer evictions guarantee shorter pauses. Adaptive admission replayed fewer positions, but the worst stream waited longer to resume. The trace shows the victim's wait until its first rescheduled prefill grew from about 2.63 to 3.53 seconds. Native watermark reservation applies when admitting both fresh and preempted waiting requests; increasing it can delay the victim too. Also, the feedback reacts after an eviction and starts fresh in every trial, so it cannot prevent all initial pressure events.

{case_tables[2]}

The Qwen seed-77 case also tests a limitation of constant reserve: full-input admission cannot predict the future KV demand of long outputs. Even reserved headroom can be exhausted after several streams begin decoding. Neither tested policy is a pause guarantee.

A next controller design could distinguish fresh admissions from resuming streams, or react to observed pause budgets before another eviction. That is an untested extension. These results evaluate one simple feedback policy and a fixed reserve, not an optimal adaptive admission policy.

## Design and measurement

- **Hardware/software:** RTX 4060 Laptop, 8,188 MiB, 55 W; local vLLM 0.30.0, Torch 2.13.0+cu132, FlashInfer 0.6.18.post1. Eager asynchronous execution; BF16 weights and KV; block size 16; max context and per-step batched tokens both 2,048; prefix caching disabled. GPU snapshots and locked evaluation source hashes are in `provenance.json`.
- **Primary:** Llama 3.2 1B, six `(nominal KV token slots, cap)` points: `(3072,3)`, `(3072,12)`, `(5120,3)`, `(5120,6)`, `(5120,12)`, `(12288,12)`. Corresponding KV allocations are 96, 160, and 384 MiB. This is a fractional six-point design, not a full cache-by-cap factorial.
- **Secondary:** Qwen 2.5 1.5B at `(3072,6)` and `(12288,12)`, using 84 and 336 MiB. Token capacity is matched, not bytes. A null block consumes 16 nominal token slots in both models.
- **Workloads:** twelve requests per trace. Balanced: inputs 128/256/512/1024 and outputs 64/128/256/512, three of each independently shuffled. Generation-heavy: inputs 128/256/512 and outputs 64/256/768, four of each independently shuffled. Exact tokenizer-ID prompts and arrival times are saved. Secondary runs use only the generation-heavy mix.
- **Arrivals/order:** clustered burst spacing 50 ms, all twelve within 0.55 seconds; staggered spacing 50 ms plus an exponential draw averaging 0.95 seconds. Seeds 23 and 77 select pairings and orders. Every paired policy shares the exact workload, and engine arrival order 0–11 is verified for every retained trial. Two short warmups per server are excluded.
- **Policies:** default watermark 0; fixed watermark 30%; adaptive starts at 5%, adds 8 percentage points per eviction, caps at 45%, and removes 2 points after 128 nonempty eviction-free scheduling steps. State resets per trial. All use the same tracing wrapper and native FCFS scheduler; no requests are dropped, cancelled, or admitted using future output lengths.
- **Streaming:** silence is the elapsed time between consecutive client token-ID delivery events after the first token. Empty decoded text does not hide delivered tokens. Output length is enforced with `ignore_eos`; emitted IDs are counted and matched to API usage. Network batching and delivery jitter remain part of the observed stream behavior.
- **Replay:** scheduler traces count previously computed token positions scheduled again after eviction. These totals exactly match lost computed positions and native eviction counters. They measure scheduled replay work, not GPU FLOPs or retired instructions. Logical prompt and generation counters do not include this extra work.

The reservation mechanism follows vLLM's native [KV admission watermark](https://github.com/vllm-project/vllm/pull/44594). Native [full-input admission](https://github.com/vllm-project/vllm/pull/37307) reserves prompt capacity, without an oracle for future output length. Congestion-based adaptation has precedent, including [CONCUR](https://arxiv.org/abs/2601.22705); this local controller is not a novelty claim.

## Limits and reproducibility

This strengthens the local phenomenon across heterogeneous traces; it does not establish universal robustness. There are only two seeds, one laptop GPU, synthetic text, forced output lengths, eager execution, six primary configuration points, and two secondary architecture points. Staggered throughput includes the arrival span and can be arrival-limited. GPU clocks were observed rather than locked, so small speed differences should be treated cautiously. The largest recorded client arrival lateness was {x['max_arrival_lateness_s']:.4f} seconds. Policy-induced first-token queueing and completion latency are reported so that moving waiting time before the first token is visible.

Reproduce with the lab environment activated. `run.py --prefix rerun` writes a new set of configuration folders; it refuses to overwrite existing ones. Then run `analyze_study.py --prefix rerun` and `extra_analysis.py` with the lab Python. Generate figures with the existing plotting environment, `/home/varish/miniconda3/envs/kvzip/bin/python plot_study.py` (Matplotlib 3.10.9), and generate the report with the lab Python and `make_report.py`. This report generator also records the retained study's order-replacement provenance. Copy the original analysis/report outputs first because these derived filenames are overwritten. `run.py` and `control_scheduler.py` import the preserved helpers in the sibling `capacity_cliff_20261002` experiment; their recorded source hashes are unchanged throughout evaluation. No external model download is needed for the saved local snapshots.

Raw configuration folders contain commands, model/cache settings, randomized case order, exact workloads, scheduler JSONL traces, server logs, per-request streaming events, GPU snapshots, and before/after native metrics. `trials.csv`, `requests.csv`, `paired.csv`, and `batching.csv` provide tidy exports; `checks.json` records verification; `design.json` and `DESIGN.md` describe the prospective scope. Diagnostic, early `eval*`, and `excluded_order_*` folders are retained for provenance and excluded from all reported results. `order_replacement.json` records the order failure and replacement; `order_audit.json` independently checks all retained arrival sequences. The results bundle includes the excluded order-failure configuration so the decision is auditable.
'''
(ROOT/'REPORT.md').write_text(content)
print('REPORT.md and provenance.json saved')








