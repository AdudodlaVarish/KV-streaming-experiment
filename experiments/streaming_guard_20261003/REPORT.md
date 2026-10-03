# Streaming-aware admission and service under KV pressure

**96 retained GPU trials / 1,152 completed requests.** The proactive policy reduced main-workload streams with client gaps over one second from **27/144 to 0/144**, at **90.0%** of matched default throughput. The larger growth allowance reached **0/144** at **91.0%**. This is a measured tradeoff, not a service guarantee: deadline misses and the held-out long tail matter as much as the ordinary-workload improvement.

In the qualifying low-concurrency comparisons, service128 retains **33.6%–60.4%** of the incremental batching gain, and service256 **41.7%–53.0%**. Retaining about 90% of total throughput therefore does not mean retaining most of the extra batching benefit in every workload. The combined objective is not established. These descriptive ranges apply to the frozen controls, not to an optimized policy.

The useful distinction is **protecting current streams before new admission versus repeatedly repairing an interruption after capacity is exhausted**. A growth-only ablation reached 11/144 stalled streams at 98.8% throughput. Adding deadline ordering, early cache handoff and temporary repeat-victim protection changed throughput to 91.1% of that ablation, evictions 18 → 183, replay 12,191 → 148,942, and worst gap 7.909 → 0.566 s. The extra machinery should be judged by that ablation rather than assumed to help.

## What was tested

The main service policy gives each resident stream a rolling allowance of 128 additional positions (256 in the second version). Fresh requests enter only if all current growth allowances plus the candidate fit free blocks, and no already-started stream is paused. The allowance includes unfinished prefills; resumed requests still need physically sufficient cache for their known context. An idle escape allows a fitting prompt when there are no resident or paused streams. The policy does not use output targets or remaining lengths.

Already-started requests precede fresh ones, ordered by observed engine-output deadline. A request evicted within the last second receives one target interval of priority credit. After half the target interval of silence, a paused stream may take cache from one less-urgent, safely evictable resident per scheduler pass. Native async completion, stale-output handling and physical allocator checks remain active. The default target is 0.5 s; the held-out study also tests 0.25 and 1 s. This is a scheduling preference and early corrective handoff, not a hard client deadline.

Controls: native FCFS; the prior adaptive reserve with resume waiver; growth admission alone; a **same-engine SLAI core-policy port**; larger-cache relief; and contemporaneous low-concurrency FCFS. The SLAI port implements critical decode, active/fresh prefill, then noncritical decode service, with published urgency and victim rules. It uses the existing vLLM stack rather than the original Sarathi/Torch2.3/CUDA12.1 engine. Native resume feasibility, output history, async completion and scheduling rollback constrain the adaptation; active-prefill ties use prompt length. Its batch-time estimate is dispatch-to-output delay, not kernel profiling. This is not a reproduction of SLAI's published performance. Exact adaptations are in [DESIGN.md](DESIGN.md) and [RELATED_WORK.md](RELATED_WORK.md); reference commit `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380`.

Hardware/runtime: RTX 4060 Laptop 8GB, Python 3.12.14, vLLM 0.30.0, Torch 2.13.0+cu132, FlashInfer 0.6.18.post1; BF16, eager async execution, block16, context/batch limit2048, prefix caching off. Two model sizes: Llama 3.2 1B and Qwen 2.5 1.5B. Main seeds 23/77 shuffle both configurations and policy/workload order. Twelve requests per trial use mixed prompt/output lengths and clustered or staggered arrivals. The pressure configurations have 3,072 nominal KV positions at cap 12 / cap 6; relief has 12,288 positions. Llama cap 3 controls measure how much batching gain remains. Held-out seed 381 uses independent prompt/output mixes with outputs up to 1,536 tokens, exceeding both allowances. All requests finish without rejection, timeout cancellation or truncation.

## Main pressure comparison

The prespecified main group has **12 workload cells / 144 requests per policy**: both seeds, models and arrival modes, plus balanced and generation-heavy Llama mixes. It is selected by configuration, not by whether default eviction occurred. Larger-cache, cap 3 and long-tail trials are reported separately.

| Policy | Throughput / default | Streams >0.5 / >1 s | Worst gap (s) | Evictions | Replay positions | Median TTFT / default |
|---|---:|---:|---:|---:|---:|---:|
| Default FCFS | 100.0% | 36 / 27 | 10.416 | 58 | 29,914 | 1.00x |
| Adaptive + resume waiver | 98.2% | 23 / 17 | 9.597 | 35 | 20,849 | 1.07x |
| SLAI core port | 102.7% | 39 / 22 | 4.752 | 58 | 35,360 | 0.68x |
| Growth only (128) | 98.8% | 13 / 11 | 7.909 | 18 | 12,191 | 1.55x |
| Service + growth128 | 90.0% | 4 / 0 | 0.566 | 183 | 148,942 | 1.50x |
| Service + growth256 | 91.0% | 4 / 0 | 0.563 | 113 | 98,839 | 3.76x |

Throughput and latency ratios are geometric means of matched workload ratios. TTFT ratios compare workload median time to first token; completion ratios below compare workload P95 completion latency. A stalled stream has at least one client token-ID delivery gap over 1 s **after its first delivery**. Each gap is a delivery event interval, not necessarily a separate token when events carry multiple IDs.

The service128 policy versus the SLAI core port: **87.7%** throughput; streams over 1 s 22 → 0; streams over0.5 s 39 → 4; evictions 58 → 183. The port favors some batching opportunities differently; retain the engine-specific scope when interpreting this comparison.

Target attainment must remain visible. Service128 misses its 0.5 s client target for **4/144** streams; service256 for **4/144**. Engine callbacks themselves exceed the target for 4 and 4 streams. Service128's worst gap is 0.566 s; reducing counts does not establish a bound. Relative P95 completion ratios are 1.125 and 1.130. Aggregate excess pause above 1 s is 99.264 s default, 53.254 resume waiver, 36.362 SLAI port, 37.120 growth-only, 0.000 service128, 0.000 service256.

### Does the effect persist across workloads?

| Service128 vs default stratum | Cells | Throughput retained | >1 s streams, default → service | Worst gap, default → service (s) |
|---|---:|---:|---:|---:|
| model=llama1b | 8 | 93.4% | 19 → 0 | 10.416 → 0.473 |
| model=qwen1.5b | 4 | 83.5% | 8 → 0 | 7.977 → 0.566 |
| mix=balanced | 4 | 95.9% | 7 → 0 | 4.343 → 0.293 |
| mix=generation | 8 | 87.2% | 20 → 0 | 10.416 → 0.566 |
| arrival=burst | 6 | 85.2% | 15 → 0 | 10.416 → 0.566 |
| arrival=stagger | 6 | 95.1% | 12 → 0 | 9.312 → 0.563 |
| seed=23 | 6 | 92.4% | 8 → 0 | 4.440 → 0.479 |
| seed=77 | 6 | 87.6% | 19 → 0 | 10.416 → 0.566 |

These overlapping strata describe the observed cells; two main seeds do not support broad statistical generalization. Per-cell ratios, regressions and every request are in the CSV/JSON exports.

### Retaining the batching benefit

The four paired Llama generation controls compare cap 12 with cap 3 at identical cache/workload settings. Incremental gain retention is `(policy throughput − cap 3 throughput) / (default cap 12 throughput − cap 3 throughput)`, reported only if default cap 12 is over 5% faster. It is stricter than total throughput retention; a value below 0 loses the observed cap 3 benefit.

| Policy | Qualifying controls | Incremental gain retained, range |
|---|---:|---:|
| Adaptive + resume waiver | 3 | 34.9%–94.1% |
| SLAI core port | 3 | 67.9%–190.5% |
| Growth only (128) | 3 | 58.9%–103.4% |
| Service + growth128 | 3 | 33.6%–60.4% |
| Service + growth256 | 3 | 41.7%–53.0% |

Cap3 controls: 0 stalled streams, 0 evictions. Relief controls: default 0 and service128 0 stalled streams, worst gaps 0.063 / 0.067 s; service128 throughput 100.1% of default.

## Held-out long tail: the failure/cost test

**Two cells / 24 requests per policy**, seed 381; long outputs exceed the fixed growth allowance. No parameters were changed after diagnostic observations or after evaluation began.

| Policy | Throughput / default | Streams >0.5 / >1 s | Worst gap (s) | Evictions | Replay positions | Median TTFT / default |
|---|---:|---:|---:|---:|---:|---:|
| Default FCFS | 100.0% | 8 / 8 | 13.661 | 15 | 9,745 | 1.00x |
| SLAI core port | 104.0% | 9 / 8 | 11.398 | 23 | 19,024 | 0.93x |
| Service + growth128 | 35.5% | 9 / 0 | 0.708 | 1,336 | 1,778,160 | 2.31x |
| Service + growth256 | 51.6% | 7 / 0 | 0.618 | 656 | 790,272 | 11.08x |
| Service target 0.25 s | 22.3% | 9 / 0 | 0.714 | 2,501 | 3,293,548 | 2.33x |
| Service target 1 s | 68.5% | 9 / 0 | 0.846 | 351 | 467,020 | 2.23x |

| Service policy | Its target (s) | Streams missing that target /24 | Early handoffs | Repeat victims | P95 completion / default |
|---|---:|---:|---:|---:|---:|
| Service + growth128 | 0.5 | 9 | 1,326 | 10 | 3.43x |
| Service + growth256 | 0.5 | 7 | 649 | 12 | 2.18x |
| Service target 0.25 s | 0.25 | 10 | 2,496 | 11 | 5.65x |
| Service target 1 s | 1 | 0 | 338 | 9 | 1.66x |

The strict 0.25 s version reaches a worst gap of **0.714 s**, with **2,501 evictions / 3,293,548 replayed positions**, and **22.3%** of default throughput. A one-second target reaches 0.846 s at 68.5%. The tradeoff cannot be summarized by the number of gaps over 1 s alone: repeated handoffs can keep streams visibly moving while spending substantial work rebuilding their context and delaying completion. Temporary victim credit does not prevent repeated eviction.

For the default 0.5 s variants, client target misses are 9/24 (128) and 7/24 (256). The selected allowance and target are fixed empirical choices; these results do not demonstrate a robust controller for unbounded generation. A next implementation would need to handle long-context replay cost explicitly before tightening service targets, and must retain this stress test and the growth-only ablation.

A **post hoc handoff audit** joins each corrective eviction to the next native scheduling pass. Service128 schedules its intended target in 1,043/1,326 such passes; the strict version in 2,012/2,496. One victim does not always free enough cache for the target's full known context. Absence of immediate dispatch does not prove a useless eviction: several victims or a later pass may be needed. These exploratory counts identify a further audit dimension; they were not used to retune or select any trial.

## Measurement and audit

The [prospective design](DESIGN.md), policy/driver hashes and workload/order rules were frozen before eight excluded diagnostic trials. Diagnostics exposed severe strict-target replay cost; settings stayed fixed. One accounting edge case required an explicit measurement correction **before the main study**: an evicted request may finish from a final in-flight output without replaying the lost prefix. [MEASUREMENT_NOTES.md](MEASUREMENT_NOTES.md) documents this. Final accounting conserves **lost positions = scheduled replay + terminal discarded context**; main evaluation has 8 such terminal episodes and 12,088 discarded positions. Scheduled replay is not executed FLOPs or billing.

All 1,152 requested outputs, API usage and native output-service counts agree; native success counts are 12 per trial. Native arrival order 0..11 is verified, preemptions match metrics, replay/terminal context conserves, every admitted growth inequality and native physical check holds, urgency thresholds/victim ordering are audited, and frozen source/runtime helper hashes are retained. Diagnostics, warmups and any order-invalid attempts are excluded and preserved, never selected by performance. Cross-process monotonic clock reads are bounded in provenance. Logical prompt/generation/iteration and extended prefill-computed counters continue to omit replay; replay is audited separately from scheduler traces.

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

Run folders are never overwritten. The analyzer/plotter update derived exports; preserve them before a rerun. `write_report.py` and `finalize.py` apply to a complete 96-trial matrix. Model snapshots are local and not bundled. Existing prior-study raw bundles remain separate. `dash.py` remains unchanged.
