# KV-cache pressure and streaming service

A focused empirical project asking whether already-started responses can keep receiving service when physical KV capacity is tight. Native vLLM experiments on an 8GB RTX 4060 Laptop GPU; `dash.py` remains the live aggregate dashboard.

The exploratory project is complete. The [technical writeup](TECHNICAL_WRITEUP.md) brings together the context, relevant work, experiments, findings, and future research goals.

## Latest findings

**96 new trials / 1,152 completed requests** compare proactive growth admission, deadline service, the prior resume waiver, and a same-engine SLAI core-policy port. Two models, two main workload/order seeds, mixed lengths, clustered/staggered arrivals, larger-cache and lower-concurrency controls, plus a held-out long-output seed.

The **12 prespecified pressure workloads /144 requests per policy** show:

| Policy | Throughput / default | Streams with >1 s gap | Worst gap (s) | Evictions |
|---|---:|---:|---:|---:|
| Default | 100.0% | 27 | 10.416 | 58 |
| Adaptive + resume waiver | 98.2% | 17 | 9.597 | 35 |
| SLAI core port | 102.7% | 22 | 4.752 | 58 |
| Growth only (128) | 98.8% | 11 | 7.909 | 18 |
| Service + growth128 | 90.0% | 0 | 0.566 | 183 |
| Service + growth256 | 91.0% | 0 | 0.563 | 113 |

Throughput retention is the geometric mean of matched ratios. Gaps measure actual client token-ID delivery **after the first delivery**. Main group selection uses configuration, not default eviction outcomes.

- **Proactive protection helps on the tested main workloads.** Reserve rolling room for active streams to grow, and pause fresh admission while a started stream is waiting. Adding early deadline service changes stalled streams 11 → 0 but also changes replay 12,191 → 148,942 positions.
- **Smoothness has admission and replay costs.** Median first-token latency for service128/256 is 1.50x / 3.76x default. Their 0.5 s client target misses are 4/144 and 4/144. Only 33.6%–60.4% /41.7%–53.0% of incremental batching gain over cap 3 remains in the three qualifying controls; retaining most total throughput does not meet the stronger batching-benefit objective.
- **The long tail exposes the limit.** On held-out outputs up to 1,536 tokens, the strict 0.25 s version retains 22.3% throughput, causes 2,501 evictions and 3,293,548 replayed positions, and still reaches a 0.714 s gap. No tested policy establishes a hard service bound.

The earlier resumption study found about 88% of adaptive resumption waiting followed physical-capacity refusals. A reserve waiver helped partly but could not create cache. The broader workload study found the batching/smoothness tradeoff in 7/8 tight-cache comparisons. This study tests proactive admission and service directly; the growth-only ablation and long-tail failure keep the attribution honest.

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

Run folders are never overwritten; analysis updates derived exports. Preserve exports before rerunning. See the report for plotting, audit and bundle scope. Runtime: Python 3.12, vLLM 0.30.0, Torch 2.13.0+cu132, FlashInfer 0.6.18.post1.

