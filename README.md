# KV-cache pressure and streaming service

This project asks a fairly simple question: **when physical KV-cache capacity gets tight, can responses that have already started keep receiving service?**

The experiments run directly on vLLM using an 8GB RTX 4060 Laptop GPU. `dash.py` remains the live aggregate dashboard.

The exploratory phase of the project is now complete. The [technical writeup](TECHNICAL_WRITEUP.md) brings together the motivation, related work, experiments, findings, limitations, and possible directions for future research.

## Latest findings

The latest study adds **96 trials and 1,152 completed requests**. It compares proactive growth admission, deadline-based service, the earlier resume waiver, and a same-engine port of SLAI's core scheduling policy.

The test matrix covers two models, two main workload/order seeds, mixed output lengths, clustered and staggered arrivals, larger-cache and lower-concurrency controls, and a held-out long-output seed.

Across the **12 prespecified pressure workloads, with 144 requests per policy**, the results are:

| Policy | Throughput / default | Streams with >1 s gap | Worst gap (s) | Evictions |
|---|---:|---:|---:|---:|
| Default | 100.0% | 27 | 10.416 | 58 |
| Adaptive + resume waiver | 98.2% | 17 | 9.597 | 35 |
| SLAI core port | 102.7% | 22 | 4.752 | 58 |
| Growth only (128) | 98.8% | 11 | 7.909 | 18 |
| Service + growth128 | 90.0% | 0 | 0.566 | 183 |
| Service + growth256 | 91.0% | 0 | 0.563 | 113 |

Throughput retention is reported as the geometric mean of matched throughput ratios. Streaming gaps measure actual client token-ID delivery **after the first token has already been delivered**. The main workload groups are selected from configuration rather than from the eviction behavior of the default policy.

- **Proactive protection helps on the main workloads tested here.** Reserving room for active streams to keep growing, while pausing new admission when a started stream is waiting, substantially reduces long stalls. Adding early deadline-based service takes the number of streams with gaps over one second from 11 to 0, but it also raises replay from 12,191 to 148,942 positions.

- **That smoother service comes with real admission and replay costs.** Median first-token latency for service128 and service256 rises to 1.50× and 3.76× the default, respectively. Both policies miss the 0.5 s client target on 4 of 144 requests. In the three qualifying controls, they preserve only 33.6%–60.4% and 41.7%–53.0% of the incremental batching gain over a concurrency cap of 3. So while most total throughput is retained, the stronger goal of preserving most of the batching benefit is not met.

- **The held-out long-output workload shows where the approach breaks down.** With outputs up to 1,536 tokens, the strict 0.25 s service version retains only 22.3% of default throughput, triggers 2,501 evictions, and replays 3,293,548 positions. Even then, the worst streaming gap reaches 0.714 s. None of the policies tested here establishes a hard service bound.

The earlier resumption study found that about 88% of adaptive resumption waits followed physical-capacity refusals. A reserve waiver helped, but only partially: it could change admission behavior, not create more cache.

The broader workload study also found the batching-versus-smoothness tradeoff in 7 of 8 tight-cache comparisons. The latest study tests proactive admission and service more directly. The growth-only ablation helps separate the effects of reservation from deadline service, while the long-tail failure makes clear that the approach has limits.

There is also important prior work here. TBT-oriented scheduling and proactive KV reservation are not new ideas. The SLAI result in this repository is a port of its core policy into the pinned engine used for these experiments; it is **not the original SLAI system and should not be interpreted as a reproduction of its published performance**.

These results are also deliberately narrow. They come from one GPU, synthetic forced outputs, short contexts, and a small number of seeds, so they should not be generalized beyond the tested setting without further validation.

## Reports and data

- [Streaming-service study](experiments/streaming_guard_20261003/REPORT.md) · [Related work](experiments/streaming_guard_20261003/RELATED_WORK.md) · [Figure](experiments/streaming_guard_20261003/streaming_results.png) · [Data/code bundle](experiments/streaming_guard_20261003/results.zip)
- [Resumption mechanism](experiments/resume_admission_20261003/REPORT.md)
- [Broader workloads](experiments/admission_robustness_20261002/REPORT.md)
- [Initial capacity cliff](experiments/capacity_cliff_20261002/REPORT.md)

Keep the four study directories together, since later studies reuse helpers from the earlier ones.

`experiments/` contains the frozen study designs, code, client records, native traces, audits, and figures. `archive/setup/` preserves the setup history. `activate.sh`, `requirements.lock.txt`, `.venv/`, and `cuda-libs/` preserve the runtime environment.

## Run

```bash
cd ~/kv-cache-lab
source activate.sh
python dash.py
```

The dashboard reads metrics from local port 8017 while an experiment server is running. Benchmark scripts shut their servers down when they finish.

Streaming-gap and replay analysis requires the client records and native traces; those details are not recoverable from dashboard aggregates alone.

To rerun the latest experiment matrix:

```bash
python experiments/streaming_guard_20261003/run.py --prefix rerun
python experiments/streaming_guard_20261003/analyze.py --prefix rerun
```

Run directories are never overwritten. Analysis may update derived exports, so preserve any exports you want to keep before rerunning. See the study report for details on plotting, audits, and bundle contents.

Runtime: Python 3.12, vLLM 0.30.0, Torch 2.13.0+cu132, FlashInfer 0.6.18.post1.
