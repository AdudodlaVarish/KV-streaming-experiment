# KV-cache admission and streaming pauses

Investigating the throughput-versus-streaming-smoothness tradeoff under KV-cache pressure, and whether adaptive admission can reduce pauses while keeping batching gains.

## Findings so far

**168 retained trials / 2,016 requests** across Llama 3.2 1B and Qwen 2.5 1.5B, mixed prompt/output lengths, bursty/staggered arrivals, several cache sizes and concurrency caps, and two randomized seeds/orders. This is a sampled configuration matrix, not a full factorial.

- **The tradeoff recurs near the cache limit.** With 3,072 KV token slots, raising Llama's cap from 3 to 12 increased throughput >3% and produced more stalled streams in **7 of 8** comparisons. Neither higher-cap comparison at 5,120 slots showed that tradeoff; the largest-cache baseline controls stayed smooth.
- **Adaptive admission helps, but does not bound pauses.** Burst stalls fell from 17 to 9; staggered stalls remained at 12. The worst pause increased from 9.54 to 10.17 seconds despite less replay.

For the **16 cases where baseline admission caused evictions** (192 requests per policy):

| Policy | Throughput retained | Streams with >1 s pauses | Evictions | Replayed token positions |
|---|---:|---:|---:|---:|
| Baseline | 100% | 29 | 62 | 31,417 |
| Adaptive headroom | 99.4% | 21 | 36 | 19,834 |
| Fixed 30% headroom | 90.9% | 9 | 12 | 8,399 |

Throughput retention is the geometric mean of paired ratios. Pauses are gaps between token deliveries after the first token. Fixed headroom reduced stalls further but also roughly doubled first-token delay under pressure.

The initial homogeneous experiment found the same local behavior: raising concurrency increased throughput 3.9% while the longest streaming pause grew from 0.166 to 5.195 seconds. Headroom or a larger cache removed those pauses in that workload. That study measured text chunks; the broader study uses token IDs.

Standard token counters hide replay work in this vLLM version; scheduler traces expose it. All retained counts and arrival orders passed verification. One configuration was repeated after an arrival-order violation, with the excluded run preserved.

**Limits:** one 8 GB laptop GPU, synthetic fixed output lengths, eager execution, and two seeds. These findings establish a conditional phenomenon, not a production pause guarantee. The next policy question is how to protect resuming streams while retaining batching gains.

## Reports and layout

- [Broader study](experiments/admission_robustness_20261002/REPORT.md) · [Figures and raw data](experiments/admission_robustness_20261002/results.zip) · [Verification](experiments/admission_robustness_20261002/checks.json)
- [Initial capacity-cliff study](experiments/capacity_cliff_20261002/REPORT.md)

```text
README.md                 Project findings and usage
dash.py                  Live aggregate metrics
activate.sh              GPU environment activation
requirements.lock.txt    Pinned dependencies
experiments/             Benchmarks, traces, results, figures, provenance
archive/setup/           Original setup README, smoke checks, help, logs
.venv/ and cuda-libs/     Existing working runtime
```

Keep the two experiment directories together: the broader study imports helpers from the initial study.

## Run

In WSL Ubuntu:

```bash
cd ~/kv-cache-lab
source activate.sh
python dash.py
```

The dashboard reads `http://127.0.0.1:8017/metrics` while an experiment server is running. Benchmarks stop their servers on completion. For another server, set `VLLM_METRICS_URL`; for example:

```bash
VLLM_METRICS_URL=http://127.0.0.1:8000/metrics python dash.py
```

Streaming pauses and replay require the saved client records and scheduler traces, beyond the dashboard's aggregate metrics.

Repeat the broader study with a fresh prefix:

```bash
python experiments/admission_robustness_20261002/run.py --prefix rerun
python experiments/admission_robustness_20261002/analyze_study.py --prefix rerun
```

Run folders are never overwritten; analysis updates derived JSON/CSV files. Copy the study directory first to preserve current exports. Full reports document figure generation and measurement details.

Runtime: RTX 4060 Laptop, Python 3.12, vLLM 0.30.0, Torch 2.13.0+cu132, FlashInfer 0.6.18.post1. Use `activate.sh` to select the required CUDA libraries/compiler and preserve the pinned environment.
