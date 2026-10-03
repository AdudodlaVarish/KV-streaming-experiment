# KV-cache pressure and streaming continuity in LLM serving

*Technical project writeup · October 3, 2026*

## Context

This project explored whether an LLM server can keep an already-started response receiving service when its key–value (KV) cache is under pressure. KV state grows during generation. Requests that fit at admission can later compete for capacity; evicting a request frees memory but requires its context to be reconstructed before decoding continues.

Throughput, time to first token (TTFT), and completion latency can conceal this interruption. We therefore measured post-first-token silence directly: intervals between client delivery events after the first delivery. Later studies used token-ID streams; an event can contain several tokens, so these intervals are delivery gaps rather than necessarily individual token intervals.

The question evolved from “Can admission control reduce evictions?” to “Can it preserve streaming continuity without losing most of the benefit from batching?”

## Relevant work

[PagedAttention](https://arxiv.org/abs/2309.06180) provides the paged KV-memory management underlying vLLM. Efficient allocation does not by itself establish a streaming-service guarantee.

[SLAI](https://arxiv.org/abs/2508.01002) prioritizes decoding requests near their time-between-token (TBT) deadlines and orders prefills using prompt lengths. [CacheOPT](https://arxiv.org/abs/2503.13773) already studies output-length estimation, proactive KV allocation, victim selection, and swapping versus recomputation. [Chronos](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1873627/full) connects TTFT/TBT constraints to schedulability and admission analysis, evaluated through a discrete-event simulator.

These works place the project within an established research area. We do not claim novelty for headroom reservation or deadline priority. Our contribution is a local empirical characterization of interruption, resumption waiting, and replay cost, supported by controlled interventions and explicit failure cases.

## What we did

Experiments ran in native vLLM 0.30.0 on an 8 GB RTX 4060 Laptop GPU, using Llama 3.2 1B and Qwen 2.5 1.5B. Follow-up studies used BF16, eager asynchronous execution, disabled prefix caching, and a 2,048-token context limit. Prompt/output lengths, clustered versus staggered arrivals, cache capacities, concurrency caps, and workload/configuration orders varied across studies. The second model changes architecture as well as size.

Four stages progressively narrowed the question:

1. **Reproduce the tradeoff.** Under a tight cache, increasing concurrency raised throughput modestly while introducing roughly five-second pauses. Lower concurrency, fixed headroom, and a larger cache removed these pauses in the initial homogeneous workload.
2. **Test workload robustness.** A 168-trial study varied admission pressure across mixed workloads and two models. The throughput-versus-smoothness tradeoff appeared in seven of eight tight-cache high-versus-low-concurrency comparisons, but disappeared in capacity-relief controls. Adaptive headroom reduced eviction burden without consistently eliminating long pauses.
3. **Identify the resumption bottleneck.** A 76-trial intervention waived the admission reserve for preempted requests. It helped partly: about 88% of measured adaptive resumption waiting followed physical-capacity refusals. Reanalysis found roughly 96% of the full duration of gaps over one second overlapped the affected request's eviction-to-first-replay wait. These are timing and sampled-decision attributions, not isolated GPU execution measurements.
4. **Control service directly.** A final 96-trial study compared native FCFS, the resume waiver, growth admission alone, two streaming-service variants, and a SLAI core-policy port. The variants reserved rolling room for 128 or 256 additional positions, delayed fresh admission while started streams were paused, prioritized observed service deadlines, and handed cache to overdue streams with temporary protection for recent eviction victims. They did not use output targets or remaining output lengths.

The SLAI comparator adapted core scheduling rules to the same pinned vLLM engine. It retained native asynchronous bookkeeping and capacity checks; it was **not a reproduction of the complete original SLAI system**. CacheOPT and Chronos were contextual references, not benchmarked implementations.

## Findings

The final study completed 1,152 requests. Its prespecified main pressure group contained 12 matched workloads and **144 requests per policy**:

| Policy | Throughput / default | Streams with >1 s gap | Worst gap (s) | Evictions |
|---|---:|---:|---:|---:|
| Native FCFS | 100.0% | 27 | 10.416 | 58 |
| Adaptive + resume waiver | 98.2% | 17 | 9.597 | 35 |
| SLAI core-policy port | 102.7% | 22 | 4.752 | 58 |
| Growth admission alone | 98.8% | 11 | 7.909 | 18 |
| Service + growth allowance 128 | 90.0% | 0 | 0.566 | 183 |
| Service + growth allowance 256 | 91.0% | 0 | 0.563 | 113 |

Throughput retention is the geometric mean of matched workload ratios; counts sum across the same workloads. Cross-study baseline differences are not policy effects.

**Eviction count is an inadequate proxy for streaming continuity.** The service policies eliminated gaps over one second in the main group while increasing eviction and replay. Both still missed their half-second target for four of 144 streams. Median TTFT ratios increased to 1.50× and 3.76×. Against low-concurrency controls, only 34–60% and 42–53% of the incremental batching gain remained in the three qualifying comparisons; retaining around 90% of total throughput did not preserve most extra batching benefit consistently.

A held-out seed with outputs up to 1,536 tokens exposed a stronger limit. Across two arrival modes and 24 requests per policy, the quarter-second service variant caused **2,501 evictions and 3.29 million scheduled replay positions**, retained **22.3% throughput**, and still reached a 0.714-second gap. Frequent service repair can keep streams visibly progressing while making completion much slower.

## Future goals

A continuation should target **resume feasibility and replay cost** together:

- Before a handoff, determine whether the proposed evictions free enough capacity for the interrupted request's full known context, and estimate whether rebuilding it can meet the service deadline.
- Make repeated eviction costly in admission and victim selection. Test whether observed growth and replay burden can adjust the active population while preserving both post-start service and first-token waiting.
- Validate against original systems or independently reviewed ports, measure tracing overhead, and extend evaluation to natural stopping, longer contexts, production arrival traces, more seeds, and additional hardware. Report total throughput and incremental batching gain separately.

These are proposed experiments, not implemented features or established guarantees.

## Evidence and scope

Policy parameters were frozen before retained evaluation. Later-study audits verified delivery counts, native arrival order, eviction counters, physical-capacity checks, and replay accounting. Warmups and diagnostics were excluded; order-invalid attempts were preserved and repeated with unchanged settings. Lost context is accounted for as scheduled replay plus terminal context discarded when an in-flight final output completes an evicted request. Replay positions are not measured FLOPs. One GPU, small models, forced output lengths, few seeds, and unmeasured tracing overhead limit generalization.

Detailed designs, raw traces, figures, source hashes, and verified bundles remain in the experiment directories:

- [Initial reproduction](experiments/capacity_cliff_20261002/REPORT.md)
- [Workload robustness](experiments/admission_robustness_20261002/REPORT.md)
- [Resumption mechanism](experiments/resume_admission_20261003/REPORT.md)
- [Streaming-service comparison and data](experiments/streaming_guard_20261003/REPORT.md)


