# Prospective evaluation design

The evaluation uses the controller constants selected before the diagnostic performance check: start reservation 0.05, add 0.08 for each eviction, cap 0.45, reduce by 0.02 after 128 nonempty eviction-free scheduling steps. Controller state resets at each trial. No output-length oracle or evaluation-specific tuning is used. Existing vLLM FCFS order, full-input admission, asynchronous execution, and prefix caching disabled are shared by all policies.

Policies: baseline watermark 0; fixed watermark 0.30; adaptive feedback watermark. All three are applied through the same wrapper and tracing code. Controls change only between completed trials. Requests are neither dropped nor cancelled by the controller.

Primary model: local Llama 3.2 1B. Six `(nominal KV token slots, max running requests)` points: `(3072,3)`, `(3072,12)`, `(5120,3)`, `(5120,6)`, `(5120,12)`, `(12288,12)`. This deliberately samples low, intermediate, and relieved pressure. It is not an exhaustive capacity/cap factorial.

Secondary model: local Qwen 2.5 1.5B at `(3072,6)` and `(12288,12)`. Token capacity is matched between models, not cache bytes: Llama uses 32 KiB/token and Qwen uses 28 KiB/token. Both weights and KV are BF16. A single null block reduces usable capacity by 16 tokens.

Two paired workload/configuration seeds: 23 and 77. Each configuration and case order is shuffled separately for each seed. Policies within a configuration share exactly the same request specs, token prompts, and arrival trace.

Each primary case contains 12 requests. Balanced mix: three each of inputs 128/256/512/1024 and outputs 64/128/256/512, independently shuffled. Generation mix: four each of inputs 128/256/512 and outputs 64/256/768, independently shuffled. Each mix is run as a burst spaced 50 ms apart (all twelve arrive within 0.55 seconds), and with staggered inter-arrivals of 50 ms plus an exponential draw with mean 0.95 seconds. The latter has mean one second and a minimum spacing that preserves request order. Engine-side arrival order is verified for every retained trial. The secondary check uses the generation mix under both arrival modes. Actual token-ID prompts and offsets are saved.

168 evaluation trials, 2,016 requests: 144 primary plus 24 secondary. Two excluded short warmups per server. Setup diagnostics, failed wrapper checks, and the early simultaneous-thread batches are excluded. The burst ordering change controls a confound and does not change controller parameters. Generation ignores EOS to guarantee the specified output length; no output-quality claims follow from this benchmark.

Primary outcomes: output tokens / trace makespan; first streamed-token delay; request completion latency; longest gap between token-delivery events after the first token; requests with a gap over one second; cumulative silence exceeding that threshold; eviction count; previously computed token positions scheduled again after eviction. Token IDs and cumulative counts are used rather than requiring nonempty text. Timing remains client token-delivery timing, not GPU kernel latency.

Analysis uses paired within-trace policy ratios. Report all cases and the transparently defined subset whose baseline evicted requests. Retaining throughput means policy throughput / paired baseline throughput. Also compare larger caps against cap 3 where the design contains both. A descriptive throughput/smoothness tradeoff is counted when a larger cap has >3% higher throughput and more requests stalled over one second; the 3% filter is a reporting convention, not a statistical significance test. For batching-benefit retention, require baseline batching gain >5% before dividing by that gain. No point is dropped because its policy outcome is unfavorable.

Single-GPU eager execution, synthetic requests, two seeds, a fractional design, and a small secondary model check limit generalization. Hardware clocks are observed rather than locked. Staggered-trace throughput includes the specified arrival span and can be arrival-limited. Policy cold starts prevent benefit from leaking across randomized workload orders, but may disadvantage feedback against policies with reserved headroom from the first request.

