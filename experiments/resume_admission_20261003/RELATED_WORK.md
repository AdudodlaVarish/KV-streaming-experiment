# Positioning the experiment

The defensible focus is **eviction/replay counts versus client interruption, and the measured delay before replay can restart**. Adaptive KV admission and smooth streaming are established objectives. These experiments do not establish publication novelty.

| Primary source | Overlap and implication |
|---|---|
| [CacheOPT (2025)](https://arxiv.org/html/2503.13773v1) | Predictive KV reservation, TTFT/TBT tradeoffs, and SLO-aware treatment of waiting and returned requests. Section 3.3.2 already distinguishes their remaining deadlines. Merely introducing resume-aware admission is also insufficient as a novelty claim. Our experiment isolates a watermark waiver without a length predictor or changes to native queue/victim selection. |
| [Chronos (2026)](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1873627/full) | Formal admission and schedulability analysis for TTFT/TBT guarantees. Our measurements trace concrete paged-cache eviction and asynchronous client delivery; they provide no hard deadline guarantee. |
| [SLAI](https://arxiv.org/abs/2508.01002) | Deadline-aware decode scheduling and prefill ordering. Accumulated streaming delay as a scheduling objective predates this project. |
| [CONCUR](https://arxiv.org/abs/2601.22705) | Feedback concurrency control using KV pressure in agentic batch inference. Feedback on congestion is established; our narrow objective is individual post-start interruption. |
| [StaticCore / Where Should Requests Wait?](https://github.com/manishraj1/StaticCore) | Author repository compares native admission, watermarking, and output-length reservation and describes moving waiting before first token. At inspection, the repository called the manuscript a draft and deferred citation until preprint submission; the pasted July 2026 publication claim was not verified. Treat it as a relevant empirical draft, not a confirmed peer-reviewed paper. |
| [Native vLLM watermark](https://github.com/vllm-project/vllm/pull/44594) | Fixed cache headroom to reduce preemption already exists upstream. Our fixed baseline uses that native allocator mechanism. |

## Observability is a narrower claim, too

[vLLM PR #13169](https://github.com/vllm-project/vllm/pull/13169), merged February 2025, explicitly discusses eviction, waiting, and recomputation raising the next inter-token interval. The causal sequence was already understood upstream. [The metrics design](https://docs.vllm.ai/en/latest/design/metrics/) says decode preemption contributes to inter-token/decode/inference intervals and describes frontend accounting of original prompt tokens and new output tokens. Consequently, the finding here is **that those token counters are unsuitable as replay-work counters**, not that vLLM has no visibility into preemption or streaming latency.

The pinned implementation additionally computes `request_prefill_kv_computed_tokens` from original prompt length minus cached tokens, rather than summing replay scheduling. Our saved native-source hashes and traces verify that distinction for vLLM 0.30.0. We have not established that it is an undocumented bug or a new discovery. Scheduled replay positions are not FLOPs or measurements of retired GPU instructions.

## What remains useful

A systems characterization can quantify where interruption time accumulates, show paired counterexamples to eviction-count optimization, identify which native allocation check blocks resumption, and test a small intervention with unchanged workloads. Strong claims should be restricted to the measured engine/version/configurations. A publication argument would still need broader platforms and workloads, tracing-overhead validation, and direct comparisons with established TBT-aware systems.

Primary sources checked October 3, 2026. The pasted MC²/CacheOPT publication equivalence and all author details were not independently established; this report cites the verified CacheOPT manuscript instead.
