# Source notices

`native_slai_schedule.py` is a transformation of the pinned vLLM `Scheduler.schedule` method. vLLM is Apache-2.0 licensed; see https://github.com/vllm-project/vllm/blob/main/LICENSE. The installed source hash and transformation scope are recorded in provenance, DESIGN.md and the generated module. The prototype package does not modify installed vLLM files.

The reference checkout `references/SLAI` is from https://github.com/agrimUT/SLAI, commit 5098a7aba05e3edbcfa3a509d6cc9cd248fc4380, and preserves its Apache-2.0 LICENSE and source notices. It was inspected for policy rules; its original serving engine was not installed or benchmarked.
