import json, os, time
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

class TraceScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = open(os.environ['LAB_SCHEDULER_TRACE'], 'w', buffering=1)
        self.replay_until = {}
        self.trace.write(json.dumps(dict(kind='config',t=time.perf_counter(),
            async_scheduling=self.scheduler_config.async_scheduling,
            watermark=self.scheduler_config.watermark,
            reserve_full_isl=self.scheduler_reserve_full_isl))+'\n')

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        high = request.num_computed_tokens
        self.replay_until[request.request_id] = max(high, self.replay_until.get(request.request_id,0))
        self.trace.write(json.dumps(dict(kind='preempt',t=time.perf_counter(),
            request=request.request_id, computed=high, tokens=request.num_tokens,
            prompt=request.num_prompt_tokens, output=request.num_output_tokens,
            preemptions_before=request.num_preemptions))+'\n')
        return super()._preempt_request(request,timestamp,drop_stale_output)

    def _update_after_schedule(self, output):
        rows = []
        for rid, n in output.num_scheduled_tokens.items():
            request = self.requests[rid]
            before = request.num_computed_tokens
            high = self.replay_until.get(rid,0)
            replay = max(0,min(n,high-before))
            rows.append([rid,before,n,replay,request.num_output_tokens])
            if before+n>=high:
                self.replay_until.pop(rid,None)
        self.trace.write(json.dumps(dict(kind='step',t=time.perf_counter(),
            kv_usage=self.kv_cache_manager.usage, running=len(self.running),
            waiting=len(self.waiting), scheduled=rows))+'\n')
        return super()._update_after_schedule(output)
