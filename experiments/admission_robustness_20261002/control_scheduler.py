"""Feedback headroom layered on the installed vLLM admission watermark.
No use of actual/future output lengths. Parameters are locked before evaluation.
"""
import json, os, time
from trace_scheduler import TraceScheduler

START, UP, DOWN, MAXIMUM, CALM_STEPS = .05, .08, .02, .45, 128

def feedback(reserve, calm, preemptions, has_work=True):
    if preemptions:
        return min(MAXIMUM, reserve+UP*preemptions), 0
    if has_work:
        calm += 1
        if calm >= CALM_STEPS:
            return max(0., reserve-DOWN), 0
    return reserve, calm

class ControlScheduler(TraceScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.control_path=os.environ['LAB_CONTROL']
        self.control_stamp=None
        self.admission_policy='baseline'
        self.trial='startup'
        self.reserve=0.
        self.calm=0
        self.preemptions_this_step=0

    def _load_control(self):
        stamp=os.stat(self.control_path).st_mtime_ns
        if stamp!=self.control_stamp:
            state=json.load(open(self.control_path))
            self.admission_policy=state['policy']
            assert self.admission_policy in ('baseline','fixed','adaptive')
            self.trial=state['trial']
            self.reserve={'baseline':0.,'fixed':.30,'adaptive':START}[self.admission_policy]
            self.calm=0
            self.control_stamp=stamp
            self.trace.write(json.dumps(dict(kind='control',t=time.perf_counter(),
                trial=self.trial,policy=self.admission_policy,reserve=self.reserve))+'\n')

    def add_request(self,request):
        self._load_control()
        self.trace.write(json.dumps(dict(kind='arrival',t=time.perf_counter(),trial=self.trial,
            request=request.request_id,prompt=request.num_prompt_tokens,policy=self.admission_policy))+'\n')
        return super().add_request(request)

    def schedule(self,*args,**kwargs):
        self._load_control()
        self.preemptions_this_step=0
        self.kv_cache_manager.watermark_blocks=int(self.reserve*self.kv_cache_manager.block_pool.num_gpu_blocks)
        used_reserve=self.reserve
        output=super().schedule(*args,**kwargs)
        if self.admission_policy=='adaptive' and not self.preemptions_this_step:
            self.reserve,self.calm=feedback(self.reserve,self.calm,0,output.total_num_scheduled_tokens>0)
        self.trace.write(json.dumps(dict(kind='admission',t=time.perf_counter(),trial=self.trial,
            policy=self.admission_policy,reserve=used_reserve,next_reserve=self.reserve,
            preemptions=self.preemptions_this_step))+'\n')
        return output

    def _preempt_request(self,request,timestamp,drop_stale_output=False):
        self.preemptions_this_step+=1
        if self.admission_policy=='adaptive':
            self.reserve,self.calm=feedback(self.reserve,self.calm,1)
        return super()._preempt_request(request,timestamp,drop_stale_output)

if __name__=='__main__':
    import inspect, re
    from vllm.v1.core.sched.scheduler import Scheduler
    own=set(re.findall(r'self\.(\w+)\s*=',inspect.getsource(ControlScheduler)))
    native=set(re.findall(r'self\.(\w+)',inspect.getsource(Scheduler)))
    assert not own & native,own & native
    inspect.signature(ControlScheduler.schedule).bind(object(),False)
    assert feedback(.05,3,3)==(.29,0)
    assert feedback(.44,0,2)==(.45,0)
    r,c=feedback(.30,127,0)
    assert abs(r-.28)<1e-9 and c==0
    assert feedback(.01,127,0)==(0.,0)
    assert feedback(.1,5,0,False)==(.1,5)
    print('feedback checks passed')




