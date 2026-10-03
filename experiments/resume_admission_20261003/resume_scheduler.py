import json, os, time
from vllm.v1.request import RequestStatus
from trace_scheduler import TraceScheduler
from control_scheduler import feedback, START

POLICIES=('baseline','fixed','adaptive','fixed_resume','adaptive_resume')

def effective_watermark(policy,status,watermark):
    return 0 if policy.endswith('_resume') and status==RequestStatus.PREEMPTED else watermark

class ResumeScheduler(TraceScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.control_path=os.environ['LAB_CONTROL']
        self.control_stamp=None
        self.admission_policy='baseline'
        self.trial='startup'
        self.reserve=0.
        self.calm=0
        self.preemptions_this_step=0
        self.native_allocator=self.kv_cache_manager.allocate_slots
        self.kv_cache_manager.allocate_slots=self._allocate_with_resume_gate

    def _load_control(self):
        stamp=os.stat(self.control_path).st_mtime_ns
        if stamp!=self.control_stamp:
            state=json.load(open(self.control_path))
            self.admission_policy=state['policy']
            assert self.admission_policy in POLICIES
            self.trial=state['trial']
            self.reserve=0. if self.admission_policy=='baseline' else .30 if self.admission_policy.startswith('fixed') else START
            self.calm=0
            self.control_stamp=stamp
            self.trace.write(json.dumps(dict(kind='control',t=time.perf_counter(),trial=self.trial,policy=self.admission_policy,reserve=self.reserve))+'\n')

    def add_request(self,request):
        self._load_control()
        self.trace.write(json.dumps(dict(kind='arrival',t=time.perf_counter(),trial=self.trial,request=request.request_id,prompt=request.num_prompt_tokens,policy=self.admission_policy))+'\n')
        return super().add_request(request)

    def _allocate_with_resume_gate(self,request,num_new_tokens,*args,**kwargs):
        manager=self.kv_cache_manager
        saved=manager.watermark_blocks
        status=request.status
        waiting=status in (RequestStatus.WAITING,RequestStatus.PREEMPTED)
        selected=effective_watermark(self.admission_policy,status,saved)
        would_apply=saved if waiting and kwargs.get('has_scheduled_reqs',True) else 0
        applied=selected if waiting and kwargs.get('has_scheduled_reqs',True) else 0
        free=need=None
        stamp=time.perf_counter()
        if waiting:
            assert not args and kwargs['full_sequence_must_fit']
            free=manager.block_pool.get_num_free_blocks()
            local=request.num_computed_tokens+kwargs.get('num_new_computed_tokens',0)
            external=kwargs.get('num_external_computed_tokens',0)
            full=min(request.num_tokens,manager.max_model_len)
            new=kwargs.get('new_computed_blocks')
            blocks=new.blocks if new is not None else manager.empty_kv_cache_blocks.blocks
            need=manager.coordinator.get_num_blocks_to_allocate(request_id=request.request_id,num_tokens=full,new_computed_blocks=blocks,num_encoder_tokens=kwargs.get('num_encoder_tokens',0),total_computed_tokens=min(local+external,manager.max_model_len),num_local_computed_tokens=local,num_tokens_main_model=full,apply_admission_cap=True)
        manager.watermark_blocks=selected
        try:
            result=self.native_allocator(request,num_new_tokens,*args,**kwargs)
        finally:
            manager.watermark_blocks=saved
        if waiting:
            outcome='admit' if result is not None else 'capacity' if need>free else 'watermark' if need+applied>free else 'other'
            self.trace.write(json.dumps(dict(kind='allocation',t=stamp,trial=self.trial,policy=self.admission_policy,request=request.request_id,status=status.name,preemptions=request.num_preemptions,output=request.num_output_tokens,computed=request.num_computed_tokens,tokens=request.num_tokens,new_tokens=num_new_tokens,free_blocks=free,full_blocks=need,global_watermark=saved,would_apply_watermark=would_apply,effective_watermark=applied,waived=applied!=would_apply,outcome=outcome))+'\n')
        return result

    def schedule(self,*args,**kwargs):
        self._load_control()
        self.preemptions_this_step=0
        self.kv_cache_manager.watermark_blocks=int(self.reserve*self.kv_cache_manager.block_pool.num_gpu_blocks)
        used=self.reserve
        output=super().schedule(*args,**kwargs)
        if self.admission_policy.startswith('adaptive') and not self.preemptions_this_step:
            self.reserve,self.calm=feedback(self.reserve,self.calm,0,output.total_num_scheduled_tokens>0)
        self.trace.write(json.dumps(dict(kind='admission',t=time.perf_counter(),trial=self.trial,policy=self.admission_policy,reserve=used,next_reserve=self.reserve,preemptions=self.preemptions_this_step))+'\n')
        return output

    def _preempt_request(self,request,timestamp,drop_stale_output=False):
        self.preemptions_this_step+=1
        if self.admission_policy.startswith('adaptive'):
            self.reserve,self.calm=feedback(self.reserve,self.calm,1)
        return super()._preempt_request(request,timestamp,drop_stale_output)

if __name__=='__main__':
    import inspect,re
    from vllm.v1.core.sched.scheduler import Scheduler
    own=set(re.findall(r'self\.(\w+)\s*=',inspect.getsource(ResumeScheduler)))
    native=set(re.findall(r'self\.(\w+)',inspect.getsource(Scheduler)))
    assert not own & native,own & native
    inspect.signature(ResumeScheduler.schedule).bind(object(),False)
    for policy in POLICIES:
        assert effective_watermark(policy,RequestStatus.WAITING,30)==30
        assert effective_watermark(policy,RequestStatus.RUNNING,30)==30
        assert effective_watermark(policy,RequestStatus.PREEMPTED,30)==(0 if policy.endswith('_resume') else 30)
    # The native allocator still decides feasibility; only its temporary watermark changes.
    from types import SimpleNamespace
    fake=ResumeScheduler.__new__(ResumeScheduler)
    fake.kv_cache_manager=SimpleNamespace(watermark_blocks=30)
    fake.admission_policy='adaptive_resume'
    seen=[]
    def original(req,n,**kwargs):
        seen.append(fake.kv_cache_manager.watermark_blocks)
        raise ValueError('sentinel')
    fake.native_allocator=original
    try: fake._allocate_with_resume_gate(SimpleNamespace(status=RequestStatus.RUNNING),1)
    except ValueError: pass
    assert seen==[30] and fake.kv_cache_manager.watermark_blocks==30
    fake.kv_cache_manager.max_model_len=2048
    fake.kv_cache_manager.block_pool=SimpleNamespace(get_num_free_blocks=lambda:99)
    fake.kv_cache_manager.empty_kv_cache_blocks=SimpleNamespace(blocks=())
    fake.kv_cache_manager.coordinator=SimpleNamespace(get_num_blocks_to_allocate=lambda **kwargs:1)
    request=SimpleNamespace(request_id='fake',status=RequestStatus.PREEMPTED,num_tokens=16,num_computed_tokens=0)
    try: fake._allocate_with_resume_gate(request,16,full_sequence_must_fit=True)
    except ValueError: pass
    assert seen==[30,0] and fake.kv_cache_manager.watermark_blocks==30
    print('Resume gate and allocator restoration checks passed')


