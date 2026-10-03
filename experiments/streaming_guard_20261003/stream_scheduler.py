import ast,copy,inspect,json,math,os,pathlib,textwrap,time
from vllm.v1.request import RequestStatus
from vllm.v1.core.sched.request_queue import SchedulingPolicy,create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
import vllm.v1.core.sched.scheduler as native_module
from resume_scheduler import ResumeScheduler
from control_scheduler import feedback

POLICIES=('baseline','adaptive_resume','slai_port','growth_only','guard128','guard256','guard_fast','guard_relaxed')
PARAMS={p:dict(growth=256 if p=='guard256' else 128,budget=.25 if p=='guard_fast' else 1. if p=='guard_relaxed' else .5) for p in POLICIES}

def blocks_for(tokens,block=16):return (tokens+block-1)//block

def slai_schedule_port():
    # Reuse the pinned native scheduling body, including all async bookkeeping.
    # Change only the running phases and victim rule, matching SLAI's core order.
    tree=ast.parse(textwrap.dedent(inspect.getsource(Scheduler.schedule)))
    function=tree.body[0]
    loop=next(n for n in function.body if isinstance(n,ast.While) and isinstance(n.test,ast.BoolOp) and 'req_index' in ast.unparse(n.test))
    assert ast.unparse(loop.body[0])=='request = self.running[req_index]'
    first=ast.parse("if request.request_id in lab_deferred:\n    req_index += 1\n    continue").body[0]
    last=ast.parse("if request.request_id not in lab_deferred:\n    req_index += 1\n    continue").body[0]
    second=copy.deepcopy(loop);second.body.insert(1,last);loop.body.insert(1,first)
    for branch in (loop,second):
        replaced=0
        for node in ast.walk(branch):
            if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='preempted_req' and ast.unparse(node.value).startswith('max(self.running'):
                node.value=ast.parse('self._slai_victim(request, scheduled_new_reqs + scheduled_resumed_reqs + scheduled_running_reqs)',mode='eval').body;replaced+=1
        assert replaced==1
    index=function.body.index(loop)
    function.body[index:index]=ast.parse('lab_deferred = self._slai_deferred()').body
    # Insert noncritical decodes after the native waiting phase, before assertions.
    insertion=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='total_num_scheduled_tokens')
    function.body[insertion:insertion]=ast.parse('req_index = 0').body+[second]
    ast.fix_missing_locations(tree)
    namespace=vars(native_module).copy();exec(compile(tree,'<native_slai_port>','exec'),namespace)
    generated='# Native vLLM Apache-2.0 scheduler body with SLAI phase/victim adaptation.\n'+ast.unparse(tree)+'\n'
    path=pathlib.Path(__file__).with_name('native_slai_schedule.py')
    if path.exists():assert path.read_text()==generated,'Pinned native body differs from saved port'
    else:path.write_text(generated)
    return namespace['schedule']

SLAI_SCHEDULE=slai_schedule_port()

class StreamScheduler(ResumeScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert not self.cache_config.enable_prefix_caching and len(self.kv_cache_config.kv_cache_groups)==1
        assert self.block_size==16 and not self.num_spec_tokens and not self.connector
        self.last_service={};self.first_service={};self.recent_eviction={}
        self.dispatch_times={};self.batch_mean=.02;self.batch_count=0
        self.scheduling_now=0.

    def _load_control(self):
        stamp=os.stat(self.control_path).st_mtime_ns
        if stamp==self.control_stamp:return
        state=json.load(open(self.control_path));self.admission_policy=state['policy'];assert self.admission_policy in POLICIES
        self.trial=state['trial'];self.reserve=.05 if self.admission_policy=='adaptive_resume' else 0.;self.calm=0;self.control_stamp=stamp
        self.last_service={};self.first_service={};self.recent_eviction={};self.dispatch_times={};self.batch_mean=.02;self.batch_count=0
        self.trace.write(json.dumps(dict(kind='control',t=time.perf_counter(),trial=self.trial,policy=self.admission_policy,reserve=self.reserve,params=PARAMS[self.admission_policy]))+'\n')

    def _deadline(self,r):return self.last_service.get(r.request_id,self.scheduling_now)+PARAMS[self.admission_policy]['budget']

    def _rank(self,r):
        now=self.scheduling_now;rid=r.request_id
        if rid not in self.first_service:return (1,r.arrival_time,0)
        # A recent eviction earns one budget of protection, capped at one credit.
        credit=PARAMS[self.admission_policy]['budget'] if now-self.recent_eviction.get(rid,-math.inf)<1. else 0.
        return (0,self._deadline(r)-credit,r.arrival_time)

    def _prepare_order(self):
        ordered=self.admission_policy.startswith('guard') or self.admission_policy=='slai_port'
        self.policy=SchedulingPolicy.PRIORITY if ordered else SchedulingPolicy.FCFS
        requests=list(self.requests.values())
        if ordered:
            if self.admission_policy=='slai_port':
                key=lambda r:(0,self._deadline(r),r.arrival_time) if r.request_id in self.first_service else (1,r.num_prompt_tokens,r.arrival_time)
            else:key=self._rank
            for priority,r in enumerate(sorted(requests,key=key)):r.priority=priority
            self.running.sort(key=lambda r:r.priority)
        # Rebuild heaps after priority updates; never mutate a live heap's keys.
        for name in ('waiting','skipped_waiting'):
            existing=list(getattr(self,name));queue=create_request_queue(self.policy)
            for r in existing:queue.add_request(r)
            setattr(self,name,queue)

    def _slai_deferred(self):
        critical_at=self.scheduling_now+2*self.batch_mean
        return {r.request_id for r in self.running if r.request_id in self.first_service and not r.is_prefill_chunk and self._deadline(r)>critical_at}

    def _slai_victim(self,request,scheduled):
        prefills=[r for r in self.running if r.request_id not in self.first_service and r not in scheduled]
        return max(prefills,key=lambda r:r.arrival_time) if prefills else request

    def _growth_extra(self,r,growth):
        allocated=len(self.kv_cache_manager.get_blocks(r.request_id).blocks[0])
        known=max(r.num_tokens,r.num_computed_tokens+r.num_output_placeholders)
        return max(0,blocks_for(min(self.max_model_len,known+growth))-allocated)

    def _allocate_with_resume_gate(self,request,num_new_tokens,*args,**kwargs):
        guarded=self.admission_policy=='growth_only' or self.admission_policy.startswith('guard')
        if guarded and request.status==RequestStatus.WAITING:
            growth=PARAMS[self.admission_policy]['growth'];free=self.kv_cache_manager.block_pool.get_num_free_blocks()
            paused=[r for r in list(self.waiting)+list(self.skipped_waiting) if r.request_id in self.first_service]
            debt=sum(self._growth_extra(r,growth) for r in self.running)
            need=self._growth_extra(request,growth)
            denied=bool(paused) or debt+need>free
            # Escape for a single near-context-limit prompt when the engine is idle.
            if not self.running and not paused:denied=blocks_for(request.num_tokens)>free
            self.trace.write(json.dumps(dict(kind='growth_gate',t=time.perf_counter(),trial=self.trial,policy=self.admission_policy,request=request.request_id,free=free,debt=debt,need=need,growth=growth,paused_streams=len(paused),denied=denied,running=len(self.running),idle_escape=not self.running and not paused))+'\n')
            if denied:return None
        return super()._allocate_with_resume_gate(request,num_new_tokens,*args,**kwargs)

    def _urgent_handoff(self):
        if not self.admission_policy.startswith('guard'):return
        now=self.scheduling_now;budget=PARAMS[self.admission_policy]['budget']
        paused=[r for r in list(self.waiting)+list(self.skipped_waiting) if r.request_id in self.first_service and not r.num_stale_output_tokens and now-self.last_service[r.request_id]>=budget*.5]
        if not paused:return
        target=min(paused,key=self._rank)
        free=self.kv_cache_manager.block_pool.get_num_free_blocks();need=blocks_for(target.num_tokens)
        if need<=free and len(self.running)<self.max_num_running_reqs:return
        candidates=[r for r in self.running if self._request_blocks_can_be_freed(r)]
        if not candidates:return
        victim=max(candidates,key=lambda r:r.priority)
        # Don't trade an older, equally protected deadline for a younger one.
        if victim.request_id in self.first_service and self._rank(victim)<=self._rank(target):return
        self.running.remove(victim)
        self.trace.write(json.dumps(dict(kind='handoff',t=now,trial=self.trial,policy=self.admission_policy,target=target.request_id,victim=victim.request_id,free=free,need=need,target_age=now-self.last_service[target.request_id]))+'\n')
        self._preempt_request(victim,time.monotonic())

    def schedule(self,*args,**kwargs):
        self._load_control();self.scheduling_now=time.perf_counter();self.preemptions_this_step=0
        self._prepare_order();self._urgent_handoff()
        self.kv_cache_manager.watermark_blocks=int(self.reserve*self.kv_cache_manager.block_pool.num_gpu_blocks);used=self.reserve
        output=SLAI_SCHEDULE(self,*args,**kwargs) if self.admission_policy=='slai_port' else Scheduler.schedule(self,*args,**kwargs)
        self.dispatch_times[id(output)]=time.perf_counter()
        if self.admission_policy=='adaptive_resume' and not self.preemptions_this_step:self.reserve,self.calm=feedback(self.reserve,self.calm,0,output.total_num_scheduled_tokens>0)
        self.trace.write(json.dumps(dict(kind='admission',t=time.perf_counter(),trial=self.trial,policy=self.admission_policy,reserve=used,next_reserve=self.reserve,preemptions=self.preemptions_this_step))+'\n')
        return output

    def _preempt_request(self,request,timestamp,drop_stale_output=False):
        self.recent_eviction[request.request_id]=time.perf_counter()
        return super()._preempt_request(request,timestamp,drop_stale_output)

    def _update_request_with_output(self,request,new_token_ids,is_stale=False):
        result,stopped=super()._update_request_with_output(request,new_token_ids,is_stale)
        if result:
            now=time.perf_counter();rid=request.request_id;self.first_service.setdefault(rid,now);self.last_service[rid]=now
            self.trace.write(json.dumps(dict(kind='service',t=now,trial=self.trial,policy=self.admission_policy,request=rid,n=len(result),stopped=stopped,stale=is_stale))+'\n')
        return result,stopped

    def update_from_output(self,scheduler_output,model_runner_output):
        result=super().update_from_output(scheduler_output,model_runner_output)
        dispatched=self.dispatch_times.pop(id(scheduler_output),None)
        if dispatched is not None and scheduler_output.total_num_scheduled_tokens:
            elapsed=time.perf_counter()-dispatched;self.batch_count+=1;self.batch_mean+=(elapsed-self.batch_mean)/self.batch_count
        return result

if __name__=='__main__':
    import re
    own=set(re.findall(r'self\.(\w+)\s*=(?!=)',inspect.getsource(StreamScheduler)));native=set(re.findall(r'self\.(\w+)',inspect.getsource(Scheduler)))
    assert own & native=={'policy'},own & native
    assert blocks_for(16)==1 and blocks_for(17)==2
    assert 'max_tokens' not in inspect.getsource(StreamScheduler)
    from types import SimpleNamespace as N
    s=StreamScheduler.__new__(StreamScheduler);s.admission_policy='guard128';s.scheduling_now=10.;s.first_service={'a':1.,'b':1.};s.last_service={'a':9.,'b':9.9};s.recent_eviction={}
    a=N(request_id='a',arrival_time=1.);b=N(request_id='b',arrival_time=2.)
    assert s._rank(a)<s._rank(b)
    s.recent_eviction['b']=9.9;assert s._rank(a)<s._rank(b)
    s.last_service['a']=9.8;assert s._rank(b)<s._rank(a)
    print('Growth rounding, deadline ordering, repeat protection and native port structure checks passed.')


