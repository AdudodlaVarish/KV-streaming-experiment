import argparse, concurrent.futures as cf, hashlib, json, os, pathlib, random
import signal, subprocess, sys, threading, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
PRIOR=ROOT.parent/'capacity_cliff_20261002'
sys.path.insert(0,str(PRIOR))
from bench import get, metrics, total, percentile, URL
MODELS={
    'llama1b':'/home/varish/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',
    'qwen1.5b':'/home/varish/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
GPU_CMD=['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm','--format=csv']

def control(path,trial,policy):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(trial=trial,policy=policy)))
    os.replace(temp,path)

def workload(seed,mix,arrival,tok):
    rng=random.Random(seed+(1000 if mix=='generation' else 0))
    ins=([128,256,512,1024]*3 if mix=='balanced' else [128,256,512]*4)
    outs=([64,128,256,512]*3 if mix=='balanced' else [64,256,768]*4)
    rng.shuffle(ins)
    rng.shuffle(outs)
    offsets=[i*.05 for i in range(12)]
    if arrival=='stagger':
        for i in range(1,12):
            offsets[i]=offsets[i-1]+.05+rng.expovariate(1/.95)
    records=[]
    for i,(p,o,a) in enumerate(zip(ins,outs,offsets)):
        text=f'Record {seed}-{i}. Explain these notes in detail. '+('A cache saves previously computed values. Memory capacity and bandwidth are different limits. '*p)
        ids=tok.encode(text,add_special_tokens=False)[:p]
        assert len(ids)==p and p+o<=2048
        records.append(dict(index=i,prompt=ids,output=o,arrival_s=a))
    return records

def request(record,origin,gate,request_id):
    gate.wait(timeout=30)
    remaining=origin+record['arrival_s']-time.perf_counter()
    if remaining>0:
        time.sleep(remaining)
    start=time.perf_counter()
    body=json.dumps(dict(model='lab-llama',prompt=record['prompt'],max_tokens=record['output'],
        temperature=0,seed=0,ignore_eos=True,stream=True,return_token_ids=True,
        stream_options={'include_usage':True})).encode()
    req=urllib.request.Request(URL+'/v1/completions',body,
        {'Content-Type':'application/json','X-Request-Id':request_id})
    token_events,text_events,ids,usage,response_id=[],[],[],None,None
    with urllib.request.urlopen(req,timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload=line[6:].strip()
            if payload==b'[DONE]':
                break
            data=json.loads(payload)
            assert not data.get('error'),data
            response_id=data.get('id',response_id)
            if data.get('usage'):
                usage=data['usage']
            for choice in data.get('choices',[]):
                stamp=time.perf_counter()-start
                delta=choice.get('token_ids') or []
                if delta:
                    ids.extend(delta)
                    token_events.append([stamp,len(ids)])
                if choice.get('text'):
                    text_events.append(stamp)
    end=time.perf_counter()
    assert usage and usage['prompt_tokens']==len(record['prompt']),usage
    assert usage['completion_tokens']==record['output']==len(ids),(usage,len(ids),record['output'])
    assert token_events
    gaps=[b[0]-a[0] for a,b in zip(token_events,token_events[1:])]
    return dict(index=record['index'],request_id=request_id,response_id=response_id,start=start,
        planned_arrival_perf_s=origin+record['arrival_s'],arrival_lateness_s=start-origin-record['arrival_s'],
        elapsed_s=end-start,ttft_s=token_events[0][0],first_text_s=text_events[0] if text_events else None,
        token_events=token_events,text_events=text_events,max_silence_s=max(gaps or [0]),
        silence_over_1s=sum(g>1 for g in gaps),silence_excess_1s_s=sum(max(0,g-1) for g in gaps),
        input_tokens=len(record['prompt']),output_tokens=record['output'],usage=usage,
        token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())

def trial(records,folder,control_path,policy,trial_id):
    folder.mkdir(parents=True,exist_ok=False)
    control(control_path,trial_id,policy)
    before_raw=get('/metrics')
    (folder/'metrics_before.txt').write_text(before_raw)
    before=metrics(before_raw)
    (folder/'gpu_before.csv').write_bytes(subprocess.check_output(GPU_CMD))
    gate=threading.Barrier(len(records)+1)
    origin=time.perf_counter()+.10
    with cf.ThreadPoolExecutor(max_workers=len(records)) as pool:
        fs=[pool.submit(request,r,origin,gate,f'{trial_id}-r{r["index"]}') for r in records]
        gate.wait(timeout=30)
        # Save completed requests even if a later request fails.
        results=[]
        for f in cf.as_completed(fs):
            r=f.result()
            results.append(r)
            with (folder/'requests.jsonl').open('a') as file:
                file.write(json.dumps(r)+'\n')
    start=min(r['start'] for r in results)
    end=max(r['start']+r['elapsed_s'] for r in results)
    after_raw=get('/metrics')
    (folder/'metrics_after.txt').write_text(after_raw)
    (folder/'gpu_after.csv').write_bytes(subprocess.check_output(GPU_CMD))
    after=metrics(after_raw)
    delta={name:total(after,name)-total(before,name) for name in (
        'vllm:num_preemptions_total','vllm:prompt_tokens_total','vllm:generation_tokens_total',
        'vllm:iteration_tokens_total_sum','vllm:request_success_total')}
    summary=dict(trial=trial_id,policy=policy,n=len(records),start_perf_s=start,end_perf_s=end,
        wall_s=end-start,throughput=sum(r['output_tokens'] for r in results)/(end-start),
        ttft_p50=percentile([r['ttft_s'] for r in results],50),ttft_p95=percentile([r['ttft_s'] for r in results],95),
        e2e_p50=percentile([r['elapsed_s'] for r in results],50),e2e_p95=percentile([r['elapsed_s'] for r in results],95),
        max_silence=max(r['max_silence_s'] for r in results),silence_p95=percentile([r['max_silence_s'] for r in results],95),
        stalled_requests=sum(r['max_silence_s']>1 for r in results),
        silence_excess_1s_s=sum(r['silence_excess_1s_s'] for r in results),metrics_delta=delta,
        max_arrival_lateness_s=max(r['arrival_lateness_s'] for r in results))
    assert delta['vllm:request_success_total']==len(records),delta
    assert delta['vllm:generation_tokens_total']==sum(r['output_tokens'] for r in results),delta
    assert delta['vllm:prompt_tokens_total']==sum(r['input_tokens'] for r in results),delta
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def server_run(job,seed,run_id,diagnostic=False):
    model,slots,cap=job
    folder=ROOT/run_id
    folder.mkdir(exist_ok=False)
    config=json.load(open(pathlib.Path(MODELS[model])/'config.json'))
    head_dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    bpt=2*config['num_hidden_layers']*config['num_key_value_heads']*head_dim*2
    control_path=folder/'control.json'
    control(control_path,'warmup','baseline')
    cmd=['vllm','serve',MODELS[model],'--served-model-name','lab-llama','--host','127.0.0.1','--port','8017',
        '--max-model-len','2048','--max-num-seqs',str(cap),'--max-num-batched-tokens','2048',
        '--kv-cache-memory-bytes',str(slots*bpt),'--gpu-memory-utilization','0.75',
        '--block-size','16','--dtype','bfloat16','--enforce-eager','--attention-backend','FLASHINFER',
        '--no-enable-prefix-caching','--generation-config','vllm','--seed','0',
        '--scheduler-cls','control_scheduler.ControlScheduler','--async-scheduling']
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=os.pathsep.join([str(ROOT),str(PRIOR),os.environ.get('PYTHONPATH','')]),
        LAB_CONTROL=str(control_path),LAB_SCHEDULER_TRACE=str(folder/'scheduler_trace.jsonl'))
    (folder/'command.json').write_text(json.dumps(cmd,indent=2))
    (folder/'config.json').write_text(json.dumps(dict(model=model,slots=slots,cap=cap,seed=seed,bytes_per_token=bpt,cache_bytes=slots*bpt),indent=2))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS[model],local_files_only=True)
    with (folder/'server.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        (folder/'server_pid.txt').write_text(str(proc.pid))
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited: {folder}/server.log')
                try:
                    get('/health')
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError('server startup')
            warm=[dict(r,output=32) for r in workload(seed,'balanced','burst',tok)[:min(4,cap)]]
            for w in range(2):
                trial(warm,folder/f'warmup{w}',control_path,'baseline',f'{run_id}-warm{w}')
            cases=[(mix,arrival,policy) for mix in ('balanced','generation') for arrival in ('burst','stagger') for policy in ('baseline','fixed','adaptive')]
            if model=='qwen1.5b':
                cases=[c for c in cases if c[0]=='generation']
            if diagnostic:
                cases=[('generation','burst',p) for p in ('baseline','fixed','adaptive')]
            rng=random.Random(seed+slots+cap*100)
            rng.shuffle(cases)
            (folder/'case_order.json').write_text(json.dumps(cases))
            for number,(mix,arrival,policy) in enumerate(cases):
                records=workload(seed,mix,arrival,tok)
                case=f'{number:02d}_{mix}_{arrival}_{policy}'
                trial_id=f'{run_id}-{case}'
                cell=folder/case
                # Every policy gets exactly the same trace for a seed/mix/arrival.
                (folder/f'{mix}_{arrival}_workload.json').write_text(json.dumps(records))
                summary=trial(records,cell,control_path,policy,trial_id)
                summary.update(model=model,slots=slots,cap=cap,seed=seed,mix=mix,arrival=arrival,run=run_id,folder=case)
                (cell/'summary.json').write_text(json.dumps(summary,indent=2))
                with (ROOT/'summaries.jsonl').open('a') as f:
                    f.write(json.dumps(summary)+'\n')
                print(json.dumps({k:summary[k] for k in ('run','folder','throughput','max_silence','stalled_requests','metrics_delta')}),flush=True)
        finally:
            try:
                os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            time.sleep(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--prefix',default='study')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    primary=[(3072,3),(3072,12),(5120,3),(5120,6),(5120,12),(12288,12)]
    # Fixed prospective design: six pressure/control points, three policies at every point.
    # Two seeds randomize both request pairings and configuration/case order.
    jobs=[]
    for seed in (23,77):
        block=[('llama1b',slots,cap) for slots,cap in primary]
        # Secondary architecture check covers both pressure and capacity relief.
        block += [('qwen1.5b',3072,6),('qwen1.5b',12288,12)]
        random.Random(seed).shuffle(block)
        jobs += [(seed,job) for job in block]
    if args.diagnostic:
        jobs=[(23,('llama1b',3072,12))]
    manifest=dict(seeds=[23,77],requests_per_trace=12,prompt_balanced=[128,256,512,1024],
        output_balanced=[64,128,256,512],prompt_generation=[128,256,512],output_generation=[64,256,768],
        burst_interval_s=.05,stagger_min_s=.05,stagger_exponential_mean_s=.95,policies=['baseline','fixed .30','feedback .05/+ .08 per eviction/- .02 per 128 calm steps/max .45'],
        jobs=jobs,diagnostic=args.diagnostic,model_paths=MODELS)
    (ROOT/('diagnostic_design.json' if args.diagnostic else 'design.json')).write_text(json.dumps(manifest,indent=2))
    for ordinal,(seed,job) in enumerate(jobs):
        model,slots,cap=job
        run_id=f'{"ordereddiag" if args.diagnostic else args.prefix}{ordinal:02d}_s{seed}_{model}_kv{slots}_cap{cap}'
        if args.resume and (ROOT/run_id/'complete.json').exists():
            continue
        server_run(job,seed,run_id,args.diagnostic)
        (ROOT/run_id/'complete.json').write_text(json.dumps(dict(complete=True)))





