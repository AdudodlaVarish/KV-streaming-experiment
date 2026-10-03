import argparse, hashlib, importlib.util, json, os, pathlib, random, signal, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'admission_robustness_20261002'
PRIOR=ROOT.parent/'capacity_cliff_20261002'
sys.path[:0]=[str(OLD),str(PRIOR)]
spec=importlib.util.spec_from_file_location('prior_client',OLD/'run.py')
prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
POLICIES=['baseline','fixed','adaptive','fixed_resume','adaptive_resume']

def cases_for(model,slots,cap):
    mixes=['balanced','generation'] if model=='llama1b' and slots==3072 and cap==12 else ['generation']
    policies=POLICIES if (slots==3072 and cap!=3) else ['baseline'] if cap==3 else ['baseline','adaptive','adaptive_resume']
    return [(mix,arrival,p) for mix in mixes for arrival in ('burst','stagger') for p in policies]

def check_order(folder):
    events=[json.loads(line) for line in (folder/'scheduler_trace.jsonl').read_text().splitlines()]
    failures=[]
    for cell in sorted(folder.glob('[0-9][0-9]_*')):
        row=json.loads((cell/'summary.json').read_text())
        requests=[json.loads(line) for line in (cell/'requests.jsonl').read_text().splitlines()]
        arrival=[e for e in events if e['kind']=='arrival' and e['trial']==row['trial']]
        order=[]
        for e in arrival:
            match=[r['index'] for r in requests if e['request'].startswith(r['response_id']+'-')]
            assert len(match)==1,(row['trial'],e)
            order+=match
        if order!=list(range(12)): failures.append(dict(trial=row['trial'],order=order))
    (folder/'order_check.json').write_text(json.dumps(dict(checked_trials=len(list(folder.glob('[0-9][0-9]_*'))),failures=failures),indent=2))
    return failures

def server_run(job,seed,name,diagnostic=False):
    model,slots,cap=job
    folder=ROOT/name; folder.mkdir(exist_ok=False)
    model_config=json.load(open(pathlib.Path(prior.MODELS[model])/'config.json'))
    head=model_config.get('head_dim',model_config['hidden_size']//model_config['num_attention_heads'])
    bpt=2*model_config['num_hidden_layers']*model_config['num_key_value_heads']*head*2
    control=folder/'control.json'; prior.control(control,'warmup','baseline')
    cmd=['vllm','serve',prior.MODELS[model],'--served-model-name','lab-llama','--host','127.0.0.1','--port','8017','--max-model-len','2048','--max-num-seqs',str(cap),'--max-num-batched-tokens','2048','--kv-cache-memory-bytes',str(slots*bpt),'--gpu-memory-utilization','0.75','--block-size','16','--dtype','bfloat16','--enforce-eager','--attention-backend','FLASHINFER','--no-enable-prefix-caching','--generation-config','vllm','--seed','0','--scheduler-cls','resume_scheduler.ResumeScheduler','--async-scheduling']
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=os.pathsep.join([str(ROOT),str(OLD),str(PRIOR),os.environ.get('PYTHONPATH','')]),LAB_CONTROL=str(control),LAB_SCHEDULER_TRACE=str(folder/'scheduler_trace.jsonl'))
    (folder/'command.json').write_text(json.dumps(cmd,indent=2))
    (folder/'config.json').write_text(json.dumps(dict(model=model,slots=slots,cap=cap,seed=seed,bytes_per_token=bpt,cache_bytes=slots*bpt),indent=2))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(prior.MODELS[model],local_files_only=True)
    with (folder/'server.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        (folder/'server_pid.txt').write_text(str(proc.pid))
        try:
            for _ in range(180):
                if proc.poll() is not None: raise RuntimeError(f'Server exited: {folder}/server.log')
                try: prior.get('/health'); break
                except Exception: time.sleep(2)
            else: raise TimeoutError('server startup')
            warm=[dict(r,output=32) for r in prior.workload(seed,'balanced','burst',tok)[:min(4,cap)]]
            for w in range(2): prior.trial(warm,folder/f'warmup{w}',control,'baseline',f'{name}-warm{w}')
            cases=cases_for(*job)
            if diagnostic: cases=[('generation','stagger',p) for p in POLICIES]
            random.Random(seed+slots+cap*100).shuffle(cases)
            (folder/'case_order.json').write_text(json.dumps(cases))
            for number,(mix,arrival,policy) in enumerate(cases):
                records=prior.workload(seed,mix,arrival,tok)
                (folder/f'{mix}_{arrival}_workload.json').write_text(json.dumps(records))
                case=f'{number:02d}_{mix}_{arrival}_{policy}'
                row=prior.trial(records,folder/case,control,policy,f'{name}-{case}')
                row.update(model=model,slots=slots,cap=cap,seed=seed,mix=mix,arrival=arrival,run=name,folder=case)
                (folder/case/'summary.json').write_text(json.dumps(row,indent=2))
                print(json.dumps({k:row[k] for k in ('run','folder','throughput','max_silence','stalled_requests','metrics_delta')}),flush=True)
        finally:
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            time.sleep(1)
    return check_order(folder)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--diagnostic',action='store_true'); parser.add_argument('--prefix',default='study'); args=parser.parse_args()
    jobs=[]
    for seed in (23,77):
        group=[('llama1b',3072,12),('qwen1.5b',3072,6),('llama1b',12288,12),('llama1b',3072,3)]
        random.Random(seed).shuffle(group); jobs += [(seed,j) for j in group]
    if args.diagnostic: jobs=[(23,('llama1b',3072,12))]
    else:
        actual={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('run.py','resume_scheduler.py','DESIGN.md')}
        assert actual==json.loads((ROOT/'evaluation_code_hashes.json').read_text()),'Frozen evaluation source differs'
    expected=sum(len(cases_for(*job)) for _,job in jobs) if not args.diagnostic else 5
    (ROOT/('diagnostic_design.json' if args.diagnostic else 'design.json')).write_text(json.dumps(dict(jobs=jobs,policies=POLICIES,expected_trials=expected,expected_requests=12*expected,prior_client=str(OLD/'run.py')),indent=2))
    for i,(seed,job) in enumerate(jobs):
        model,slots,cap=job
        for attempt in range(1,4):
            name=f'{"diag" if args.diagnostic else args.prefix}{i:02d}_a{attempt}_s{seed}_{model}_kv{slots}_cap{cap}'
            failures=server_run(job,seed,name,args.diagnostic)
            if not failures:
                (ROOT/name/'complete.json').write_text(json.dumps(dict(complete=True,order_verified=True)))
                break
            (ROOT/name).rename(ROOT/('excluded_order_'+name))
            print(json.dumps(dict(excluded=name,reason='engine arrival order',failures=failures)),flush=True)
        else: raise RuntimeError('Three order-invalid attempts; no invalid run retained')
