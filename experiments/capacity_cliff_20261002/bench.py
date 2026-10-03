import argparse, concurrent.futures as cf, hashlib, json, os, pathlib
import signal, subprocess, threading, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = '/home/varish/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
URL = 'http://127.0.0.1:8017'

def get(path):
    return urllib.request.urlopen(URL + path, timeout=10).read().decode()

def metrics(raw):
    vals = {}
    for line in raw.splitlines():
        if line and not line.startswith('#'):
            key, value, *_ = line.split()
            vals[key] = float(value)
    return vals

def total(vals, prefix):
    return sum(v for k, v in vals.items() if k.split('{')[0] == prefix)

def percentile(values, p):
    vals = sorted(values)
    x = (len(vals)-1)*p/100
    a = int(x)
    return vals[a] + (vals[min(a+1, len(vals)-1)]-vals[a])*(x-a)

def request(prompt, output, gate=None):
    if gate:
        gate.wait(timeout=30)
    start = time.perf_counter()
    body = json.dumps(dict(model='lab-llama', prompt=prompt, max_tokens=output,
        temperature=0, seed=0, ignore_eos=True, stream=True,
        stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(URL+'/v1/completions', body,
        {'Content-Type': 'application/json'})
    events, fragments, usage = [], [], None
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            data = json.loads(payload)
            if data.get('usage'):
                usage = data['usage']
            for choice in data.get('choices', []):
                if choice.get('text'):
                    events.append(time.perf_counter()-start)
                    fragments.append(choice['text'])
    end = time.perf_counter()-start
    assert usage and usage['prompt_tokens'] == len(prompt), usage
    assert usage['completion_tokens'] == output, usage
    return dict(start=start, start_unix_s=time.time()-end, elapsed_s=end, ttft_s=events[0],
        stream_event_times_s=events, max_stream_gap_s=max([b-a for a,b in zip(events,events[1:])] or [0]),
        usage=usage, text_sha256=hashlib.sha256(''.join(fragments).encode()).hexdigest())

def burst(prompts, output, folder):
    folder.mkdir(parents=True, exist_ok=False)
    gpu_cmd = ['nvidia-smi','--query-gpu=timestamp,name,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm','--format=csv']
    (folder/'gpu_before.csv').write_bytes(subprocess.check_output(gpu_cmd))
    before_raw = get('/metrics')
    (folder/'metrics_before.txt').write_text(before_raw)
    before = metrics(before_raw)
    gate = threading.Barrier(len(prompts)+1)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futs = [pool.submit(request, p, output, gate) for p in prompts]
        gate.wait(timeout=30)
        t0 = time.perf_counter()
        results = [f.result() for f in futs]
    t0 = min(r['start'] for r in results)
    elapsed = max(r['start']+r['elapsed_s'] for r in results)-t0
    (folder/'gpu_after.csv').write_bytes(subprocess.check_output(gpu_cmd))
    after_raw = get('/metrics')
    (folder/'metrics_after.txt').write_text(after_raw)
    (folder/'requests.json').write_text(json.dumps(results, indent=2))
    after = metrics(after_raw)
    delta = {name: total(after, name)-total(before,name) for name in (
        'vllm:num_preemptions_total', 'vllm:prompt_tokens_total',
        'vllm:prompt_tokens_by_source_total', 'vllm:generation_tokens_total',
        'vllm:iteration_tokens_total_sum', 'vllm:request_success_total')}
    summary = dict(wall_s=elapsed, output_tok_s=len(prompts)*output/elapsed,
        n=len(prompts), input_tokens=len(prompts[0]), output_tokens=output,
        ttft_p50_s=percentile([r['ttft_s'] for r in results],50),
        ttft_p95_s=percentile([r['ttft_s'] for r in results],95),
        e2e_p50_s=percentile([r['elapsed_s'] for r in results],50),
        e2e_p95_s=percentile([r['elapsed_s'] for r in results],95),
        max_stream_gap_s=max(r['max_stream_gap_s'] for r in results),
        request_gap_p50_s=percentile([r['max_stream_gap_s'] for r in results],50),
        request_gap_p95_s=percentile([r['max_stream_gap_s'] for r in results],95),
        stalled_requests_1s=sum(r['max_stream_gap_s']>1 for r in results),
        start_perf_s=t0, metrics_delta=delta)
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary

def run(config, prompts, output, reps, tag):
    name = f'{tag}_kv{config[0]}_cap{config[1]}'
    folder = ROOT/name
    folder.mkdir(exist_ok=False)
    cmd = ['vllm','serve',MODEL,'--served-model-name','lab-llama','--host','127.0.0.1','--port','8017',
        '--max-model-len','2048','--max-num-seqs',str(config[1]),
        '--max-num-batched-tokens','2048','--kv-cache-memory-bytes',str(config[0]*1024**2),
        '--gpu-memory-utilization','0.75','--block-size','16','--dtype','bfloat16',
        '--enforce-eager','--attention-backend','FLASHINFER','--no-enable-prefix-caching',
        '--generation-config','vllm','--seed','0','--watermark',str(args.watermark)]
    if args.trace:
        cmd += ['--scheduler-cls','trace_scheduler.TraceScheduler','--async-scheduling']
    env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
        LAB_SCHEDULER_TRACE=str(folder/'scheduler_trace.jsonl'))
    (folder/'command.json').write_text(json.dumps(cmd,indent=2))
    with (folder/'server.log').open('w') as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
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
                raise TimeoutError('startup')
            # Exercise actual prompt length and mixed prefill/decode before measuring.
            for w in range(2):
                burst(prompts[:min(config[1],4)],64,folder/f'warmup{w}')
            for i in range(reps):
                summary = burst(prompts,output,folder/f'rep{i}')
                summary.update(config=name, rep=i)
                with (ROOT/'summaries.jsonl').open('a') as f:
                    f.write(json.dumps(summary)+'\n')
                print(json.dumps(summary),flush=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
            time.sleep(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',default='pilot')
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--watermark',type=float,default=0.0)
    parser.add_argument('--configs',default='128:3,128:8,256:8')
    parser.add_argument('--reps',type=int,default=1)
    parser.add_argument('--input',type=int,default=768)
    parser.add_argument('--output',type=int,default=384)
    parser.add_argument('--requests',type=int,default=12)
    args = parser.parse_args()
    assert percentile([1,2,3],50)==2
    assert total(metrics('x{a="b"} 4\nx{a="c"} 5\n'), 'x')==9
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    prompts = []
    for i in range(args.requests):
        text = f'Record {i}. Explain the following notes in detail. '+ ('A cache saves previously computed values. Memory capacity and bandwidth are different limits. ' * args.input)
        tokens = tok.encode(text,add_special_tokens=False)[:args.input]
        assert len(tokens)==args.input
        prompts.append(tokens)
    (ROOT/f'{args.tag}_prompts.json').write_text(json.dumps(prompts))
    for config in args.configs.split(','):
        run(tuple(map(int,config.split(':'))),prompts,args.output,args.reps,args.tag)




