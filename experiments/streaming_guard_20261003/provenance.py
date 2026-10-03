import datetime,hashlib,importlib.metadata,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent;LAB=ROOT.parent.parent;native=LAB/'.venv/lib/python3.12/site-packages/vllm'
files=[LAB/'requirements.lock.txt',LAB/'activate.sh']+[ROOT.parent/'capacity_cliff_20261002'/n for n in ('bench.py','analyze.py','trace_scheduler.py')]+[ROOT.parent/'admission_robustness_20261002'/n for n in ('run.py','control_scheduler.py')]+[ROOT.parent/'resume_admission_20261003'/n for n in ('resume_scheduler.py','analyze.py')]+[native/n for n in ('v1/core/sched/scheduler.py','v1/core/sched/async_scheduler.py','v1/core/sched/request_queue.py','v1/core/kv_cache_manager.py','v1/metrics/loggers.py','v1/metrics/stats.py')]
info=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,clock=vars(time.get_clock_info('perf_counter')),versions={n:importlib.metadata.version(n) for n in ('vllm','torch','flashinfer-python','transformers')},source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv'],text=True))
repo=ROOT/'references/SLAI';info['slai_reference']=dict(url='https://github.com/agrimUT/SLAI',commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),scheduler_sha256=hashlib.sha256((repo/'sarathi/core/scheduler/slai_scheduler.py').read_bytes()).hexdigest())
bounds=[]
for _ in range(5):
    before=time.perf_counter();child=float(subprocess.check_output([sys.executable,'-c','import time; print(time.perf_counter())'],text=True));after=time.perf_counter();assert before<=child<=after;bounds.append(dict(before=before,child=child,after=after))
info['cross_process_clock_bounds']=bounds
(ROOT/'provenance.json').write_text(json.dumps(info,indent=2));print('Saved provenance and verified cross-process clock bounds.')
