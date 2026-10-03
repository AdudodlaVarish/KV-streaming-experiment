import datetime,hashlib,json,pathlib,subprocess,zipfile
ROOT=pathlib.Path(__file__).resolve().parent;LAB=ROOT.parent.parent
read=lambda p:json.loads(p.read_text())
checks=read(ROOT/'checks.json'); design=read(ROOT/'design.json')
assert checks['completed_trials']==design['expected_trials']==76
assert checks['completed_requests']==design['expected_requests']==912
assert not checks['unfinished_cells']
assert all(checks[k] for k in ('token_counts_verified','arrival_order_verified','preemptions_match_metrics','replay_matches_lost_positions','policy_and_admission_checks','extended_prefill_metric_excludes_replay','source_hashes_verified','diagnostics_excluded'))
folders=[p for p in ROOT.glob('study*') if p.is_dir()]
assert len(folders)==8
for folder in folders:
    assert read(folder/'complete.json')['order_verified']
    assert not read(folder/'order_check.json')['failures']
rows=read(ROOT/'analysis.json')
assert all(r['trace_preemptions']==0 and r['stalled_requests']==0 for r in rows if r['slots']==12288)
from collections import Counter
assert Counter(r['policy'] for r in rows)==dict(baseline=20,adaptive=16,adaptive_resume=16,fixed=12,fixed_resume=12)
for r in rows:
    rr=[q for q in read(ROOT/'requests.json') if q['trial']==r['trial']]
    assert r['metrics_delta']['vllm:prompt_tokens_total']==sum(q['input'] for q in rr)
    assert r['metrics_delta']['vllm:generation_tokens_total']==sum(q['output'] for q in rr)
    assert r['metrics_delta']['vllm:iteration_tokens_total_sum']==sum(q['input']+q['output'] for q in rr)
provenance=read(ROOT/'provenance.json')
for name,digest in provenance['source_hashes'].items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
# Driver process groups have exited; confirm every recorded server pid is absent.
alive=[]
for folder in folders:
    pid=int((folder/'server_pid.txt').read_text())
    status=pathlib.Path('/proc')/str(pid)/'cmdline'
    if status.exists() and status.read_bytes():alive.append(pid)
assert not alive,alive
final=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_checks_passed=True,retained_trials=76,retained_requests=912,accepted_server_configs=8,excluded_order_runs=[p.name for p in ROOT.glob('excluded_order_study*')],servers_stopped=True,policy_counts=dict(Counter(r['policy'] for r in rows)),native_source_and_helpers_unchanged=True,gpu_final=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,power.draw','--format=csv'],text=True))
(ROOT/'completion.json').write_text(json.dumps(final,indent=2))
# Preserve sibling paths to support the unchanged helper imports and prior reanalysis.
files=[]
for directory in (ROOT,ROOT.parent/'capacity_cliff_20261002',ROOT.parent/'admission_robustness_20261002'):
    files.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts and p.name not in ('bundle_manifest.json','finalization.log'))
files += [LAB/'README.md',LAB/'activate.sh',LAB/'requirements.lock.txt',LAB/'dash.py']
manifest={str(p.relative_to(LAB)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
(ROOT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(ROOT/'bundle_manifest.json')
with zipfile.ZipFile(ROOT/'results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(files):archive.write(p,str(p.relative_to(LAB)))
with zipfile.ZipFile(ROOT/'results.zip') as archive:
    assert archive.testzip() is None
    for name,meta in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==meta['sha256'],name
print(json.dumps(dict(completion=final,bundle_bytes=(ROOT/'results.zip').stat().st_size,bundle_files=len(files)),indent=2))

