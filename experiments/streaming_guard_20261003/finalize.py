import collections,datetime,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parent;LAB=ROOT.parent.parent
load=lambda n:json.loads((ROOT/(n+'.json')).read_text())
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=load('checks');rows=load('analysis');design=load('design')
assert (checks['completed_trials'],checks['completed_requests'],checks['completed_configs'])==(96,1152,9)
assert not checks['unfinished'] and all(v for v in checks.values() if isinstance(v,bool))
assert collections.Counter(r['policy'] for r in rows)==dict(baseline=22,adaptive_resume=12,slai_port=14,growth_only=12,guard128=18,guard256=14,guard_fast=2,guard_relaxed=2)
assert all(r['metrics_delta']['vllm:request_success_total']==12 for r in rows)
folders=[p for p in ROOT.glob('study*') if p.is_dir() and (p/'config.json').exists()]
assert len(folders)==9 and all((p/'complete.json').exists() for p in folders)
for p in folders:
    order=json.loads((p/'order_check.json').read_text());assert not order['failures']
    pid=int((p/'server_pid.txt').read_text());proc=pathlib.Path('/proc')/str(pid)/'cmdline'
    assert not proc.exists() or b'vllm' not in proc.read_bytes(),p
for name,digest in load('evaluation_code_hashes').items():assert hashfile(ROOT/name)==digest,name
for path,digest in load('provenance')['source_hashes'].items():assert hashfile(pathlib.Path(path))==digest,path
assert hashfile(LAB/'dash.py')=='dc17167abd9f7e63d86a62b8a567dff9c619fb58e28f094bbdbe3acd9dde9254'
for name in ['REPORT.md','RELATED_WORK.md','MEASUREMENT_NOTES.md','streaming_results.png','streaming_results.pdf','long_tail_cost.png','long_tail_cost.pdf']:assert (ROOT/name).exists(),name
excluded=[p.name for p in ROOT.glob('excluded_order_*') if p.is_dir()]
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in {'.git','__pycache__'} for part in p.parts) and p.suffix not in {'.zip','.pyc'} and p.name not in {'bundle_manifest.json','finalization.log'}]
helpers={'capacity_cliff_20261002':['bench.py','analyze.py','trace_scheduler.py'],'admission_robustness_20261002':['run.py','control_scheduler.py'],'resume_admission_20261003':['resume_scheduler.py','analyze.py']}
files += [LAB/'README.md',LAB/'dash.py',LAB/'activate.sh',LAB/'requirements.lock.txt']
files += [ROOT.parent/folder/name for folder,names in helpers.items() for name in names]
files=sorted(set(files));manifest=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),trials=96,requests=1152,configs=9,excluded_order_configs=excluded,diagnostic_trials=8,servers_stopped=True,dash_unchanged=True,files={str(p.relative_to(LAB)):dict(sha256=hashfile(p),bytes=p.stat().st_size) for p in files},scope='Current study, diagnostics and exclusions, frozen reference checkout without Git metadata, runtime lock and required prior helper code. Prior study raw data/model snapshots/runtime binaries are not included.')
(ROOT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(ROOT/'bundle_manifest.json')
archive=ROOT/'results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,str(p.relative_to(LAB)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,metadata in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==metadata['sha256'],name
print(json.dumps(dict(verified=True,trials=96,requests=1152,configs=9,excluded_order_configs=excluded,bundle_files=len(files),bundle_bytes=archive.stat().st_size,bundle_sha256=hashfile(archive)),indent=2))
