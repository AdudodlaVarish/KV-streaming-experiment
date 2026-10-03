import hashlib, json, pathlib, zipfile
ROOT=pathlib.Path(__file__).resolve().parent
checks=json.loads((ROOT/'checks.json').read_text())
assert checks['completed_trials']==168 and checks['completed_requests']==2016 and not checks['unfinished_cells']
files={}
for name in ('run.py','control_scheduler.py','analyze_study.py','extra_analysis.py','plot_study.py','make_report.py','bundle_results.py','verify_orders.py','repeat_order_failure.py','order_audit.json','order_replacement.json','replacement.log','REPORT.md','DESIGN.md','design.json','evaluation_code_hashes.json','provenance.json','completion.json','gpu_final.csv','environment.lock.txt','base_environment_provenance.json','analysis.json','paired.json','summary.json','batching.json','extra_analysis.json','checks.json','trials.csv','requests.csv','paired.csv','batching.csv','robustness.png','robustness.svg','pressure_timeline.png','pressure_timeline.svg','study.log'):
    path=ROOT/name
    assert path.is_file(),path
    files[str(path.relative_to(ROOT.parent))]=path
folders=sorted(p for p in ROOT.glob('study*') if p.is_dir())
assert len(folders)==16 and all((p/'complete.json').is_file() for p in folders)
for folder in folders+sorted(ROOT.glob('excluded_order_*')):
    for path in folder.rglob('*'):
        if path.is_file(): files[str(path.relative_to(ROOT.parent))]=path
for name in ('bench.py','analyze.py','trace_scheduler.py'):
    path=ROOT.parent/'capacity_cliff_20261002'/name
    files[str(path.relative_to(ROOT.parent))]=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
manifest_path=ROOT/'bundle_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2))
files[str(manifest_path.relative_to(ROOT.parent))]=manifest_path
with zipfile.ZipFile(ROOT/'results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in sorted(files.items()): archive.write(path,name)
with zipfile.ZipFile(ROOT/'results.zip') as archive:
    assert archive.testzip() is None
print(json.dumps(dict(files=len(files),uncompressed_bytes=sum(p.stat().st_size for p in files.values()),zip_bytes=(ROOT/'results.zip').stat().st_size),indent=2))


