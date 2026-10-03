import csv, hashlib, json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parent
rows=json.loads((ROOT/'analysis.json').read_text())
pairs=json.loads((ROOT/'paired.json').read_text())
thresholds={}
for threshold in (.5,1,2,5):
    counts={}
    for row in rows:
        requests=[json.loads(line) for line in (ROOT/row['run']/row['folder']/'requests.jsonl').read_text().splitlines()]
        counts[row['trial']]=sum(r['max_silence_s']>threshold for r in requests)
    result={}
    for policy in ('fixed','adaptive'):
        chosen=[p for p in pairs if p['policy']==policy and p['baseline_evictions']>0]
        btotal=ptotal=0
        improved=worsened=0
        for p in chosen:
            key=tuple(p[k] for k in ('model','slots','cap','seed','mix','arrival'))
            matched={r['policy']:r for r in rows if tuple(r[k] for k in ('model','slots','cap','seed','mix','arrival'))==key}
            b=counts[matched['baseline']['trial']]
            v=counts[matched[policy]['trial']]
            btotal+=b; ptotal+=v; improved+=v<b; worsened+=v>b
        result[policy]=dict(pairs=len(chosen),baseline_stalled=btotal,policy_stalled=ptotal,improved_pairs=improved,worsened_pairs=worsened)
    thresholds[str(threshold)]=result
slices={}
for dim in ('model','mix','arrival','seed'):
    for value in sorted({r[dim] for r in rows},key=str):
        group=[r for r in rows if r[dim]==value and r['policy']=='baseline']
        slices[f'{dim}={value}']=dict(trials=len(group),trials_with_evictions=sum(r['trace_preemptions']>0 for r in group),trials_with_stalls=sum(r['stalled_requests']>0 for r in group),stalled_requests=sum(r['stalled_requests'] for r in group),requests=sum(r['n'] for r in group),max_silence=max(r['max_silence'] for r in group))
evicted=sum(r['stalled_requests'] for r in rows if r['policy']=='baseline' and r['trace_preemptions']>0)
not_evicted=sum(r['stalled_requests'] for r in rows if r['policy']=='baseline' and r['trace_preemptions']==0)
request_rows=list(csv.DictReader((ROOT/'requests.csv').open()))
base_stalled=[r for r in request_rows if r['policy']=='baseline' and float(r['max_silence'])>1]
base_own_evicted=sum(int(r['trace_evictions'])>0 for r in base_stalled)
gpu=[]
for row in rows:
    for filename in ('gpu_before.csv','gpu_after.csv'):
        for entry in csv.DictReader((ROOT/row['run']/row['folder']/filename).open(),skipinitialspace=True):
            gpu.append({k.strip():v.strip() for k,v in entry.items()})
ranges={}
for label,key in [('temperature_c','temperature.gpu'),('power_w','power.draw [W]'),('sm_clock_mhz','clocks.current.sm [MHz]')]:
    vals=[]
    for g in gpu:
        if key in g:
            try: vals.append(float(g[key].split()[0]))
            except ValueError: pass
    if vals: ranges[label]=dict(min=min(vals),max=max(vals),median=statistics.median(vals))
locked=json.loads((ROOT/'evaluation_code_hashes.json').read_text())
actual={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('run.py','control_scheduler.py')}
print('Locked hashes:',locked)
print('Actual hashes:',actual)
# Preserve both formats without guessing the recorded schema.
(ROOT/'extra_analysis.json').write_text(json.dumps(dict(thresholds=thresholds,baseline_slices=slices,baseline_stalled_in_eviction_trials=evicted,baseline_stalled_in_non_eviction_trials=not_evicted,baseline_stalled_requests_ever_evicted=base_own_evicted,baseline_stalled_requests_total=len(base_stalled),gpu_snapshot_ranges=ranges,max_arrival_lateness_s=max(r['max_arrival_lateness_s'] for r in rows),locked_hashes=locked,actual_hashes=actual),indent=2))
print(json.dumps(dict(completed=len(rows),thresholds=thresholds,gpu_snapshot_ranges=ranges),indent=2))
