import json, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
failures=[]; checked=0
for folder in sorted(ROOT.glob('study*')):
    if not folder.is_dir(): continue
    arrivals=[]
    for line in (folder/'scheduler_trace.jsonl').open():
        try: e=json.loads(line)
        except json.JSONDecodeError: break
        if e['kind']=='arrival': arrivals.append(e)
    for cell in sorted(folder.glob('[0-9][0-9]_*')):
        if not (cell/'summary.json').exists(): continue
        row=json.loads((cell/'summary.json').read_text())
        requests=[json.loads(line) for line in (cell/'requests.jsonl').read_text().splitlines()]
        actual=[]
        times=[]
        for e in arrivals:
            if e['trial']!=row['trial']: continue
            matches=[r['index'] for r in requests if e['request'].startswith(r['response_id']+'-')]
            assert len(matches)==1
            actual+=matches; times.append([matches[0],e['t']-row['start_perf_s']])
        checked+=1
        if actual!=list(range(12)):
            failures.append(dict(run=folder.name,trial=row['trial'],actual_order=actual,engine_arrivals=times,client_starts=[[r['index'],r['start']-row['start_perf_s'],r['arrival_lateness_s']] for r in sorted(requests,key=lambda r:r['index'])]))
(ROOT/'order_audit.json').write_text(json.dumps(dict(checked_trials=checked,failures=failures),indent=2))
print(json.dumps(dict(checked_trials=checked,failures=failures),indent=2))
