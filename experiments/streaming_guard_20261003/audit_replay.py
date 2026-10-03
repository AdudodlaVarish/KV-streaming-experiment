import collections,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
f=ROOT/'diag00_a1_s23_llama1b_kv3072_cap12';row=json.loads((f/'07_long_burst_guard_fast/summary.json').read_text());events=[json.loads(line) for line in (f/'scheduler_trace.jsonl').open()];events=[e for e in events if row['start_perf_s']<=e['t']<=row['end_perf_s']]
lost=collections.Counter();replay=collections.Counter()
for e in events:
    if e['kind']=='preempt':lost[e['request']]+=e['computed']
    if e['kind']=='step':
        for r,b,n,x,o in e['scheduled']:replay[r]+=x
print('total',sum(lost.values()),sum(replay.values()))
for rid in sorted(lost):
    if lost[rid]!=replay[rid]:
        own=[e for e in events if e.get('request')==rid]
        print(rid,'lost',lost[rid],'replay',replay[rid],'deficit',lost[rid]-replay[rid]);print(json.dumps(own[-9:],indent=2))
