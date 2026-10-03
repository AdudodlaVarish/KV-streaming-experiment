"""Verify saved requests/counters and summarize logging-only scheduler traces."""
import csv, json, pathlib, statistics
ROOT = pathlib.Path(__file__).resolve().parent

def trace_summary(events, start, end):
    selected = [e for e in events if start <= e['t'] <= end]
    steps = [e for e in selected if e['kind']=='step']
    evictions = [e for e in selected if e['kind']=='preempt']
    return dict(trace_preemptions=len(evictions),
        scheduled_tokens=sum(row[2] for e in steps for row in e['scheduled']),
        rescheduled_token_positions=sum(row[3] for e in steps for row in e['scheduled']),
        evicted_computed_positions=sum(e['computed'] for e in evictions),
        max_running=max([e['running'] for e in steps] or [0]),
        max_kv_usage=max([e['kv_usage'] for e in steps] or [0]),
        evicted_generated_tokens=[e['output'] for e in evictions])

if __name__ == '__main__':
    assert trace_summary([dict(kind='step',t=2,scheduled=[['r',0,8,5,0]],running=1,kv_usage=.5)],1,3)['rescheduled_token_positions']==5
    rows = []
    for folder in sorted(ROOT.glob('*_kv*_cap*')):
        trace_path=folder/'scheduler_trace.jsonl'
        events=[json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
        for rep in sorted(folder.glob('rep*')):
            if not (rep/'summary.json').exists():
                continue
            summary=json.loads((rep/'summary.json').read_text())
            requests=json.loads((rep/'requests.json').read_text())
            assert len(requests)==summary['n']
            assert all(r['usage']['prompt_tokens']==summary['input_tokens'] and r['usage']['completion_tokens']==summary['output_tokens'] for r in requests)
            assert all(r['elapsed_s']>=r['ttft_s']>0 for r in requests)
            assert all(all(a<=b for a,b in zip(r['stream_event_times_s'],r['stream_event_times_s'][1:])) for r in requests)
            assert summary['metrics_delta']['vllm:request_success_total']==summary['n']
            assert summary['metrics_delta']['vllm:prompt_tokens_total']==summary['n']*summary['input_tokens']
            assert summary['metrics_delta']['vllm:generation_tokens_total']==summary['n']*summary['output_tokens']
            summary.update(config=folder.name,rep=rep.name)
            summary['stalled_requests_1s']=sum(r['max_stream_gap_s']>1 for r in requests)
            start=min(r['start'] for r in requests)
            end=max(r['start']+r['elapsed_s'] for r in requests)
            if events:
                trace=trace_summary(events,start,end)
                assert trace['trace_preemptions']==summary['metrics_delta']['vllm:num_preemptions_total'], (folder,rep,trace,summary)
                assert trace['rescheduled_token_positions']==trace['evicted_computed_positions'], (folder,rep,trace)
                summary.update(trace)
                evictions=[e for e in events if e['kind']=='preempt' and start<=e['t']<=end]
                pauses=[]
                for r in requests:
                    times=r['stream_event_times_s']
                    for i,(a,b) in enumerate(zip(times,times[1:])):
                        if b-a>1:
                            gap_start=r['start']+a
                            nearest=min(evictions,key=lambda e:abs(e['t']-gap_start),default=None)
                            pauses.append(dict(gap_s=b-a,preemption_after_last_chunk_ms=((nearest['t']-gap_start)*1000 if nearest else None),
                                nonempty_chunks_before_pause=i+1,engine_output_before_preemption=nearest['output'] if nearest else None))
                summary['pause_evidence']=pauses
            rows.append(summary)
    (ROOT/'analysis.json').write_text(json.dumps(rows,indent=2))
    keys=['config','rep','output_tok_s','ttft_p50_s','e2e_p95_s','max_stream_gap_s','stalled_requests_1s','trace_preemptions','rescheduled_token_positions','max_running','max_kv_usage']
    with (ROOT/'analysis.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    groups={}
    for r in rows:
        groups.setdefault(r['config'],[]).append(r)
    aggregated={name: {key:dict(median=statistics.median([r[key] for r in group]),min=min(r[key] for r in group),max=max(r[key] for r in group)) for key in keys[2:] if all(key in r for r in group)} for name,group in groups.items()}
    (ROOT/'aggregated.json').write_text(json.dumps(aggregated,indent=2))
    (ROOT/'checks.json').write_text(json.dumps(dict(completed_bursts=len(rows),completed_requests=sum(r['n'] for r in rows),traced_bursts=sum('trace_preemptions' in r for r in rows),token_counts_verified=True,success_counters_verified=True,trace_preemptions_match_metrics=True,rescheduled_positions_match_evicted_positions=True),indent=2))
    print(json.dumps(aggregated,indent=2))



