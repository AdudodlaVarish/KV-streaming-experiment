import argparse, csv, json, math, pathlib, statistics, sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'capacity_cliff_20261002'))
from analyze import trace_summary
from bench import percentile

def ratio(a,b):
    return a/b if b else None

def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None

def write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k],(dict,list))))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def summarize_pairs(pairs):
    if not pairs:
        return {}
    output={}
    for policy in ('fixed','adaptive'):
        chosen=[p for p in pairs if p['policy']==policy]
        if not chosen:
            continue
        output[policy]=dict(pairs=len(chosen),throughput_ratio_geomean=geometric([p['throughput_ratio'] for p in chosen]),
            throughput_ratio_median=statistics.median(p['throughput_ratio'] for p in chosen),
            throughput_ratio_min=min(p['throughput_ratio'] for p in chosen),
            throughput_ratio_max=max(p['throughput_ratio'] for p in chosen),
            ttft_p50_ratio_geomean=geometric([p['ttft_ratio'] for p in chosen]),
            e2e_p95_ratio_geomean=geometric([p['e2e_ratio'] for p in chosen]),
            baseline_stalled_requests=sum(p['baseline_stalled'] for p in chosen),
            policy_stalled_requests=sum(p['policy_stalled'] for p in chosen),
            baseline_evictions=sum(p['baseline_evictions'] for p in chosen),
            policy_evictions=sum(p['policy_evictions'] for p in chosen),
            baseline_replay=sum(p['baseline_replay'] for p in chosen),
            policy_replay=sum(p['policy_replay'] for p in chosen),
            baseline_excess_s=sum(p['baseline_excess_s'] for p in chosen),
            policy_excess_s=sum(p['policy_excess_s'] for p in chosen),
            fewer_stalls_retains_90pct=sum(p['policy_stalled']<p['baseline_stalled'] and p['throughput_ratio']>=.9 for p in chosen),
            fewer_stalls_retains_95pct=sum(p['policy_stalled']<p['baseline_stalled'] and p['throughput_ratio']>=.95 for p in chosen),
            count_stall_improved=sum(p['policy_stalled']<p['baseline_stalled'] for p in chosen),
            count_stall_worsened=sum(p['policy_stalled']>p['baseline_stalled'] for p in chosen))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',default='study')
    args=parser.parse_args()
    assert geometric([.5,2])==1
    rows,request_rows=[],[]
    missing=[]
    for folder in sorted(ROOT.glob(args.prefix+'*')):
        if not folder.is_dir():
            continue
        config=json.loads((folder/'config.json').read_text())
        path=folder/'scheduler_trace.jsonl'
        if not path.exists():
            continue
        events=[]
        for line in path.open():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break # Only a possibly incomplete tail from an active writer.
        cells=sorted(folder.glob('[0-9][0-9]_*'))
        for cell in cells:
            if not (cell/'summary.json').exists():
                missing.append(str(cell.relative_to(ROOT)))
                continue
            row=json.loads((cell/'summary.json').read_text())
            requests=[json.loads(line) for line in (cell/'requests.jsonl').read_text().splitlines()]
            assert len(requests)==row['n']==12
            assert len({r['index'] for r in requests})==12
            planned=json.loads((folder/f'{row["mix"]}_{row["arrival"]}_workload.json').read_text())
            expected={r['index']:r for r in planned}
            arrivals=[e for e in events if e['kind']=='arrival' and e['trial']==row['trial']]
            actual_order=[]
            for e in arrivals:
                matches=[r['index'] for r in requests if e['request'].startswith(r['response_id']+'-')]
                assert len(matches)==1,(row['trial'],e,matches)
                actual_order+=matches
            assert actual_order==list(range(12)),(row['trial'],actual_order)
            row['arrival_order_verified']=True
            trial_events=[e for e in events if row['start_perf_s']<=e['t']<=row['end_perf_s']]
            evictions=[e for e in trial_events if e['kind']=='preempt']
            steps=[e for e in trial_events if e['kind']=='step']
            resume_waits=[]
            for e in evictions:
                next_steps=[step for step in steps if step['t']>=e['t'] and any(work[0]==e['request'] and work[1]==0 for work in step['scheduled'])]
                assert next_steps,(row['trial'],e)
                resume_waits.append(next_steps[0]['t']-e['t'])
            row['preempt_to_resume_mean_s']=statistics.mean(resume_waits) if resume_waits else 0
            row['preempt_to_resume_max_s']=max(resume_waits or [0])
            row['evictions_after_client_first_token']=0
            row['eviction_associated_stalled_requests']=0
            for r in requests:
                own_evictions=[e for e in evictions if e['request'].startswith(r['response_id']+'-')]
                own_replay=sum(work[3] for step in steps for work in step['scheduled'] if work[0].startswith(r['response_id']+'-'))
                row['evictions_after_client_first_token']+=sum(e['t']>=r['start']+r['ttft_s'] for e in own_evictions)
                gaps=[(r['start']+a[0],r['start']+b[0]) for a,b in zip(r['token_events'],r['token_events'][1:]) if b[0]-a[0]>1]
                associated=any(a-.05<=e['t']<=b for a,b in gaps for e in own_evictions)
                row['eviction_associated_stalled_requests']+=associated
                r['trace_evictions']=len(own_evictions)
                r['trace_replay']=own_replay
                assert r['input_tokens']==len(expected[r['index']]['prompt'])
                assert r['output_tokens']==expected[r['index']]['output']
                assert r['usage']['prompt_tokens']==r['input_tokens']
                assert r['usage']['completion_tokens']==r['output_tokens']==r['token_events'][-1][1]
                assert r['elapsed_s']>=r['ttft_s']>0
                assert all(a[0]<=b[0] and a[1]<b[1] for a,b in zip(r['token_events'],r['token_events'][1:]))
                request_rows.append(dict(model=row['model'],slots=row['slots'],cap=row['cap'],seed=row['seed'],mix=row['mix'],arrival=row['arrival'],policy=row['policy'],
                    trial=row['trial'],index=r['index'],input=r['input_tokens'],output=r['output_tokens'],ttft=r['ttft_s'],e2e=r['elapsed_s'],max_silence=r['max_silence_s'],
                    silence_excess_s=r['silence_excess_1s_s'],trace_evictions=r['trace_evictions'],trace_replay=r['trace_replay'],token_event_count=len(r['token_events']),lateness_s=r['arrival_lateness_s']))
            trace=trace_summary(events,row['start_perf_s'],row['end_perf_s'])
            assert trace['trace_preemptions']==row['metrics_delta']['vllm:num_preemptions_total'],(row['trial'],trace,row['metrics_delta'])
            assert trace['rescheduled_token_positions']==trace['evicted_computed_positions'],(row['trial'],trace)
            admissions=[e for e in events if e['kind']=='admission' and e['trial']==row['trial']]
            assert admissions and all(e['policy']==row['policy'] for e in admissions)
            assert all(0<=e['reserve']<=.45 for e in admissions)
            if row['policy']!='adaptive':
                expected=.3 if row['policy']=='fixed' else 0.
                assert all(e['reserve']==expected for e in admissions)
            row.update(trace)
            row['peak_headroom']=max(e['reserve'] for e in admissions)
            row['mean_headroom']=statistics.mean(e['reserve'] for e in admissions)
            row['stalled_fraction']=row['stalled_requests']/row['n']
            row['request_silence_p95']=percentile([r['max_silence_s'] for r in requests],95)
            rows.append(row)
    keyed={}
    for row in rows:
        key=tuple(row[k] for k in ('model','slots','cap','seed','mix','arrival'))
        assert row['policy'] not in keyed.setdefault(key,{}),key
        keyed[key][row['policy']]=row
    pairs=[]
    for key,policies in keyed.items():
        if 'baseline' not in policies:
            continue
        b=policies['baseline']
        for policy in ('fixed','adaptive'):
            if policy not in policies:
                continue
            p=policies[policy]
            assert (ROOT/b['run']/f'{b["mix"]}_{b["arrival"]}_workload.json').read_text()==(ROOT/p['run']/f'{p["mix"]}_{p["arrival"]}_workload.json').read_text()
            pairs.append(dict(model=b['model'],slots=b['slots'],cap=b['cap'],seed=b['seed'],mix=b['mix'],arrival=b['arrival'],policy=policy,
                throughput_ratio=p['throughput']/b['throughput'],ttft_ratio=p['ttft_p50']/b['ttft_p50'],e2e_ratio=p['e2e_p95']/b['e2e_p95'],
                baseline_throughput=b['throughput'],policy_throughput=p['throughput'],baseline_gap=b['max_silence'],policy_gap=p['max_silence'],
                baseline_stalled=b['stalled_requests'],policy_stalled=p['stalled_requests'],
                baseline_evictions=b['trace_preemptions'],policy_evictions=p['trace_preemptions'],
                baseline_replay=b['rescheduled_token_positions'],policy_replay=p['rescheduled_token_positions'],
                baseline_excess_s=b['silence_excess_1s_s'],policy_excess_s=p['silence_excess_1s_s']))
    comparisons=[]
    for key,policies in keyed.items():
        model,slots,cap,seed,mix,arrival=key
        if model!='llama1b' or cap==3 or 'baseline' not in policies:
            continue
        low=keyed.get((model,slots,3,seed,mix,arrival),{}).get('baseline')
        if not low:
            continue
        high=policies['baseline']
        gain=high['throughput']/low['throughput']
        row=dict(slots=slots,cap=cap,seed=seed,mix=mix,arrival=arrival,throughput_ratio_high_to_cap3=gain,
            low_stalled=low['stalled_requests'],high_stalled=high['stalled_requests'],low_gap=low['max_silence'],high_gap=high['max_silence'],
            low_e2e=low['e2e_p95'],high_e2e=high['e2e_p95'],
            tradeoff=gain>1.03 and high['stalled_requests']>low['stalled_requests'])
        row['low_throughput']=low['throughput']
        row['high_throughput']=high['throughput']
        if 'adaptive' in policies:
            row['adaptive_stalled']=policies['adaptive']['stalled_requests']
            row['adaptive_gap']=policies['adaptive']['max_silence']
            row['adaptive_throughput']=policies['adaptive']['throughput']
        if 'adaptive' in policies and gain>1.05:
            row['adaptive_batching_gain_retained']=(policies['adaptive']['throughput']-low['throughput'])/(high['throughput']-low['throughput'])
        comparisons.append(row)
    all_summary=summarize_pairs(pairs)
    pressure=summarize_pairs([p for p in pairs if p['baseline_evictions']>0])
    stalled=summarize_pairs([p for p in pairs if p['baseline_stalled']>0])
    stratified={}
    for dimension in ('model','slots','cap','mix','arrival','seed'):
        for value in sorted({p[dimension] for p in pairs},key=str):
            group=[p for p in pairs if p[dimension]==value]
            stratified[f'{dimension}={value}']=dict(all=summarize_pairs(group),pressure=summarize_pairs([p for p in group if p['baseline_evictions']>0]))
    summary=dict(completed_trials=len(rows),completed_requests=len(request_rows),all=all_summary,pressure=pressure,stalled_baseline=stalled,
        configuration_tradeoffs=dict(pairs=len(comparisons),tradeoff_pairs=sum(r['tradeoff'] for r in comparisons),
            adaptive_batching_gain_retained_median=statistics.median([r['adaptive_batching_gain_retained'] for r in comparisons if 'adaptive_batching_gain_retained' in r]) if any('adaptive_batching_gain_retained' in r for r in comparisons) else None),
        strata=stratified)
    (ROOT/'analysis.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'paired.json').write_text(json.dumps(pairs,indent=2))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'batching.json').write_text(json.dumps(comparisons,indent=2))
    write_csv(ROOT/'trials.csv',rows)
    write_csv(ROOT/'requests.csv',request_rows)
    write_csv(ROOT/'paired.csv',pairs)
    write_csv(ROOT/'batching.csv',comparisons)
    checks=dict(completed_trials=len(rows),completed_requests=len(request_rows),paired_comparisons=len(pairs),token_counts_verified=True,
        preemption_trace_matches_metrics=True,replayed_positions_match_evicted=True,policy_state_verified=True,diagnostics_excluded=not args.prefix.startswith(('diag','ordereddiag','eval')),early_simultaneous_batches_excluded=True,engine_arrival_order_verified=True,unfinished_cells=missing)
    (ROOT/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({k:summary[k] for k in ('completed_trials','completed_requests','all','pressure','configuration_tradeoffs')},indent=2))





