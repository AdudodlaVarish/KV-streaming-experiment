import argparse,csv,hashlib,json,math,pathlib,statistics,sys
ROOT=pathlib.Path(__file__).resolve().parent
PRIOR=ROOT.parent/'capacity_cliff_20261002'
sys.path.insert(0,str(PRIOR))
from analyze import trace_summary
from bench import percentile,metrics,total

def geo(values): return math.exp(statistics.mean(math.log(v) for v in values)) if values else None

def csv_write(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k,v in row.items() if not isinstance(v,(list,dict))))
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=keys,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)

def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a: continue
        if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(b,out[-1][1]))
        else: out.append((a,b))
    return out

def overlap(intervals,windows):
    return sum(max(0,min(b,d)-max(a,c)) for a,b in union(intervals) for c,d in union(windows))

def episode(e,steps,allocations,preempts,request,row):
    matched=[s for s in steps if s['t']>=e['t'] and any(w[0]==e['request'] and w[1]==0 and w[2]>0 for w in s['scheduled'])]
    assert matched,(row['trial'],e)
    replay=matched[0]['t']
    deliveries=[request['start']+t for t,_ in request['token_events']]
    later=[t for t in deliveries if t>=replay]
    assert later,(row['trial'],e,replay,deliveries[-1])
    delivery=later[0]
    attempts=[a for a in allocations if e['t']<=a['t']<=replay and a['request']==e['request']]
    duration={'no_attempt':0.,'capacity':0.,'watermark':0.,'other':0.,'admit':0.}
    previous=e['t']; state='no_attempt'
    for a in attempts:
        duration[state]+=max(0,a['t']-previous); previous=a['t']; state=a['outcome']
    duration[state]+=max(0,replay-previous)
    assert abs(sum(duration.values())-(replay-e['t']))<1e-6
    interrupted=sum(replay<=p['t']<delivery and p['request']==e['request'] for p in preempts)
    result=dict(trial=row['trial'],model=row['model'],slots=row['slots'],cap=row['cap'],seed=row['seed'],mix=row['mix'],arrival=row['arrival'],policy=row['policy'],index=request['index'],engine_request=e['request'],preempt_perf_s=e['t'],replay_start_perf_s=replay,next_delivery_perf_s=delivery,wait_s=replay-e['t'],replay_to_delivery_s=delivery-replay,preempt_to_delivery_s=delivery-e['t'],computed_lost=e['computed'],generated_at_eviction=e['output'],client_started_at_eviction=deliveries[0]<=e['t'],client_started_before_replay=deliveries[0]<=replay,intervening_preemptions=interrupted,allocation_attempts=len(attempts),waived_attempts=sum(a['waived'] for a in attempts),binding_waiver_admissions=sum(a['waived'] and a['outcome']=='admit' and a['full_blocks']+a['would_apply_watermark']>a['free_blocks'] for a in attempts))
    result['admission_tracing_available']=bool(allocations)
    result.update({k+'_decision_s':v if allocations else None for k,v in duration.items()})
    return result

def paired_summary(pairs):
    if not pairs: return {}
    return dict(pairs=len(pairs),throughput_ratio_geo=geo([p['throughput_ratio'] for p in pairs]),throughput_ratio_min=min(p['throughput_ratio'] for p in pairs),throughput_ratio_max=max(p['throughput_ratio'] for p in pairs),ttft_ratio_geo=geo([p['ttft_ratio'] for p in pairs]),completion_ratio_geo=geo([p['completion_ratio'] for p in pairs]),reference_stalled=sum(p['reference_stalled'] for p in pairs),policy_stalled=sum(p['policy_stalled'] for p in pairs),reference_evictions=sum(p['reference_evictions'] for p in pairs),policy_evictions=sum(p['policy_evictions'] for p in pairs),reference_replay=sum(p['reference_replay'] for p in pairs),policy_replay=sum(p['policy_replay'] for p in pairs),reference_excess_s=sum(p['reference_excess_s'] for p in pairs),policy_excess_s=sum(p['policy_excess_s'] for p in pairs),reference_worst_gap=max(p['reference_gap'] for p in pairs),policy_worst_gap=max(p['policy_gap'] for p in pairs),stalled_improved=sum(p['policy_stalled']<p['reference_stalled'] for p in pairs),stalled_worsened=sum(p['policy_stalled']>p['reference_stalled'] for p in pairs),gap_improved=sum(p['policy_gap']<p['reference_gap']-.05 for p in pairs),gap_worsened=sum(p['policy_gap']>p['reference_gap']+.05 for p in pairs),fewer_evictions_but_worse_gap=sum(p['policy_evictions']<p['reference_evictions'] and p['policy_gap']>p['reference_gap']+.05 for p in pairs),same_evictions_gap_improved=sum(p['policy_evictions']==p['reference_evictions'] and p['policy_gap']<p['reference_gap']-.05 for p in pairs))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prefix',default='study'); parser.add_argument('--previous',action='store_true'); args=parser.parse_args()
    assert union([(0,2),(1,3),(4,5)])==[(0,3),(4,5)]
    assert overlap([(0,2),(1,3)],[(1,2)])==1
    source=ROOT.parent/'admission_robustness_20261002' if args.previous else ROOT
    output_prefix='previous_' if args.previous else ''
    rows=[]; request_rows=[]; episodes=[]; unfinished=[]
    for folder in sorted(source.glob(args.prefix+'*')):
        if not folder.is_dir() or not (folder/'config.json').exists(): continue
        events=[]
        trace=folder/'scheduler_trace.jsonl'
        if not trace.exists(): continue
        for line in trace.open():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: break
        for cell in sorted(folder.glob('[0-9][0-9]_*')):
            if not (cell/'summary.json').exists(): unfinished.append(str(cell.relative_to(source))); continue
            row=json.loads((cell/'summary.json').read_text())
            requests=[json.loads(line) for line in (cell/'requests.jsonl').read_text().splitlines()]
            assert len(requests)==12 and len({r['index'] for r in requests})==12
            expected={r['index']:r for r in json.loads((folder/f'{row["mix"]}_{row["arrival"]}_workload.json').read_text())}
            selected=[e for e in events if row['start_perf_s']<=e['t']<=row['end_perf_s']]
            arrivals=[e for e in selected if e['kind']=='arrival' and e['trial']==row['trial']]
            order=[]
            for e in arrivals:
                match=[r['index'] for r in requests if e['request'].startswith(r['response_id']+'-')]
                assert len(match)==1; order+=match
            assert order==list(range(12)),(row['trial'],order)
            preempts=[e for e in selected if e['kind']=='preempt']; steps=[e for e in selected if e['kind']=='step']
            allocations=[e for e in selected if e['kind']=='allocation']
            if not args.previous:
                assert allocations
                for a in allocations:
                    assert a['trial']==row['trial'] and a['policy']==row['policy']
                    effective=0 if a['status']=='PREEMPTED' and row['policy'].endswith('_resume') else a['would_apply_watermark']
                    assert a['effective_watermark']==effective
                    assert a['waived']==(a['effective_watermark']!=a['would_apply_watermark'])
                    assert a['outcome']!='admit' or a['full_blocks']+effective<=a['free_blocks']
                    assert a['outcome']!='watermark' or a['full_blocks']<=a['free_blocks']<a['full_blocks']+effective
                    assert a['outcome']!='capacity' or a['full_blocks']>a['free_blocks']
            ads=[e for e in selected if e['kind']=='admission' and e['trial']==row['trial']]
            assert ads and all(e['policy']==row['policy'] and 0<=e['reserve']<=.45 for e in ads)
            if row['policy']=='baseline': assert all(e['reserve']==0 for e in ads)
            if row['policy'].startswith('fixed'): assert all(e['reserve']==.30 for e in ads)
            tr=trace_summary(events,row['start_perf_s'],row['end_perf_s'])
            assert tr['trace_preemptions']==row['metrics_delta']['vllm:num_preemptions_total']
            assert tr['rescheduled_token_positions']==tr['evicted_computed_positions']
            row.update(tr)
            trial_episodes=[]; wait_covered=long_gap_s=0.; evicted_stalled=0
            for r in requests:
                assert r['input_tokens']==len(expected[r['index']]['prompt'])==r['usage']['prompt_tokens']
                assert r['output_tokens']==expected[r['index']]['output']==r['usage']['completion_tokens']==r['token_events'][-1][1]
                assert all(a[0]<=b[0] and a[1]<b[1] for a,b in zip(r['token_events'],r['token_events'][1:]))
                own=[e for e in preempts if e['request'].startswith(r['response_id']+'-')]
                replayed=sum(w[3] for s in steps for w in s['scheduled'] if w[0].startswith(r['response_id']+'-'))
                own_episodes=[episode(e,steps,allocations,preempts,r,row) for e in own]
                gaps=[(r['start']+a[0],r['start']+b[0]) for a,b in zip(r['token_events'],r['token_events'][1:]) if b[0]-a[0]>1]
                waits=[(e['preempt_perf_s'],e['replay_start_perf_s']) for e in own_episodes]
                covered=overlap(waits,gaps); gap_s=sum(b-a for a,b in gaps)
                assert covered<=gap_s+1e-6
                wait_covered+=covered; long_gap_s+=gap_s
                evicted_stalled+=bool(own and r['max_silence_s']>1)
                trial_episodes+=own_episodes
                rr=dict(trial=row['trial'],model=row['model'],slots=row['slots'],cap=row['cap'],seed=row['seed'],mix=row['mix'],arrival=row['arrival'],policy=row['policy'],index=r['index'],input=r['input_tokens'],output=r['output_tokens'],ttft=r['ttft_s'],completion=r['elapsed_s'],max_gap=r['max_silence_s'],evictions=len(own),replay=replayed,wait_s=sum(e['wait_s'] for e in own_episodes),max_wait_s=max([e['wait_s'] for e in own_episodes] or [0]),wait_covered_gt1_gaps_s=covered,total_gt1_gaps_s=gap_s,lateness_s=r['arrival_lateness_s'])
                rr.update({f'gap_gt_{threshold}':r['max_silence_s']>threshold for threshold in (.5,1,2,5)})
                request_rows.append(rr)
            episodes+=trial_episodes
            row.update(wait_sum_s=sum(e['wait_s'] for e in trial_episodes),wait_max_s=max([e['wait_s'] for e in trial_episodes] or [0]),watermark_decision_s=sum(e['watermark_decision_s'] or 0 for e in trial_episodes) if not args.previous else None,capacity_decision_s=sum(e['capacity_decision_s'] or 0 for e in trial_episodes) if not args.previous else None,no_attempt_decision_s=sum(e['no_attempt_decision_s'] or 0 for e in trial_episodes) if not args.previous else None,binding_waiver_admissions=sum(e['binding_waiver_admissions'] for e in trial_episodes),repeatedly_evicted_requests=sum(r['evictions']>1 for r in request_rows if r['trial']==row['trial']),evicted_stalled_requests=evicted_stalled,wait_covered_gt1_gaps_s=wait_covered,total_gt1_gaps_s=long_gap_s,allocation_attempts=len(allocations),mean_headroom=statistics.mean(e['reserve'] for e in ads),peak_headroom=max(e['reserve'] for e in ads))
            before=metrics((cell/'metrics_before.txt').read_text()); after=metrics((cell/'metrics_after.txt').read_text())
            row['prefill_kv_computed_metric_delta']=total(after,'vllm:request_prefill_kv_computed_tokens_sum')-total(before,'vllm:request_prefill_kv_computed_tokens_sum')
            assert row['prefill_kv_computed_metric_delta']==sum(r['input_tokens'] for r in requests),(row['trial'],row['prefill_kv_computed_metric_delta'])
            rows.append(row)
    # Aggregates below consume only verified complete trials.
    keyed={}
    dimensions=('model','slots','cap','seed','mix','arrival')
    for row in rows:
        key=tuple(row[k] for k in dimensions)
        assert row['policy'] not in keyed.setdefault(key,{})
        keyed[key][row['policy']]=row
    pairs=[]
    for key,policies in keyed.items():
        references=[('baseline',p) for p in policies if p!='baseline']
        references += [(p.removesuffix('_resume'),p) for p in policies if p.endswith('_resume') and p.removesuffix('_resume') in policies]
        for reference,policy in references:
            if reference not in policies: continue
            b,p=policies[reference],policies[policy]
            pairs.append(dict(zip(dimensions,key))|dict(reference=reference,policy=policy,throughput_ratio=p['throughput']/b['throughput'],ttft_ratio=p['ttft_p50']/b['ttft_p50'],completion_ratio=p['e2e_p95']/b['e2e_p95'],reference_throughput=b['throughput'],policy_throughput=p['throughput'],reference_stalled=b['stalled_requests'],policy_stalled=p['stalled_requests'],reference_gap=b['max_silence'],policy_gap=p['max_silence'],reference_evictions=b['trace_preemptions'],policy_evictions=p['trace_preemptions'],reference_replay=b['rescheduled_token_positions'],policy_replay=p['rescheduled_token_positions'],reference_excess_s=b['silence_excess_1s_s'],policy_excess_s=p['silence_excess_1s_s'],reference_wait_max=b['wait_max_s'],policy_wait_max=p['wait_max_s'],reference_watermark_wait_s=b['watermark_decision_s'],policy_watermark_wait_s=p['watermark_decision_s']))
    pressure_keys={key for key,policies in keyed.items() if 'baseline' in policies and policies['baseline']['trace_preemptions']>0}
    pressure_trials={r['trial'] for r in rows if tuple(r[k] for k in dimensions) in pressure_keys}
    policy_aggregates={}
    for label,chosen in [('all',rows),('baseline_pressure',[r for r in rows if r['trial'] in pressure_trials])]:
        policy_aggregates[label]={}
        for policy in sorted({r['policy'] for r in chosen}):
            group=[r for r in chosen if r['policy']==policy]
            rr=[r for r in request_rows if r['trial'] in {g['trial'] for g in group}]
            ee=[e for e in episodes if e['trial'] in {g['trial'] for g in group}]
            clean=[e for e in ee if not e['intervening_preemptions']]
            gap_s=sum(r['total_gt1_gaps_s'] for r in group)
            policy_aggregates[label][policy]=dict(trials=len(group),requests=12*len(group),stalled_requests=sum(r['stalled_requests'] for r in group),evictions=sum(r['trace_preemptions'] for r in group),replay=sum(r['rescheduled_token_positions'] for r in group),worst_gap=max(r['max_silence'] for r in group),excess_s=sum(r['silence_excess_1s_s'] for r in group),total_long_gap_s=gap_s,wait_covered_gap_s=sum(r['wait_covered_gt1_gaps_s'] for r in group),wait_coverage_fraction=sum(r['wait_covered_gt1_gaps_s'] for r in group)/gap_s if gap_s else None,wait_sum_s=sum(r['wait_sum_s'] for r in group),wait_max_s=max(r['wait_max_s'] for r in group),watermark_decision_s=sum(r['watermark_decision_s'] or 0 for r in group) if not args.previous else None,capacity_decision_s=sum(r['capacity_decision_s'] or 0 for r in group) if not args.previous else None,no_attempt_decision_s=sum(r['no_attempt_decision_s'] or 0 for r in group) if not args.previous else None,binding_waiver_admissions=sum(r['binding_waiver_admissions'] for r in group),repeatedly_evicted_requests=sum(r['repeatedly_evicted_requests'] for r in group),evicted_stalled_requests=sum(r['evicted_stalled_requests'] for r in group),uncensored_episodes=len(clean),censored_episodes=len(ee)-len(clean),episode_wait_median=statistics.median(e['wait_s'] for e in ee) if ee else None,clean_replay_to_delivery_median=statistics.median(e['replay_to_delivery_s'] for e in clean) if clean else None,clean_replay_to_delivery_p95=percentile([e['replay_to_delivery_s'] for e in clean],95) if clean else None,thresholds={str(t):sum(r[f'gap_gt_{t}'] for r in rr) for t in (.5,1,2,5)})
    paired={}
    for reference,policy in sorted({(p['reference'],p['policy']) for p in pairs}):
        group=[p for p in pairs if p['reference']==reference and p['policy']==policy]
        pressure=[p for p in group if tuple(p[k] for k in dimensions) in pressure_keys]
        paired[f'{policy}_vs_{reference}']=dict(all=paired_summary(group),baseline_pressure=paired_summary(pressure),strata={f'{dim}={value}':paired_summary([p for p in pressure if p[dim]==value]) for dim in ('model','mix','arrival','seed') for value in sorted({p[dim] for p in pressure},key=str)})
    batching=[]
    for key,policies in keyed.items():
        model,slots,cap,seed,mix,arrival=key
        if model!='llama1b' or slots!=3072 or cap!=12 or mix!='generation' or 'baseline' not in policies: continue
        low=keyed.get((model,slots,3,seed,mix,arrival),{}).get('baseline')
        if not low: continue
        b=policies['baseline']; gain=b['throughput']-low['throughput']
        for policy,p in policies.items():
            if policy=='baseline': continue
            batching.append(dict(seed=seed,arrival=arrival,policy=policy,low_throughput=low['throughput'],high_throughput=b['throughput'],policy_throughput=p['throughput'],default_gain_ratio=b['throughput']/low['throughput']-1,gain_retained=(p['throughput']-low['throughput'])/gain if b['throughput']/low['throughput']>1.05 else None,default_stalled=b['stalled_requests'],policy_stalled=p['stalled_requests']))
    frozen=json.loads((ROOT/'evaluation_code_hashes.json').read_text())
    assert frozen=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in frozen}
    checks=dict(completed_trials=len(rows),completed_requests=len(request_rows),token_counts_verified=True,arrival_order_verified=True,preemptions_match_metrics=True,replay_matches_lost_positions=True,policy_and_admission_checks=True,extended_prefill_metric_excludes_replay=True,source_hashes_verified=True,unfinished_cells=unfinished,diagnostics_excluded=args.prefix=='study',previous_source=args.previous)
    summary=dict(completed_trials=len(rows),completed_requests=len(request_rows),baseline_pressure_cases=len(pressure_keys),policies=policy_aggregates,paired=paired,episode_count=len(episodes),max_arrival_lateness_s=max([r['lateness_s'] for r in request_rows] or [0]))
    for name,value in [('analysis',rows),('requests',request_rows),('episodes',episodes),('pairs',pairs),('batching',batching),('summary',summary),('checks',checks)]:
        (ROOT/f'{output_prefix}{name}.json').write_text(json.dumps(value,indent=2))
    for name,value in [('trials',rows),('requests',request_rows),('episodes',episodes),('pairs',pairs),('batching',batching)]: csv_write(ROOT/f'{output_prefix}{name}.csv',value)
    print(json.dumps(dict(checks=checks,policies=policy_aggregates['baseline_pressure'],paired={k:v['baseline_pressure'] for k,v in paired.items()}),indent=2))

