import argparse,csv,hashlib,importlib.util,json,math,pathlib,statistics,sys
ROOT=pathlib.Path(__file__).resolve().parent;PRIOR=ROOT.parent/'capacity_cliff_20261002';RESUME=ROOT.parent/'resume_admission_20261003'
sys.path.insert(0,str(PRIOR))
from bench import metrics,total,percentile
from analyze import trace_summary
spec=importlib.util.spec_from_file_location('resume_analysis',RESUME/'analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
PARAMS={p:dict(growth=256 if p=='guard256' else 128,budget=.25 if p=='guard_fast' else 1. if p=='guard_relaxed' else .5) for p in ('baseline','adaptive_resume','slai_port','growth_only','guard128','guard256','guard_fast','guard_relaxed')}
DIMS=('model','slots','cap','seed','mix','arrival')

def geo(values):return math.exp(statistics.mean(math.log(v) for v in values)) if values else None

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k,v in row.items() if not isinstance(v,(dict,list))))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def pair_summary(pp):
    if not pp:return {}
    out=dict(pairs=len(pp),throughput_ratio_geo=geo([p['throughput_ratio'] for p in pp]),ttft_ratio_geo=geo([p['ttft_ratio'] for p in pp]),completion_ratio_geo=geo([p['completion_ratio'] for p in pp]),throughput_ratio_min=min(p['throughput_ratio'] for p in pp),throughput_ratio_max=max(p['throughput_ratio'] for p in pp))
    for key in ('stalled','budget_misses','evictions','replay','excess','repeat_victims'):
        for side in ('reference','policy'):out[side+'_'+key]=sum(p[side+'_'+key] for p in pp)
    out.update(reference_worst=max(p['reference_gap'] for p in pp),policy_worst=max(p['policy_gap'] for p in pp),gap_improved=sum(p['policy_gap']<p['reference_gap']-.05 for p in pp),gap_worsened=sum(p['policy_gap']>p['reference_gap']+.05 for p in pp),stalls_improved=sum(p['policy_stalled']<p['reference_stalled'] for p in pp),stalls_worsened=sum(p['policy_stalled']>p['reference_stalled'] for p in pp))
    return out

def replay_accounting(events):
    high={};lost={};scheduled={};discarded={}
    for e in events:
        if e['kind']=='preempt':
            rid=e['request'];lost[rid]=lost.get(rid,0)+e['computed'];high[rid]=max(e['computed'],high.get(rid,0))
        elif e['kind']=='step':
            for rid,before,n,replayed,output in e['scheduled']:
                expected=max(0,min(n,high.get(rid,0)-before));assert replayed==expected
                scheduled[rid]=scheduled.get(rid,0)+replayed
                if before+n>=high.get(rid,0):high.pop(rid,None)
        elif e['kind']=='service' and e['stopped']:
            rid=e['request'];remaining=high.pop(rid,0)
            discarded[rid]=discarded.get(rid,0)+remaining
            if remaining:assert e['stale'],e
    assert not high,high
    for rid in lost:assert lost[rid]==scheduled.get(rid,0)+discarded.get(rid,0),(rid,lost[rid],scheduled.get(rid,0),discarded.get(rid,0))
    return lost,scheduled,discarded

def joined_episode(e,steps,alloc,preempts,services,request,row):
    has_replay=any(s['t']>=e['t'] and any(w[0]==e['request'] and w[1]==0 and w[2]>0 for w in s['scheduled']) for s in steps)
    if has_replay:
        result=old.episode(e,steps,alloc,preempts,request,row);result['terminal_without_replay']=False;return result
    terminal=[s for s in services if s['request']==e['request'] and s['t']>=e['t'] and s['stopped']]
    assert terminal and terminal[0]['stale'],(row['trial'],e)
    deliveries=[request['start']+t for t,n in request['token_events']]
    return dict(trial=row['trial'],model=row['model'],slots=row['slots'],cap=row['cap'],seed=row['seed'],mix=row['mix'],arrival=row['arrival'],policy=row['policy'],index=request['index'],engine_request=e['request'],preempt_perf_s=e['t'],replay_start_perf_s=None,next_delivery_perf_s=min(t for t in deliveries if t>=e['t']),wait_s=None,replay_to_delivery_s=None,computed_lost=e['computed'],terminal_without_replay=True,terminal_service_perf_s=terminal[0]['t'],client_started_at_eviction=deliveries[0]<=e['t'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='study');args=parser.parse_args();outprefix='' if args.prefix=='study' else args.prefix+'_'
    rows=[];rr=[];episodes=[];unfinished=[];growth=[];handoffs=[];handoff_effects=[];completed_configs=0
    for folder in sorted(ROOT.glob(args.prefix+'*')):
        if not folder.is_dir() or not (folder/'config.json').exists():continue
        completed_configs+=int((folder/'complete.json').exists());events=[]
        for line in (folder/'scheduler_trace.jsonl').open():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:break
        for cell in sorted(folder.glob('[0-9][0-9]_*')):
            if not (cell/'summary.json').exists():unfinished.append(str(cell.relative_to(ROOT)));continue
            row=json.loads((cell/'summary.json').read_text());reqs=[json.loads(line) for line in (cell/'requests.jsonl').read_text().splitlines()];assert len(reqs)==12
            expected={r['index']:r for r in json.loads((folder/f'{row["mix"]}_{row["arrival"]}_workload.json').read_text())}
            selected=[e for e in events if row['start_perf_s']<=e['t']<=row['end_perf_s']];arrivals=[e for e in selected if e['kind']=='arrival' and e['trial']==row['trial']]
            order=[]
            for e in arrivals:
                match=[r['index'] for r in reqs if e['request'].startswith(r['response_id']+'-')];assert len(match)==1;order+=match
            assert order==list(range(12)),(row['trial'],order)
            preempts=[e for e in selected if e['kind']=='preempt'];steps=[e for e in selected if e['kind']=='step'];alloc=[e for e in selected if e['kind']=='allocation'];services=[e for e in selected if e['kind']=='service'];gates=[e for e in selected if e['kind']=='growth_gate'];swaps=[e for e in selected if e['kind']=='handoff']
            assert services and alloc
            tr=trace_summary(events,row['start_perf_s'],row['end_perf_s']);row.update(tr)
            assert tr['trace_preemptions']==row['metrics_delta']['vllm:num_preemptions_total'];lost_by_req,replay_by_req,terminal_discard=replay_accounting(selected);assert tr['rescheduled_token_positions']+sum(terminal_discard.values())==tr['evicted_computed_positions'],row['trial']
            for g in gates:
                assert g['growth']==PARAMS[row['policy']]['growth']
                assert g['denied'] or g['idle_escape'] or (g['paused_streams']==0 and g['debt']+g['need']<=g['free'])
            for a in alloc:
                assert a['trial']==row['trial'] and a['policy']==row['policy']
                assert a['outcome']!='admit' or a['full_blocks']+a['effective_watermark']<=a['free_blocks']
                if row['policy']=='adaptive_resume' and a['status']=='PREEMPTED':assert a['effective_watermark']==0
                if row['policy']!='adaptive_resume':assert a['global_watermark']==0
            for h in swaps:
                assert h['target_age']>=PARAMS[row['policy']]['budget']*.5
                prior_service={}
                for e in services:
                    if e['t']<h['t']:prior_service[e['request']]=e['t']
                assert h['target'] in prior_service
                assert abs(h['target_age']-(h['t']-prior_service[h['target']]))<1e-5
                if h['victim'] in prior_service:
                    credits={rid:PARAMS[row['policy']]['budget'] if any(p['request']==rid and 0<=h['t']-p['t']<1. for p in preempts) else 0. for rid in (h['target'],h['victim'])}
                    assert prior_service[h['victim']]-credits[h['victim']]>=prior_service[h['target']]-credits[h['target']]-1e-5
            pending=None;effects=[]
            for e in selected:
                if e['kind']=='handoff':
                    assert pending is None
                    pending={**{k:row[k] for k in DIMS},**e,'target_allocation_outcome':None}
                elif pending and e['kind']=='allocation' and e['request']==pending['target']:
                    pending['target_allocation_outcome']=e['outcome']
                elif pending and e['kind']=='step':
                    dispatched={w[0] for w in e['scheduled'] if w[2]>0}
                    pending.update(target_scheduled_next_pass=pending['target'] in dispatched,victim_scheduled_next_pass=pending['victim'] in dispatched,next_step_delay_s=e['t']-pending['t'])
                    effects.append(pending);pending=None
            assert pending is None
            handoff_effects+=effects
            row.update(handoff_target_next_pass=sum(e['target_scheduled_next_pass'] for e in effects),handoff_victim_next_pass=sum(e['victim_scheduled_next_pass'] for e in effects))
            ownrows=[];trial_eps=[]
            for r in reqs:
                assert r['input_tokens']==len(expected[r['index']]['prompt'])==r['usage']['prompt_tokens'];assert r['output_tokens']==expected[r['index']]['output']==r['usage']['completion_tokens']==r['token_events'][-1][1]
                assert all(a[0]<=b[0] and a[1]<b[1] for a,b in zip(r['token_events'],r['token_events'][1:]))
                own=[e for e in preempts if e['request'].startswith(r['response_id']+'-')];svc=[e for e in services if e['request'].startswith(r['response_id']+'-')];assert sum(e['n'] for e in svc)==r['output_tokens']
                ee=[joined_episode(e,steps,alloc,preempts,services,r,row) for e in own];trial_eps+=ee
                times=[r['start']+e[0] for e in r['token_events']];windows=[(a,b) for a,b in zip(times,times[1:]) if b-a>1];covered=old.overlap([(e['preempt_perf_s'],e['replay_start_perf_s']) for e in ee if e['replay_start_perf_s'] is not None],windows)
                q={k:row[k] for k in DIMS};q.update(trial=row['trial'],policy=row['policy'],index=r['index'],input=r['input_tokens'],output=r['output_tokens'],ttft=r['ttft_s'],completion=r['elapsed_s'],max_gap=r['max_silence_s'],evictions=len(own),replay=sum(replay_by_req.get(e['request'],0) for e in own[:1]),lost=sum(e['computed'] for e in own),terminal_discard=sum(terminal_discard.get(e['request'],0) for e in own[:1]),engine_max_gap=max([b['t']-a['t'] for a,b in zip(svc,svc[1:])] or [0]),wait_covered_gt1_gaps_s=covered,total_gt1_gaps_s=sum(b-a for a,b in windows),budget=PARAMS[row['policy']]['budget'],budget_miss=r['max_silence_s']>PARAMS[row['policy']]['budget'],token_sha256=r['token_sha256'])
                q.update({f'gap_gt_{threshold}':r['max_silence_s']>threshold for threshold in (.25,.5,1,2,5)});ownrows.append(q)
            rr+=ownrows;episodes+=trial_eps
            for label,items in (('growth',gates),('handoff',swaps)):
                target=growth if label=='growth' else handoffs
                target.extend({**{k:row[k] for k in DIMS},**e} for e in items)
            row.update(budget=PARAMS[row['policy']]['budget'],budget_misses=sum(q['budget_miss'] for q in ownrows),repeat_victims=sum(q['evictions']>1 for q in ownrows),growth_attempts=len(gates),growth_denials=sum(g['denied'] for g in gates),handoffs=len(swaps),terminal_discard=sum(terminal_discard.values()),terminal_evictions=sum(e['terminal_without_replay'] for e in trial_eps),wait_sum_s=sum(e['wait_s'] or 0 for e in trial_eps),wait_max_s=max([e['wait_s'] or 0 for e in trial_eps] or [0]),wait_covered_gt1_gaps_s=sum(q['wait_covered_gt1_gaps_s'] for q in ownrows),total_gt1_gaps_s=sum(q['total_gt1_gaps_s'] for q in ownrows),evicted_stalled=sum(q['evictions']>0 and q['max_gap']>1 for q in ownrows))
            row.update({f'stalled_gt_{t}':sum(q[f'gap_gt_{t}'] for q in ownrows) for t in (.25,.5,1,2,5)})
            before=metrics((cell/'metrics_before.txt').read_text());after=metrics((cell/'metrics_after.txt').read_text());row['prefill_computed_delta']=total(after,'vllm:request_prefill_kv_computed_tokens_sum')-total(before,'vllm:request_prefill_kv_computed_tokens_sum');assert row['prefill_computed_delta']==sum(q['input'] for q in ownrows)
            assert row['metrics_delta']['vllm:iteration_tokens_total_sum']==sum(q['input']+q['output'] for q in ownrows)
            rows.append(row)
    keyed={}
    for r in rows:
        key=tuple(r[k] for k in DIMS);assert r['policy'] not in keyed.setdefault(key,{});keyed[key][r['policy']]=r
    pairs=[]
    for key,policies in keyed.items():
        comparisons=[('baseline',p) for p in policies if p!='baseline']+[(ref,p) for ref,p in [('slai_port','guard128'),('adaptive_resume','guard128'),('growth_only','guard128'),('guard128','guard256'),('guard128','guard_fast'),('guard128','guard_relaxed')] if ref in policies and p in policies]
        for reference,policy in comparisons:
            if reference not in policies:continue
            b,p=policies[reference],policies[policy];pair=dict(zip(DIMS,key))|dict(reference=reference,policy=policy,throughput_ratio=p['throughput']/b['throughput'],ttft_ratio=p['ttft_p50']/b['ttft_p50'],completion_ratio=p['e2e_p95']/b['e2e_p95'],reference_gap=b['max_silence'],policy_gap=p['max_silence'])
            for side,r in [('reference',b),('policy',p)]:
                pair.update({side+'_stalled':r['stalled_requests'],side+'_budget_misses':r['stalled_gt_0.5'],side+'_evictions':r['trace_preemptions'],side+'_replay':r['rescheduled_token_positions'],side+'_excess':r['silence_excess_1s_s'],side+'_repeat_victims':r['repeat_victims']})
            pairs.append(pair)
    summary=dict(completed_trials=len(rows),completed_requests=len(rr),completed_configs=completed_configs,policies={},paired={})
    for label,chosen in [('main_pressure',[r for r in rows if r['seed']!=381 and r['slots']==3072 and r['cap']!=3]),('long_tail',[r for r in rows if r['seed']==381]),('relief',[r for r in rows if r['slots']==12288])]:
        summary['policies'][label]={};selected_trials={r['trial'] for r in chosen}
        for policy in sorted({r['policy'] for r in chosen}):
            group=[r for r in chosen if r['policy']==policy];own=[q for q in rr if q['trial'] in selected_trials and q['policy']==policy]
            g=dict(trials=len(group),requests=len(own),stalled=sum(r['stalled_requests'] for r in group),evictions=sum(r['trace_preemptions'] for r in group),replay=sum(r['rescheduled_token_positions'] for r in group),excess=sum(r['silence_excess_1s_s'] for r in group),worst=max(r['max_silence'] for r in group),budget_misses=sum(r['budget_misses'] for r in group),repeat_victims=sum(r['repeat_victims'] for r in group),handoffs=sum(r['handoffs'] for r in group),growth_denials=sum(r['growth_denials'] for r in group),wait_sum_s=sum(r['wait_sum_s'] for r in group),handoff_target_next_pass=sum(r['handoff_target_next_pass'] for r in group),handoff_victim_next_pass=sum(r['handoff_victim_next_pass'] for r in group),terminal_discard=sum(r['terminal_discard'] for r in group),terminal_evictions=sum(r['terminal_evictions'] for r in group),engine_over_budget=sum(q['engine_max_gap']>q['budget'] for q in own),client_only_budget_misses=sum(q['budget_miss'] and q['engine_max_gap']<=q['budget'] for q in own))
            totalgap=sum(r['total_gt1_gaps_s'] for r in group);g['wait_coverage']=sum(r['wait_covered_gt1_gaps_s'] for r in group)/totalgap if totalgap else None;g.update({f'stalled_gt_{t}':sum(r[f'stalled_gt_{t}'] for r in group) for t in (.25,.5,1,2,5)});summary['policies'][label][policy]=g
        summary['paired'][label]={f'{policy}_vs_{ref}':pair_summary([p for p in pairs if p['reference']==ref and p['policy']==policy and tuple(p[k] for k in DIMS) in {tuple(r[k] for k in DIMS) for r in chosen}]) for ref,policy in {(p['reference'],p['policy']) for p in pairs}}
    main=[p for p in pairs if p['slots']==3072 and p['cap']!=3 and p['seed']!=381 and p['policy']=='guard128' and p['reference']=='baseline']
    summary['guard_strata']={f'{dim}={value}':pair_summary([p for p in main if p[dim]==value]) for dim in ('model','mix','arrival','seed') for value in sorted({p[dim] for p in main},key=str)}
    batching=[]
    for key,policies in keyed.items():
        model,slots,cap,seed,mix,arrival=key
        if model!='llama1b' or slots!=3072 or cap!=12 or mix!='generation' or 'baseline' not in policies:continue
        low=keyed.get((model,slots,3,seed,mix,arrival),{}).get('baseline')
        if not low:continue
        base=policies['baseline'];gain=base['throughput']-low['throughput']
        for policy,r in policies.items():
            if policy=='baseline':continue
            batching.append(dict(seed=seed,arrival=arrival,policy=policy,default_gain_ratio=base['throughput']/low['throughput']-1,gain_retained=(r['throughput']-low['throughput'])/gain if base['throughput']/low['throughput']>1.05 else None,low_throughput=low['throughput'],default_throughput=base['throughput'],policy_throughput=r['throughput']))
    frozen=json.loads((ROOT/'evaluation_code_hashes.json').read_text());assert frozen=={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in frozen}
    checks=dict(completed_trials=len(rows),completed_requests=len(rr),completed_configs=completed_configs,token_usage_and_engine_service_verified=True,arrival_order_verified=True,preemptions_match_metrics=True,replay_plus_terminal_discard_equals_lost=True,growth_admission_invariants=True,handoff_urgency_invariants=True,native_capacity_checks=True,extended_prefill_metric_excludes_replay=True,frozen_code_verified=True,unfinished=unfinished,diagnostics_excluded=not args.prefix.startswith('diag'))
    for name,contents in [('analysis',rows),('requests',rr),('episodes',episodes),('pairs',pairs),('growth',growth),('handoffs',handoffs),('handoff_effects',handoff_effects),('batching',batching),('summary',summary),('checks',checks)]:
        (ROOT/f'{outprefix}{name}.json').write_text(json.dumps(contents,indent=2))
        if isinstance(contents,list):csv_write(ROOT/f'{outprefix}{name}.csv',contents)
    print(json.dumps(dict(checks=checks,policies=summary['policies'],paired_main=summary['paired']['main_pressure']),indent=2))



