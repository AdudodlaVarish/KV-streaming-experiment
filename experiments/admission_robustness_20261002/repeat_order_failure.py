"""Replace entire configuration only for failed arrival-order control, never performance."""
import hashlib, json, pathlib, subprocess, sys
import run
ROOT=pathlib.Path(__file__).resolve().parent
locked=json.loads((ROOT/'evaluation_code_hashes.json').read_text())
assert locked=={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in locked}
original=ROOT/'study13_s77_llama1b_kv3072_cap12'
assert original.is_dir() and (original/'complete.json').is_file()
assert all((ROOT/f'study{i:02d}_s{seed}_{model}_kv{slots}_cap{cap}'/'complete.json').is_file() for i,(seed,(model,slots,cap)) in enumerate(json.loads((ROOT/'design.json').read_text())['jobs']))
subprocess.run([sys.executable,str(ROOT/'verify_orders.py')],check=True)
audit=json.loads((ROOT/'order_audit.json').read_text())
assert len(audit['failures'])==1 and audit['failures'][0]['run']==original.name,audit
metadata=dict(reason='Engine arrival order differed from prospective request-index order. Entire configuration replaced without outcome selection.',frozen_hashes=locked,failed_order=audit['failures'],excluded_configurations=[],accepted_replacement=None)
original.rename(ROOT/('excluded_order_'+original.name))
metadata['excluded_configurations'].append('excluded_order_'+original.name)
for attempt in range(1,4):
    name=f'study13r{attempt:02d}_s77_llama1b_kv3072_cap12'
    (ROOT/'order_replacement.json').write_text(json.dumps(metadata,indent=2))
    run.server_run(('llama1b',3072,12),77,name)
    (ROOT/name/'complete.json').write_text(json.dumps(dict(complete=True,replacement_for=original.name)))
    subprocess.run([sys.executable,str(ROOT/'verify_orders.py')],check=True)
    audit=json.loads((ROOT/'order_audit.json').read_text())
    if not audit['failures']:
        metadata['accepted_replacement']=name
        metadata['retained_trials']=168
        metadata['retained_requests']=2016
        metadata['additional_configuration_attempts']=attempt
        (ROOT/'order_replacement.json').write_text(json.dumps(metadata,indent=2))
        print('Order-controlled replacement accepted:',name,flush=True)
        break
    assert all(f['run']==name for f in audit['failures']),audit
    (ROOT/name).rename(ROOT/('excluded_order_'+name))
    metadata['excluded_configurations'].append('excluded_order_'+name)
    metadata['failed_order']+=audit['failures']
else:
    (ROOT/'order_replacement.json').write_text(json.dumps(metadata,indent=2))
    raise RuntimeError('Three same-code repeats failed engine order check; no invalid run retained')
