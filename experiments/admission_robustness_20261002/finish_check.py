import datetime, json, pathlib, subprocess
root=pathlib.Path('/home/varish/kv-cache-lab/experiments/admission_robustness_20261002')
checks=json.loads((root/'checks.json').read_text())
assert checks['completed_trials']==168 and not checks['unfinished_cells']
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
matches=[line for line in processes.splitlines() if 'vllm serve ' in line or 'VLLM::EngineCore' in line]
assert not matches,matches
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu','--format=csv'],text=True)
(root/'gpu_final.csv').write_text(gpu)
(root/'completion.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_runner_exit=0,replacement_runner_exit=0,retained_trials=168,retained_requests=2016,excluded_order_configuration_trials=12,experiment_servers_stopped=True,gpu_final=gpu),indent=2))
print('Completion and server cleanup verified')
