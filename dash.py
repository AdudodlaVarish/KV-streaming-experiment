import time, urllib.request, os
from prometheus_client.parser import text_string_to_metric_families

URL = os.environ.get("VLLM_METRICS_URL", "http://127.0.0.1:8017/metrics")

def fetch():
    text = urllib.request.urlopen(URL).read().decode()
    v = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            n = s.name
            if not n.startswith("vllm:") or n.endswith(("_created", "_bucket")):
                continue
            for lab in ("source", "finished_reason"):
                if lab in s.labels:
                    n += f"[{s.labels[lab]}]"
            v[n] = s.value
    return v

def avg(v, base):
    c = v.get(base + "_count", 0)
    return v.get(base + "_sum", 0) / c if c else 0

def show(v):
    q = v.get("vllm:prefix_cache_queries_total", 0)
    h = v.get("vllm:prefix_cache_hits_total", 0)
    rows = [
        ("SERVER", None),
        ("Requests running", int(v.get("vllm:num_requests_running", 0))),
        ("Requests waiting", int(v.get("vllm:num_requests_waiting", 0))),
        ("Requests finished (ok)", int(v.get("vllm:request_success_total[stop]", 0) + v.get("vllm:request_success_total[length]", 0))),
        ("Preemptions", int(v.get("vllm:num_preemptions_total", 0))),
        ("KV CACHE", None),
        ("KV cache usage", f"{v.get('vllm:kv_cache_usage_perc', 0) * 100:.2f}%"),
        ("Prefix cache queries (tokens)", int(q)),
        ("Prefix cache hits (tokens)", int(h)),
        ("Prefix cache hit rate", f"{(h / q * 100) if q else 0:.1f}%"),
        ("TOKENS", None),
        ("Prompt tokens computed", int(v.get("vllm:prompt_tokens_by_source_total[local_compute]", 0))),
        ("Prompt tokens from cache", int(v.get("vllm:prompt_tokens_by_source_total[local_cache_hit]", 0))),
        ("Generated tokens", int(v.get("vllm:generation_tokens_total", 0))),
        ("LATENCY (avg per request)", None),
        ("Time to first token", f"{avg(v, 'vllm:time_to_first_token_seconds') * 1000:.0f} ms"),
        ("Prefill time", f"{avg(v, 'vllm:request_prefill_time_seconds') * 1000:.0f} ms"),
        ("Inter-token latency", f"{avg(v, 'vllm:inter_token_latency_seconds') * 1000:.1f} ms"),
        ("End-to-end", f"{avg(v, 'vllm:e2e_request_latency_seconds'):.2f} s"),
    ]
    for label, val in rows:
        if val is None:
            print(f"\n== {label} ==")
        else:
            print(f"  {label:<32}{val}")

while True:
    try:
        v = fetch()
        os.system("clear")
        print("vLLM live metrics  (Ctrl+C to quit)")
        show(v)
    except Exception as e:
        print("Can't reach server:", e)
    time.sleep(2)
