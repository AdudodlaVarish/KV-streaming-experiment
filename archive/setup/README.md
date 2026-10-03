# KV cache lab

Installed on WSL2 Ubuntu 26.04.1 LTS for the RTX 4060 Laptop GPU (8 GB).
Python 3.12, vLLM 0.30.0, FlashInfer 0.6.18.post1 (required by vLLM), PyTorch 2.13.0 with CUDA 13.2.

From PowerShell, enter Ubuntu:

```powershell
wsl -d Ubuntu
```

Then in Ubuntu:

```bash
cd ~/kv-cache-lab
source activate.sh
python check_flashinfer.py
python check_vllm.py
```

The activation script selects the environment's CUDA compiler and libraries; use it instead of only activating .venv. Existing system CUDA and Python environments are untouched.
The FlashInfer check compares GPU decode attention against PyTorch. The vLLM check generates text using the FlashInfer backend. The small Qwen model is stored in the Hugging Face cache.

Start a local server:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.65 --enforce-eager --attention-backend FLASHINFER --enable-prefix-caching
```

In another activated Ubuntu terminal:

```bash
vllm chat --url http://127.0.0.1:8000/v1
curl -s http://127.0.0.1:8000/metrics
vllm bench latency --help
vllm bench throughput --help
vllm bench serve --help
```

Run a small latency benchmark after stopping the server (Ctrl+C):

```bash
vllm bench latency --model Qwen/Qwen2.5-0.5B-Instruct --input-len 512 --output-len 64 --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 2048 --gpu-memory-utilization 0.65 --max-num-seqs 4 --enforce-eager --attention-backend FLASHINFER --no-enable-prefix-caching --output-json latency.json
```

Vary --input-len (128, 512, 1024) and --batch-size (1, 2, 4). The server metrics expose KV usage and prefix-cache behavior. The server preallocates a KV pool, so total GPU memory alone does not measure active KV usage. Disable prefix caching for an independent-request baseline. Keep requests plus outputs within max-model-len.
Eager mode keeps initial startup simple; remove --enforce-eager when measuring optimized performance (additional compilation and graph memory may be needed). First-use kernel compilation and model download are not steady-state timings.

Installed implementation code is under .venv/lib/python3.12/site-packages/vllm and flashinfer. See vllm/v1/core/kv_cache_manager.py and the adjacent scheduler/cache files.
requirements.lock.txt records installed versions. Do not upgrade FlashInfer independently: vLLM pins its compatible version.

Documentation:
- https://docs.vllm.ai/en/latest/
- https://docs.flashinfer.ai/
