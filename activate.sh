source ~/kv-cache-lab/.venv/bin/activate
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
export LIBRARY_PATH="$HOME/kv-cache-lab/cuda-libs:$CUDA_HOME/lib:/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
