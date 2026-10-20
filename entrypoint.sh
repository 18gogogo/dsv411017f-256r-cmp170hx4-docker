#!/usr/bin/env bash
# 鏡像內 entrypoint：將上游生產啟動參數內建，用戶只需掛載模型並啟動容器。
# 注意：Docker Hub 現成鏡像未設定 entrypoint — 請使用 DEPLOY.md 步驟 4 的完整 docker run 指令。
# 自行以 ./build.sh 構建的鏡像才會使用此 entrypoint。
set -euo pipefail
: "${MODEL_PATH:=/model}"
: "${HOST:=127.0.0.1}"
: "${PORT:=18050}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset DSV41_FORCE_LOCAL_CACHE || true
export DSV41_HOST_STAGING_CACHE=1
export NCCL_P2P_DISABLE=1
export NCCL_PROTO=Simple
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/dsv41-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/dsv41-triton}"
export HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export CUDA_DEVICE_MAX_CONNECTIONS=16
exec /workspace/vllm-backport/.venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host "$HOST" --port "$PORT" \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --max-model-len 65536 \
  --kv-cache-tokens 131072 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort":50}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[6,6,6,22],"cpu_threads":[32,28],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
