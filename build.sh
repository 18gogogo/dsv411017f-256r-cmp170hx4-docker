#!/usr/bin/env bash
# 自動 build：clone vllm-backport（釘上游實測提交）→ apply x99-stable.patch → docker build
# 需時視乎 CPU（實測機 32 核 ≈ 1 小時）。需要 Docker + BuildKit。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[1/3] clone vllm-backport @ 實測提交..."
rm -rf vllm-backport
git clone --branch codex/deepseek-v41-cpu-hybrid https://github.com/buliaoyin/vllm-backport vllm-backport
cd vllm-backport
git fetch --depth 1 origin c3d3104de0163c389d26d9fcba0180f55ad7964d
git checkout FETCH_HEAD

echo "[2/3] apply x99-stable.patch..."
git apply --check "$ROOT/patches/x99-stable.patch"
git apply "$ROOT/patches/x99-stable.patch"

echo "[3/3] docker build（CUDA kernel 編譯最慢，請 patiently 等）..."
cd "$ROOT"
docker build -t dsv41-vllm:local .

echo "BUILD 完成。下一步：./download-model.sh 然後照 DEPLOY.md 步驟 4/5 啟動。"
echo "（自行 build 嘅鏡像已內建 entrypoint，啟動可簡化為：）"
echo "docker run -d --name dsv41 --gpus all --ipc=host --cap-add=SYS_NICE --network host \\"
echo "  -v \$HOME/models/DeepSeek-V4.1-Flash-1017:/model:ro --restart no dsv41-vllm:local"
