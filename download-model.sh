#!/usr/bin/env bash
# 下載 deepseek-ai/DeepSeek-V4.1-Flash-1017（510.3 GiB，非 gated MIT）
# 支援斷點續傳：中斷後重跑同款指令會跳過已完成檔案。
# 有 HF token 可解除匿名頻寬限制：export HF_TOKEN=*** 再跑（token 唔使有，模型係 public）。
set -euo pipefail
DIR="${1:-$HOME/models/DeepSeek-V4.1-Flash-1017}"
mkdir -p "$DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "未搵到 hf CLI — 嘗試安裝 huggingface_hub..."
  pip3 install -U huggingface_hub 2>/dev/null \
    || pip3 install -U --user --break-system-packages huggingface_hub
  export PATH="$HOME/.local/bin:$PATH"
fi

export HF_XET_HIGH_PERFORMANCE=1
echo "下載 510.3 GiB → $DIR （需時視乎網速；中斷冇驚，重跑接返）"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$DIR"
echo "完成。照 DEPLOY.md 步驟 3 驗證鏡像後啟動。"
