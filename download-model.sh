#!/usr/bin/env bash
# 下載 deepseek-ai/DeepSeek-V4.1-Flash-1017（510.3 GiB，非 gated MIT）
# 支援斷點續傳：中斷後重跑同款指令會跳過已完成檔案。
# 有 HF token 可解除匿名頻寬限制：export HF_TOKEN=xxx 再跑（token 唔使有，模型係 public）。
set -euo pipefail
DIR="${1:-$HOME/models/DeepSeek-V4.1-Flash-1017}"
mkdir -p "$DIR"

# --- 安裝 hf CLI（pipx 方式；Ubuntu 24+/26.04 有 PEP 668，pip 直接裝會失敗）---
if ! command -v hf >/dev/null 2>&1; then
  echo "未搵到 hf CLI，開始安裝..."

  # 1) 裝 pipx
  if ! command -v pipx >/dev/null 2>&1; then
    echo "安裝 pipx..."
    sudo apt update && sudo apt install -y pipx
  fi

  # 2) 配置 PATH（關鍵步驟）
  pipx ensurepath

  # 3) 裝 huggingface_hub（注意：新版本已移除 huggingface-cli，改用 hf 命令）
  pipx install huggingface_hub

  # 4) 載入新 PATH（或直接重新開一個 shell）
  # shellcheck disable=SC1091
  source "$HOME/.bashrc"
fi

# --- 擁有模型目錄（避開權限問題）---
sudo chown -R "${USER}:${USER}" "$DIR"

# --- 下載 ---
export PATH="$HOME/.local/bin:$PATH"
export HF_XET_HIGH_PERFORMANCE=1
echo "下載 510.3 GiB → $DIR （需時視乎網速；中斷冇驚，重跑接返）"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$DIR"

echo "完成。照 DEPLOY.md 步驟 3 驗證鏡像後啟動。"
