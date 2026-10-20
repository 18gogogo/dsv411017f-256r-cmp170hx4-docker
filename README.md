# DeepSeek V4.1 Flash-1017 · 4× CMP 170HX X99 混合推理 Docker 鏡像

[![Docker Pulls](https://img.shields.io/docker/pulls/18gogogo/170hx4-256r-dsv41-1017-flash.svg)](https://hub.docker.com/r/18gogogo/170hx4-256r-dsv41-1017-flash)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

在 **X99 平台（TR 級 CPU + 256GB DDR4）+ 4 張 CMP 170HX 64GB** 上以 vLLM 混合推理運行 **DeepSeek-V4.1-Flash-1017**（510.3 GiB MoE）的 OpenAI 相容服務。部分 MoE 專家層由 CPU + NVMe Engram 緩衝分擔，跨卡通訊經 pinned host staging（170HX 沒有 NVLink，P2P 必須禁用）。

*Run DeepSeek-V4.1-Flash-1017 (510.3 GiB MoE) with vLLM hybrid CPU/GPU inference on 4× CMP 170HX 64GB (X99, 256GB DDR4). OpenAI-compatible API. Measured single-stream decode 47.84–58.65 tok/s.*

---

## 🙏 鳴謝 Credits（核心功勞屬於原作者）

本 repo 僅將以下原作者的成果包裝為 Docker 鏡像並加以參數化，請支持他們：

- **[LocalFrameLab/deepseek-v41-cmp170hx-x99](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99)**（Apache-2.0）— `x99-stable.patch`、生產啟動參數、完整 Decode/Prefill/TTFT 實測基準。本鏡像的核心
- **[buliaoyin/vllm-backport](https://github.com/buliaoyin/vllm-backport)** 分支 `codex/deepseek-v41-cpu-hybrid`，釘選實測提交 `c3d3104de0163c389d26d9fcba0180f55ad7964d` — CPU/GPU 混合推理引擎本體（含 `libdsv41_ik.so` / `libdsv41_cuda.so` Engram SSD 介面）
- **[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**（MIT）— 模型權重 510.3 GiB，非 gated
- **[vLLM 專案](https://github.com/vllm-project/vllm)** — 推理引擎上游

本 repo 相對上游的改動：Dockerfile 容器化、`entrypoint.sh` 參數化、`build.sh` 自動化（clone + 釘選提交 + apply patch + build）、`download-model.sh`、Docker Hub 鏡像發布。授權條款沿用 [Apache-2.0](LICENSE)。

## ☕ 支持本專案

[![Buy me a coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://buymeacoffee.com/tenantreview)

## ⚠️ 硬體門檻（請看清楚後再嘗試）

| 要求 | 值 |
|---|---|
| GPU | **4 張** CMP 170HX 64GB（實測機為破解韌體執行 sm_80；標準 Ampere sm80 卡本專案未實測） |
| 主機記憶體 | **≥ 256GB**（啟動瞬間需要 ≥236GB available；Qwen／其他模型必須先停止） |
| CPU | X99 / TR 級（實測機 TR1900X 8C/16T） |
| NVMe | 模型 510.3 GiB + 緩衝 ≈ 600GB 以上可用空間 |
| 軟體 | Linux + NVIDIA driver + nvidia-container-toolkit |

**只有 1–3 張卡啟動會直接失敗（PP4 設計）— 請勿嘗試。**

## 📦 快速開始（三步）

**1. 拉取鏡像（~30.6 GB）**

```bash
docker pull 18gogogo/170hx4-256r-dsv41-1017-flash:sm80
```

**2. 下載模型（510.3 GiB，支援斷點續傳）**

```bash
./download-model.sh
```

> Script 內部會自動以 **pipx** 安裝 hf CLI（Ubuntu 24+/26.04 的 PEP 668 限制使 pip 直接安裝失敗）。想使用手動逐步版，請見 [DEPLOY.md](DEPLOY.md) 步驟 2「手動逐步版」。

**3. 啟動**
請按照 [DEPLOY.md](DEPLOY.md) 步驟 4 的完整 `docker run` 指令啟動（Hub 鏡像未設定 entrypoint，需寫完整啟動命令）。自行以 `./build.sh` 構建的鏡像已內建 entrypoint，一行即可啟動。

啟動後**請先發送 8 個 warmup 請求**再評估速度（見 DEPLOY.md 步驟 5）。API 預設僅綁定 `127.0.0.1:18050`（隱私設計）；跨機器存取請使用 SSH tunnel：`ssh -L 18050:127.0.0.1:18050 user@host`。

## 📊 實測效能（LocalFrameLab 2026-10-18 四卡數據）

- 單流 Decode：**47.84–58.65 tok/s**（完整故事／程式任務分別約 45.46 / 49.24）
- Prefill / TTFT：1K → 640–931 tok/s / 1.1–1.6s；4K → 1784–2130 / 2.2–2.3s；8K → 3100–3670 / 2.3–2.7s；16K → 2920–5455 / 3.1–5.7s
- 8 路並行：每路 11–16 tok/s，聚合 91.28–116.96 tok/s
- ⚠️ 單流 80 tok/s 目標**未達成** — CMP 170HX 沒有 NVLink + PCIe Gen2 屬物理極限，不是配置問題
- ⚠️ 實測機曾出現可修正的 PCIe AER 錯誤（原因未證實，未見資料損壞）

## 🔒 隱私與安全設計

- 服務預設僅綁定 `127.0.0.1` — 不會暴露區域網路
- `HF_HUB_OFFLINE=1` — 執行時期零外連
- 鏡像內不含任何個人憑證、URL、路徑

## 📄 授權與轉發聲明

`patches/x99-stable.patch` 與 `patches/tested-source-hashes.json` 轉發自 [LocalFrameLab/deepseek-v41-cmp170hx-x99](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99)（Apache-2.0），未經修改。詳見 [DEPLOY.md](DEPLOY.md)。
