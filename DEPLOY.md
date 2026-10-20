# 部署指南：DeepSeek V4.1 Flash-1017（4× CMP 170HX 混合推理版）

目標機：**X99 + TR 級 CPU + 256GB DDR4 + 4× CMP 170HX 64GB**，Linux + NVIDIA driver + nvidia-container-toolkit
鏡像：`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`（Docker Hub public，30.6 GB；基於上游實測提交 + x99-stable.patch）
模型：`deepseek-ai/DeepSeek-V4.1-Flash-1017`（510.3 GiB，MIT，非 gated — 不需要 HF token，有 token 下載更快）

---

## 步驟 0：前置檢查

```bash
nvidia-smi                      # 4 張卡全部可見，且未被其他行程佔用
free -g                         # available 必須 ≥ 236 GiB（其他模型容器須先停止）
df -h                           # NVMe 可用空間 ≥ 600 GiB
docker ps                       # 確認沒有其他 GPU 模型容器正在執行
```

## 步驟 1：取得鏡像（二選一）

**A. 拉取現成鏡像（~30.6 GB，10–30 分鐘視網路速度）**

```bash
docker pull 18gogogo/170hx4-256r-dsv41-1017-flash:sm80
```

**B. 自行 build（需時約 1 小時起，視 CPU 核心數）**

```bash
./build.sh    # 自動 clone vllm-backport @ 釘選實測提交、apply patch、docker build
```

## 步驟 2：下載模型（510.3 GiB）

```bash
./download-model.sh            # 預設儲存至 $HOME/models/DeepSeek-V4.1-Flash-1017
# 或自訂目錄： ./download-model.sh /path/to/models/DeepSeek-V4.1-Flash-1017
```

支援斷點續傳 — 中斷後重跑相同指令會跳過已完成檔案。如有 HF token，請先 `export HF_TOKEN=<你的>` 再執行，可解除匿名頻寬限制。

### 手動逐步版（不想使用 script 的用戶）

> Ubuntu 24.04+/26.04 有 PEP 668 限制，`pip` 直接安裝會失敗 — 正確方法是 **pipx**。

```bash
# 1. 安裝 pipx
sudo apt update && sudo apt install -y pipx

# 2. 設定 PATH（關鍵步驟；之後請重新開啟一個 shell 或執行 source ~/.bashrc）
pipx ensurepath

# 3. 安裝 huggingface_hub（注意：新版本已移除 huggingface-cli，改用 hf 命令）
pipx install huggingface_hub

# 4. 取得模型目錄擁有權（避免權限問題）
sudo chown -R $USER:$USER $HOME/models/DeepSeek-V4.1-Flash-1017

# 5. 下載（510.3 GiB；斷點續傳 — 中斷後重跑會跳過已完成檔案）
export HF_XET_HIGH_PERFORMANCE=1
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $HOME/models/DeepSeek-V4.1-Flash-1017
```

## 步驟 3：鏡像驗證（首次啟動前執行一次）

上游第七輪實錄教訓：CUDA 函式庫漏鏈 `engram_ssd.cpp` → 服務在模型載入前即退出。驗證兩個函式庫 + Engram 介面：

```bash
docker run --rm 18gogogo/170hx4-256r-dsv41-1017-flash:sm80 bash -lc '
ls -d /workspace/vllm-backport
find /workspace/vllm-backport -name "libdsv41_*.so"
for f in $(find /workspace/vllm-backport -name libdsv41_cuda.so); do
  echo "$f engram_symbols=$(nm -D "$f" | grep -ci engram)"; done'
```

兩個 `.so` 皆可見 + `engram_symbols` > 0 = 通過。（自行 build 的用戶請將鏡像名改為 `dsv41-vllm:local`，做法相同）

## 步驟 4：啟動

```bash
docker run -d --name deepseek-v41f1017 \
  --gpus all \
  --ipc=host \
  --cap-add=SYS_NICE \
  --network host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e DSV41_HOST_STAGING_CACHE=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_PROTO=Simple \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e VLLM_CACHE_ROOT=/tmp/dsv41-cache \
  -e TRITON_CACHE_DIR=/tmp/dsv41-triton-cache \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=36000 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=16 \
  -v $HOME/models/DeepSeek-V4.1-Flash-1017:/model:ro \
  -v $HOME/dsv41-cache:/tmp/dsv41-cache \
  -v $HOME/dsv41-triton:/tmp/dsv41-triton \
  --restart no \
  18gogogo/170hx4-256r-dsv41-1017-flash:sm80 \
  /workspace/vllm-backport/.venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name DeepSeek-V4.1-Flash-1017 \
    --host 127.0.0.1 --port 18050 \
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
```

> 自行 build 鏡像的用戶：將鏡像名改為 `dsv41-vllm:local` 並移除最後整段啟動命令（entrypoint 已內建全部參數），即可簡化為：
> `docker run -d --name deepseek-v41f1017 --gpus all --ipc=host --cap-add=SYS_NICE --network host -v $HOME/models/DeepSeek-V4.1-Flash-1017:/model:ro -v $HOME/dsv41-cache:/tmp/dsv41-cache -v $HOME/dsv41-triton:/tmp/dsv41-triton --restart no dsv41-vllm:local`

### 部署參數逐項參考（為什麼 + 好處／壞處）

| 參數 | 原因 |
|---|---|
| `NCCL_P2P_DISABLE=1` + `NCCL_PROTO=Simple` | 170HX 沒有 NVLink，P2P 實測不可靠 — 强行啟用會產生資料錯誤。代價：跨卡通訊慢，是 decode 速度上限的物理成因 |
| `DSV41_HOST_STAGING_CACHE=1` | 由 patch #3 提供；禁用 P2P 時以 pinned host 緩衝執行跨卡傳輸，缺少它 CUDA graph capture 會直接崩潰 |
| `--ipc=host` | 多 worker 共用大型共享記憶體（跨卡傳輸走主機記憶體） |
| `--cap-add=SYS_NICE` | NUMA 記憶體配置所需的最小權限（上游明確警告：不要為了消除提示而開放所有權限） |
| `--network host` + `127.0.0.1` | 隱私預設：僅本機可見。區域網路存取請以 `ssh -L 18050:127.0.0.1:18050 user@host` 再連線 `http://localhost:18050/v1` |
| `--pipeline-parallel-size 4` + `pipeline_layers [6,6,6,22]` | 非平均分配是上游調校過的負載平衡（末卡 22 層）— 自行修改會變慢 |
| `--gpu-memory-utilization 0.97` | KV cache 最大化；顯存上不要再佔用其他東西 |
| `--max-num-seqs 8` | 上游以 8 路為設計點；超過不會更快，只會每路更慢 |
| `cpu_threads [32,28]` | 上游實測機數值；較少核心的 CPU（如 8C/16T）屬於超額訂閱，上游結論「修改執行緒數沒有穩定提升」— 建議照用為基線，可自行 A/B 測試 |
| `engram_storage ssd` + `engram_cache_gib 4` + `host_cache_gib 8` | MoE 專家緩衝存放 NVMe + 釘選 8GiB 主機記憶體；將顯存保留給 KV cache |
| `dspark` 投機解码 3 tokens | 上游測試過修改草稿數量沒有穩定提升，照用 3 |
| `VLLM_ENGINE_READY_TIMEOUT_S=36000` | 510GB 冷讀入顯存可能需要數小時 — 防止 timeout 誤殺 |
| `--restart no` | 建議手動管理重啟（`docker start deepseek-v41f1017`） |
| cache bind mount（`$HOME/dsv41-cache`、`$HOME/dsv41-triton`） | 緩存跨重啟持久化 → 第二次啟動不必重新編譯 kernel，顯著加快 |

## 步驟 5：首次啟動 + warmup

```bash
docker logs -f deepseek-v41f1017     # 模型載入可能需要 30–90 分鐘（510GB 冷讀）
```

看到 startup complete 之後，**請先發送 8 個 dummy 請求**才能達到正常速度：

```bash
for i in $(seq 8); do
  curl -s http://127.0.0.1:18050/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}' > /dev/null
done
```

驗證 API：

```bash
curl -s http://127.0.0.1:18050/v1/models | head -c 200
```

## 步驟 6：效能預期

請見 README 的實測表。單流 80 tok/s 未達成屬 170HX 物理極限（沒有 NVLink + PCIe Gen2），不應期待配置層面的奇蹟。

## FAQ

| 症狀 | 處理 |
|---|---|
| 僅 1–3 張卡啟動即失敗 | PP4 需要恰好 4 張 — 沒有 workaround |
| 服務在模型載入前退出 | 依步驟 3 驗證：`engram_symbols=0` → 鏡像缺少 Engram SSD 連結，需以 build.sh 重建 |
| CUDA OOM | 確認顯卡未被其他行程佔用；不要調低 `--gpu-memory-utilization` 以下試錯 |
| 啟動 timeout / 記憶體 OOM | 以 `free -g` 確認 available ≥236GiB；停止其他大型應用 |
| 想啟用 P2P 提速 | 不要 — 上游實測基線就是禁用 P2P，啟用會產生資料錯誤 |
| dmesg 出現可修正的 PCIe AER 錯誤 | 上游實測機同樣出現過，未見資料損壞；屬可觀察範圍 |
| 切換回其他模型 | `docker stop deepseek-v41f1017 && docker start <你的其他模型容器>` |

## LAN 直接存取變體（選用）

將啟動命令中 `--host 127.0.0.1 --port 18050` 改為 `--host 0.0.0.0 --port 9494`（端口可自訂）。

**必做 — 開放防火牆**（UFW 預設拒絕所有進入連接）：

```bash
sudo ufw allow 9494/tcp
```

未開放防火牆時，綁定 `0.0.0.0` 仍然無法從區域網路連線。

**隱私取捨**：綁定 `0.0.0.0` 後區域網路內任何設備皆可直接存取 API（vLLM 預設無身分驗證）。建議加上 `--api-key <自訂密鑰>` 作最低限度防護，客戶端需攜帶 `Authorization: Bearer <密鑰>`。LAN 版 API 位址：`http://<主機IP>:9494/v1`。

## 單流速度上限說明

上游第七輪系統性測試（緩存調整、投機解碼草稿數、CPU 預取、GPU 分塊、主機傳輸打包、橋接替換）結論：**皆無穩定提升**。在相同精度、相同上下文前提下，單流 45–59 tok/s 即为此硬體（無 NVLink + PCIe Gen2）之穩定上限；突破需重構引擎層分工，非參數調整可達。唯一值得依自身 CPU 核心數 A/B 測試嘅參數为 `cpu_threads`（例如 8C/16T 機器可試 `[16,14]`）。
