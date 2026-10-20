# 部署指南：DeepSeek V4.1 Flash-1017（4× CMP 170HX 混合推理版）

目標機：**X99 + TR 級 CPU + 256GB DDR4 + 4× CMP 170HX 64GB**，Linux + NVIDIA driver + nvidia-container-toolkit
鏡像：`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`（Docker Hub public，30.6 GB；基於上游實測提交 + x99-stable.patch）
模型：`deepseek-ai/DeepSeek-V4.1-Flash-1017`（510.3 GiB，MIT，非 gated — 唔使 HF token，有 token 會快啲）

---

## 步驟 0：前置檢查

```bash
nvidia-smi                      # 4 張卡全部見到、冇被其他進程佔用
free -g                         # available 必須 ≥ 236 GiB（其他模型容器要先停）
df -h                           # NVMe 空位 ≥ 600 GiB
docker ps                       # 確認冇其他 GPU 模型容器喺度行
```

## 步驟 1：取得鏡像（二揀其一）

**A. 拉現成鏡像（~30.6 GB，10–30 分鐘視乎網速）**

```bash
docker pull 18gogogo/170hx4-256r-dsv41-1017-flash:sm80
```

**B. 自行 build（需時約 1 小時起，視乎 CPU 核數）**

```bash
./build.sh    # 自動 clone vllm-backport @ 釘實測提交、apply patch、docker build
```

## 步驟 2：下載模型（510.3 GiB）

```bash
./download-model.sh            # 預設去 $HOME/models/DeepSeek-V4.1-Flash-1017
# 或自訂目錄： ./download-model.sh /path/to/models/DeepSeek-V4.1-Flash-1017
```

支援斷點續傳 — 中斷後重跑同款指令會跳過已完成檔案。有 HF token 嘅話 `export HF_TOKEN=<你的>` 再跑，可解除匿名頻寬限制。

### 手動逐步版（不想用 script 嘅用家）

> Ubuntu 24.04+/26.04 有 PEP 668 限制，`pip` 直接裝會失敗 — 正確方法係 **pipx**。

```bash
# 1. 安裝 pipx
sudo apt update && sudo apt install -y pipx

# 2. 配置 PATH（關鍵步驟；之後重新開一個 shell 或 source ~/.bashrc）
pipx ensurepath

# 3. 安裝 huggingface_hub（注意：新版本已移除 huggingface-cli，改用 hf 命令）
pipx install huggingface_hub

# 4. 擁有模型目錄（避免權限問題）
sudo chown -R $USER:$USER $HOME/models/DeepSeek-V4.1-Flash-1017

# 5. 下載（510.3 GiB；斷點續傳 — 中斷後重跑跳過已完成檔案）
export HF_XET_HIGH_PERFORMANCE=1
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $HOME/models/DeepSeek-V4.1-Flash-1017
```

## 步驟 3：鏡像驗證（首次啟動前做一次）

上游第七輪實錄教訓：CUDA 庫漏鏈 `engram_ssd.cpp` → 服務喺 load 模型前就死。驗證兩個庫 + Engram 接口：

```bash
docker run --rm dsv41-vllm:local bash -lc '
ls -d /workspace/vllm-backport
find /workspace/vllm-backport -name "libdsv41_*.so"
for f in $(find /workspace/vllm-backport -name libdsv41_cuda.so); do
  echo "$f engram_symbols=$(nm -D "$f" | grep -ci engram)"; done'
```

兩個 `.so` 都見到 + `engram_symbols` > 0 = OK。（Hub 鏡像改住名 `18gogogo/170hx4-256r-dsv41-1017-flash:sm80` 同樣做法）

## 步驟 4：啟動

```bash
docker run -d --name dsv41 \
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

參數逐項解釋（點解 + 好處/壞處）：

| 參數 | 原因 |
|---|---|
| `NCCL_P2P_DISABLE=1` + `NCCL_PROTO=Simple` | 170HX 冇 NVLink，P2P 實測唔可靠 — 强行開會出數據錯誤。代價：跨卡通信慢，decode 上限嘅物理成因 |
| `DSV41_HOST_STAGING_CACHE=1` | patch #3 提供；禁 P2P 下用 pinned host 緩衝做跨卡傳輸，冇佢 CUDA graph capture 即崩 |
| `--ipc=host` | 多 worker 共用大共享記憶體（跨卡傳輸行主機 RAM） |
| `--cap-add=SYS_NICE` | NUMA 記憶體放置所需嘅最小權限（上游明確警告唔好開放所有權限） |
| `--network host` + `127.0.0.1` | 私隱預設：只喺本機可見。LAN 用 `ssh -L 18050:127.0.0.1:18050 user@host` 再連 `http://localhost:18050/v1` |
| `--pipeline-parallel-size 4` + `pipeline_layers [6,6,6,22]` | 唔平均分係上游調過嘅負載平衡（尾卡 22 層）— 自己改會慢 |
| `--gpu-memory-utilization 0.97` | KV cache 最大化；卡上唔好再佔其他嘢 |
| `--max-num-seqs 8` | 上游以 8 路為設計點；超過唔會更快只會每路更慢 |
| `cpu_threads [32,28]` | 上游實測機嘅值；細核數 CPU（如 8C/16T）屬超額訂閱，上游結論「改线程数冇穩定提升」— 照用做基線，可自行 A/B |
| `engram_storage ssd` + `engram_cache_gib 4` + `host_cache_gib 8` | MoE 专家缓存放 NVMe + 釘住 8GiB 主機 RAM；GPU 顯存留俾 KV |
| `dspark` 投機解码 3 tokens | 上游測過改草稿數冇穩定提升，照用 3 |
| `VLLM_ENGINE_READY_TIMEOUT_S=36000` | 510GB 冷讀入顯存可要幾個鐘 — 防 timeout 誤殺 |
| `--restart no` | 建議手動管理重啟（`docker start dsv41`） |
| cache bind mount（可選） | 加 `-v /path/cache:/tmp/dsv41-cache -v /path/triton:/tmp/dsv41-triton` → 第二次啟動唔使重新編譯 kernel，快好多 |

## 步驟 5：首次啟動 + warmup

```bash
docker logs -f dsv41     # 模型載入可要 30–90 分鐘（510GB 冷讀）
```

見到 startup complete 之後，**先發 8 個 dummy 請求**先至到正常速度：

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

## 步驟 6：性能預期

見 README 嘅實測表。單流 80 tok/s 未達成屬 170HX 物理極限（冇 NVLink + PCIe Gen2），唔好期待配置層面 magic。

## FAQ

| 症狀 | 處理 |
|---|---|
| 得 1–3 張卡啟動即敗 | PP4 需要恰好 4 張 — 冇 workaround |
| 服務喺 load 模型前就死 | 步驟 3 驗證：`engram_symbols=0` → 鏡像缺 Engram SSD 鏈接，要由 build.sh 重新 build |
| CUDA OOM | 確認張卡冇被其他進程佔用；唔好降 `--gpu-memory-utilization` 以下試錯 |
| 啟動 timeout / OOM on RAM | `free -g` 睇 available ≥236GiB；停其他大應用 |
| 想開 P2P 快啲 | 唔好 — 上游實測基線就係禁 P2P，開住會出數據錯誤 |
| dmesg 見到 PCIe AER 可糾正錯誤 | 上游實測機都有，未見數據損壞；屬可觀察範圍 |
| 切返其他模型 | `docker stop dsv41 && docker start <你其他模型容器>` |
