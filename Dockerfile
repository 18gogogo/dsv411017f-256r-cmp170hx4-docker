FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=8 \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv build-essential git cmake ninja-build curl ca-certificates \
      && rm -rf /var/lib/apt/lists/* \
      && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
COPY vllm-backport /workspace/vllm-backport

RUN cd /workspace/vllm-backport \
    && uv venv --python 3.12 .venv \
    && VIRTUAL_ENV=/workspace/vllm-backport/.venv uv pip install torch==2.13.0 -q \
    && VIRTUAL_ENV=/workspace/vllm-backport/.venv .venv/bin/python -c "import torch; print('DIAG torch.version.cuda:', repr(torch.version.cuda), 'xpu:', repr(torch.version.xpu))"

RUN cd /workspace/vllm-backport \
    && unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION \
    && VIRTUAL_ENV=/workspace/vllm-backport/.venv \
       uv pip install -e . --torch-backend=auto

ENV PATH="/workspace/vllm-backport/.venv/bin:$PATH"

# 以下為本 repo 加嘅 entrypoint（僅自行 build 嘅鏡像用到；Hub 鏡像照 DEPLOY.md 完整指令）
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
