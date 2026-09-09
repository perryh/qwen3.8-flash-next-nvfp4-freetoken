# FreeToken Dockerfile for Qwen3.8-Flash-Next-NVFP4
# CUDA 13 devel base (FreeToken needs r580+ driver / CUDA 13 toolkit for JIT kernels)
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
      ca-certificates python3.12 python3.12-venv python3-pip curl git uv && \
    rm -rf /var/lib/apt/lists/* || true

# uv via standalone installer (Ubuntu 24.04 repos may lack the uv package)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /app
# Install FreeToken from source (CLI `ft`)
RUN git clone --depth 1 https://github.com/FlashML-org/FreeToken.git /app/FreeToken && \
    cd /app/FreeToken && \
    uv venv .venv && \
    VIRTUAL_ENV=/app/FreeToken/.venv uv pip install -e ".[accel]"

ENV PATH="/app/FreeToken/.venv/bin:${PATH}"
ENV HF_HOME=/home/perryh/.cache/huggingface

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 1919
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD [ "curl", "-f", "http://localhost:1919/health" ]
ENTRYPOINT [ "/app/entrypoint.sh" ]
