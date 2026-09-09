# FreeToken Dockerfile for Qwen3.8-Flash-Next-NVFP4
# CUDA 13 devel base (FreeToken needs r580+ driver / CUDA 13 toolkit for JIT kernels)
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

# python3.12 + git + curl + uv (git needed for source install; python3.12-dev for C++ extension builds)
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
      ca-certificates git curl python3.12 python3.12-dev python3.12-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install uv (standalone; also gives python management)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /app
# Install FreeToken from source (CLI `ft`)
RUN git clone --depth 1 https://github.com/FlashML-org/FreeToken.git /app/FreeToken && \
    cd /app/FreeToken && \
    uv venv .venv --python python3.12 && \
    VIRTUAL_ENV=/app/FreeToken/.venv uv pip install -e ".[accel]"

ENV PATH="/app/FreeToken/.venv/bin:${PATH}"

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 1919
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD [ "curl", "-f", "http://localhost:1919/health" ]
ENTRYPOINT [ "/app/entrypoint.sh" ]
