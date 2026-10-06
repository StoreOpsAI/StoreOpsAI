#!/usr/bin/env bash
set -euo pipefail

# WSL의 사용자 가상환경에서 Qwen3-VL FP8 OpenAI 호환 서버를 실행한다.
VENV_PATH="${QWEN_VLLM_VENV:-$HOME/qwen-vllm}"
MODEL_NAME="${QWEN_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
PORT="${QWEN_VLLM_PORT:-8001}"

if [[ ! -x "$VENV_PATH/bin/vllm" ]]; then
  echo "vLLM 실행 파일을 찾을 수 없습니다: $VENV_PATH/bin/vllm" >&2
  exit 1
fi

# uv/pip로 설치한 CUDA Toolkit도 FlashInfer JIT가 찾도록 명시한다.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in "$VENV_PATH"/lib/python*/site-packages/nvidia/cu13; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_HOME="$candidate"
      break
    fi
  done
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA nvcc를 찾을 수 없습니다. Qwen vLLM 가상환경 또는 CUDA Toolkit을 확인하세요." >&2
  exit 1
fi

# pip CUDA wheel은 cudart의 versioned library만 제공하므로 linker용 이름을 만든다.
mkdir -p "$CUDA_HOME/lib64/stubs"
ln -sfn "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
  ln -sfn /usr/lib/wsl/lib/libcuda.so "$CUDA_HOME/lib64/stubs/libcuda.so"
fi

export CUDA_HOME
export PATH="$VENV_PATH/bin:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

exec "$VENV_PATH/bin/vllm" serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --limit-mm-per-prompt '{"image":3,"video":0}' \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
