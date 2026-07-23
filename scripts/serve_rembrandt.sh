#!/usr/bin/env bash
# Rembrandt 680M (gfx1035) / RDNA2 working recipe — what upstream docs miss.
#
# Truth:
#  - Compile/run as gfx1030 (Tensile exists); HSA_OVERRIDE maps 680M → 10.3.0
#  - NEVER default bf16 on RDNA2 (software emulates → ~1–4 tok/s). This fork
#    forces float16 first for gfx103x via RocmPlatform.supported_dtypes.
#  - UMA: keep --gpu-memory-utilization low (0.5–0.7)
#
# Prereq: ROCm/PyTorch HIP that can see /dev/kfd (stock ROCm + override, or
# TheRock / community stack). Without HIP, this script exits with residual.
set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1030}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Avoid AITER CDNA kernels on consumer RDNA
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
MEM_UTIL="${MEM_UTIL:-0.55}"
MAX_LEN="${MAX_LEN:-2048}"
DTYPE="${DTYPE:-float16}"

echo "== rembrandt vLLM serve =="
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "  MODEL=$MODEL dtype=$DTYPE mem_util=$MEM_UTIL max_len=$MAX_LEN port=$PORT"

python3 - <<'PY'
import sys
try:
    import torch
    print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
    print("cuda_is_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))
        try:
            print("gcn", torch.cuda.get_device_properties(0).gcnArchName)
        except Exception as e:
            print("gcn_err", e)
    else:
        print("RESIDUAL: no HIP/CUDA device — install ROCm PyTorch first", file=sys.stderr)
        sys.exit(2)
except Exception as e:
    print("RESIDUAL: torch import failed:", e, file=sys.stderr)
    sys.exit(2)
PY

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-model-len "$MAX_LEN" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enforce-eager \
  "$@"
