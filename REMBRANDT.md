# vllm-rembrandt — Radeon 680M (gfx1035) that actually works

Upstream ROCm product docs + HIP allowlists treat Rembrandt iGPU as nonexistent.
Their CMake lists **gfx1030** (Navi21 dGPU) but not **gfx1035** (680M). Their
`supported_dtypes` prefers **bfloat16**, which RDNA2 **software-emulates**
→ abysmal tok/s ([vllm#38107](https://github.com/vllm-project/vllm/issues/38107), AMD staff).

## The recipe (not optional)

| Knob | Value | Why |
|------|-------|-----|
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Map 680M → gfx1030 Tensile path |
| `PYTORCH_ROCM_ARCH` | `gfx1030` | Build/run kernels that exist |
| `--dtype` | `float16` | No HW bf16 on RDNA2 |
| `--gpu-memory-utilization` | `0.5–0.7` | UMA / shared RAM |
| `VLLM_ROCM_USE_AITER` | `0` | CDNA kernels break consumer |

## Fork changes

1. `HIP_SUPPORTED_ARCHS` includes **gfx1035** (compile-time allow).
2. `RocmPlatform.supported_dtypes()` → **float16 first** on gfx103x (auto dtype).
3. `check_if_supports_dtype` **rejects bf16** on gfx103x with a clear error.

## Serve

```bash
# needs ROCm-enabled PyTorch that sees /dev/kfd
cd ~/vllm-rembrandt
pip install -e .   # or use a ROCm venv
./scripts/serve_rembrandt.sh
# other terminal:
./scripts/smoke_rembrandt.sh
```

## Honest residual

If rocBLAS/Tensile still abort on your ROCm point release, the stack is broken
upstream for iGPUs — this fork cannot invent missing library files. Use
**llama.cpp Vulkan** / Ollama on the same silicon for production GGUF until
HIP math libs catch up. This fork still does the software fixes upstream skipped.
