# Machete-style W4A16 GEMM

A standalone implementation of mixed-precision GEMM (INT4 weights, BF16 activations) for NVIDIA Hopper GPUs, inspired by the Machete kernel from vLLM/Neural Magic.

## Overview

This implementation computes `Y = X @ W^T` where:
- **X**: Activation matrix (BF16, RowMajor)
- **W**: Weight matrix (INT4, quantized with per-group scales)
- **Y**: Output matrix (FP16)

### Key Optimizations

1. **Transpose Trick**: Compute `Y^T = W^T @ X^T` to put quantized weights in registers (required for efficient dequantization)

2. **Weight Pre-shuffling**: Offline reorder INT4 weights to enable 128-bit shared memory loads instead of multiple 8-bit loads

3. **TMA + Warp Specialization**: Use Hopper's Tensor Memory Accelerator with producer/consumer warp separation for optimal memory/compute overlap

4. **Value Shuffle Layout**: `[0,2,4,6,1,3,5,7]` interleaving for efficient INT4→BF16 conversion using `prmt` instructions

## Building

```bash
cd machete
make
```

## Running

```bash
# Single test (M=1, typical for single-token decode)
./machete_gemm_example --m=1 --n=2048 --k=2048 --g=128

# Sweep multiple sizes
make sweep

# LLaMA-7B FFN sizes
make llama
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--m` | M dimension (batch size / sequence length) | 1 |
| `--n` | N dimension (output features) | 2048 |
| `--k` | K dimension (input features) | 2048 |
| `--g` | Group size for quantization scales | 128 |
| `--mode` | 0=convert, 1=scale, 2=scale+zero | 1 |
| `--iterations` | Profiling iterations | 100 |
| `--warmup` | Warmup iterations | 10 |

## Requirements

- CUDA 12.0+
- NVIDIA Hopper GPU (H100, H200, etc.)
- CUTLASS 3.5+

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Tensors                             │
│  X: (M, K) BF16    W: (N, K) INT4    Scale: (N, K/g) BF16   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Offline Weight Pre-packing                      │
│  - Reorder for 128-bit aligned loads                        │
│  - Value shuffle for efficient INT4→BF16 conversion          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GEMM Kernel                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Producer   │    │  Consumer   │    │  Epilogue   │      │
│  │   Warps     │───▶│   Warps     │───▶│   (TMA)     │      │
│  │  (TMA Load) │    │  (wgmma)    │    │             │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                              │
│  Pipeline Stages: Auto-tuned for SMEM capacity               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output: Y (M, N) FP16                     │
└─────────────────────────────────────────────────────────────┘
```

## Performance Notes

- For M=1 (single-token decode), performance is memory-bound
- For larger M (prefill), compute becomes the bottleneck
- Group size 128 provides good accuracy/performance tradeoff
- Per-channel quantization (g=K) is more compute-efficient but less accurate

## References

- [CUTLASS 3.5 Mixed-Dtype GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm)
- [Machete: Mixed-input GEMM kernel for vLLM](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
- [Neural Magic Blog](https://neuralmagic.com/)
