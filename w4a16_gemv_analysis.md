# W4A16 GEMV Kernel 分析与实现方案

## 1. 问题定义

**W4A16 GEMV**：
- **W**: INT4 权重矩阵 `[M, K]`
- **A**: FP16 激活向量 `[K, 1]`（batch=1）
- **输出**: FP16 向量 `[M, 1]`

计算：`Y[M,1] = W[M,K] × A[K,1]`

## 2. 为什么需要专用 GEMV Kernel？

### 2.1 CUTLASS 55 Example 的限制

CUTLASS `55_hopper_int4_bf16_gemm` 使用 TileShape `<128, 128, 64>`：

```cpp
using TileShape = Shape<_128, _128, cute::Int<TileShapeK>>;  // M=128, N=128, K=64
```

当 N=1 时的问题：

| 方面 | GEMM (N=128) | GEMV (N=1) |
|------|-------------|------------|
| TileShape | `<128, 128, 64>` | N=1 无法匹配 |
| Tensor Core 利用 | 高效（wgmma 需要 N≥8） | N=1 几乎无法利用 |
| 计算特性 | Compute-bound | **Memory-bound** |
| K-reduction | 无需跨线程规约 | 需要 warp shuffle |
| 并行利用率 | 100% | ~1/128 (浪费) |

### 2.2 GEMV 的特点

1. **Memory-bound**：瓶颈在权重加载带宽，不是计算
2. **K-reduction**：需要跨线程规约 K 维度
3. **不用 Tensor Core**：batch=1 时 CUDA core 更高效

## 3. 业界高性能方案调研

### 3.1 主要实现

| 方案 | 特点 | 链接 |
|------|------|------|
| **Machete** (vLLM) | wgmma + TMA + warp specialization，batch≥32 性能好 | [Neural Magic Blog](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) |
| **QServe** | W4A8KV4，compute-aware weight reordering | [GitHub](https://github.com/nyunAI/qserve) |
| **AutoAWQ** | AWQ 算法，2x 加速 | [GitHub](https://github.com/casper-hansen/AutoAWQ) |
| **gemlite** | 简单易懂的 GEMV 实现 | [PyTorch AO](https://github.com/pytorch/ao/issues/697) |
| **cuda_hgemv** | 纯 FP16 GEMV 优化方法 | [GitHub](https://github.com/Bruce-Lee-LY/cuda_hgemv) |

### 3.2 关键发现

- **小 batch (< 32)**：需要专门的 GEMV kernel，Tensor Core 不适用
- **Dequantization 开销**：INT4→FP16 转换需要 20-90% 额外开销
- **Memory-bound**：小 batch 时，量化模型的优势在于减少内存带宽

## 4. CUTLASS 现有资源

### 4.1 相关文件

| 文件 | 用途 |
|------|------|
| `examples/55_hopper_mixed_dtype_gemm/` | INT4×BF16 GEMM (Hopper) |
| `examples/91_fp4_gemv/` | FP4 GEMV (Blackwell SM100) |
| `include/cutlass/gemm/kernel/gemv_blockscaled.h` | GEMV kernel 模板 |
| `include/cutlass/numeric_conversion.h` | INT4→FP16 转换器 |

### 4.2 INT4→FP16 转换（已实现）

CUTLASS 在 `numeric_conversion.h` (Line 5842-5983) 已实现：

```cpp
// 使用方式
NumericArrayConverter<cutlass::half_t, cutlass::int4b_t, 32> converter;
Array<half_t, 32> fp16_result = converter(int4_source);
```

转换原理：
1. `prmt.b32` 指令进行字节重排
2. `lop3.b32` 指令进行 XOR/AND 组合操作
3. `hfma` 指令完成最终转换（构造 FP16 magic number 然后减法）

### 4.3 Weight Shuffle 优化

55 example 使用 `[0,2,4,6,1,3,5,7]` 顺序重排 INT4 权重：

```cpp
using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>; // order [0,2,4,6,1,3,5,7]
```

好处：多个 nibble 可以并行提取和转换，减少指令数。

## 5. 高效 GEMV Kernel 设计

### 5.1 线程组织

```
Thread Block: (kThreadsPerRow=16, kThreadsPerCol=8) = 128 threads
- kThreadsPerRow: 负责 K 维度并行加载 + shuffle reduction
- kThreadsPerCol: 负责 M 维度不同行

Grid: (ceil(M / kThreadsPerCol), 1, batch)
```

### 5.2 核心配置

```cpp
// 数据类型
using ElementA = cutlass::half_t;           // Activation: FP16
using ElementB = cutlass::int4b_t;          // Weight: INT4
using ElementC = cutlass::half_t;           // Output: FP16
using ElementAccumulator = float;           // 累加器: FP32
using ElementScale = cutlass::half_t;       // Scale: FP16

// 线程配置
static constexpr int kThreadCount = 128;
static constexpr int kThreadsPerRow = 16;   // K-reduction threads
static constexpr int kThreadsPerCol = 8;    // M-output threads
static constexpr int kElementsPerAccess = 32; // 128 bits / 4 bits
```

### 5.3 K-Reduction（Warp Shuffle）

```cpp
// Butterfly reduction across kThreadsPerRow threads
CUTLASS_PRAGMA_UNROLL
for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
    accum += __shfl_xor_sync(0xFFFFFFFF, accum, mask, 32);
}
// Only threadIdx.x == 0 writes output
```

### 5.4 主循环伪代码

```cpp
// 每个 thread block 处理 kThreadsPerCol 个 M 行
for (int m_idx = blockIdx.x * kThreadsPerCol + threadIdx.y;
     m_idx < M;
     m_idx += gridDim.x * kThreadsPerCol) {

    float accum = 0.0f;

    // 沿 K 维度迭代
    for (int k_tile = 0; k_tile < K; k_tile += kTileK) {
        // 1. 加载 activation A[k:k+kTileK] 到 smem (所有 threads 共享)
        // 2. 加载 weight B[m_idx, k:k+kTileK]
        // 3. 加载 scale factor
        // 4. Dequant INT4 → FP16
        // 5. 累加: accum += A * B * scale
    }

    // Warp shuffle reduction
    for (int mask = kThreadsPerRow/2; mask > 0; mask >>= 1) {
        accum += __shfl_xor_sync(0xFFFFFFFF, accum, mask);
    }

    // 只有 threadIdx.x == 0 写输出
    if (threadIdx.x == 0 && m_idx < M) {
        output[m_idx] = static_cast<ElementC>(accum);
    }
}
```

## 6. 优化策略

### Phase 1: 基础实现
- 正确性优先
- 使用 shared memory 缓存 activation
- INT4 dequant + scale
- Warp shuffle reduction

### Phase 2: 性能优化
- Double buffering (2 buffer × 4 stages)
- cp.async 异步加载
- 寄存器复用
- Weight shuffle 预处理（offline）

### Phase 3: 扩展功能
- 支持 group-wise scale
- 支持 zero-point
- 支持 batched GEMV

## 7. 预期性能

对于 Memory-bound GEMV：
- 理论带宽利用率目标: >80%
- INT4 比 FP16 权重减少 4x 内存带宽需求
- 预期 2-4x 加速 vs FP16 GEMV

## 8. 文件规划

```
examples/92_hopper_int4_fp16_gemv/
├── 92_hopper_int4_fp16_gemv.cu    # 主程序
├── CMakeLists.txt
└── README.md
```

## 9. 参考资料

- [Machete Kernel - Red Hat Developer](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
- [PyTorch AO Issue #697](https://github.com/pytorch/ao/issues/697)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [QServe](https://github.com/nyunAI/qserve)
- [cuda_hgemv](https://github.com/Bruce-Lee-LY/cuda_hgemv)
- [vLLM INT4 W4A16](https://docs.vllm.ai/en/latest/features/quantization/int4/)
