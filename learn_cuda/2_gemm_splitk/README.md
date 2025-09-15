# Split-K GEMM 详解

Split-K 是一种 GEMM 优化技术，通过将 K 维度分割成多个部分并行计算，特别适用于 K 维度很大的情况。

## 什么是 Split-K？

在标准 GEMM 中：
```
C[M×N] = A[M×K] × B[K×N]
```

Split-K 将 K 维度分成 P 个部分：
```
K = K₀ + K₁ + ... + Kₚ₋₁

C = A[M×K₀] × B[K₀×N] +
    A[M×K₁] × B[K₁×N] +
    ... +
    A[M×Kₚ₋₁] × B[Kₚ₋₁×N]
```

## 工作原理

```
标准 GEMM:
┌─────────┐     ┌─────────┐     ┌─────────┐
│    A    │  ×  │    B    │  =  │    C    │
│  M × K  │     │  K × N  │     │  M × N  │
└─────────┘     └─────────┘     └─────────┘

Split-K GEMM (P=4):
┌─────┐         ┌─────┐         ┌─────────┐
│ A₀  │    ×    │ B₀  │    →    │   C₀    │
├─────┤         ├─────┤         ├─────────┤
│ A₁  │    ×    │ B₁  │    →    │   C₁    │  → Reduction → C
├─────┤         ├─────┤         ├─────────┤
│ A₂  │    ×    │ B₂  │    →    │   C₂    │
├─────┤         ├─────┤         ├─────────┤
│ A₃  │    ×    │ B₃  │    →    │   C₃    │
└─────┘         └─────┘         └─────────┘
M×(K/4)         (K/4)×N           M × N
```

## 优势

1. **更好的并行性**
   - 多个线程块可以同时处理不同的 K 分片
   - 减少线程块间的同步开销

2. **改善负载均衡**
   - 当 M×N 较小而 K 很大时，Split-K 能更好地利用 GPU

3. **减少共享内存压力**
   - 每个分片处理更小的 K 维度
   - 降低每个线程块的共享内存需求

4. **提高占用率**
   - 更多的线程块可以同时运行
   - 更好地隐藏内存延迟

## 适用场景

Split-K 在以下情况下特别有效：

- **K >> M, N**: K 维度远大于 M 和 N
- **瘦长矩阵**: 例如 M=128, N=128, K=8192
- **批量小矩阵**: 多个小矩阵的批处理
- **内存带宽受限**: 通过并行化减少内存访问瓶颈

## CUTLASS 实现

CUTLASS 提供了两种 Split-K 实现：

### 1. GemmSplitKParallel
```cpp
using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    ReductionOp,
    SwizzleThreadBlock
>;
```

### 2. 关键参数

- `split_k_slices`: K 维度分片数量
- `ReductionOp`: 用于合并部分结果的归约操作
- 通常使用 `ReduceAdd` 进行求和

## 性能调优

### 选择最佳分片数

```cpp
// 经验法则
int optimal_slices = K / (ThreadblockShape::kK * stages);

// 确保能整除
if (K % optimal_slices != 0) {
    optimal_slices = find_nearest_divisor(K, optimal_slices);
}
```

### 性能考虑因素

1. **分片数过少**：无法充分利用并行性
2. **分片数过多**：归约开销增加
3. **不能整除**：导致负载不均衡

## 示例代码结构

```cpp
// 1. 定义 Split-K GEMM 类型
using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<...>;

// 2. 设置参数（包括分片数）
GemmSplitK::Arguments args(
    problem_size,
    ref_A, ref_B, ref_C, ref_D,
    epilogue_params,
    split_k_slices  // 分片数量
);

// 3. 执行 kernel
GemmSplitK gemm_op;
cutlass::Status status = gemm_op(args);
```

## 性能对比

典型性能提升（M=1024, N=1024）：

| K 维度 | Regular GEMM | Split-K (8 slices) | Speedup |
|--------|--------------|-------------------|---------|
| 1024   | 100%         | 95%               | 0.95x   |
| 4096   | 100%         | 110%              | 1.10x   |
| 8192   | 100%         | 125%              | 1.25x   |
| 16384  | 100%         | 140%              | 1.40x   |

## 实验建议

1. **测试不同的分片数**
   ```bash
   ./gemm_splitk 1024 1024 8192 4   # 4 slices
   ./gemm_splitk 1024 1024 8192 8   # 8 slices
   ./gemm_splitk 1024 1024 8192 16  # 16 slices
   ```

2. **观察 K 维度的影响**
   ```bash
   ./gemm_splitk 1024 1024 1024 8   # Small K
   ./gemm_splitk 1024 1024 4096 8   # Medium K
   ./gemm_splitk 1024 1024 16384 8  # Large K
   ```

3. **不同矩阵形状**
   ```bash
   ./gemm_splitk 128 128 8192 8     # Skinny matrices
   ./gemm_splitk 4096 4096 512 8    # Wide matrices
   ```

## 高级优化

### 1. 动态分片选择
```cpp
int auto_select_slices(int M, int N, int K) {
    if (K < 2048) return 1;      // No split for small K
    if (K < 4096) return 2;
    if (K < 8192) return 4;
    if (K < 16384) return 8;
    return 16;
}
```

### 2. 混合策略
- 对于批量 GEMM，可以结合 Split-K 和批处理
- 使用 Stream-K 进一步优化

### 3. 内存布局优化
- 考虑使用不同的内存布局减少归约开销
- 优化部分结果的存储位置

## 调试技巧

1. **验证正确性**：始终与非 Split-K 版本对比
2. **性能分析**：使用 nvprof/nsight 分析瓶颈
3. **检查占用率**：确保足够的线程块并发执行

## 总结

Split-K 是优化大 K 维度 GEMM 的有效技术：
- ✅ 适合 K >> M, N 的场景
- ✅ 提高 GPU 利用率
- ✅ CUTLASS 提供了易用的实现
- ⚠️ 需要根据问题规模调整分片数
- ⚠️ 归约操作会带来额外开销

通过合理配置，Split-K 可以在特定场景下提供显著的性能提升。