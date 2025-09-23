# CUTLASS 技术详解与示例指南

本文档详细介绍 CUTLASS（CUDA Templates for Linear Algebra Subroutines）库的核心概念、技术细节和示例程序。

## 目录
1. [核心概念](#核心概念)
2. [内存层次结构](#内存层次结构)
3. [Tensor Core 技术](#tensor-core-技术)
4. [示例程序详解](#示例程序详解)
5. [性能优化策略](#性能优化策略)

## 核心概念

### 1. 层次化分解 (Hierarchical Decomposition)

CUTLASS 使用三层嵌套的形状定义来组织计算：

#### ThreadblockShape
- **定义**：线程块处理的矩阵块大小
- **典型值**：`<128, 128, 32>` 表示 M=128, N=128, K=32
- **作用**：决定了一个线程块负责计算的输出矩阵区域
- **选择原则**：
  - 太大：共享内存不足，占用率降低
  - 太小：无法充分利用 GPU 资源
  - 推荐：128×128 或 256×128（取决于架构）

#### WarpShape
- **定义**：一个 Warp（32个线程）处理的矩阵块大小
- **典型值**：`<64, 64, 32>`
- **关系**：ThreadblockShape 必须是 WarpShape 的整数倍
- **作用**：控制 Warp 级别的数据复用和同步

#### InstructionShape
- **定义**：单条 Tensor Core 指令处理的矩阵大小
- **架构相关**：
  - Volta (V100): `<8, 8, 4>`（半精度）
  - Turing (T4): `<8, 8, 16>` 或 `<16, 8, 8>`
  - Ampere (A100): `<16, 8, 16>`（半精度）
- **作用**：直接对应硬件 Tensor Core 指令

### 2. 内存布局 (Memory Layouts)

#### RowMajor（行主序）
```cpp
// 元素按行连续存储
// A[i][j] 的地址 = base + i * ldm + j
using LayoutA = cutlass::layout::RowMajor;
```
- 适用场景：C/C++ 默认布局，适合行遍历

#### ColumnMajor（列主序）
```cpp
// 元素按列连续存储
// A[i][j] 的地址 = base + j * ldm + i
using LayoutB = cutlass::layout::ColumnMajor;
```
- 适用场景：FORTRAN 布局，BLAS 库标准

#### TensorOpMultiplicand（Tensor Core 专用布局）
```cpp
// 针对 Tensor Core 优化的交错布局
using LayoutA = cutlass::layout::TensorOpMultiplicand<
    ElementSizeInBits,  // 元素位宽
    CrosswiseElements   // 交错元素数
>;
```
- 特点：数据重排以匹配 Tensor Core 访问模式
- 优势：减少 bank 冲突，提高吞吐量

### 3. Fragment 与 Iterator

#### Fragment（片段）
- **定义**：每个线程持有的寄存器数组
- **作用**：存储线程负责的矩阵数据片段
- **示例**：
```cpp
// 每个线程持有的累加器片段
typename MmaWarp::FragmentC accum;
accum.clear();  // 初始化为零
```

#### Iterator（迭代器）
- **作用**：系统化地访问内存中的数据
- **类型**：
  - PredicatedTileIterator：带边界检查的迭代器
  - TileIterator：不带边界检查（性能更高）
- **示例**：
```cpp
// 创建访问共享内存的迭代器
typename MmaWarp::IteratorA iter_A(ref_A, extent, lane_id);
```

## 内存层次结构

### 存储层次与延迟
```
寄存器 (Register)     : ~0 cycles    | 每线程私有，最快
共享内存 (Shared)      : ~30 cycles   | 线程块共享，bank 冲突敏感
L1 缓存               : ~100 cycles  | SM 级缓存
L2 缓存               : ~200 cycles  | 全局缓存
全局内存 (Global)      : ~500 cycles  | 最慢，带宽受限
```

### 优化策略
1. **数据复用最大化**：利用共享内存缓存频繁访问的数据
2. **合并内存访问**：连续线程访问连续地址
3. **双缓冲技术**：计算与数据传输重叠
4. **Bank 冲突避免**：使用 padding 或交错访问模式

## Tensor Core 技术

### 工作原理
Tensor Core 是专门的矩阵乘累加单元，一条指令可完成：
```
D = A × B + C
```
其中 A、B、C、D 是小矩阵块。

### 数据类型支持
- **FP16**：半精度，最常用
- **BF16**：Brain Float 16，动态范围更大
- **TF32**：TensorFloat-32，兼容 FP32 范围
- **INT8/INT4**：整数运算，用于量化推理
- **FP64**：双精度（仅 A100 及以上）

### 编程模型
```cpp
// 1. 数据准备：加载到寄存器
FragmentA frag_A;
FragmentB frag_B;
FragmentC accum;

// 2. 执行 Tensor Core 运算
mma_op(accum, frag_A, frag_B, accum);

// 3. 结果存储
store(accum, output_ptr);
```

## 示例程序详解

### Example 0: Basic GEMM（基础矩阵乘法）
**核心技术**：
- 最简单的 GEMM 实现
- 使用 CUTLASS Device API
- 自动选择最优核函数

**关键代码**：
```cpp
// 定义 GEMM 操作类型
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator
>;

// 执行 GEMM
Gemm gemm_op;
gemm_op({problem_size, ref_A, ref_B, ref_C, ref_D, {alpha, beta}});
```

### Example 2: 完整 GEMM（通用矩阵乘法）
**核心技术**：
- 支持任意精度组合
- 流水线优化（多级缓冲）
- Split-K 分解支持

**优化特点**：
- 4 级流水线隐藏内存延迟
- 自适应 tile 大小选择
- 支持混合精度计算

### Example 3: GEMM_128（128-bit 对齐优化）
**核心技术**：
- 强制 128-bit 内存对齐
- 向量化内存访问
- 减少内存事务数

**性能提升**：
- 内存带宽利用率提升 ~20%
- 特别适合大矩阵计算

### Example 4: Tile Iterator（数据迭代器）
**核心技术**：
- 展示如何遍历矩阵 tiles
- Predicated 访问（边界检查）
- Swizzle 模式（避免 bank 冲突）

**应用场景**：
- 自定义数据布局
- 稀疏矩阵处理
- 特殊边界条件

### Example 5: Batched GEMM（批量矩阵乘法）
**核心技术**：
- 批量处理多个小矩阵
- 共享内存复用
- Grid-stride 循环

**优化策略**：
```cpp
// 批量 GEMM 参数
BatchedGemmCoord problem(
    {M, N, K},      // 单个 GEMM 大小
    batch_count     // 批次数量
);
```

### Example 6: Split-K GEMM（K 维度分解）
**核心技术**：
- 将 K 维度分成多个部分
- 并行计算部分和
- 最后规约得到结果

**适用场景**：
- K 维度特别大
- 需要更多并行度
- 减少单个线程块的工作量

### Example 12: GEMM + Bias + ReLU（融合操作）
**核心技术**：
- Epilogue 融合
- 一次核函数完成多个操作
- 减少内存往返

**融合流程**：
```cpp
// D = ReLU(alpha * A * B + beta * C + bias)
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    AlignmentOutput,
    ElementAccumulator,
    ElementCompute
>;
```

### Example 19: Tensor Core Canonical（规范化布局）
**核心技术**：
- Warp 级 Tensor Core 编程
- Canonical 内存布局
- Fragment 操作详解

**关键概念**：
- Lane ID：线程在 Warp 中的索引（0-31）
- Fragment：每个线程的寄存器数据
- MMA 指令：矩阵乘累加操作

### Example 23: GEMM + Operand Reduction（操作数规约）
**核心技术**：
- GEMM 同时进行行/列求和
- 减少额外的规约核函数
- Warp 级规约优化

**应用场景**：
- BatchNorm 计算
- 统计信息收集
- 注意力机制

### Example 35: GEMM + Softmax（Softmax 融合）
**核心技术**：
- 数值稳定的 Softmax 实现
- 两遍扫描算法
- 共享内存缓存中间结果

**算法流程**：
1. 计算每行最大值（数值稳定性）
2. 计算 exp 和求和
3. 归一化得到 Softmax

### Example 36: Gather-Scatter Fusion（聚散操作融合）
**核心技术**：
- 索引式内存访问
- 稀疏矩阵支持
- 不规则数据模式优化

**应用场景**：
- Embedding 查找
- 图神经网络
- 稀疏注意力

### Example 37: GEMM + LayerNorm + GEMM（三重融合）
**核心技术**：
- Transformer FFN 层完整实现
- 三个操作融合成一个核函数
- 统计量计算与归一化

**性能优势**：
- 减少 2 次核函数启动
- 节省中间结果存储
- 提高数据局部性

### Example 39: GEMM + Permute（维度变换）
**核心技术**：
- 输出张量重排
- Layout 转换融合
- 批处理维度交换

**典型用途**：
- NHWC ↔ NCHW 转换
- Attention 中的矩阵转置
- 维度重排优化

## 性能优化策略

### 1. 选择合适的 Tile 大小
```cpp
// 根据问题规模选择
if (M * N < 10000) {
    // 小矩阵：使用小 tile
    using ThreadblockShape = GemmShape<64, 64, 32>;
} else {
    // 大矩阵：使用大 tile
    using ThreadblockShape = GemmShape<128, 256, 32>;
}
```

### 2. 流水线级数优化
```cpp
// 根据共享内存大小选择
constexpr int Stages = (sizeof_shared_memory > 48KB) ? 4 : 2;
```

### 3. 数据预取策略
- 使用 `__pipeline_memcpy_async` (CUDA 11+)
- 双缓冲或多缓冲
- 软件流水线

### 4. Bank 冲突避免
```cpp
// 添加 padding 避免 bank 冲突
__shared__ ElementA smem_A[M][K + 1];  // +1 for padding
```

### 5. 向量化访问
```cpp
// 使用向量类型提高带宽利用率
using AccessType = cutlass::AlignedArray<ElementA, 8>;  // 128-bit 访问
```

## 调试与分析

### 常用工具
1. **Nsight Compute**：核函数性能分析
2. **nvprof/nsys**：应用级性能分析
3. **CUTLASS Profiler**：自动性能调优

### 性能指标
- **算术强度**：FLOPs / Bytes
- **占用率**：活跃 warps / 最大 warps
- **内存效率**：有效带宽 / 理论带宽
- **Tensor Core 利用率**：实际 vs 理论峰值

### 常见问题
1. **性能不达预期**
   - 检查内存对齐
   - 验证 Tensor Core 是否启用
   - 分析 bank 冲突

2. **精度问题**
   - 使用混合精度时注意数值范围
   - 考虑使用 TF32 代替 FP32
   - 必要时使用更高精度累加器

3. **编译错误**
   - 确保 CUDA 版本匹配
   - 检查架构兼容性
   - 验证模板参数合法性

## 总结

CUTLASS 提供了高度优化和可定制的 GPU 线性代数原语，通过：
- 层次化分解实现高效并行
- Tensor Core 加速矩阵运算
- 操作融合减少内存访问
- 模板化设计提供灵活性

掌握这些技术可以实现接近硬件极限的性能。