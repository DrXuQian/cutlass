# Raster Order 和 Swizzle 对缓存命中的影响分析

## 1. 基础概念

### GEMM 计算中的数据访问模式
```
C = A × B
[M×N] = [M×K] × [K×N]
```

每个 CTA（threadblock）计算输出矩阵 C 的一个 tile：
- 需要从 A 矩阵读取对应的行块
- 需要从 B 矩阵读取对应的列块

## 2. Raster Order 实现分析

根据代码 `sm90_tile_scheduler_group.hpp:403-408`：

```cpp
if (raster_order == RasterOrder::AlongN) {
    return {minor_work_idx, major_work_idx, ...};  // (m, n) 顺序
} else {
    return {major_work_idx, minor_work_idx, ...};  // (n, m) 顺序
}
```

### 2.1 Raster Order 对缓存的影响

#### AlongM（沿 M 维度优先）
```
假设 4×4 个 CTAs，执行顺序：
时间 →
t0: CTA[0,0] → 需要 A[0,:], B[:,0]
t1: CTA[1,0] → 需要 A[1,:], B[:,0] ← B 重用！
t2: CTA[2,0] → 需要 A[2,:], B[:,0] ← B 重用！
t3: CTA[3,0] → 需要 A[3,:], B[:,0] ← B 重用！
t4: CTA[0,1] → 需要 A[0,:], B[:,1]
...

L2 Cache 状态：
- B[:,0] 在 t0-t3 期间驻留缓存
- 连续 4 个 CTA 共享相同的 B 数据
- 适合 M >> N 的情况
```

#### AlongN（沿 N 维度优先）
```
执行顺序：
t0: CTA[0,0] → 需要 A[0,:], B[:,0]
t1: CTA[0,1] → 需要 A[0,:], B[:,1] ← A 重用！
t2: CTA[0,2] → 需要 A[0,:], B[:,2] ← A 重用！
t3: CTA[0,3] → 需要 A[0,:], B[:,3] ← A 重用！
t4: CTA[1,0] → 需要 A[1,:], B[:,0]
...

L2 Cache 状态：
- A[0,:] 在 t0-t3 期间驻留缓存
- 连续 4 个 CTA 共享相同的 A 数据
- 适合 N >> M 的情况
```

## 3. Swizzle 实现分析

根据代码 `sm90_tile_scheduler_group.hpp:388-396`：

```cpp
// Swizzle 算法核心
offset = cluster_id & ((1 << log_swizzle_size) - 1);
extra = cluster_id >> log_swizzle_size;

cluster_idx_minor_div_swizzle = extra / curr_group_cluster_blk_major;
cluster_idx_major = extra % curr_group_cluster_blk_major;

cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;
```

### 3.1 Swizzle 的数学原理

对于 cluster_id 序列 [0, 1, 2, 3, 4, 5, 6, 7, ...]：

**log_swizzle_size = 0 (无 swizzle)**：
```
cluster_idx = cluster_id
映射：0→0, 1→1, 2→2, 3→3, 4→4, ...
```

**log_swizzle_size = 1 (swizzle_size = 2)**：
```
offset = cluster_id & 1  // 取最低位
extra = cluster_id >> 1   // 右移一位

重排结果：
0 → offset=0, extra=0 → idx=0
1 → offset=1, extra=0 → idx=1
2 → offset=0, extra=1 → idx=2
3 → offset=1, extra=1 → idx=3
4 → offset=0, extra=2 → idx=4
...

2D 布局（假设 4×4）：
原始：         Swizzle=2：
[0  1  2  3]  [0  2  1  3]
[4  5  6  7]  [4  6  5  7]
[8  9 10 11]  [8 10  9 11]
[12 13 14 15] [12 14 13 15]
```

### 3.2 Swizzle 对 L2 缓存的影响

#### 无 Swizzle 时的问题
```
时间线（16 个 SM，每个时刻运行 16 个 CTA）：
t0: CTA[0-15] → 都在访问矩阵的前 16 个连续 tiles
t1: CTA[16-31] → 都在访问矩阵的下 16 个连续 tiles

问题：
- 16 个 CTA 同时竞争相邻的数据区域
- L2 缓存行竞争激烈
- 缓存抖动（thrashing）
```

#### Swizzle=2 的改善
```
时间线：
t0: CTA[0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15]
    → 访问的 tiles 在空间上更分散

优势：
- 减少了对相邻缓存行的同时竞争
- 更好的缓存行利用率
- 降低缓存抖动
```

## 4. 组合效果分析

### 4.1 最佳实践场景

#### 场景 1：M=16384, N=1024, K=4096（M >> N）
```
推荐配置：
- raster_order=AlongM
- swizzle_size=4 或 8

原因：
- AlongM 让多个 CTA 共享 B 矩阵数据（B 较小）
- 大 swizzle 防止多个 CTA 竞争相同的 A 矩阵区域
```

#### 场景 2：M=1024, N=16384, K=4096（N >> M）
```
推荐配置：
- raster_order=AlongN
- swizzle_size=4 或 8

原因：
- AlongN 让多个 CTA 共享 A 矩阵数据（A 较小）
- 大 swizzle 防止多个 CTA 竞争相同的 B 矩阵区域
```

#### 场景 3：M=4096, N=4096, K=4096（方阵）
```
推荐配置：
- raster_order=Heuristic（自动选择）
- swizzle_size=2 或 4

原因：
- 对称的问题规模，两种遍历顺序效果相似
- 中等 swizzle 提供平衡的缓存利用
```

## 5. 性能调优建议

### 5.1 L2 缓存容量考虑

Hopper H100 的 L2 缓存：50MB

```cpp
// 来自 sm90_tile_scheduler_group.hpp:275-276
auto problem_blocks_m = round_up(ctas_along_m,
    (1 << log_swizzle_size) * cluster_shape.m());
```

这确保了 swizzle 模式与 cluster 形状对齐。

### 5.2 调优策略

1. **先确定 raster order**：
   - 根据 M/N 比例选择
   - 测量不同配置的内存带宽利用率

2. **再调整 swizzle size**：
   - 从 swizzle_size=1 开始测试
   - 逐步增加到 2, 4, 8
   - 观察 L2 缓存命中率变化

3. **使用 CUTLASS Profiler 验证**：
```bash
# 测试所有组合
for raster in M N heuristic; do
  for swizzle in 1 2 4 8; do
    ./cutlass_profiler --operation=Gemm \
      --m=$M --n=$N --k=$K \
      --raster_order=$raster \
      --swizzle_size=$swizzle \
      --profiling-iterations=100
  done
done
```

## 6. CUTLASS Profiler 详细分析

### 6.1 CUTLASS Profiler 概述

CUTLASS Profiler 是一个性能分析和调优工具，用于：
- 测试不同的 GEMM 配置组合
- 自动搜索最优参数
- 比较 CUTLASS 和 cuBLAS 性能
- 生成性能报告

### 6.2 基本使用方法

#### 基础命令
```bash
# 基本 GEMM 测试
./cutlass_profiler --operation=Gemm \
                   --m=4096 --n=4096 --k=4096 \
                   --A=f16 --B=f16 --C=f32 --D=f32 \
                   --profiling-iterations=100
```

#### 支持的参数类型
```bash
# 数据类型
--A, --B, --C, --D : f16, bf16, f32, f64, s8, u8, s32, e4m3, e5m2

# 布局
--layout_A, --layout_B, --layout_C : RowMajor (T), ColumnMajor (N)

# 累加器精度
--accumulator : f32, f64, s32
```

### 6.3 高级配置选项

#### 架构和内核选择
```bash
# 指定计算能力
--compute-capability=90  # Hopper
--compute-capability=80  # Ampere

# 指定内核类型
--kernels=gemm           # 普通 GEMM
--kernels=sparse         # 稀疏 GEMM
--kernels=conv2d         # 卷积

# 使用特定内核库
--library-algo-mode=all           # 所有算法
--library-algo-mode=default       # 默认算法
--library-algo-mode=heuristic     # 启发式选择
```

#### 性能调优参数
```bash
# Tile 形状配置
--tile_m=128,256         # CTA tile M 维度
--tile_n=128,256         # CTA tile N 维度
--tile_k=32,64           # K 维度分块

# Cluster 形状（Hopper）
--cluster_m=1,2          # Cluster M 维度
--cluster_n=1,2          # Cluster N 维度
--cluster_k=1            # Cluster K 维度

# Warp 形状
--warp_m=64,128
--warp_n=64,128
--warp_k=16,32
```

### 6.4 Raster Order 和 Swizzle 配置

```bash
# 测试不同的 raster order
--raster_order=AlongM     # M 维度优先
--raster_order=AlongN     # N 维度优先
--raster_order=Heuristic  # 自动选择

# 测试不同的 swizzle size
--swizzle_size=1         # 无 swizzle
--swizzle_size=2         # 2-way swizzle
--swizzle_size=4         # 4-way swizzle
--swizzle_size=8         # 8-way swizzle

# 组合测试
--raster_order=AlongM,AlongN,Heuristic \
--swizzle_size=1,2,4,8
```

### 6.5 批量测试和性能扫描

#### 参数扫描脚本
```bash
#!/bin/bash
# sweep_parameters.sh

M_SIZES="1024 2048 4096 8192 16384"
N_SIZES="1024 2048 4096 8192 16384"
K_SIZES="1024 2048 4096"
RASTER_ORDERS="AlongM AlongN Heuristic"
SWIZZLE_SIZES="1 2 4 8"

for M in $M_SIZES; do
  for N in $N_SIZES; do
    for K in $K_SIZES; do
      for RASTER in $RASTER_ORDERS; do
        for SWIZZLE in $SWIZZLE_SIZES; do
          echo "Testing M=$M N=$N K=$K raster=$RASTER swizzle=$SWIZZLE"
          ./cutlass_profiler \
            --operation=Gemm \
            --m=$M --n=$N --k=$K \
            --A=f16 --B=f16 --C=f32 --D=f32 \
            --raster_order=$RASTER \
            --swizzle_size=$SWIZZLE \
            --profiling-iterations=100 \
            --output=results_${M}_${N}_${K}_${RASTER}_${SWIZZLE}.csv
        done
      done
    done
  done
done
```

#### 使用配置文件
```json
// gemm_configs.json
{
  "operations": [{
    "operation": "Gemm",
    "A": "f16", "B": "f16", "C": "f32", "D": "f32",
    "m": [1024, 2048, 4096],
    "n": [1024, 2048, 4096],
    "k": [1024, 2048],
    "raster_order": ["AlongM", "AlongN", "Heuristic"],
    "swizzle_size": [1, 2, 4, 8],
    "tile_m": [128, 256],
    "tile_n": [128, 256],
    "tile_k": [32, 64]
  }]
}
```

```bash
# 使用配置文件运行
./cutlass_profiler --configuration=gemm_configs.json
```

### 6.6 性能分析和报告

#### 输出格式选项
```bash
# CSV 输出（便于分析）
--output=results.csv
--append=true            # 追加到现有文件

# 详细输出
--verbose=true           # 显示详细信息
--report=summary         # 摘要报告
--report=verbose         # 详细报告
--report=csv             # CSV 格式

# 性能指标
--metrics=gflops         # GFLOPS
--metrics=runtime_ms     # 运行时间
--metrics=bytes          # 内存带宽
```

#### 性能比较
```bash
# 与 cuBLAS 比较
--reference=cublas       # 使用 cuBLAS 作为基准
--compare=relative       # 相对性能
--compare=absolute       # 绝对性能

# 示例：比较 CUTLASS 和 cuBLAS
./cutlass_profiler \
  --operation=Gemm \
  --m=4096 --n=4096 --k=4096 \
  --A=f16 --B=f16 --C=f32 \
  --reference=cublas \
  --kernels=all \
  --report=verbose
```

### 6.7 达到 cuBLAS 性能的策略

#### 步骤 1：基准测试
```bash
# 首先获取 cuBLAS 基准性能
./cutlass_profiler \
  --operation=Gemm \
  --m=8192 --n=8192 --k=8192 \
  --A=f16 --B=f16 --C=f32 \
  --reference=cublas \
  --profiling-iterations=100 \
  --output=cublas_baseline.csv
```

#### 步骤 2：自动调优
```bash
# 让 profiler 自动搜索最优配置
./cutlass_profiler \
  --operation=Gemm \
  --m=8192 --n=8192 --k=8192 \
  --A=f16 --B=f16 --C=f32 \
  --kernels=all \
  --library-algo-mode=all \
  --profiling-iterations=20 \
  --output=cutlass_all_kernels.csv
```

#### 步骤 3：精细调优
```bash
# 基于初步结果，精细调优特定参数
./cutlass_profiler \
  --operation=Gemm \
  --m=8192 --n=8192 --k=8192 \
  --A=f16 --B=f16 --C=f32 \
  --tile_m=128,256 \
  --tile_n=128,256 \
  --tile_k=32,64 \
  --cluster_m=1,2 \
  --cluster_n=1,2 \
  --raster_order=AlongM,AlongN,Heuristic \
  --swizzle_size=1,2,4,8 \
  --stages=3,4,5 \
  --profiling-iterations=50 \
  --output=fine_tuning_results.csv
```

### 6.8 实际优化案例

#### 案例 1：大矩阵优化（M=16384, N=16384, K=4096）
```bash
# 问题：初始性能只有 cuBLAS 的 70%
# 解决方案：调整 tile 大小和 swizzle

# 测试不同配置
./cutlass_profiler \
  --operation=Gemm \
  --m=16384 --n=16384 --k=4096 \
  --A=f16 --B=f16 --C=f32 \
  --tile_m=256 --tile_n=256 --tile_k=64 \
  --swizzle_size=8 \
  --raster_order=Heuristic \
  --stages=4 \
  --profiling-iterations=100

# 结果：性能提升到 cuBLAS 的 95%
```

#### 案例 2：批量小矩阵（M=512, N=512, K=512, batch=1000）
```bash
# 问题：小矩阵批量计算效率低
# 解决方案：调整 tile 和 warp 配置

./cutlass_profiler \
  --operation=GemmBatched \
  --m=512 --n=512 --k=512 \
  --batch=1000 \
  --A=f16 --B=f16 --C=f32 \
  --tile_m=64 --tile_n=64 --tile_k=32 \
  --warp_m=32 --warp_n=32 --warp_k=16 \
  --swizzle_size=2 \
  --profiling-iterations=100
```

### 6.9 性能分析脚本

```python
# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results.csv')

# 找出最佳配置
best_config = df.loc[df['GFLOPS'].idxmax()]
print(f"最佳配置：")
print(f"  Raster Order: {best_config['raster_order']}")
print(f"  Swizzle Size: {best_config['swizzle_size']}")
print(f"  Tile Shape: {best_config['tile_m']}x{best_config['tile_n']}x{best_config['tile_k']}")
print(f"  性能: {best_config['GFLOPS']:.2f} GFLOPS")

# 分析 swizzle 影响
swizzle_perf = df.groupby('swizzle_size')['GFLOPS'].mean()
plt.figure(figsize=(10, 6))
swizzle_perf.plot(kind='bar')
plt.xlabel('Swizzle Size')
plt.ylabel('Average GFLOPS')
plt.title('Swizzle Size vs Performance')
plt.show()

# 分析 raster order 影响
raster_perf = df.groupby('raster_order')['GFLOPS'].mean()
plt.figure(figsize=(10, 6))
raster_perf.plot(kind='bar')
plt.xlabel('Raster Order')
plt.ylabel('Average GFLOPS')
plt.title('Raster Order vs Performance')
plt.show()
```

### 6.10 常见性能问题和解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 性能显著低于 cuBLAS | Tile 配置不当 | 使用 profiler 扫描不同 tile 大小 |
| L2 缓存命中率低 | Swizzle 配置不当 | 增加 swizzle_size |
| 内存带宽利用率低 | Raster order 不匹配 | 根据 M/N 比例调整 raster order |
| 小矩阵性能差 | Tile 太大 | 减小 tile 大小，调整 warp 配置 |
| 大矩阵性能差 | 缓存竞争 | 增加 swizzle，优化 stages |

## 7. 总结

- **Raster Order** 控制 CTA 的宏观遍历模式，影响 A/B 矩阵的缓存驻留
- **Swizzle** 在微观层面打乱 CTA 执行顺序，减少缓存竞争
- **CUTLASS Profiler** 是达到 cuBLAS 性能的关键工具，通过系统化的参数扫描和分析可以找到最优配置
- 最优配置取决于问题规模、硬件缓存容量和具体应用场景
- 通过合理使用 profiler 的自动调优功能，大多数情况下可以达到 cuBLAS 90% 以上的性能