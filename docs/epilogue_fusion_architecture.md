# CUTLASS Epilogue Fusion 架构详解

本文档详细解释 SM90 Epilogue 和 Fusion 系统的完整架构。

## 1. 总体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CUTLASS SM90 Epilogue + Fusion 完整架构                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    用户代码                                               │   │
│  │                                                                                          │   │
│  │  // 选择融合操作类型                                                                      │   │
│  │  using FusionOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<                    │   │
│  │      cutlass::epilogue::thread::ReLU,  // 激活函数                                       │   │
│  │      half_t,                            // 输出类型                                       │   │
│  │      float,                             // 计算类型                                       │   │
│  │      half_t                             // Bias 类型                                      │   │
│  │  >;                                                                                       │   │
│  │                                                                                          │   │
│  │  // 配置 Epilogue                                                                        │   │
│  │  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<           │   │
│  │      Sm90TmaWarpSpecialized<4, 4, 2, false, false>,  // DispatchPolicy                   │   │
│  │      TileShape, EpilogueTile,                                                            │   │
│  │      ElementC, StrideC, ElementD, StrideD,                                               │   │
│  │      FusionCallbacks<..., FusionOp, ...>,            // 融合回调                         │   │
│  │      ...                                                                                 │   │
│  │  >;                                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                   │
│                                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Template Dispatch Layer                                      │   │
│  │                                                                                          │   │
│  │  fusion/operations.hpp          fusion/callbacks.hpp       sm90_callbacks_*.hpp          │   │
│  │  ┌─────────────────┐           ┌─────────────────┐       ┌─────────────────────┐        │   │
│  │  │ FusionOperation │           │ FusionCallbacks │       │ FusionCallbacks     │        │   │
│  │  │ (元数据定义)     │ ────────► │ (分发接口)       │ ─────►│ 特化 (EVT 实现)     │        │   │
│  │  │                 │           │                 │       │                     │        │   │
│  │  │ • LinearComb    │           │ 匹配 Policy +   │       │ 继承自 Sm90EVT<...> │        │   │
│  │  │ • LinCombBias   │           │ Operation       │       │                     │        │   │
│  │  │ • LinCombEltAct │           │                 │       │                     │        │   │
│  │  └─────────────────┘           └─────────────────┘       └─────────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                   │
│                                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              EVT (Expression Visitor Tree)                                │   │
│  │                                                                                          │   │
│  │  示例: LinCombPerRowBiasEltAct = ReLU(α * acc + β * C + bias)                           │   │
│  │                                                                                          │   │
│  │                          Sm90Compute<ReLU>                    ← 激活函数                 │   │
│  │                                │                                                         │   │
│  │                    Sm90Compute<plus>                          ← 加 bias                  │   │
│  │                   /                    \                                                 │   │
│  │       Sm90RowBroadcast            Sm90Compute<multiply_add>   ← β*C + α*acc             │   │
│  │           (bias)                 /         |         \                                   │   │
│  │                       Sm90Scalar    Sm90SrcFetch   Sm90EVT<multiplies>                  │   │
│  │                          (β)           (C)        /              \                       │   │
│  │                                            Sm90Scalar      Sm90AccFetch                  │   │
│  │                                               (α)             (acc)                      │   │
│  │                                                                                          │   │
│  │  EVT 节点类型:                                                                           │   │
│  │  • Leaf nodes: Sm90AccFetch, Sm90SrcFetch, Sm90ScalarBroadcast, Sm90RowBroadcast        │   │
│  │  • Internal nodes: Sm90Compute<Op> (Op = plus, multiplies, ReLU, GELU, etc.)            │   │
│  │  • Output nodes: Sm90AuxStore (可选的辅助输出)                                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                   │
│                                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          CollectiveEpilogue 执行引擎                                      │   │
│  │                                                                                          │   │
│  │    ┌─────────────────────────┐              ┌─────────────────────────┐                 │   │
│  │    │   Producer Load Warp    │              │  Consumer Store Warps   │                 │   │
│  │    │        (pld)            │              │        (cst)            │                 │   │
│  │    │                         │              │                         │                 │   │
│  │    │  • TMA load C           │    SMEM      │  • SMEM → Register      │                 │   │
│  │    │  • TMA load bias        │ ──────────►  │  • EVT.visit() 计算     │                 │   │
│  │    │  • TMA load scale       │   Pipeline   │  • Register → SMEM      │                 │   │
│  │    │                         │              │  • TMA store D          │                 │   │
│  │    └─────────────────────────┘              └─────────────────────────┘                 │   │
│  │                                                                                          │   │
│  │    Pipeline 同步 (mbarrier):                                                             │   │
│  │    • Full Barrier: Producer → Consumer (数据就绪)                                        │   │
│  │    • Empty Barrier: Consumer → Producer (空间释放)                                       │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                   │
│                                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                 硬件层 (SM90/Hopper)                                      │   │
│  │                                                                                          │   │
│  │    GMEM ◄────── TMA ──────► SMEM ◄────── Register File                                  │   │
│  │                              ▲                                                           │   │
│  │                              │                                                           │   │
│  │                         mbarrier (64-bit)                                                │   │
│  │                    ┌──────────────────────┐                                              │   │
│  │                    │ Phase │ PendingTX │ Arrival │                                       │   │
│  │                    │ 1-bit │  ~20-bit  │ ~20-bit │                                       │   │
│  │                    └──────────────────────┘                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 具体示例：GEMM + Bias + ReLU

### 2.1 用户代码示例

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>

// 定义 GEMM 问题
// D = ReLU(alpha * (A @ B) + beta * C + bias)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementBias = cutlass::half_t;
using ElementCompute = float;  // 累加和计算使用 FP32

// Step 1: 选择融合操作
using EpilogueOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
    cutlass::epilogue::thread::ReLU,  // 激活函数
    ElementD,                          // 输出类型
    ElementCompute,                    // 计算类型
    ElementBias,                       // Bias 类型
    ElementC,                          // Source (C) 类型
    ElementCompute                     // Scalar (alpha/beta) 类型
>;

// Step 2: 使用 CollectiveBuilder 构建 Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,           // e.g., Shape<_128, _128, _64>
    ClusterShape,        // e.g., Shape<_1, _1, _1>
    EpilogueTileType,
    ElementCompute,      // Accumulator type
    ElementCompute,      // Compute type
    ElementC, LayoutC,
    128 / cutlass::sizeof_bits<ElementC>::value,  // Alignment
    ElementD, LayoutD,
    128 / cutlass::sizeof_bits<ElementD>::value,
    EpilogueOp           // 融合操作
>::CollectiveOp;

// Step 3: 配置运行时参数
typename CollectiveEpilogue::Arguments epilogue_args {
    {
        alpha,                    // ElementCompute alpha
        beta,                     // ElementCompute beta
        nullptr,                  // alpha_ptr (optional)
        nullptr,                  // beta_ptr (optional)
        ptr_bias,                 // ElementBias const* ptr_bias
        {_0{}, _1{}, 0}          // Stride for bias (per-row)
    },
    ptr_C,                        // ElementC const* ptr_C
    {ldC, _1{}, 0},              // StrideC
    ptr_D,                        // ElementD* ptr_D
    {ldD, _1{}, 0}               // StrideD
};
```

### 2.2 EVT 展开过程

上述 `LinCombPerRowBiasEltAct` 在编译时展开为以下 EVT 结构：

```cpp
// 文件: sm90_callbacks_tma_warpspecialized.hpp

// D = ReLU(alpha * acc + beta * C + bias)
using Sm90LinCombPerRowBiasEltAct =
  Sm90EVT<Sm90Compute<ReLU, ElementD, ElementCompute, RoundStyle>,  // 根节点: ReLU
    Sm90EVT<Sm90Compute<plus, ElementCompute, ElementCompute>,       // 加 bias
      Sm90RowBroadcast<ElementBias, Stride<_1, _0, int64_t>>,        // bias (per-row)
      Sm90EVT<Sm90Compute<multiply_add, ElementCompute, ElementCompute>,  // β*C + α*acc
        Sm90ScalarBroadcast<ElementScalar>,                          // β
        Sm90SrcFetch<ElementC>,                                      // C
        Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute>,  // α*acc
          Sm90ScalarBroadcast<ElementScalar>,                        // α
          Sm90AccFetch                                               // acc (累加器)
        >
      >
    >
  >;
```

### 2.3 运行时执行流程

```
时间线
──────────────────────────────────────────────────────────────────────────────►

Producer Load Warp (pld)                    Consumer Store Warps (cst)
        │                                            │
        ▼                                            │
┌───────────────────────┐                            │
│ producer_acquire(s0)  │                            │
│ • wait(empty[0])      │                            │
│ • expect_tx(full[0])  │                            │
└───────────────────────┘                            │
        │                                            │
        ▼                                            │
┌───────────────────────┐                            │
│ pld_callbacks.step()  │                            │
│ • TMA load bias[0]    │                            │
│ • TMA load C[0]       │                            │
└───────────────────────┘                            │
        │                                            │
        │  (TMA 完成, 硬件自动 complete_tx)           │
        │  full_barrier[0] phase flip ──────────────►│
        │                                            ▼
        │                              ┌───────────────────────┐
        │                              │ consumer_wait(s0)     │
        │                              │ • wait(full[0])       │
        │                              └───────────────────────┘
        │                                            │
        │                                            ▼
        │                              ┌───────────────────────┐
        │                              │ S2R: SMEM → Register  │
        │                              │ • copy C to rC        │
        │                              └───────────────────────┘
        │                                            │
        │                                            ▼
        │                              ┌───────────────────────┐
        │                              │ cst_callbacks.previsit│
        │                              │ • copy bias to rBias  │
        │                              └───────────────────────┘
        │                                            │
        │                                            ▼
        │                              ┌───────────────────────────────────┐
        │                              │ cst_callbacks.visit()             │
        │                              │                                   │
        │                              │ for (v = 0; v < FragmentSize; v++)│
        │                              │   // EVT 后序遍历执行:            │
        │                              │   acc_v = rAcc[v]                 │
        │                              │   t1 = α * acc_v                  │
        │                              │   t2 = β * rC[v] + t1             │
        │                              │   t3 = t2 + rBias[row]            │
        │                              │   rD[v] = ReLU(t3)                │
        │                              └───────────────────────────────────┘
        │                                            │
        │                                            ▼
        │                              ┌───────────────────────┐
        │                              │ R2S: Register → SMEM  │
        │                              │ • copy rD to sD       │
        │                              └───────────────────────┘
        │                                            │
◄───────────────────────────────────── │◄────────────┘
        │   consumer_release(s0)                     │
        │   arrive(empty[0])                         ▼
        ▼                              ┌───────────────────────┐
┌───────────────────────┐              │ TMA store D           │
│ producer_acquire(s1)  │              │ • fence_async         │
│ • wait(empty[1])      │              │ • copy sD → GMEM      │
│ (可以继续加载下一个)   │              └───────────────────────┘
└───────────────────────┘
```

## 3. 数据流详解

### 3.1 SMEM 布局示例

假设配置：`StagesC=4, StagesD=4, EpilogueTile=(64, 64), ElementC=half_t`

```
SharedStorage 内存布局:
┌─────────────────────────────────────────────────────────────────────────┐
│                           TensorStorage                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  CollectiveStorage                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  smem_C[4 stages]                                                   ││
│  │  ┌──────────┬──────────┬──────────┬──────────┐                      ││
│  │  │ Stage 0  │ Stage 1  │ Stage 2  │ Stage 3  │                      ││
│  │  │ 64x64    │ 64x64    │ 64x64    │ 64x64    │                      ││
│  │  │ = 8 KB   │ = 8 KB   │ = 8 KB   │ = 8 KB   │  Total: 32 KB        ││
│  │  └──────────┴──────────┴──────────┴──────────┘                      ││
│  ├─────────────────────────────────────────────────────────────────────┤│
│  │  smem_D[4 stages] (或与 smem_C 共用 if ReuseSmemC)                  ││
│  │  ┌──────────┬──────────┬──────────┬──────────┐                      ││
│  │  │ Stage 0  │ Stage 1  │ Stage 2  │ Stage 3  │                      ││
│  │  │ 64x64    │ 64x64    │ 64x64    │ 64x64    │  Total: 32 KB        ││
│  │  └──────────┴──────────┴──────────┴──────────┘                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  FusionStorage (for bias, scale, aux)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  smem_bias[4 stages]                                                ││
│  │  ┌──────────┬──────────┬──────────┬──────────┐                      ││
│  │  │ Stage 0  │ Stage 1  │ Stage 2  │ Stage 3  │                      ││
│  │  │ 64 elems │ 64 elems │ 64 elems │ 64 elems │  Total: 512 B        ││
│  │  └──────────┴──────────┴──────────┴──────────┘                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  PipelineStorage                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  full_barrier[4]   = 4 × 8 bytes = 32 bytes                         ││
│  │  empty_barrier[4]  = 4 × 8 bytes = 32 bytes                         ││
│  │                                               Total: 64 bytes        ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
Total SMEM: ~64 KB (without ReuseSmemC) or ~32 KB (with ReuseSmemC)
```

### 3.2 寄存器使用示例

```
Thread 寄存器分配 (假设 FragmentSize=2, MMA 输出 8x8 per thread):

┌─────────────────────────────────────────────────────────────────────────┐
│                      Per-Thread Register Usage                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Accumulator (from mainloop):                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  rAcc[64]  = 64 × FP32 = 256 bytes (from MMA, 8x8 elements)         ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Source C (loaded from SMEM):                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  rC[64]    = 64 × FP16 = 128 bytes (converted to FP32 for compute)  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Bias (loaded from SMEM, broadcast per row):                             │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  rBias[8]  = 8 × FP16 = 16 bytes (8 rows)                           ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Output D (computed result):                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  rD[64]    = 64 × FP16 = 128 bytes (output elements)                ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Scalar constants:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  alpha, beta = 2 × FP32 = 8 bytes                                   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Approximate total: ~550 bytes per thread                                │
│  For 256 threads/CTA: ~140 KB register usage                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Pipeline 状态机

### 4.1 PipelineState 结构

```cpp
template <int Stages>
struct PipelineState {
    int index_ = 0;       // 当前 stage (0 到 Stages-1)
    uint32_t phase_ = 0;  // 当前 phase (0 或 1)
    uint32_t count_ = 0;  // 总迭代次数

    // 推进状态
    void operator++() {
        ++index_;
        ++count_;
        if (index_ == Stages) {
            index_ = 0;      // 循环回 0
            phase_ ^= 1;     // 翻转 phase
        }
    }
};
```

### 4.2 4-Stage Pipeline 状态转换示例

```
时间   Producer State          Consumer State         备注
────   ──────────────          ──────────────         ────
T0     (idx=0, ph=0, cnt=0)    (idx=0, ph=0, cnt=0)   初始状态
       ↓ ++
T1     (idx=1, ph=0, cnt=1)    (idx=0, ph=0, cnt=0)   Producer 领先
       ↓ ++
T2     (idx=2, ph=0, cnt=2)    (idx=0, ph=0, cnt=0)
       ↓ ++                    ↓ ++
T3     (idx=3, ph=0, cnt=3)    (idx=1, ph=0, cnt=1)   Consumer 开始追赶
       ↓ ++                    ↓ ++
T4     (idx=0, ph=1, cnt=4)    (idx=2, ph=0, cnt=2)   Producer 绕回, phase 翻转
       │                       ↓ ++
       │ wait(empty[0], ph=1)  (idx=3, ph=0, cnt=3)   Producer 必须等待
       │                       ↓ ++
       │                       (idx=0, ph=1, cnt=4)   Consumer 绕回
       │                       │
       │◄──── arrive(empty[0]) ┘                      Consumer 释放 stage 0
       ↓ ++
T5     (idx=1, ph=1, cnt=5)    ...                    Producer 可以继续
```

## 5. EVT 节点详解

### 5.1 Leaf 节点

| 节点类型 | 数据来源 | 回调时机 | 示例 |
|---------|---------|---------|------|
| `Sm90AccFetch` | 寄存器 (mainloop) | visit() | 获取累加器值 |
| `Sm90SrcFetch` | SMEM (C matrix) | previsit() 加载 | 获取 C 矩阵值 |
| `Sm90ScalarBroadcast` | 参数 (alpha, beta) | begin() 加载 | 广播标量到所有元素 |
| `Sm90RowBroadcast` | SMEM (bias) | previsit() 加载 | 广播行向量 |
| `Sm90ColBroadcast` | SMEM (scale) | previsit() 加载 | 广播列向量 |

### 5.2 Internal 节点

```cpp
// Sm90Compute 模板
template <
    template<class> class ComputeFn,  // 计算函数 (plus, multiplies, ReLU, etc.)
    class ElementOutput,               // 输出类型
    class ElementCompute,              // 计算类型
    FloatRoundStyle RoundStyle         // 舍入模式
>
struct Sm90Compute {
    // visit() 执行计算
    template <typename Accumulator, typename... Inputs>
    CUTLASS_DEVICE auto visit(
        Accumulator const& acc,        // 累加器片段
        int epi_v, int epi_m, int epi_n,  // 位置
        Inputs const&... inputs        // 子节点输出
    ) {
        // 应用计算函数
        return ComputeFn<ElementCompute>{}(inputs...);
    }
};
```

### 5.3 常用计算函数

```cpp
// 二元操作
template<class T> struct plus        { T operator()(T a, T b) { return a + b; } };
template<class T> struct multiplies  { T operator()(T a, T b) { return a * b; } };
template<class T> struct multiply_add{ T operator()(T a, T b, T c) { return a * b + c; } };

// 激活函数
template<class T> struct ReLU        { T operator()(T x) { return max(T(0), x); } };
template<class T> struct GELU        { T operator()(T x) { return x * 0.5f * (1 + erf(x/sqrt(2))); } };
template<class T> struct SiLU        { T operator()(T x) { return x / (1 + exp(-x)); } };
template<class T> struct Sigmoid     { T operator()(T x) { return 1 / (1 + exp(-x)); } };
```

## 6. 回调函数调用时序

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Fusion Callbacks 调用时序                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Producer Load Warp:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │  pld_callbacks = fusion.get_producer_load_callbacks(args)                      │ │
│  │                                                                                │ │
│  │  pld_callbacks.begin()           // 初始化, 加载 scalar 到寄存器               │ │
│  │  │                                                                             │ │
│  │  for each subtile (epi_m, epi_n):                                              │ │
│  │  │   pld_callbacks.step(barrier, epi_m, epi_n, ...)                            │ │
│  │  │   │                           // TMA 加载 bias, scale 等到 SMEM             │ │
│  │  │   └── TMA load C (if needed)                                                │ │
│  │                                                                                │ │
│  │  pld_callbacks.end()             // 清理 (通常为空)                            │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  Consumer Store Warps:                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │  cst_callbacks = fusion.get_consumer_store_callbacks<RefSrc>(args)             │ │
│  │                                                                                │ │
│  │  cst_callbacks.begin()           // 初始化                                     │ │
│  │  │                                                                             │ │
│  │  for each subtile (epi_m, epi_n):                                              │ │
│  │  │   cst_callbacks.begin_loop(epi_m, epi_n)                                    │ │
│  │  │   │                           // per-subtile 初始化                         │ │
│  │  │   │                                                                         │ │
│  │  │   │   [consumer_wait for C]                                                 │ │
│  │  │   │   [S2R copy C to register]                                              │ │
│  │  │   │                                                                         │ │
│  │  │   cst_callbacks.previsit(epi_m, epi_n, ...)                                 │ │
│  │  │   │                           // SMEM → Register: bias, scale               │ │
│  │  │   │                                                                         │ │
│  │  │   for each fragment v:                                                      │ │
│  │  │   │   result[v] = cst_callbacks.visit(acc[v], v, epi_m, epi_n)              │ │
│  │  │   │   │                       // ★ 核心计算: EVT 树求值 ★                   │ │
│  │  │   │   │                       // D = ReLU(α*acc + β*C + bias)               │ │
│  │  │   │                                                                         │ │
│  │  │   [R2S copy result to SMEM]                                                 │ │
│  │  │                                                                             │ │
│  │  │   cst_callbacks.tma_store(epi_m, epi_n, ...)                                │ │
│  │  │                               // TMA store 前的回调 (可选 aux store)         │ │
│  │  │                                                                             │ │
│  │  │   [TMA store D]                                                             │ │
│  │                                                                                │ │
│  │  cst_callbacks.end()             // 最终处理 (如 absmax reduction)             │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 7. 性能调优参数

| 参数 | 默认值 | 影响 | 调优建议 |
|------|-------|------|---------|
| `StagesC` | 4 | C 加载 pipeline 深度 | 增加可隐藏延迟，但消耗更多 SMEM |
| `StagesD` | 4 | D 存储 pipeline 深度 | 通常与 StagesC 相同 |
| `FragmentSize` | 2 | 向量化宽度 | 2 for FP16, 1 for FP32 |
| `ReuseSmemC` | false | C/D 共用 SMEM | 节省 SMEM 但需更复杂同步 |
| `DelayTmaStore` | false | 延迟 TMA store | 提高指令交错，但增加延迟 |
| `EpilogueTile` | (64, 64) | Subtile 大小 | 平衡并行度和寄存器压力 |

## 8. 相关源文件

| 文件 | 内容 |
|------|------|
| `epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` | CollectiveEpilogue 实现 |
| `epilogue/fusion/operations.hpp` | FusionOperation 定义 |
| `epilogue/fusion/callbacks.hpp` | FusionCallbacks 分发接口 |
| `epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` | EVT 特化 |
| `epilogue/fusion/sm90_visitor_*.hpp` | EVT 节点实现 |
| `pipeline/sm90_pipeline.hpp` | Pipeline 实现 |
| `arch/barrier.h` | mbarrier PTX 封装 |
