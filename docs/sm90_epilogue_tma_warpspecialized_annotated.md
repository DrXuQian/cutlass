# SM90 Epilogue TMA Warp-Specialized 详解

本文档对 `sm90_epilogue_tma_warpspecialized.hpp` 中的核心代码进行详细注释说明。

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SM90 Epilogue TMA Warp-Specialized                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Collective Epilogue                            │ │
│  │                                                                        │ │
│  │  模板参数:                                                             │ │
│  │  • DispatchPolicy: Sm90TmaWarpSpecialized<StagesC, StagesD, ...>      │ │
│  │  • CtaTileMNK: CTA 处理的 tile 大小 (CTA_M, CTA_N, CTA_K)             │ │
│  │  • EpilogueTile: Epilogue subtile 大小 (EPI_TILE_M, EPI_TILE_N)       │ │
│  │  • FusionCallbacks: 融合操作回调 (LinearCombination, Bias, etc.)      │ │
│  │  • CopyOpG2S/S2G/S2R/R2S: TMA 和寄存器拷贝操作                        │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │  Producer Load Warp  │    │  Consumer Store Warps │                       │
│  │  (pld_callbacks)     │    │  (cst_callbacks)      │                       │
│  │                      │    │                       │                       │
│  │  • TMA load C        │    │  • SMEM → Register    │                       │
│  │  • TMA load bias     │    │  • Fusion compute     │                       │
│  │  • TMA load aux      │    │  • Register → SMEM    │                       │
│  │                      │    │  • TMA store D        │                       │
│  └──────────────────────┘    └───────────────────────┘                       │
│            │                           ▲                                     │
│            │      SMEM Pipeline        │                                     │
│            └───────────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 模板参数详解

```cpp
template <
  int StagesC_,           // C 矩阵加载的 pipeline stage 数量 (SMEM buffer 数量)
  int StagesD_,           // D 矩阵存储的 pipeline stage 数量
  int FragmentSize_,      // 向量化操作的 fragment 大小 (通常为 2)
  bool ReuseSmemC_,       // 是否复用 C 的 SMEM 存储 D (节省 SMEM)
  bool DelayTmaStore_,    // 是否延迟 TMA store (提高指令交错)
  class CtaTileMNK_,      // CTA tile 形状 (CTA_M, CTA_N, CTA_K)
  class EpilogueTile_,    // Epilogue subtile 形状 (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,        // C 矩阵元素类型 (void 表示不加载 C)
  class StrideC_,         // C 矩阵 stride (M, N, L)
  class ElementD_,        // D 矩阵元素类型 (void 表示不存储 D)
  class StrideD_,         // D 矩阵 stride (M, N, L)
  class FusionCallbacks_, // 融合回调类型 (定义 D = f(acc, C, bias, ...))
  class CopyOpG2S_,       // GMEM → SMEM 拷贝操作 (TMA load)
  class SmemLayoutAtomC_, // C 在 SMEM 中的 layout atom
  class CopyOpS2R_,       // SMEM → Register 拷贝操作
  class CopyOpS2G_,       // SMEM → GMEM 拷贝操作 (TMA store)
  class SmemLayoutAtomD_, // D 在 SMEM 中的 layout atom
  class CopyOpR2S_,       // Register → SMEM 拷贝操作
  class CopyAtomC_,       // C 的拷贝 atom (用于 partition)
  class CopyOpR2R_        // Register → Register 变换 (可选)
>
```

## 3. SharedStorage 结构

```cpp
struct SharedStorage {
  struct TensorStorage {
    // Collective 存储: C 和 D 的 SMEM buffer
    // 根据配置选择不同的存储策略:
    // - CollectiveStorageWithC:    C 和 D 分开存储
    // - CollectiveStorageWithoutC: 只存储 D (C 为 void)
    // - CollectiveStorageReuseC:   C 和 D 复用同一块 SMEM (union)
    CollectiveStorage collective;

    // Fusion 回调的共享存储 (bias, scale 等)
    FusionStorage thread;
  } tensors;

  // Pipeline 的 barrier 存储
  PipelineStorage pipeline;
};
```

**SMEM 布局示意:**

```
┌─────────────────────────────────────────────────────────────┐
│                        SharedStorage                         │
├─────────────────────────────────────────────────────────────┤
│  TensorStorage                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ CollectiveStorage                                       ││
│  │ ┌───────────────────┐  ┌───────────────────┐           ││
│  │ │     smem_C        │  │     smem_D        │           ││
│  │ │ (EPI_M, EPI_N,    │  │ (EPI_M, EPI_N,    │           ││
│  │ │     StagesC)      │  │     StagesD)      │           ││
│  │ └───────────────────┘  └───────────────────┘           ││
│  │ 或 ReuseSmemC 模式: smem_C 和 smem_D 共用同一块内存     ││
│  ├─────────────────────────────────────────────────────────┤│
│  │ FusionStorage (bias, scale, aux 等的 SMEM 缓存)         ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  PipelineStorage                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ full_barrier_[StagesC]   // 64-bit mbarrier 数组        ││
│  │ empty_barrier_[StagesC]  // 用于 Producer-Consumer 同步 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 4. load() 函数详解

```cpp
// Producer Load Warp 执行此函数
// 负责将 C 矩阵和辅助数据从 GMEM 加载到 SMEM
template<...>
CUTLASS_DEVICE auto
load(
    LoadPipeline load_pipeline,           // Pipeline 接口 (barrier 操作)
    LoadPipelineState load_pipe_producer_state,  // Producer 当前状态 (stage, phase)
    ProblemShapeMNKL problem_shape_mnkl,  // 问题形状 (M, N, K, L)
    TileShapeMNK tile_shape_MNK,          // Tile 形状
    TileCoordMNKL tile_coord_mnkl,        // 当前 tile 坐标
    TiledMma tiled_mma,                   // MMA 配置 (用于 partition)
    int thread_idx,                       // 线程 ID
    TensorStorage& shared_tensors,        // SMEM 存储
    int subtile_idx=-1)                   // 可选: 只处理特定 subtile
{
```

**load() 执行流程:**

```cpp
// Step 1: 创建 GMEM tensor 并 partition
Tensor mC_mn = params.tma_load_c.get_tma_tensor(make_shape(M,N,L));
Tensor gC = local_tile(mC, take<0,2>(CtaTileMNK{}), coord_shape);

// Step 2: 划分为 epilogue subtiles
Tensor gC_epi = flat_divide(gC, EpilogueTile{});  // (EPI_M, EPI_N, EPI_M_tiles, EPI_N_tiles)
Tensor sC_epi = make_tensor(..., SmemLayoutC{});  // (EPI_M, EPI_N, PIPE_C)

// Step 3: TMA partition
ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);  // Source (GMEM)
Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);  // Destination (SMEM)

// Step 4: 获取 fusion callbacks
auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
pld_callbacks.begin();  // 预处理 (如果有)

// Step 5: 循环处理每个 subtile
for (int epi_n = 0; epi_n < size<3>(gC_epi); ++epi_n) {
  for (int epi_m = 0; epi_m < size<2>(gC_epi); ++epi_m) {

    // 5a: 获取 barrier 并 acquire (等待 Consumer 释放 SMEM)
    uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
    load_pipeline.producer_acquire(load_pipe_producer_state);
    //   └─→ empty_barrier[stage].wait(phase)  等待空间可用
    //   └─→ full_barrier[stage].arrive_and_expect_tx(bytes)  设置期望字节数

    // 5b: Fusion callback step (加载 bias, scale 等)
    pld_callbacks.step(tma_barrier, epi_m, epi_n, ...);

    // 5c: TMA 加载 C (如果需要)
    if (issue_tma_load && is_C_load_needed) {
      copy(params.tma_load_c.with(*tma_barrier, mcast_mask),
           bGS_gC(_,_,_,epi_m,epi_n),           // Source: GMEM
           bGS_sC(_,_,_,load_pipe_producer_state.index()));  // Dest: SMEM
      // TMA 完成后硬件自动执行: full_barrier[stage].complete_tx(bytes)
    }

    // 5d: 推进 pipeline state
    ++load_pipe_producer_state;
  }
}
```

## 5. store() 函数详解

```cpp
// Consumer Store Warps (MMA warps) 执行此函数
// 负责: 从 SMEM 读 C → 融合计算 → 写回 SMEM → TMA store D
template<...>
CUTLASS_DEVICE auto
store(
    LoadPipeline load_pipeline,           // Load pipeline (用于 consumer_wait)
    LoadPipelineState load_pipe_consumer_state,   // Consumer 状态
    StorePipeline store_pipeline,         // Store pipeline (用于 TMA store 同步)
    StorePipelineState store_pipe_producer_state, // Store producer 状态
    ProblemShapeMNKL problem_shape_mnkl,
    TileShapeMNK tile_shape_MNK,
    TileCoordMNKL tile_coord_mnkl,
    AccumulatorTensor accumulators,       // 来自 mainloop 的累加器 (Register)
    TiledMma tiled_mma,
    int thread_idx,
    TensorStorage& shared_tensors,
    int subtile_idx=-1)
{
```

**store() 执行流程:**

```cpp
// Step 1: 创建 GMEM/SMEM tensors
Tensor gD_epi = flat_divide(gD, EpilogueTile{});
Tensor sC_epi = make_tensor(..., SmemLayoutC{});
Tensor sD_epi = make_tensor(..., SmemLayoutD{});

// Step 2: 创建 TiledCopy 并 partition
// S2R: SMEM → Register (读 C)
TiledCopy tiled_s2r = make_tiled_copy_S(Copy_Atom<CopyOpS2R, SmemElementC>{}, ...);
Tensor tSR_sC = thread_s2r.partition_S(sC_epi);  // SMEM source
Tensor tSR_rC = thread_s2r.retile_D(...);         // Register dest

// R2S: Register → SMEM (写 D)
TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<CopyOpR2S, SmemElementD>{}, ...);
Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);  // Register source (累加器)
Tensor tRS_sD = thread_r2s.partition_D(sD_epi);       // SMEM dest

// S2G: SMEM → GMEM (TMA store D)
ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);
Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);

// Step 3: 获取 fusion callbacks
auto cst_callbacks = fusion_callbacks.get_consumer_store_callbacks<RefSrc>(cst_args);
cst_callbacks.begin();

// Step 4: 主循环 - 处理每个 subtile
for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n) {
  for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {

    // 4a: Begin loop callback
    cst_callbacks.begin_loop(epi_m, epi_n);

    // 4b: 等待 Producer 加载完成
    if (is_producer_load_needed) {
      load_pipeline.consumer_wait(load_wait_state);
      //   └─→ full_barrier[stage].wait(phase)  等待数据就绪

      // 从 SMEM 拷贝 C 到 Register
      if (is_C_load_needed) {
        copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
      }
    }

    // 4c: Previsit callback (从 SMEM 加载 bias/scale 到寄存器)
    cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

    // 4d: 释放 load pipeline (如果不复用 SMEM)
    if (is_producer_load_needed && !ReuseSmemC) {
      load_pipeline.consumer_release(load_pipe_consumer_state);
      //   └─→ empty_barrier[stage].arrive()  通知 Producer 空间已释放
      ++load_pipe_consumer_state;
    }

    // 4e: 核心计算 - visit callback
    // 向量化循环处理每个 fragment
    for (int epi_v = 0; epi_v < size(tRS_rCompute_frg); ++epi_v) {
      // 调用融合回调执行: D = α * acc + β * C + bias + ...
      tRS_rCompute_frg(epi_v) = cst_callbacks.visit(
        tRS_rAcc_frg_mn(r2s_v + epi_v),  // 输入: 累加器 fragment
        epi_v, epi_m, epi_n              // 位置信息
      );
      // visit 内部执行 EVT (Expression Visitor Tree):
      //   Sm90Compute<multiply_add>
      //     ├─ Sm90ScalarBroadcast (β)
      //     ├─ Sm90SrcFetch (C)
      //     └─ Sm90Compute<multiplies>
      //          ├─ Sm90ScalarBroadcast (α)
      //          └─ Sm90AccFetch (acc)
    }

    // 4f: 如果需要类型转换 (FP32 → FP16 等)
    if constexpr (!IsDirectR2S) {
      // Convert: ElementCompute → SmemElementD
      transform(tRS_rCompute_frg, tRS_rD_frg, ...);
    }

    // 4g: 将结果从 Register 拷贝到 SMEM
    copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_pipe_producer_state.index()));

    // 4h: TMA store D
    tma_store_fn(epi_m, epi_n);
    // 内部执行:
    //   fence_view_async_shared();  // 确保 SMEM 写入可见
    //   synchronize();              // 线程同步
    //   copy(params.tma_store_d, bSG_sD, bSG_gD);  // TMA store
    //   store_pipeline.producer_commit(state);    // tma_store_arrive()
    //   ++store_pipe_producer_state;

    ++load_wait_state;
  }
}

// Step 5: 结束回调
cst_callbacks.end();
```

## 6. Pipeline 同步详解

```
时序图: Producer Load Warp 与 Consumer Store Warps 的同步

Producer Load Warp                    Consumer Store Warps
       │                                     │
       ▼                                     │
  producer_acquire(state)                    │
    ├─ wait(empty_barrier, phase)            │
    │    ← 等待 Consumer 释放空间            │
    └─ arrive_and_expect_tx(full_barrier)    │
         ← 告知期望的传输字节数              │
       │                                     │
       ▼                                     │
  TMA Load C → SMEM                          │
       │                                     │
       ▼                                     │
  (TMA 完成, 硬件自动)                        │
  complete_tx(full_barrier)                  │
    ← pending_tx 减到 0                      │
    ← phase 翻转!                            │
       │                                     │
       │                                     ▼
       │                            consumer_wait(state)
       │                              └─ wait(full_barrier, phase)
       │                                   ← 等待数据就绪
       │                                     │
       │                                     ▼
       │                            SMEM → Register (读 C)
       │                            Fusion compute (visit)
       │                            Register → SMEM (写 D)
       │                            TMA Store D
       │                                     │
       │                                     ▼
       │                            consumer_release(state)
       │                              └─ arrive(empty_barrier)
       │                                   ← arrival_count 减到 0
       │                                   ← phase 翻转!
       │                                     │
       ▼                                     │
  producer_acquire(next_state)               │
    ← 现在可以写入这个 stage                 │
```

## 7. Fusion Callbacks 调用点

```cpp
// 在 load() 中 (Producer Load Warp):
auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
pld_callbacks.begin();                    // 预处理
pld_callbacks.step(barrier, epi_m, epi_n, ...);  // 每个 subtile

// 在 store() 中 (Consumer Store Warps):
auto cst_callbacks = fusion_callbacks.get_consumer_store_callbacks<RefSrc>(cst_args);
cst_callbacks.begin();                    // 预处理
cst_callbacks.begin_loop(epi_m, epi_n);   // 每个 subtile 开始
cst_callbacks.previsit(...);              // visit 前 (SMEM → Reg 加载 bias 等)
cst_callbacks.visit(acc_frg, ...);        // 核心计算! D = f(acc, C, ...)
cst_callbacks.reduce(...);                // 可选: reduction 操作
cst_callbacks.postreduce(...);            // 可选: reduction 后处理
cst_callbacks.tma_store(...);             // TMA store 前
cst_callbacks.end();                      // 结束处理
```

## 8. 关键数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            数据流总览                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GMEM                    SMEM                     Register                   │
│  ┌─────┐                ┌─────┐                  ┌─────────────┐            │
│  │  C  │ ─── TMA ─────► │sC[i]│ ──── S2R ──────► │    rC       │            │
│  └─────┘    (G2S)       └─────┘    (SMEM→Reg)    └─────────────┘            │
│                                                         │                    │
│                                                         ▼                    │
│                                                  ┌─────────────┐            │
│  From Mainloop ─────────────────────────────────►│    rAcc     │            │
│  (Accumulators)                                  └─────────────┘            │
│                                                         │                    │
│                                                         ▼                    │
│                                           ┌──────────────────────┐          │
│  GMEM (bias等)──TMA──►SMEM──S2R──►Reg ──►│   Fusion Compute     │          │
│                                           │  D = α*acc + β*C     │          │
│                                           │      + bias + ...    │          │
│                                           └──────────────────────┘          │
│                                                         │                    │
│                                                         ▼                    │
│  GMEM                    SMEM                     Register                   │
│  ┌─────┐                ┌─────┐                  ┌─────────────┐            │
│  │  D  │ ◄─── TMA ───── │sD[j]│ ◄──── R2S ────── │    rD       │            │
│  └─────┘    (S2G)       └─────┘    (Reg→SMEM)    └─────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 9. 配置参数影响

| 参数 | 影响 |
|------|------|
| `StagesC` | C 加载的 pipeline 深度，更多 stage = 更好的延迟隐藏，更多 SMEM |
| `StagesD` | D 存储的 pipeline 深度 |
| `FragmentSize` | 向量化宽度，通常为 2 (FP16x2) |
| `ReuseSmemC` | true: C 和 D 共用 SMEM，省内存但需要更复杂的同步 |
| `DelayTmaStore` | true: 延迟 TMA store 一次迭代，提高指令交错 |
| `EpilogueTile` | Subtile 大小，影响循环次数和 SMEM 使用 |
