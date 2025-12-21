# CUTLASS Pipeline 接口到 PTX 指令映射

本文档详细说明 CUTLASS SM90 Pipeline 接口如何映射到底层的 mbarrier PTX 指令。

## 1. 核心概念

### 1.1 双 Barrier 架构

Pipeline 使用双 Barrier 实现生产者-消费者同步：

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Stage [i]                        │
│                                                              │
│   ┌──────────────────┐          ┌──────────────────┐        │
│   │   Full Barrier   │          │  Empty Barrier   │        │
│   │ (数据就绪信号)    │          │ (空间释放信号)    │        │
│   │                  │          │                  │        │
│   │ ClusterTransaction│         │  ClusterBarrier  │        │
│   │     Barrier      │          │                  │        │
│   └──────────────────┘          └──────────────────┘        │
│          ↑                              ↑                   │
│     Producer 写完                  Consumer 用完             │
│     数据后 signal                  数据后 signal             │
└─────────────────────────────────────────────────────────────┘
```

| Barrier | 类型 | 谁 Signal | 谁 Wait | 含义 |
|---------|------|----------|---------|------|
| **Full Barrier** | `ClusterTransactionBarrier` | Producer (TMA) | Consumer | "数据准备好了，可以读" |
| **Empty Barrier** | `ClusterBarrier` | Consumer | Producer | "空间腾出来了，可以写" |

### 1.2 SMEM 中的 Barrier 存储

```cpp
// sm90_pipeline.hpp
struct SharedStorage {
    FullBarrier full_barrier_[Stages];   // ClusterTransactionBarrier (64-bit mbarrier)
    EmptyBarrier empty_barrier_[Stages]; // ClusterBarrier (64-bit mbarrier)
};
```

### 1.3 PipelineState 结构

```cpp
template <int Stages_>
struct PipelineState {
    int index_ = 0;       // 当前 stage (0 ~ Stages-1)，循环取模
    uint32_t phase_ = 0;  // 当前 phase (0 或 1)，每绕回一次翻转
    uint32_t count_ = 0;  // 总迭代次数
};
```

## 2. Pipeline 接口到 PTX 映射总览

| Pipeline 接口 | Barrier 类型 | 底层方法 | PTX 指令 |
|--------------|-------------|---------|---------|
| `producer_acquire` | EmptyBarrier | `wait(phase)` | `mbarrier.try_wait.parity` (spin) |
| `producer_acquire` (leader) | FullBarrier | `arrive_and_expect_tx(bytes)` | `mbarrier.arrive.expect_tx` |
| `producer_commit` | FullBarrier | TMA 硬件触发 | `mbarrier.complete_tx` |
| `producer_get_barrier` | FullBarrier | 返回指针 | (无 PTX) |
| `producer_tail` | EmptyBarrier | 循环 `wait` 所有 stages | `mbarrier.try_wait.parity` |
| `consumer_try_wait` | FullBarrier | `try_wait(phase)` | `mbarrier.try_wait.parity` (单次) |
| `consumer_wait` | FullBarrier | `wait(phase)` | `mbarrier.try_wait.parity` (spin) |
| `consumer_release` | EmptyBarrier | `arrive(cta_id)` | `mbarrier.arrive` |

## 3. Producer 接口详解

### 3.1 producer_acquire

**功能**：等待 stage 空间可用，并设置期望的传输字节数

```cpp
// sm90_pipeline.hpp:512
void producer_acquire(uint32_t stage, uint32_t phase) {
    // Step 1: 等待 Consumer 释放空间
    empty_barrier_ptr_[stage].wait(phase);

    // Step 2: Leader 线程设置期望传输字节数
    if (params_.is_leader) {
        full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
}
```

**PTX (wait - 阻塞式 spin loop):**
```asm
LAB_WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [smem_addr], phase, 0x989680;
    @P1 bra DONE;
    bra LAB_WAIT;
DONE:
```

**PTX (arrive_and_expect_tx):**
```asm
mbarrier.arrive.expect_tx.shared::cta.b64 _, [smem_addr], transaction_bytes;
```

### 3.2 producer_commit

**功能**：TMA 传输完成后，硬件自动触发 barrier 完成

```cpp
// TMA 硬件自动执行，等价于:
full_barrier_ptr_[stage].complete_transaction(bytes);
```

**PTX:**
```asm
mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [smem_addr], transaction_bytes;
```

> **注意**：通常不需要软件显式调用，TMA 指令会自动在传输完成后触发此操作。

### 3.3 producer_get_barrier

**功能**：返回 Full Barrier 指针，用于 TMA descriptor

```cpp
// sm90_pipeline.hpp:555
ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
}
```

**无 PTX 指令**，仅返回 SMEM 地址供 TMA 使用。

### 3.4 producer_tail

**功能**：防止 Producer block 过早退出，等待所有 stages 被 Consumer 释放

```cpp
// sm90_pipeline.hpp:448
void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
        empty_barrier_ptr_[state.index()].wait(state.phase());
        ++state;
    }
}
```

**PTX:** 循环 Stages 次执行 `mbarrier.try_wait.parity`

## 4. Consumer 接口详解

### 4.1 consumer_try_wait

**功能**：非阻塞尝试等待数据就绪

```cpp
// sm90_pipeline.hpp:590
ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
        return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
}
```

**PTX (单次尝试，无 spin):**
```asm
mbarrier.try_wait.parity.shared::cta.b64 P1, [smem_addr], phase;
selp.b32 result, 1, 0, P1;
```

**返回值**：
- `BarrierStatus::WaitDone` (1): 数据已就绪
- `BarrierStatus::WaitAgain` (0): 数据未就绪，需要继续等待

### 4.2 consumer_wait

**功能**：阻塞等待数据就绪

```cpp
// sm90_pipeline.hpp:611
void consumer_wait(uint32_t stage, uint32_t phase) {
    full_barrier_ptr_[stage].wait(phase);
}

// 带 token 版本
void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
        full_barrier_ptr_[stage].wait(phase);
    }
    // 如果 WaitDone，直接跳过
}
```

**PTX (阻塞式 spin loop):**
```asm
LAB_WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [smem_addr], phase, 0x989680;
    @P1 bra DONE;
    bra LAB_WAIT;
DONE:
```

### 4.3 consumer_release

**功能**：通知 Producer 空间已释放

```cpp
// sm90_pipeline.hpp:628
void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
}
```

**PTX (本地 CTA arrive):**
```asm
mbarrier.arrive.shared::cta.b64 _, [smem_addr];
```

**PTX (远程 Cluster arrive):**
```asm
mapa.shared::cluster.u32 remAddr32, smem_addr, cta_id;
mbarrier.arrive.shared::cluster.b64 _, [remAddr32];
```

## 5. mbarrier 64-bit 结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    mbarrier (64-bit)                            │
├─────────────┬─────────────────────┬─────────────────────────────┤
│  Phase Bit  │   Pending TX Count  │     Arrival Count           │
│   (1 bit)   │     (~20 bits)      │      (~20 bits)             │
├─────────────┼─────────────────────┼─────────────────────────────┤
│ 0 或 1      │ 期望的传输字节数     │ 剩余需要 arrive 的线程数     │
│ 每次完成翻转 │ TMA 完成时递减       │ 每次 arrive 递减             │
└─────────────┴─────────────────────┴─────────────────────────────┘

完成条件: Pending TX Count == 0 AND Arrival Count == 0
         → Phase Bit 翻转
```

## 6. 完整工作流程

```
Producer Warp                              Consumer Warp
     │                                          │
     ▼                                          │
producer_acquire(state)                         │
  ├─ wait(empty_barrier[stage], phase)          │
  │    ← 等待 Consumer 释放空间                  │
  └─ arrive_and_expect_tx(full_barrier, bytes)  │
       ← 告知期望多少字节                        │
     │                                          │
     ▼                                          │
发起 TMA 加载                                    │
     │                                          │
     ▼                                          │
TMA 完成 → 硬件自动 complete_tx                  │
     │         ← pending_tx 减到 0               │
     │         ← phase 翻转!                     │
     │                                          ▼
     │                              consumer_try_wait(state)
     │                                ← try_wait(full_barrier, phase)
     │                                          │
     │                                          ▼
     │                              consumer_wait(state, token)
     │                                ← 如果 WaitAgain，继续等待
     │                                          │
     │                                          ▼
     │                              执行 MMA 计算
     │                                          │
     │                                          ▼
     │                              consumer_release(state)
     │                                ← arrive(empty_barrier)
     │                                ← arrival_count 减到 0
     │                                ← phase 翻转!
     ▼                                          │
producer_acquire(next_state)                    │
  ← 现在可以写入这个 stage                       │
```

## 7. 多 Consumer 场景

当有多个 Consumer Warp Group 时：

```cpp
// 初始化时
empty_barrier[stage].init(arrival_count = num_consumers);  // 例如 2

// 运行时
Consumer0: empty_barrier[stage].arrive()  // count: 2→1
Consumer1: empty_barrier[stage].arrive()  // count: 1→0 → phase flip!

// 只有所有 Consumer 都 arrive 后，Producer 才能继续
```

## 8. 源码参考

- Pipeline 实现: `include/cutlass/pipeline/sm90_pipeline.hpp`
- Barrier 封装: `include/cutlass/arch/barrier.h`
- ClusterBarrier: `barrier.h:341-532`
- ClusterTransactionBarrier: `barrier.h:538-690`
