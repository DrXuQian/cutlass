# CUTLASS Hopper INT8 支持实现指南

## 概述

本指南详细说明了如何在类CUTLASS代码库中添加 Hopper (SM90) INT8 支持。通过分析 CUTLASS 3.x 代码库，我们发现**当已有 FP8 支持时，数据打包布局问题已经得到解决**，因为 FP8 和 INT8 都是 8 位类型，在张量内存加速器 (TMA) 和 warp-group MMA 指令中的处理方式相似。

## 现有代码分析

### CUTLASS 当前支持情况

1. **FP8 支持 (SM89+)**
   - `float_e4m3_t` 和 `float_e5m2_t` 类型定义
   - 完整的 MMA 指令支持：`/include/cute/arch/mma_sm89.hpp`
   - **重要发现：TMA 内部将 FP8 作为 `uint8_t` 处理**
   - 代码位置：[mma_sm89.hpp:64-149](include/cute/arch/mma_sm89.hpp#L64-L149)

2. **INT8 支持 (仅限 SM80/Ampere)**
   - 支持 INT8/UINT8 的 MMA 指令
   - 代码位置：[mma_traits_sm80.hpp:224-415](include/cute/atom/mma_traits_sm80.hpp#L224-L415)
   - 支持组合：`S8S8S32`, `S8U8S32`, `U8S8S32`, `U8U8S32`
   - **缺失：SM90 的 INT8 GMMA 支持**

3. **亚字节整数类型**
   - 已定义 `int4b_t`, `uint4b_t`
   - **未定义 `int8b_t` 或 `uint8b_t`**
   - 代码位置：[integer_subbyte.h:200-225](include/cutlass/integer_subbyte.h#L200-L225)

### 关键发现：TMA 统一使用 uint8_t

在混合输入的集体主循环实现中：

```cpp
// 来自 sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:202-203
static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
```

**FP8 和 INT8 都是 8 位类型**，因此：
- 不需要特殊的打包/解包逻辑（非亚字节）
- 使用相同的 TMA 描述符处理
- 共享相似的对齐要求（通常 16 字节）
- **TMA 内部统一使用 `uint8_t` 传输所有 8 位类型**

## 实现要求

### 1. 类型系统层

**文件**: `/include/cutlass/numeric_types.h`
```cpp
// INT8 不需要亚字节处理，直接使用原生类型
using int8b_t = int8_t;    // 可以直接用标准 int8_t
using uint8b_t = uint8_t;  // 可以直接用标准 uint8_t
```

### 2. Atom MMA 层（PTX 指令）

**文件**: 创建 `/include/cute/arch/mma_sm90.hpp` 或扩展现有文件
```cpp
// Hopper 的 INT8 MMA 指令
struct SM90_16x8x32_S32S8S8S32_TN {
    using DRegisters = int32_t[4];
    using ARegisters = uint32_t[4];  // 打包的 INT8
    using BRegisters = uint32_t[2];  // 打包的 INT8
    using CRegisters = int32_t[4];

    CUTE_HOST_DEVICE static void
    fma(int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        int32_t const& c0, int32_t const& c1, int32_t const& c2, int32_t const& c3)
    {
        // 注意：需要验证实际的 PTX 指令
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    }
};
```

### 3. MMA Traits 层

**文件**: `/include/cute/atom/mma_traits_sm90_gmma.hpp`
```cpp
template <>
struct MMA_Traits<SM90_16x8x32_S32S8S8S32_TN> {
    using ValTypeD = int32_t;
    using ValTypeA = int8_t;
    using ValTypeB = int8_t;
    using ValTypeC = int32_t;

    using Shape_MNK = Shape<_16, _8, _32>;
    using ThrID = Layout<_128>;  // Warpgroup = 4 个 warp

    // 根据 GMMA 要求定义布局
    using ALayout = /* 根据 GMMA 要求定义 */;
    using BLayout = /* 根据 GMMA 要求定义 */;
    using CLayout = /* 根据 GMMA 要求定义 */;
};
```

### 4. TMA Copy 支持

**重要发现：TMA 将所有 8 位类型作为 `uint8_t` 处理**

基于代码分析：
- FP8 类型（`float_e4m3_t`, `float_e5m2_t`）在 TMA 中作为 `uint8_t` 传输
- 混合输入集体主循环中：`TmaElementA = uint8_t`（对于所有 8 位类型）

**文件**: `/include/cute/atom/copy_traits_sm90_tma.hpp`
```cpp
// INT8 复用 FP8/uint8_t 的 TMA 基础设施
// 实际上 TMA 硬件层面不区分具体的 8 位类型
template <>
struct Copy_Traits<SM90_TMA_LOAD, int8_t> {
    // TMA 描述符创建时使用 uint8_t
    // 实际数据传输时硬件不关心具体类型
    using TmaType = uint8_t;
    // 复用现有的 TMA 实现...
};
```

### 5. 集体主循环构建器

**文件**: `/include/cutlass/gemm/collective/builders/sm90_common.inl`

由于 INT8 不是亚字节，处理很简单：
```cpp
// 在元素类型分发逻辑中
template <class ElementA, class ElementB>
struct CollectiveBuilder<..., int8_t, int8_t, ...> {
    // TMA 将 int8_t 作为 uint8_t 处理
    using TmaElementA = uint8_t;  // 关键：TMA 内部类型
    using TmaElementB = uint8_t;

    // 使用标准的 8 位类型布局
    using SmemLayoutAtomA = /* 8 位类型的标准布局 */;
    using SmemLayoutAtomB = /* 8 位类型的标准布局 */;

    // 不需要特殊打包 - INT8 是字节对齐的
    static constexpr int IsSubbyteA = false;
    static constexpr int IsSubbyteB = false;
};
```

### 6. 数值转换

**文件**: `/include/cutlass/numeric_conversion.h`
```cpp
// INT8 到累加器的转换
template <>
struct NumericConverter<int32_t, int8_t> {
    CUTLASS_HOST_DEVICE
    int32_t operator()(int8_t x) const {
        return static_cast<int32_t>(x);
    }
};

// 混合精度支持（INT8 x FP16 → FP32 等）
template <>
struct NumericConverter<float, int8_t> {
    CUTLASS_HOST_DEVICE
    float operator()(int8_t x) const {
        return static_cast<float>(x);
    }
};
```

## 代码位置参考

### 现有 INT8 相关代码（SM80）
- MMA traits：[mma_traits_sm80.hpp:224-415](include/cute/atom/mma_traits_sm80.hpp#L224-L415)
- MMA 指令：[mma_sm80.hpp](include/cute/arch/mma_sm80.hpp)
- **注意**：这些只支持 SM80/Ampere，不支持 SM90/Hopper GMMA

### FP8 实现（参考）
- 类型定义：[float8.h:170-220](include/cutlass/float8.h#L170-L220)
- MMA 指令：[mma_sm89.hpp:64-149](include/cute/arch/mma_sm89.hpp#L64-L149)
- 混合输入处理：[sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:202-204](include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp#L202-L204)

### TMA 使用 uint8_t 的证据
```cpp
// sm103_blockscaled_mma_array_warpspecialized.hpp:278-279
using TmaInternalElementA = uint8_t;
using TmaInternalElementB = uint8_t;

// sm103_blockscaled_mma_array_warpspecialized.hpp:488
typename Params::TMA_A tma_load_a = make_tma_atom<uint8_t>(...);

// sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:203
using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
```

### 关键集体主循环文件
- 基础 warp specialized：[sm90_mma_tma_gmma_rs_warpspecialized.hpp](include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp)
- 混合输入变体：[sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp](include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp)
- 构建器逻辑：[sm90_common.inl](include/cutlass/gemm/collective/builders/sm90_common.inl)

### 调度和内核文件
- 调度策略：[dispatch_policy.hpp:118-152](include/cutlass/gemm/dispatch_policy.hpp#L118-L152)
- 内核实现：[sm90_gemm_tma_warpspecialized.hpp](include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp)

## 关键洞察

1. **FP8 存在时无打包问题**：由于 FP8 和 INT8 都是 8 位类型，处理字节对齐数据的基础设施已经存在。亚字节打包逻辑仅用于 < 8 位的类型（int4 等）。

2. **TMA 处理是统一的**：
   - TMA 硬件将所有 8 位类型视为相同处理
   - **内部统一使用 `uint8_t` 作为传输类型**
   - FP8 也是通过 `uint8_t` 传输，然后在计算时解释为具体类型

3. **对齐要求**：FP8 和 INT8 都需要 16 字节对齐以获得最佳 TMA 性能。

4. **混合精度支持**：现有的混合输入基础设施可以扩展支持 INT8 与其他类型的混合。

## 实现步骤总结

1. **添加 PTX 指令封装**（必需）
   - 为 INT8 GMMA 添加 PTX 指令包装器
   - 参考 FP8 的实现模式

2. **定义 MMA Traits**（必需）
   - 为 SM90 INT8 操作定义 MMA traits
   - 包括布局和形状定义

3. **TMA 支持**（最小改动）
   - 复用现有的 `uint8_t` TMA 基础设施
   - 不需要新的 Copy_Traits，只需类型映射

4. **数值转换**（必需）
   - 确保 INT8 到累加器的转换
   - 支持混合精度场景

5. **集体构建器**（最小改动）
   - 主要是类型分发
   - 设置 `TmaElementA/B = uint8_t`

## 测试建议

1. 从简单的 INT8×INT8→INT32 GEMM 开始
2. 验证 INT8 张量的 TMA 描述符生成
3. 测试对齐要求（通常 16 字节）
4. 验证溢出情况下的数值转换
5. 与 FP8 对比性能，验证类似的性能表现

## 总结

在已有 FP8 支持的 CUTLASS 类代码库中添加 Hopper INT8 支持相对简单。主要工作包括：

1. **最关键**：添加 INT8 MMA 的 PTX 指令包装器和 MMA traits
2. **最简单**：TMA 层面复用 `uint8_t` 基础设施
3. **最小改动**：集体构建器主要做类型分发

由于 FP8 支持的存在，8 位类型的基础设施已经完备，避免了亚字节类型所需的复杂打包/解包逻辑。INT8 和 FP8 在 TMA 层面的处理完全一致，都作为 `uint8_t` 传输。