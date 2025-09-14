# Custom CUTLASS Epilogue 设计详解

本文档详细解释 `CustomLinearCombinationRelu` 的设计和实现。

## 1. 类模板定义

```cpp
template <
    typename ElementOutput_,      // 输出元素类型 (如 float)
    int Count,                    // 每次处理的元素数量
    typename ElementAccumulator_, // 累加器元素类型
    typename ElementCompute_      // 计算时使用的类型
>
class CustomLinearCombinationRelu
```

### 参数说明：
- `ElementOutput_`: 最终输出矩阵的数据类型
- `Count`: 向量化处理的元素数量，用于 SIMD 优化
- `ElementAccumulator_`: GEMM 计算过程中累加器的类型
- `ElementCompute_`: 执行计算时的精度类型

## 2. 类型别名定义

```cpp
using ElementOutput = ElementOutput_;
using ElementAccumulator = ElementAccumulator_;
using ElementCompute = ElementCompute_;
using ElementC = ElementOutput_;
```
**作用**：为模板参数创建简短的别名，提高代码可读性。

```cpp
static int const kCount = Count;
```
**作用**：定义编译时常量，表示每次处理的元素数量。

## 3. Fragment 类型定义

```cpp
using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
using FragmentC = cutlass::Array<ElementC, kCount>;
```

### Fragment 概念：
- **Fragment** 是 CUTLASS 中的核心概念，表示一个线程处理的数据片段
- 使用 `cutlass::Array` 实现向量化存储
- 每个 Fragment 包含 `kCount` 个元素
- 允许编译器进行 SIMD 优化

## 4. 参数结构体

```cpp
struct Params {
    ElementCompute alpha;  // GEMM 的 alpha 缩放因子
    ElementCompute beta;   // 原始 C 矩阵的 beta 缩放因子

    // 默认构造函数：alpha=1, beta=0
    CUTLASS_HOST_DEVICE
    Params() : alpha(1), beta(0) {}

    // 参数化构造函数
    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta) {}
};
```

### 说明：
- `alpha`: 控制 A×B 结果的缩放
- `beta`: 控制原始 C 矩阵的缩放
- `CUTLASS_HOST_DEVICE`: 宏定义，使函数可在 CPU 和 GPU 上运行

## 5. 成员变量

```cpp
private:
    ElementCompute alpha_;  // 存储 alpha 值
    ElementCompute beta_;   // 存储 beta 值
```

## 6. 构造函数

```cpp
CUTLASS_HOST_DEVICE
CustomLinearCombinationRelu(Params const &params)
    : alpha_(params.alpha), beta_(params.beta) {}
```
**作用**：从参数结构体初始化 epilogue functor。

## 7. 辅助函数

```cpp
CUTLASS_HOST_DEVICE
bool is_source_needed() const {
    return beta_ != ElementCompute(0);
}
```
**作用**：判断是否需要读取源矩阵 C。如果 beta=0，则不需要读取原始 C 矩阵，可以优化内存访问。

```cpp
CUTLASS_HOST_DEVICE
void set_k_partition(int k_partition, int k_partition_count) {}
```
**作用**：用于分片 K 维度的 GEMM。在基础实现中为空函数。

## 8. 核心操作符（带源矩阵）

```cpp
CUTLASS_HOST_DEVICE
FragmentOutput operator()(
    FragmentAccumulator const &accumulator,  // GEMM 计算结果
    FragmentC const &source) const {         // 原始 C 矩阵值

    FragmentOutput output;
    CustomReLU<ElementCompute> relu_op;

    CUTLASS_PRAGMA_UNROLL  // 循环展开优化
    for (int i = 0; i < kCount; ++i) {
        // 步骤1：线性组合
        // result = alpha * (A×B) + beta * C
        ElementCompute compute_result =
            alpha_ * ElementCompute(accumulator[i]) +
            beta_ * ElementCompute(source[i]);

        // 步骤2：应用 ReLU
        // output = max(0, result)
        output[i] = ElementOutput(relu_op(compute_result));
    }

    return output;
}
```

### 执行流程：
1. **输入**：
   - `accumulator`: GEMM 的计算结果 (A×B)
   - `source`: 原始 C 矩阵的值

2. **计算**：
   - 执行线性组合：`alpha * accumulator + beta * source`
   - 应用 ReLU 激活：`max(0, result)`

3. **优化**：
   - `CUTLASS_PRAGMA_UNROLL`: 指示编译器展开循环
   - 向量化处理 `kCount` 个元素

## 9. 核心操作符（不带源矩阵）

```cpp
CUTLASS_HOST_DEVICE
FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    FragmentOutput output;
    CustomReLU<ElementCompute> relu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
        // 当 beta=0 时，只需要处理 accumulator
        ElementCompute compute_result =
            alpha_ * ElementCompute(accumulator[i]);

        // 应用 ReLU
        output[i] = ElementOutput(relu_op(compute_result));
    }

    return output;
}
```

### 使用场景：
- 当 `beta = 0` 时调用此版本
- 避免不必要的内存读取
- 提高性能

## 10. 在 CUTLASS GEMM 中的集成

```cpp
using CutlassGemmCustom = cutlass::gemm::device::Gemm<
    float,                          // ElementA
    RowMajor,                       // LayoutA
    float,                          // ElementB
    RowMajor,                       // LayoutB
    float,                          // ElementC
    RowMajor,                       // LayoutC
    float,                          // ElementAccumulator
    cutlass::arch::OpClassSimt,    // OpClass
    cutlass::arch::Sm80,            // ArchTag
    cutlass::gemm::GemmShape<128, 128, 8>,  // ThreadblockShape
    cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape
    CustomLinearCombinationRelu<    // 自定义 Epilogue
        float, 1, float, float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                               // Stages
>;
```

## 11. 数据流程图

```
输入矩阵 A, B
    ↓
GEMM 主循环计算 (A × B)
    ↓
Accumulator (累加结果)
    ↓
CustomLinearCombinationRelu Epilogue:
    1. 读取源矩阵 C (如果 beta ≠ 0)
    2. 计算: result = alpha * accumulator + beta * C
    3. 应用 ReLU: output = max(0, result)
    ↓
输出矩阵 D
```

## 12. 性能优化要点

1. **向量化处理**：每次处理 `kCount` 个元素
2. **循环展开**：使用 `CUTLASS_PRAGMA_UNROLL`
3. **条件内存访问**：通过 `is_source_needed()` 避免不必要的读取
4. **融合操作**：在一个 kernel 中完成 GEMM + 线性组合 + ReLU
5. **寄存器优化**：使用 Fragment 保持数据在寄存器中

## 13. 扩展性

可以轻松修改为其他激活函数：

```cpp
// Sigmoid
output[i] = 1.0f / (1.0f + expf(-compute_result));

// Tanh
output[i] = tanhf(compute_result);

// LeakyReLU
output[i] = compute_result > 0 ? compute_result : 0.1f * compute_result;

// ELU
output[i] = compute_result > 0 ? compute_result : alpha * (expf(compute_result) - 1);
```

## 总结

`CustomLinearCombinationRelu` 展示了如何在 CUTLASS 中实现高效的自定义 epilogue：

- **灵活性**：完全控制后处理逻辑
- **性能**：通过融合避免额外的内存访问
- **可扩展**：易于添加新的激活函数或操作
- **优化**：利用向量化和循环展开提高吞吐量

这种设计模式是 CUTLASS 高性能的关键，通过在 GEMM 计算的最后阶段融合多个操作，最大化了计算密度并最小化了内存带宽需求。