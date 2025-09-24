/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
 * CUTLASS GEMM + Softmax 融合实现
 * ==============================
 *
 * 此文件定义了完整的 GEMM+Softmax 融合实现，包括:
 *
 * 1. ApplySoftmax 内核 - 实现数值稳定的 softmax 计算
 * 2. GemmSoftmax 模板 - 统一的 GEMM+Softmax 融合接口
 * 3. 两遍 softmax 算法 - 确保数值稳定性
 *
 * SOFTMAX 算法原理:
 * ================
 * 传统的 softmax 计算: softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
 * 存在数值不稳定性问题：当 x_i 很大时，exp(x_i) 可能溢出
 *
 * 数值稳定的 softmax 算法:
 * softmax(x_i) = exp(x_i - max_j(x_j)) / sum_j(exp(x_j - max_j(x_j)))
 *
 * 两遍算法实现:
 * 第一遍: 找到每行的最大值 max_j(x_j)
 * 第二遍: 计算 exp(x_i - max) 并求和，然后正则化
 *
 * 融合优势:
 * =========
 * - 单次内核调用完成 GEMM+Softmax
 * - 减少内存带宽使用（无需存储中间 GEMM 结果）
 * - 提高数据局部性（在寄存器/共享内存中处理数据）
 * - 数值稳定性保证（通过最大值减法）
 */

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

// 标准 C++ 库
#include <cmath>        // 数学函数 (exp, max 等)
#include <iostream>     // I/O 流
#include <vector>       // 动态数组
#include <limits>       // 数值限制

// CUTLASS 核心库
#include "cutlass/cutlass.h"              // CUTLASS 基础定义
#include "cutlass/arch/memory.h"          // 内存架构抽象
#include "cutlass/arch/memory_sm75.h"     // SM75 架构特定内存操作

// CUTLASS GEMM 核心组件
#include "cutlass/gemm/kernel/default_gemm.h"              // 默认 GEMM 内核
#include "cutlass/gemm/kernel/default_gemm_complex.h"      // 复数 GEMM 内核
#include "cutlass/gemm/device/default_gemm_configuration.h" // GEMM 默认配置

// CUTLASS Epilogue 组件 - 用于 GEMM 后的融合操作
#include "cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h"  // Softmax epilogue visitor
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"          // 通用 epilogue visitor 支持
#include "cutlass/reduction/kernel/reduce_softmax_final.h"               // Softmax 最终归约内核

/////////////////////////////////////////////////////////////////////////////////////////////////

// 本示例的自定义 epilogue visitor 实现
#include "gemm_with_epilogue_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Softmax 应用内核 - 执行最终的 softmax 计算
// =============================================
//
// 此内核实现 softmax 的最终步骤:
// 计算: Softmax[m, n] = exp(D[m, n] - Norm[m]) / Sum[m]
//
// 输入:
// - D[m, n]: GEMM 输出矩阵
// - Norm[m]: 每行的最大值 (数值稳定性)
// - Sum[m]: 每行的指数和 (用于歒一化)
//
// 输出:
// - Softmax[m, n]: 最终的 softmax 结果
//
template <
  typename ElementD_,                 // GEMM 输出元素类型
  typename ElementNorm_,              // 归一化元素类型 (最大值)
  typename ElementSum_,               // 求和元素类型
  typename ElementSoft_,              // Softmax 输出元素类型
  typename ElementSoftmaxCompute_,    // Softmax 中间计算类型 (通常为 float)
  int Alignment,                      // 内存对齐要求（向量化访问）
  typename ApplyShape_ = MatrixShape<1, 1024>  // 工作分块形状
>
class ApplySoftmax {
public:

  // 元素类型定义
  using ElementD = ElementD_;                           // GEMM 输出元素类型
  using ElementNorm = ElementNorm_;                     // 最大值元素类型
  using ElementSum = ElementSum_;                       // 求和元素类型
  using ElementSoft = ElementSoft_;                     // Softmax 输出元素类型
  using ElementSoftmaxCompute = ElementSoftmaxCompute_; // Softmax 计算精度类型

  // 内存访问配置
  static int const kAlignment = Alignment;              // 向量化访问的对齐要求
  using ApplyShape = ApplyShape_;                       // 计算块的形状配置

  // 内存布局配置 - 所有矩阵使用行主序布局
  using Layout = cutlass::layout::RowMajor;

  // 张量引用类型定义
  using TensorRefD = TensorRef<ElementD, Layout>;       // GEMM 输出矩阵引用
  using TensorRefN = TensorRef<ElementNorm, Layout>;    // 最大值数组引用
  using TensorRefSum = TensorRef<ElementSum, Layout>;   // 求和数组引用
  using TensorRefSoft = TensorRef<ElementSoft, Layout>; // Softmax 输出矩阵引用

  // 向量化计算片段类型 (提高内存带宽利用率)
  using FragmentSoftmax = Array<ElementSoftmaxCompute, kAlignment>;

  //
  // 内核参数结构
  // ===============

  struct Arguments {

    MatrixCoord     extent;             ///< D 和 Softmax 矩阵的维度 (M x N)
    int             batch_count;        ///< 批处理数量
    TensorRefD      ref_D;              ///< GEMM+Max 计算的 D 矩阵 (输入)
    TensorRefN      ref_N;              ///< 最大值张量 (输入，数值稳定性)
    TensorRefSum    ref_S;              ///< 求和张量 (输入，歒一化)
    TensorRefSoft   ref_Soft;           ///< Softmax 结果张量 (输出)
    int64_t         batch_stride_D;     ///< D 张量的批步长
    int64_t         batch_stride_N;     ///< N 张量的批步长
    int64_t         batch_stride_S;     ///< S 张量的批步长
    int64_t         batch_stride_Soft;  ///< Softmax 张量的批步长

    //
    // 构造函数和方法
    //
    Arguments():
      batch_count(1),                 // 默认批大小为 1
      batch_stride_D(0),              // 默认无批步长
      batch_stride_N(0),
      batch_stride_S(0),
      batch_stride_Soft(0)
    { }

    Arguments(
      MatrixCoord     extent_,             ///< Extent of D and Softmax matrices
      int             batch_count_,        ///< Batch count
      TensorRefD      ref_D_,              ///< D matrix computed by GEMM+PartialReduce
      TensorRefN      ref_N_,              ///< Output parameter for N
      TensorRefSum    ref_S_,              ///< Output parameter for N
      TensorRefSoft   ref_Soft_,           ///< Softmax
      int64_t         batch_stride_D_ = 0,
      int64_t         batch_stride_N_ = 0,
      int64_t         batch_stride_S_ = 0,
      int64_t         batch_stride_Soft_ = 0
    ):
      extent(extent_),
      batch_count(batch_count_),
      ref_D(ref_D_),
      ref_N(ref_N_),
      ref_S(ref_S_),
      ref_Soft(ref_Soft_),
      batch_stride_D(batch_stride_D_),
      batch_stride_N(batch_stride_N_),
      batch_stride_S(batch_stride_S_),
      batch_stride_Soft(batch_stride_Soft_)
    {

    }
  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

  //
  // SharedStorage
  //

  struct SharedStorage {

  };

private:

public:

  CUTLASS_DEVICE
  ApplySoftmax() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    apply(params, shared_storage);
  }

private:


  /// 计算 Softmax - 数值稳定的两遍算法实现
  /// =============================================
  /// 此函数实现 softmax 的最终计算步骤:
  /// softmax(x_i) = exp(x_i - max) / sum_j(exp(x_j - max))
  /// 其中 max 和 sum 已由之前的内核计算完成
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;

    // 线程块和线程索引计算
    int block_batch = blockIdx.z;                        // 批处理维度索引
    int block_m = blockIdx.x * ApplyShape::kRow;         // 线程块在 M 维度的起始位置
    int block_n = 0;                                     // N 维度从 0 开始 (处理整行)

    int thread_m = threadIdx.y;                          // 线程在 M 维度的局部索引
    int thread_n = threadIdx.x * kAlignment;             // 线程在 N 维度的局部索引（向量化）

    // 全局坐标计算
    int idx_m = block_m + thread_m;                      // 全局 M 维度索引
    int idx_n = block_n + thread_n;                      // 全局 N 维度索引

    // 批处理偏移量计算（用于访问归一化和求和数据）
    int batch_offset_norm = block_batch * params.args.batch_stride_N;
    int batch_offset_sum = block_batch * params.args.batch_stride_S;

    // 边界检查 - 如果线程超出行边界，则提前退出
    if (params.args.extent.row() <= idx_m) {
      return;
    }

    //
    // 设置指针以重新加载 D 矩阵数据
    // =================================
    // 定义所需的数据类型和转换器

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;         // D 矩阵向量化访问类型
    using AccessTypeSoft = AlignedArray<ElementSoft, kAlignment>;   // Softmax 输出向量化访问类型
    using FragmentSoft = Array<ElementSoft, kAlignment>;           // Softmax 片段类型

    // 数值类型转换器 - 在不同精度之间转换
    using ConvertSoftCompute = cutlass::NumericArrayConverter<ElementSoftmaxCompute, ElementD, kAlignment>;
    using ConvertSoftOutput = cutlass::NumericArrayConverter<ElementSoft, ElementSoftmaxCompute, kAlignment>;

    // Softmax 计算所需的数学运算器
    using Mul = cutlass::multiplies<FragmentSoftmax>;     // 乘法运算（与倒数相乘）
    using Minus = cutlass::minus<FragmentSoftmax>;        // 减法运算（数值稳定性）
    using Exp   = cutlass::fast_exp_op<FragmentSoftmax>;  // 快速指数运算

    // 初始化运算器实例
    ConvertSoftCompute   convert_soft_compute;  // 输入转换器
    ConvertSoftOutput  convert_soft_output;     // 输出转换器

    Minus     minus;        // 减法运算器（实现 x - max 操作）
    Mul       mul;          // 乘法运算器（实现歒一化）
    Exp       exponential;  // 指数运算器（实现 exp 操作）

    // 标量转换器 - 用于处理归一化参数
    using ConvertSum = cutlass::NumericConverter<ElementSoftmaxCompute, ElementSum>;
    using ConvertNorm = cutlass::NumericConverter<ElementSoftmaxCompute, ElementNorm>;

    ConvertSum   convert_sum;     // 求和转换器
    ConvertNorm  convert_norm;    // 最大值转换器

    // 计算向量化内存访问指针
    // D 矩阵输入指针（GEMM 结果）
    AccessTypeD *access_d = reinterpret_cast<AccessTypeD *>(
      params.args.ref_D.data() +                              // D 矩阵基地址
      params.args.batch_stride_D * block_batch +               // 批处理偏移
      params.args.ref_D.layout()({idx_m, idx_n}));            // 元素偏移

    // Softmax 输出指针
    AccessTypeSoft *access_soft = reinterpret_cast<AccessTypeSoft *>(
      params.args.ref_Soft.data() +                           // Softmax 矩阵基地址
      params.args.batch_stride_Soft * block_batch +           // 批处理偏移
      params.args.ref_Soft.layout()({idx_m, idx_n}));         // 元素偏移

    // 加载当前行的归一化参数（已由前面的内核计算完成）
    ElementSum inv_sum = (params.args.ref_S.data())[idx_m + batch_offset_sum];    // 倒数和（1/sum）
    ElementNorm norm = (params.args.ref_N.data())[idx_m + batch_offset_norm];     // 行最大值

    //
    // 主计算循环 - 数值稳定的 Softmax 计算
    // =========================================
    // 此循环实现了 softmax 的核心计算:
    // softmax(x_i) = exp(x_i - max) / sum
    //              = exp(x_i - max) * (1 / sum)
    //
    CUTLASS_PRAGMA_UNROLL  // 编译器指示: 展开循环以提高性能
    for (
      int idx = 0;
      idx < params.args.extent.column();                    // 遍历整行
      idx += ApplyShape::kColumn * kAlignment) {             // 按块步进

      // 边界检查和计算
      if (idx_n < params.args.extent.column()) {
        AccessTypeD fetch;  // 临时存储 GEMM 输出数据

        // 从全局内存加载数据（向量化访问）
        arch::global_load<AccessTypeD, sizeof(AccessTypeD)>(fetch, access_d, true);

        // 数值稳定的 Softmax 计算：
        // 1. convert_soft_compute(fetch) - 将 GEMM 输出转为计算精度
        // 2. minus(..., convert_norm(norm)) - 减去行最大值确保数值稳定
        // 3. exponential(...) - 计算指数
        // 4. mul(..., convert_sum(inv_sum)) - 与倒数和相乘实现歒一化
        FragmentSoftmax result = mul(
          exponential(
            minus(convert_soft_compute(fetch), convert_norm(norm))
          ),
          convert_sum(inv_sum)
        );

        // 将结果转换为输出类型
        FragmentSoft soft = convert_soft_output(result);

        // 将结果存储到全局内存（向量化访问）
        arch::global_store<FragmentSoft, sizeof(FragmentSoft)>(soft, access_soft, true);
      }

      // 更新指针位置，移动到下一个块
      access_d += ApplyShape::kColumn;      // D 矩阵指针前进
      access_soft += ApplyShape::kColumn;   // Softmax 输出指针前进
      idx_n += ApplyShape::kColumn * kAlignment;  // 更新列索引
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

///
/// GEMM + Softmax 融合模板类
/// ============================
///
/// 此模板实现了完整的 GEMM+Softmax 融合操作，包括:
/// 1. 高性能 GEMM 计算 (使用 Tensor Core)
/// 2. 数值稳定的两遍 softmax 算法
/// 3. 内存带宽优化的融合实现
///
/// 模板参数说明:
/// - ElementA_, LayoutA_: 输入矩阵 A 的类型和布局
/// - ElementB_, LayoutB_: 输入矩阵 B 的类型和布局
/// - ElementC_: 输出矩阵和偏置矩阵的元素类型
/// - ElementCompute_: GEMM 内部计算精度 (通常为 float)
/// - OperatorClass_: 计算类型 (TensorOp 使用 Tensor Core)
/// - ArchTag_: 目标 GPU 架构 (Sm80 用于 Ampere)
/// - ThreadblockShape_, WarpShape_, InstructionShape_: 分层的并行结构
/// - EpilogueFunctorOp_: GEMM 结束后的线性变换操作
/// - kStages_: 流水线阶段数
/// - ApplyShape_: Softmax 计算块的形状
/// - AlignmentA_, AlignmentB_, AlignmentSoftmax_: 对齐要求
/// - ElementNorm_, ElementSum_: Softmax 计算中间结果类型
/// - ElementSoftmax_: 最终 Softmax 输出类型
///
template <
  typename ElementA_,                // A 矩阵元素类型
  typename LayoutA_,                 // A 矩阵布局
  typename ElementB_,                // B 矩阵元素类型
  typename LayoutB_,                 // B 矩阵布局
  typename ElementC_,                // C/D 矩阵元素类型
  typename ElementCompute_,          // GEMM 内部计算精度
  typename OperatorClass_,           // 计算单元类型 (TensorOp/SIMT)
  typename ArchTag_,                 // 目标架构标签
  typename ThreadblockShape_,        // 线程块级分块形状
  typename WarpShape_,               // Warp 级分块形状
  typename InstructionShape_,        // 指令级分块形状
  typename EpilogueFunctorOp_,       // Epilogue 线性组合操作
  int kStages_,                      // 流水线阶段数
  typename ApplyShape_ = MatrixShape<1, 1024>,  // Softmax 计算块形状
  int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,      // A 矩阵对齐
  int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value,      // B 矩阵对齐
  int AlignmentSoftmax_ = 128 / cutlass::sizeof_bits<ElementC_>::value, // Softmax 对齐
  typename ElementNorm_ = float,     // 最大值类型 (数值稳定性)
  typename ElementSum_ = float,      // 求和类型 (归一化)
  typename ElementSoftmax_ = ElementC_  // Softmax 输出类型
>
class GemmSoftmax {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // 类型定义 - 模板参数的别名
  // =========================
  //

  using ElementA = ElementA_;                    // 输入矩阵 A 元素类型
  using ElementB = ElementB_;                    // 输入矩阵 B 元素类型
  using ElementC = ElementC_;                    // 输出矩阵 C/D 元素类型
  using ElementCompute = ElementCompute_;        // GEMM 内部计算精度
  using ElementSum = ElementSum_;                // Softmax 求和类型
  using ElementSoft = ElementSoftmax_;           // Softmax 最终输出类型
  using ElementSoftmaxCompute = float;           // Softmax 中间计算精度 (固定为 float)

  // 布局类型
  using LayoutA = LayoutA_;                      // A 矩阵布局 (RowMajor/ColumnMajor)
  using LayoutB = LayoutB_;                      // B 矩阵布局

  // 功能组件类型
  using EpilogueFunctorOp = EpilogueFunctorOp_;  // GEMM epilogue 线性组合操作
  using ElementNorm = ElementNorm_;              // 最大值元素类型

  // 计算块配置
  using ApplyShape = ApplyShape_;                // Softmax 计算的工作分块形状

  // 强制性布局类型 - Softmax 计算需要使用行主序布局
  // ===========================================================
  // 所有 Softmax 相关的矩阵都必须使用行主序，因为 softmax 操作是按行进行的
  using LayoutC = cutlass::layout::RowMajor;     // C/D 矩阵布局 (输入/输出)
  using LayoutN = cutlass::layout::RowMajor;     // 最大值数组布局
  using LayoutS = cutlass::layout::RowMajor;     // 求和数组布局
  using LayoutSoft = cutlass::layout::RowMajor;  // Softmax 输出矩阵布局

  // 张量引用类型 - 用于高效的张量访问
  using TensorRefA = TensorRef<ElementA, LayoutA>;        // A 矩阵引用
  using TensorRefB = TensorRef<ElementB, LayoutB>;        // B 矩阵引用
  using TensorRefC = TensorRef<ElementC, LayoutC>;        // C/D 矩阵引用
  using TensorRefN = TensorRef<ElementNorm, LayoutN>;     // 最大值数组引用
  using TensorRefSum = TensorRef<ElementSum, LayoutS>;    // 求和数组引用
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>; // Softmax 输出矩阵引用

  // 分层并行结构定义
  using ThreadblockShape = ThreadblockShape_;    // 线程块级分块形状 (M x N x K)
  using WarpShape        = WarpShape_;           // Warp 级分块形状
  using InstructionShape = InstructionShape_;    // Tensor Core 指令形状

  // 架构配置
  using OperatorClass = OperatorClass_;          // 计算单元类型 (TensorOp/SIMT)
  using ArchTag = ArchTag_;                      // GPU 架构标签 (Sm80/Sm75)

  // 编译时常量配置
  static int const kStages  = kStages_;                  // 流水线阶段数 (影响占用的共享内存)
  static int const AlignmentA = AlignmentA_;             // A 矩阵对齐要求 (向量化访问)
  static int const AlignmentB = AlignmentB_;             // B 矩阵对齐要求
  static int const AlignmentSoftmax = AlignmentSoftmax_; // Softmax 对齐要求

  // 线程块调度策略 - 优化内存访问模式
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementC,
    LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    ThreadblockSwizzle,
    kStages,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorSoftmax<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    ElementCompute,
    ElementNorm,
    ElementSum,
    ElementSoftmaxCompute,
    EpilogueFunctorOp
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;

  // Softmax kernel
  using SoftmaxApplyKernel = kernel::ApplySoftmax<
    ElementC,
    ElementNorm,
    ElementSum,
    ElementSoft,
    ElementSoftmaxCompute,
    AlignmentSoftmax,
    ApplyShape
  >;

  using ApplyFinalReductionKernel = cutlass::reduction::kernel::ApplySoftmaxFinalReduction<
    ElementNorm,
    ElementSum,
    ElementSoftmaxCompute,
    ThreadblockShape
  >;

public:

  /// Arguments class
  struct Arguments {

    typename GemmKernel::Arguments         gemm;
    typename SoftmaxApplyKernel::Arguments softmax;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      int32_t    batch_count_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,
      TensorRefC ref_D_,
      typename EpilogueFunctorOp::Params linear_scaling,
      TensorRefN ref_N_,
      TensorRefSum ref_S_,
      TensorRefSoft ref_Softmax_,
      int64_t batch_stride_A_ = 0,
      int64_t batch_stride_B_ = 0,
      int64_t batch_stride_C_ = 0,
      int64_t batch_stride_D_ = 0,
      int64_t batch_stride_Max_ = 0,
      int64_t batch_stride_Sum_ = 0,
      int64_t batch_stride_Softmax_ = 0
    ):
      gemm(
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        batch_count_,
        ref_A_,
        ref_B_,
        ref_C_,
        ref_D_,
        ref_N_.data(),
        ref_S_.data(),
        batch_stride_A_,
        batch_stride_B_,
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          batch_stride_C_,
          batch_stride_D_,
          batch_stride_Max_,
          batch_stride_Sum_
        )
      ),
      reduction(
        problem_size,
        ref_N_.data(),
        ref_S_.data(),
        batch_stride_Max_,
        batch_stride_Sum_
      ), 
      softmax(
        MatrixCoord(problem_size.m(), problem_size.n()),
        batch_count_,
        ref_D_,
        ref_N_,
        ref_S_,
        ref_Softmax_,
        batch_stride_D_,
        batch_stride_Max_,
        batch_stride_Sum_,
        batch_stride_Softmax_
      ),
      extend(problem_size)
    {

    }
  };

  struct Params {

    typename GemmKernel::Params         gemm;
    typename SoftmaxApplyKernel::Params softmax;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm(args.gemm),
      reduction(args.reduction),
      softmax(args.softmax),
      extend(MatrixCoord(args.extend.m(), args.extend.n()))
    {

    }
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Params params_;

public:

  /// Ctor
  GemmSoftmax() {

  }

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// 运行融合 GEMM+Softmax 算法
  /// =============================
  /// 此函数执行完整的三阶段 softmax 计算:
  /// 1. GEMM + 第一遍归约 (找最大值和部分求和)
  /// 2. 最终归约 (合并部分结果得到最终最大值和求和)
  /// 3. Softmax 应用 (计算最终 softmax 值)
  Status run(cudaStream_t stream) {

    //
    // 第一阶段: 启动 GEMM + 部分最大值内核
    // ========================================
    // 此内核执行 GEMM 计算并同时计算每行的部分最大值和部分指数和
    //

    // 计算 CUDA 网格和线程块配置
    dim3 gemm_grid = ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);   // 每个线程块的线程数

    // 计算所需的共享内存大小
    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    // 如果需要的共享内存超过默认限制 (48KB)，需要手动设置
    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    // 启动 GEMM+部分最大值内核
    cutlass::Kernel<GemmKernel><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }


    //
    // 第二阶段: 启动最终归约内核
    // ==============================
    // 此内核将所有线程块产生的部分结果合并，
    // 得到每行的最终最大值和最终指数和
    //

    // 动态计算最优的线程块配置
    int thread_per_block = 128;  // 初始线程数 (高占用率)
    int block_per_row = (params_.extend.row() + thread_per_block - 1) / thread_per_block;

    // 如果行数太少，减少线程数以提高占用率
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row = (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    }

    // 最终归约网格配置: (blocks_per_row, 1, batch_count)
    dim3 final_reduction_grid(block_per_row, 1, params_.softmax.args.batch_count);
    dim3 final_reduction_block(thread_per_block);

    // 启动最终归约内核
    Kernel<ApplyFinalReductionKernel><<<
      final_reduction_grid, final_reduction_block, sizeof(typename ApplyFinalReductionKernel::SharedStorage), stream
    >>>(params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // 第三阶段: 启动 Softmax 应用内核
    // ===============================
    // 此内核执行最终的 softmax 计算:
    // softmax(x_i) = exp(x_i - max) / sum
    // 使用之前计算的最大值和求和结果
    //

    // Softmax 应用线程块配置
    dim3 apply_block(SoftmaxApplyKernel::ApplyShape::kColumn,   // X 维度线程数
                     SoftmaxApplyKernel::ApplyShape::kRow);    // Y 维度线程数

    // 计算每个线程块处理的元素数量
    int threadblock_rows = SoftmaxApplyKernel::ApplyShape::kRow;
    int threadblock_columns = SoftmaxApplyKernel::ApplyShape::kColumn * SoftmaxApplyKernel::kAlignment;

    // Softmax 应用网格配置: 覆盖整个输出矩阵
    dim3 apply_grid(
      (params_.softmax.args.extent.row() + threadblock_rows - 1) / threadblock_rows,        // M 维度网格
      (params_.softmax.args.extent.column() + threadblock_columns - 1) / threadblock_columns, // N 维度网格
      params_.softmax.args.batch_count);                                                      // 批处理维度

    // 启动 Softmax 应用内核 - 最终阶段
    Kernel<SoftmaxApplyKernel><<<
      apply_grid, apply_block, sizeof(typename SoftmaxApplyKernel::SharedStorage), stream
    >>>(params_.softmax);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;  // 所有阶段均成功完成
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
