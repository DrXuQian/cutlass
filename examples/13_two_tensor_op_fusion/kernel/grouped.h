/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief High-level interface for running a grouped version of a CUTLASS kernel
    \brief 运行分组版本CUTLASS内核的高层接口
*/

#pragma once

#include "cutlass/cutlass.h"                                      // CUTLASS核心头文件
#include "cutlass/fast_math.h"                                   // 快速数学运算
#include "cutlass/gemm/gemm.h"                                   // GEMM基础定义
#include "cutlass/matrix_coord.h"                                // 矩阵坐标
#include "cutlass/complex.h"                                     // 复数支持
#include "cutlass/semaphore.h"                                   // 信号量支持

#include "cutlass/layout/matrix.h"                              // 矩阵布局
#include "cutlass/trace.h"                                      // 跟踪调试
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"        // GEMM操作数转置
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"   // GEMM分组问题访问器

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// High-level interface for running a grouped version of a CUTLASS kernel
/// 运行分组版本CUTLASS内核的高层接口
template <
  typename BaseKernel_   ///! Kernel-scoped matrix multiply-accumulate  // 内核级矩阵乘加累加
>
struct GroupedKernel {
public:

  using BaseKernel = BaseKernel_;                                // 基础内核类型
  using Epilogue = typename BaseKernel::Epilogue;                // Epilogue类型

  /// Types that need to be exported to work properly with device::BaseGrouped
  /// 需要导出的类型，以便与device::BaseGrouped正常工作
  // A矩阵相关类型
  using ElementA = typename BaseKernel::ElementA;                 // A矩阵元素类型
  using LayoutA = typename BaseKernel::LayoutA;                   // A矩阵布局类型
  using TensorRefA = TensorRef<ElementA const, LayoutA>;          // A矩阵张量引用
  static ComplexTransform const kTransformA = BaseKernel::kTransformA;  // A矩阵复数变换
  static int const kAlignmentA = BaseKernel::kAlignmentA;         // A矩阵对齐要求

  // B矩阵相关类型
  using ElementB = typename BaseKernel::ElementB;                 // B矩阵元素类型
  using LayoutB = typename BaseKernel::LayoutB;                   // B矩阵布局类型
  using TensorRefB = TensorRef<ElementB const, LayoutB>;          // B矩阵张量引用
  static ComplexTransform const kTransformB = BaseKernel::kTransformB;  // B矩阵复数变换
  static int const kAlignmentB = BaseKernel::kAlignmentB;         // B矩阵对齐要求

  // C/D矩阵相关类型
  using ElementC = typename BaseKernel::ElementC;                 // C矩阵元素类型
  using LayoutC = typename BaseKernel::LayoutC;                   // C矩阵布局类型
  using TensorRefC = TensorRef<ElementC const, LayoutC>;          // C矩阵张量引用（输入）
  using TensorRefD = TensorRef<ElementC, LayoutC>;                // D矩阵张量引用（输出）
  static int const kAlignmentC = BaseKernel::kAlignmentC;         // C矩阵对齐要求

  // 计算相关类型
  using ElementAccumulator = typename BaseKernel::Mma::Policy::Operator::ElementC;  // 累加器元素类型

  using EpilogueOutputOp = typename BaseKernel::EpilogueOutputOp;       // Epilogue输出操作
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;   // 线程块调度策略

  using Operator = typename BaseKernel::Operator;                       // 操作符类型
  using WarpMmaOperator = typename BaseKernel::Mma::Policy::Operator;   // Warp级MMA操作符

  // 架构和数学操作相关
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;    // 架构MMA操作符
  using MathOperator = typename WarpMmaOperator::MathOperator;          // 数学操作符
  using OperatorClass = typename WarpMmaOperator::OperatorClass;        // 操作符类别
  using ArchTag = typename WarpMmaOperator::ArchTag;                    // 架构标签
  using ThreadblockShape = typename BaseKernel::Mma::Shape;             // 线程块形状
  using WarpShape = typename BaseKernel::WarpShape;                     // Warp形状
  using InstructionShape = typename BaseKernel::InstructionShape;       // 指令形状
  static int const kStages = BaseKernel::Mma::kStages;                  // 流水线阶段数

  using Mma = typename BaseKernel::Mma;                           // MMA操作类型

  using Arguments = typename BaseKernel::GroupedArguments;        // 分组参数类型
  using Params = typename BaseKernel::GroupedParams;              // 分组参数类型
  using ProblemVisitor = typename ThreadblockSwizzle::ProblemVisitor;  // 问题访问器类型

  static int const kThreadCount = BaseKernel::kThreadCount;       // 线程数

  /// Shared memory storage structure
  /// 共享内存存储结构
  struct SharedStorage {
    typename BaseKernel::SharedStorage kernel;             // 基础内核的共享存储

    // ProblemVisitor shared storage can't be overlapped with others
    // ProblemVisitor的共享存储不能与其他重叠
    typename ProblemVisitor::SharedStorage problem_visitor;  // 问题访问器的共享存储
  };

public:

  //
  // Methods  // 方法
  //

  // 设备端构造函数
  CUTLASS_DEVICE
  GroupedKernel() { }

  /// Determines whether kernel satisfies alignment
  /// 确定内核是否满足对齐要求
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;  // 总是返回成功（子类可重写）
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;  // 总是返回成功（子类可重写）
  }

  /// Executes a kernel-level GEMM in a loop
  /// 在循环中执行内核级GEMM
  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {

    // 创建线程块调度器
    ThreadblockSwizzle swizzle(params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    // 如果需要转置，转置参数
    if (ProblemVisitor::kTransposed) {
      params.transpose();
    }

    BaseKernel mma;  // 创建基础MMA内核

    // Outer 'persistent' loop to iterate over tiles
    // 外层“持久”循环，遍历所有tile
    while (swizzle.problem_visitor.next_tile()) {  // 获取下一个tile

      // 从分组参数转换为单个问题的参数
      typename BaseKernel::Params mma_params = params.to_single_params(swizzle.problem_visitor);
      // 使用调度器运行MMA
      mma.run_with_swizzle(mma_params, shared_storage.kernel, swizzle);

      // Next tile
      // 前进到下一个tile
      swizzle.problem_visitor.advance(gridDim.x);  // 以网格大小为步长前进
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
