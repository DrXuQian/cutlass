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
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
    \brief 双缓冲线程块级GEMM内核模板。
*/

#pragma once

#include "cutlass/aligned_buffer.h"  // 对齐缓冲区
#include "cutlass/arch/memory.h"      // 架构相关内存操作
#include "cutlass/array.h"            // 数组工具
#include "cutlass/cutlass.h"          // CUTLASS核心头文件
#include "cutlass/gemm/gemm.h"        // GEMM相关定义
#include "cutlass/matrix_shape.h"     // 矩阵形状定义
#include "cutlass/numeric_types.h"    // 数值类型定义
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
/// 计算矩阵乘积的结构，针对CUDA核心和SIMT数学指令。
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>  // 第一个GEMM问题的大小
    typename Shape0_,
    /// Size of the Gemm problem - concept: gemm::GemmShape<>  // 第二个GEMM问题的大小
    typename Shape1_,
    /// Policy describing tuning details (concept: MmaPolicy)  // 第一个GEMM的调优策略
    typename Policy0_,
    /// Policy describing tuning details (concept: MmaPolicy)  // 第二个GEMM的调优策略
    typename Policy1_,
    /// Number of stages,  // 流水线阶段数
    int Stages,
    /// Used for partial specialization  // 用于偏特化
    typename Enable = bool>
class B2bMmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>  // GEMM问题大小
  using Shape0 = Shape0_;  // 第一个GEMM的形状
  using Shape1 = Shape1_;  // 第二个GEMM的形状

  ///< Policy describing tuning details  // 调优策略
  using Policy0 = Policy0_;  // 第一个GEMM的策略
  using Policy1 = Policy1_;  // 第二个GEMM的策略

  //
  // Dependent types  // 依赖类型
  //

  /// Warp-level Mma  // Warp级MMA操作
  using Operator0 = typename Policy0::Operator;  // 第一个GEMM的操作符
  using Operator1 = typename Policy1::Operator;  // 第二个GEMM的操作符

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  /// 每个warp从共享内存计算的整体GEMM形状。
  using WarpGemm0 = typename Policy0::Operator::Shape;  // 第一个GEMM的warp形状
  using WarpGemm1 = typename Policy1::Operator::Shape;  // 第二个GEMM的warp形状

  /// Shape describing the number of warps filling the CTA
  /// 描述填充CTA（合作线程数组）的warp数量的形状。
  using WarpCount0 = GemmShape<Shape0::kM / WarpGemm0::kM,  // M维度的warp数
                               Shape0::kN / WarpGemm0::kN,   // N维度的warp数
                               Shape0::kK / WarpGemm0::kK>;  // K维度的warp数
  using WarpCount1 = GemmShape<Shape1::kM / WarpGemm1::kM,  // M维度的warp数
                               Shape1::kN / WarpGemm1::kN,   // N维度的warp数
                               Shape1::kK / WarpGemm1::kK>;  // K维度的warp数

  /// Number of warp-level GEMM oeprations  // Warp级GEMM操作数量
  static int const kWarpGemmIterations0 =
      (WarpGemm0::kK / Operator0::Policy::MmaShape::kK);  // 第一个GEMM的warp迭代次数
  static int const kWarpGemmIterations1 =
      (WarpGemm1::kK / Operator1::Policy::MmaShape::kK);  // 第二个GEMM的warp迭代次数

  /// Number of stages  // 流水线阶段数
  static int const kStages = Stages;

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  /// 线程块级GEMM所需的共享存储对象
  template<
    typename Shape_,   // GEMM形状
    typename Policy_   // 策略
  >
  class SharedStorage {
   public:
    //
    // Type definitions  // 类型定义
    //
    using Shape = Shape_;      // 形状类型
    using Policy = Policy_;    // 策略类型
    using Operator = typename Policy::Operator;  // 操作符类型

    /// Tensor reference to the A operand  // A操作数的张量引用
    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

    /// Tensor reference to the B operand  // B操作数的张量引用
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;


    /// Shape of the A matrix operand in shared memory  // 共享内存中A矩阵操作数的形状
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,      // 行数 + 行填充
                               Shape::kK * kStages +                        // 列数 * 阶段数
                                   Policy::SmemPaddingA::kColumn>;          // + 列填充

    /// Shape of the B matrix operand in shared memory  // 共享内存中B矩阵操作数的形状
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,  // 行数 * 阶段数 + 行填充
                    Shape::kN + Policy::SmemPaddingB::kColumn>;        // 列数 + 列填充

   public:
    //
    // Data members  // 数据成员
    //

    /// Buffer for A operand  // A操作数的缓冲区
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
  };

  using SharedStorage0 = SharedStorage<Shape0, Policy0>;
  using SharedStorage1 = SharedStorage<Shape1, Policy1>;
  union B2bMmaSharedStorage {
    SharedStorage0 shared_storage0;
    SharedStorage1 shared_storage1;
  };


 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A0 operand from shared memory
  typename Operator0::IteratorA warp_tile_iterator_A0_;

  /// Iterator to load a warp-scoped tile of B0 operand from shared memory
  typename Operator0::IteratorB warp_tile_iterator_B0_;

  /// Iterator to load a warp-scoped tile of B1 operand from shared memory
  typename Operator1::IteratorB warp_tile_iterator_B1_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  B2bMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      B2bMmaSharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_A0_(shared_storage.shared_storage0.operand_A_ref(), lane_idx),
      warp_tile_iterator_B0_(shared_storage.shared_storage0.operand_B_ref(), lane_idx),
      warp_tile_iterator_B1_(shared_storage.shared_storage1.operand_B_ref(), lane_idx) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
