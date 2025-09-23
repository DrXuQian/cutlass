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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
    \brief 流水线GEMM内核模板。不计算批处理或支持split-K。
*/

#pragma once

#include "cutlass/cutlass.h"                                // CUTLASS核心头文件
#include "cutlass/numeric_types.h"                         // 数值类型定义
#include "cutlass/arch/arch.h"                             // 架构相关定义
#include "cutlass/device_kernel.h"                         // 设备内核基类

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"  // 线程块调度策略

#include "cutlass/gemm/device/default_gemm_configuration.h" // 默认GEMM配置
#include "cutlass/epilogue/thread/linear_combination_relu.h" // ReLU epilogue操作

#include "kernel/b2b_gemm.h"                               // B2B GEMM内核
#include "kernel/default_b2b_gemm.h"                       // 默认B2B GEMM配置
#include "kernel/default_b2b_gemm_smem_accumulator.h"      // 使用共享内存累加器的B2B GEMM

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape0_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp0_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Epilogue output operator
    typename EpilogueOutputOp1_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Stage accumulator in shared memory
    bool SmemAccumulator = false,
    /// Access granularity of A matrix in units of elements  // A矩阵访问粒度（以元素为单位）
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements  // B矩阵访问粒度（以元素为单位）
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM  // GEMM执行的操作类型
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator>
// B2B GEMM设备端操作类
class B2bGemm {
 public:

  using ElementA = ElementA_;                                      // A矩阵元素类型
  using LayoutA = LayoutA_;                                        // A矩阵布局类型
  using TensorRefA = TensorRef<ElementA const, LayoutA>;           // A矩阵张量引用类型
  using ElementB = ElementB_;                                      // B矩阵元素类型
  using LayoutB = LayoutB_;                                        // B矩阵布局类型
  using TensorRefB = TensorRef<ElementB const, LayoutB>;           // B矩阵张量引用类型
  using ElementC = ElementC_;                                      // C矩阵元素类型
  using LayoutC = LayoutC_;                                        // C矩阵布局类型
  using TensorRefC = TensorRef<ElementC const, LayoutC>;           // C矩阵张量引用类型（输入）
  using TensorRefD = TensorRef<ElementC, LayoutC>;                 // D矩阵张量引用类型（输出）
  using ElementAccumulator = ElementAccumulator_;                  // 累加器元素类型
  using OperatorClass = OperatorClass_;                            // 操作类
  using ArchTag = ArchTag_;                                        // 架构标签
  using ThreadblockShape0 = ThreadblockShape0_;                    // 第一个GEMM的线程块形状
  using ThreadblockShape1 = ThreadblockShape1_;                    // 第二个GEMM的线程块形状
  using WarpShape0 = WarpShape0_;                                  // 第一个GEMM的Warp形状
  using WarpShape1 = WarpShape1_;                                  // 第二个GEMM的Warp形状
  using InstructionShape = InstructionShape_;                      // 指令形状
  using EpilogueOutputOp0 = EpilogueOutputOp0_;                    // 第一个GEMM的epilogue操作
  using EpilogueOutputOp1 = EpilogueOutputOp1_;                    // 第二个GEMM的epilogue操作
  using ThreadblockSwizzle = ThreadblockSwizzle_;                  // 线程块调度策略
  using Operator = Operator_;                                      // 操作类型
  static int const kStages = Stages;                               // 流水线阶段数
  static int const kAlignmentA = AlignmentA;                       // A矩阵对齐要求
  static int const kAlignmentB = AlignmentB;                       // B矩阵对齐要求
  static int const kAlignmentC = EpilogueOutputOp1::kCount;        // C矩阵对齐要求
  static ComplexTransform const kTransformA = ComplexTransform::kNone;  // A矩阵复数变换（无）
  static ComplexTransform const kTransformB = ComplexTransform::kNone;  // B矩阵复数变换（无）

  /// Derived types  // 派生类型
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;  // 缩放和偏置的元素类型
  using LayoutScaleBias = layout::RowMajor;                             // 缩放和偏置的布局（行主序）

  /// Define the kernel  // 定义内核
  using B2bGemmKernel = typename kernel::DefaultB2bGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    ThreadblockSwizzle,
    kStages,
    Operator,
    SmemAccumulator
  >::B2bGemmKernel;

  using Arguments = typename B2bGemmKernel::Arguments;

private:

  /// Kernel parameters object
  typename B2bGemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  B2bGemm() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    Status status = B2bGemmKernel::can_implement(
      args.problem_size_0,
      args.problem_size_1,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size_0,
      {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
      args.batch_count);

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size_0,
      {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
      args.batch_count);
//    cutlass::gemm::GemmCoord grid_shape_1 = threadblock_swizzle.get_tiled_shape(
//      args.problem_size_1,
//      {ThreadblockShape1::kM, ThreadblockShape1::kN, ThreadblockShape1::kK},
//      args.batch_count);

    // Initialize the Params structure
    params_ = typename B2bGemmKernel::Params{
      args.mode,
      args.problem_size_0,
      args.problem_size_1,
      grid_shape,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_Scale0.non_const_ref(),
      args.ref_Bias0.non_const_ref(),
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1,
      args.batch_stride_A0,
      args.batch_stride_B0,
      args.batch_stride_B1,
      args.batch_stride_C1,
      args.batch_stride_D1,
      args.batch_stride_Bias0,
      args.batch_stride_Scale0,
      args.epilogue0,
      args.epilogue1,
      static_cast<int *>(workspace),
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    params_.ref_A0.reset(args.ref_A0.non_const_ref().data());
    params_.ref_B0.reset(args.ref_B0.non_const_ref().data());
    params_.ref_C0.reset(args.ref_C0.non_const_ref().data());
    params_.ref_Scale0.reset(args.ref_Scale0.non_const_ref().data());
    params_.ref_Bias0.reset(args.ref_Bias0.non_const_ref().data());
    params_.ref_B1.reset(args.ref_B1.non_const_ref().data());
    params_.ref_C1.reset(args.ref_C1.non_const_ref().data());
    params_.ref_D1.reset(args.ref_D1.data());
    params_.output_op_0 = args.epilogue0;
    params_.output_op_1 = args.epilogue1;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(B2bGemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename B2bGemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<B2bGemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<B2bGemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
