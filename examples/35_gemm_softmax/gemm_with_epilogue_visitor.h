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
    \brief 支持 epilogue visitor 模型的 GEMM 内核
    用于自定义 softmax 部分归约 epilogue 融合。

    一旦使用方式稳定，此源文件可能会移动到 `include/cutlass/gemm/kernel/`。
    目前包含在此示例中，以演示一些基本的输出融合选项。

    EPILOGUE VISITOR 模式解释:
    ==========================
    Epilogue Visitor 是 CUTLASS 中的一种设计模式，允许用户在 GEMM 计算完成后
    立即执行自定义操作，而无需将中间结果写回全局内存。这对于融合操作
    (如 softmax) 特别有用，因为它:

    1. 减少内存带宽使用 - 避免存储中间 GEMM 结果
    2. 提高缓存局部性 - 在数据仍在寄存器/共享内存中时处理
    3. 减少内核启动开销 - 单次内核调用而非多次
    4. 支持复杂的归约操作 - 如 softmax 中的行级最大值和求和
*/

#pragma once

// CUTLASS 核心组件
#include "cutlass/cutlass.h"        // CUTLASS 基础定义和宏
#include "cutlass/fast_math.h"       // 快速数学运算（针对 GPU 优化）
#include "cutlass/gemm/gemm.h"       // GEMM 核心定义和枚举
#include "cutlass/matrix_coord.h"    // 矩阵坐标系统
#include "cutlass/complex.h"         // 复数类型支持
#include "cutlass/semaphore.h"       // 线程同步原语

#include "cutlass/trace.h"           // 调试跟踪工具

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// 带有 Epilogue Visitor 的 GEMM 内核模板
// =======================================
// 此内核扩展了标准 GEMM，添加了自定义的 epilogue visitor 支持
// 用于在 GEMM 计算完成后立即执行融合操作（如 softmax）
template <
  typename Mma_,                  ///! 线程块范围的矩阵乘累加器
  typename Epilogue_,             ///! Epilogue 处理器（包含 visitor）
  typename ThreadblockSwizzle_    ///! 线程块调度函数（优化内存访问模式）
>
struct GemmWithEpilogueVisitor {
public:

  // 核心组件类型定义
  using Mma = Mma_;                                     // 矩阵乘法累加器
  using Epilogue = Epilogue_;                           // Epilogue 处理器
  using EpilogueVisitor = typename Epilogue::Visitor;   // 自定义 visitor（执行 softmax）
  using ThreadblockSwizzle = ThreadblockSwizzle_;       // 线程块调度策略

  // 输入矩阵 A 的类型配置
  using ElementA = typename Mma::IteratorA::Element;    // A 矩阵元素类型 (通常为 half_t)
  using LayoutA = typename Mma::IteratorA::Layout;      // A 矩阵内存布局 (RowMajor/ColumnMajor)
  using TensorRefA = TensorRef<ElementA, LayoutA>;      // A 矩阵张量引用类型

  // 输入矩阵 B 的类型配置
  using ElementB = typename Mma::IteratorB::Element;    // B 矩阵元素类型 (通常为 half_t)
  using LayoutB = typename Mma::IteratorB::Layout;      // B 矩阵内存布局
  using TensorRefB = TensorRef<ElementB, LayoutB>;      // B 矩阵张量引用类型

  // 输出矩阵 C/D 的类型配置
  using ElementC = typename EpilogueVisitor::ElementOutput; // 输出元素类型
  using LayoutC = typename Epilogue::Layout;            // 输出矩阵内存布局
  using TensorRefC = TensorRef<ElementC, LayoutC>;      // 输出矩阵张量引用类型

  // 矩阵变换配置（用于复数支持）
  static ComplexTransform const kTransformA = Mma::kTransformA;  // A 矩阵复数变换类型
  static ComplexTransform const kTransformB = Mma::kTransformB;  // B 矩阵复数变换类型
  using Operator = typename Mma::Operator;                       // 底层数学运算符类型

  // 计算层次配置 - 定义不同级别的并行结构
  using OperatorClass = typename Mma::Operator::OperatorClass;    // 运算类别 (TensorOp/SIMT)
  using ThreadblockShape = typename Mma::Shape;                   // 线程块级分块形状
  using WarpShape = typename Mma::Operator::Shape;                // Warp 级分块形状
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape; // 指令级形状
  using ArchTag = typename Mma::ArchTag;                          // 目标 GPU 架构标签

  // Softmax 计算所需的辅助数据类型
  // 这些类型用于实现数值稳定的 softmax 算法
  using ElementNorm = typename EpilogueVisitor::ElementNorm;  // 行最大值类型 (数值稳定性)
  using ElementSum = typename EpilogueVisitor::ElementSum;    // 行指数和类型 (归一化)

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value,
    128 / sizeof_bits<ElementB>::value
  );

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode;
    GemmCoord problem_size;
    int batch_count;

    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;

    ElementNorm *ptr_Max; 
    ElementSum  *ptr_Sum;

    int64_t    batch_stride_A;
    int64_t    batch_stride_B;

    typename EpilogueVisitor::Arguments epilogue_visitor;

    //
    // Methods
    //

    Arguments():
      mode(GemmUniversalMode::kGemm),
      batch_count(1)
    { }


    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode_,
      GemmCoord problem_size_,
      int batch_count_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,
      TensorRefC ref_D_,
      ElementNorm *ptr_Max_,
      ElementSum *ptr_Sum_,
      int64_t batch_stride_A_,
      int64_t batch_stride_B_,
      typename EpilogueVisitor::Arguments epilogue_visitor_
    ):
      mode(mode_),
      problem_size(problem_size_),
      batch_count(batch_count_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ptr_Max(ptr_Max_),
      ptr_Sum(ptr_Sum_),
      batch_stride_A(batch_stride_A_),
      batch_stride_B(batch_stride_B_),
      epilogue_visitor(epilogue_visitor_)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;

    GemmUniversalMode mode;
    int batch_count;
    int gemm_k_size;

    void * ptr_A;
    void * ptr_B;
    ElementC * ptr_C;
    ElementC * ptr_D;

    ElementNorm * ptr_Max;
    ElementSum * ptr_Sum;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    typename EpilogueVisitor::Params epilogue_visitor;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      swizzle_log_tile(0),
      params_A(0),
      params_B(0),
      params_C(0),
      params_D(0),
      batch_count(0),
      gemm_k_size(0),
      mode(cutlass::gemm::GemmUniversalMode::kGemm),
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr),
      ptr_Max(nullptr),
      ptr_Sum(nullptr),
      batch_stride_A(0),
      batch_stride_B(0)
    { }


    Params(
      Arguments const &args
    ):
      problem_size(args.problem_size),
      swizzle_log_tile(0),
      params_A(args.ref_A.layout()),
      params_B(args.ref_B.layout()),
      params_C(args.ref_C.layout()),
      params_D(args.ref_D.layout()),
      mode(args.mode),
      batch_count(args.batch_count),
      gemm_k_size(args.problem_size.k()),
      ptr_A(args.ref_A.data()),
      ptr_B(args.ref_B.data()),
      ptr_C(args.ref_C.data()),
      ptr_D(args.ref_D.data()),
      ptr_Max(args.ptr_Max),
      ptr_Sum(args.ptr_Sum),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      epilogue_visitor(args.epilogue_visitor)
    {

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.batch_count);

      if (args.mode == GemmUniversalMode::kGemm || args.mode == GemmUniversalMode::kGemmSplitKParallel) {

        int const kAlignK = const_max(const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value), 1);

        gemm_k_size = round_up(ceil_div(args.problem_size.k(), args.batch_count), kAlignK);

        if (gemm_k_size) {
          grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
        }
      }

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {

    typename Mma::SharedStorage main_loop;

    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epilogue;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmWithEpilogueVisitor() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmWithEpilogueVisitor::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  #define SPLIT_K_ENABLED 1

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);


    #if SPLIT_K_ENABLED
    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm ||
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
    }
    #endif

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations,
      accumulators,
      iterator_A,
      iterator_B,
      accumulators);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    //
    // Construct the epilogue visitor
    //

    EpilogueVisitor epilogue_visitor(
      params.epilogue_visitor,
      shared_storage.epilogue.visitor,
      params.problem_size.mn(),
      thread_idx,
      warp_idx,
      lane_idx,
      params.params_C,
      params.params_D,
      params.ptr_C,
      params.ptr_D,
      params.ptr_Max,
      params.ptr_Sum,
      threadblock_offset,
      blockIdx.y *params.problem_size.m() );

    if (params.mode == GemmUniversalMode::kGemm) {
      // Indicate which position in a serial reduction the output operator is currently updating
      epilogue_visitor.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }
    else if (params.mode == GemmUniversalMode::kBatched || params.mode == GemmUniversalMode::kArray) {
      epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
    }

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(epilogue_visitor, accumulators);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
