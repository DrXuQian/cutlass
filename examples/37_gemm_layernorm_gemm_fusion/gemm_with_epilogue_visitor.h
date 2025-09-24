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
    \brief GEMM kernel to support the epilogue visitor model
    for customized layernorm partial reduction epilogue fusion.
    \brief 支持epilogue visitor模式的GEMM核函数，用于自定义LayerNorm部分规约epilogue融合

    This source file will likely be moved to `include/cutlass/gemm/kernel/` in the future once
    its usage has been stabilized. For now, it is included in this example to demonstrate
    some basic output fusion options.

    这个源文件未来可能会移动到`include/cutlass/gemm/kernel/`目录中。
    目前它包含在这个示例中，用于演示一些基本的输出融合选项。
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// GEMM核函数，支持Epilogue Visitor模式
// Epilogue Visitor是一种设计模式，允许在epilogue阶段执行自定义操作
template <
  typename Mma_,                  ///! 线程块级别的矩阵乘累加单元
  typename Epilogue_,             ///! Epilogue操作
  typename ThreadblockSwizzle_    ///! 线程块交织函数（用于优化内存访问模式）
>
struct GemmWithEpilogueVisitor {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp计数（GemmShape概念）
  using WarpCount = typename Mma::WarpCount;
  // 线程总数 = 32线程/warp * warp数量
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K保持128位对齐的分割
  // Split-K是一种将K维度分割成多个部分的优化技术
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value,
    128 / sizeof_bits<ElementB>::value
  );

  //
  // Structures
  //

  /// 参数结构体 - 包含从CPU传递到GPU的所有参数
  struct Arguments {

    //
    // 数据成员
    //

    GemmUniversalMode mode;              // GEMM模式（普通/批处理/Split-K等）
    GemmCoord problem_size;              // 问题规模 [M, N, K]

    TensorRefA ref_A;                    // 输入矩阵A的引用
    TensorRefB ref_B;                    // 输入矩阵B的引用

    typename EpilogueVisitor::Arguments epilogue_visitor;  // Epilogue visitor的参数

    //
    // Methods
    //

    Arguments():
      mode(GemmUniversalMode::kGemm)
    { }


    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode_,
      GemmCoord problem_size_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      typename EpilogueVisitor::Arguments epilogue_visitor_
    ):
      mode(mode_),
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      epilogue_visitor(epilogue_visitor_)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// 参数结构体 - 预计算的参数，直接传递给kernel
  struct Params {

    cutlass::gemm::GemmCoord problem_size;       // 问题规模
    cutlass::gemm::GemmCoord grid_tiled_shape;   // Grid的tile形状
    int swizzle_log_tile;                        // 交织的log tile大小

    typename Mma::IteratorA::Params params_A;    // A矩阵迭代器参数
    typename Mma::IteratorB::Params params_B;    // B矩阵迭代器参数

    GemmUniversalMode mode;                      // GEMM模式
    int gemm_k_size;                             // K维度大小

    void * ptr_A;                                // A矩阵指针
    void * ptr_B;                                // B矩阵指针

    typename EpilogueVisitor::Params epilogue_visitor;  // Epilogue visitor参数

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      swizzle_log_tile(0),
      params_A(0),
      params_B(0),
      gemm_k_size(0),
      mode(cutlass::gemm::GemmUniversalMode::kGemm),
      ptr_A(nullptr),
      ptr_B(nullptr)
    { }


    Params(
      Arguments const &args
    ):
      problem_size(args.problem_size),
      swizzle_log_tile(0),
      params_A(args.ref_A.layout()),
      params_B(args.ref_B.layout()),
      mode(args.mode),
      gemm_k_size(args.problem_size.k()),
      ptr_A(args.ref_A.data()),
      ptr_B(args.ref_B.data()),
      epilogue_visitor(args.epilogue_visitor)
    {

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, 1);

      if (args.mode == GemmUniversalMode::kGemm || args.mode == GemmUniversalMode::kGemmSplitKParallel) {

        int const kAlignK = const_max(const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value), 1);

        gemm_k_size = round_up(args.problem_size.k(), kAlignK);

        if (gemm_k_size) {
          grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
        }
      }

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// 共享内存存储结构
  // 使用union节省共享内存：主循环和epilogue不会同时使用
  union SharedStorage {

    typename Mma::SharedStorage main_loop;        // 主循环使用的共享内存

    struct {
      typename Epilogue::SharedStorage epilogue;  // Epilogue使用的共享内存
      typename EpilogueVisitor::SharedStorage visitor;  // Visitor使用的共享内存
    } epilogue;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmWithEpilogueVisitor() { }

  /// 检查kernel是否满足对齐要求
  // 对齐是高性能的关键，不满足对齐要求会导致性能下降或错误
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmWithEpilogueVisitor::can_implement()");

    // 获取各个操作数的对齐要求
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

  /// 执行一个GEMM操作
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // 计算线程块位置
    // Swizzle技术用于打乱线程块执行顺序，减少内存bank冲突
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // 早期退出：如果CTA（线程块）超出范围
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    // 计算逻辑坐标中的初始位置
    // 每个线程块处理A矩阵的一个[M, K]块
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,  // M方向偏移
      offset_k,                                       // K方向偏移
    };

    // 每个线程块处理B矩阵的一个[K, N]块
    cutlass::MatrixCoord tb_offset_B{
      offset_k,                                       // K方向偏移
      threadblock_tile_offset.n() * Mma::Shape::kN   // N方向偏移
    };

    // 计算线程在线程块内的位置
    int thread_idx = threadIdx.x;

    // 构造A和B操作数的迭代器
    // 迭代器负责从全局内存加载数据到共享内存
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},  // A矩阵形状
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},  // B矩阵形状
      thread_idx,
      tb_offset_B);

    // 广播lane 0计算的warp_id，确保相关代码被编译为warp统一的
    // Warp统一执行可以避免分支分歧，提高性能
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    // 计算线程在warp内的位置（0-31）
    int lane_idx = threadIdx.x % 32;

    //
    // 主循环 - 执行矩阵乘法
    //

    // 构造线程级别的矩阵乘法对象
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    // 定义累加器片段（存储部分结果）
    typename Mma::FragmentC accumulators;

    // 清零累加器
    accumulators.clear();

    // 计算K维度的迭代次数
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // 执行线程块级别的矩阵乘累加
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
    // 构造epilogue visitor
    // Visitor模式允许在epilogue阶段执行自定义操作
    //

    EpilogueVisitor epilogue_visitor(
      params.epilogue_visitor,
      shared_storage.epilogue.visitor,
      params.problem_size.mn(),          // 输出矩阵大小
      thread_idx,
      warp_idx,
      lane_idx,
      threadblock_offset);

    if (params.mode == GemmUniversalMode::kGemm) {
      // 指示在串行规约中输出操作器当前更新的位置
      // 用于Split-K模式
      epilogue_visitor.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }
    else if (params.mode == GemmUniversalMode::kBatched || params.mode == GemmUniversalMode::kArray) {
      // 设置批次索引（用于批处理GEMM）
      epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
    }

    // 构造epilogue对象
    Epilogue epilogue(
      shared_storage.epilogue.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // 执行epilogue操作器以更新目标张量
    // 这里会调用epilogue_visitor的方法来完成自定义操作
    // 对于LayerNorm融合，这里会计算部分和与部分平方和
    epilogue(epilogue_visitor, accumulators);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
