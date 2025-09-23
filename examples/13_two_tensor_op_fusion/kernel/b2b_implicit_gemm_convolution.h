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
    \brief Template for a pipelined Implicit GEMM kernel.
    \brief 流水线隐式GEMM内核的模板
*/

#pragma once

#include "cutlass/cutlass.h"                            // CUTLASS核心头文件

#include "cutlass/aligned_buffer.h"                      // 对齐缓冲区
#include "cutlass/array.h"                               // 数组容器
#include "cutlass/numeric_types.h"                       // 数值类型
#include "cutlass/matrix_shape.h"                        // 矩阵形状
#include "cutlass/semaphore.h"                          // 信号量支持
#include "cutlass/tensor_ref.h"                         // 张量引用
#include "cutlass/layout/tensor.h"                      // 张量布局
#include "cutlass/gemm/gemm.h"                          // GEMM相关定义
#include "cutlass/conv/convolution.h"                   // 卷积操作定义
#include "cutlass/conv/conv2d_problem_size.h"           // 2D卷积问题规模
#include "cutlass/conv/conv3d_problem_size.h"           // 3D卷积问题规模
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"  // 输出迭代器参数

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename B2bMma_,                               ///! Threadblock-scoped matrix multiply-accumulate  // 线程块级矩阵乘加累加
  typename Epilogue_,                             ///! Epilogue  // Epilogue操作
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function  // 线程块调度函数
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)  // 卷积操作类型（前向、梯度、权重梯度）
  typename ConvProblemSize_ = Conv2dProblemSize   ///! Convolutional operator on 2D or 3D problem  // 2D或3D卷积问题规模
>
struct B2bImplicitGemmConvolution {

  using B2bMma = B2bMma_;                                        // B2B矩阵乘加类型
  using Epilogue = Epilogue_;                                    // Epilogue类型
  using EpilogueOutputOp0 = typename B2bMma::OutputOp;          // 第一个GEMM的输出操作
  using EpilogueOutputOp1 = typename Epilogue::OutputOp;        // 第二个GEMM的输出操作
  using ThreadblockSwizzle = ThreadblockSwizzle_;               // 线程块调度策略
  static Operator const kConvolutionalOperator = ConvOperator;  // 卷积操作类型常量

  using ElementA = typename B2bMma::IteratorA0::Element;        // A矩阵元素类型
  using LayoutA = typename B2bMma::IteratorA0::Layout;          // A矩阵布局类型
  using ElementB = typename B2bMma::IteratorB0::Element;        // B矩阵元素类型
  using LayoutB = typename B2bMma::IteratorB0::Layout;          // B矩阵布局类型
  using ElementC = typename EpilogueOutputOp1::ElementOutput;   // C矩阵元素类型

  /// Set output tensor C layout
  /// 设置输出张量C的布局
  using LayoutC = LayoutA;  // C矩阵布局与A矩阵相同

  using ElementAccumulator = typename EpilogueOutputOp0::ElementAccumulator;  // 累加器元素类型
  using ElementCompute = typename EpilogueOutputOp0::ElementCompute;          // 计算元素类型

  /// Scale and Bias
  /// 缩放和偏置
  using ElementScaleBias = typename B2bMma::IteratorAccumulatorScaleBias::Element;  // 缩放/偏置元素类型
  using LayoutScaleBias = typename B2bMma::IteratorAccumulatorScaleBias::Layout;    // 缩放/偏置布局类型

  using WarpMmaOperator0 = typename B2bMma::Policy0::Operator;        // 第一个GEMM的Warp MMA操作符
  using WarpMmaOperator1 = typename B2bMma::Policy1::Operator;        // 第二个GEMM的Warp MMA操作符

  using ArchMmaOperator = typename WarpMmaOperator0::ArchMmaOperator; // 架构级MMA操作符
  using MathOperator = typename ArchMmaOperator::Operator;            // 数学操作符

  using OperatorClass = typename WarpMmaOperator0::OperatorClass;     // 操作符类别
  using ArchTag = typename WarpMmaOperator0::ArchTag;                 // 架构标签

  using ThreadblockShape0 = typename B2bMma::Shape0;          // 第一个GEMM的线程块形状
  using ThreadblockShape1 = typename B2bMma::Shape1;          // 第二个GEMM的线程块形状
  using WarpShape0 = typename WarpMmaOperator0::Shape;        // 第一个GEMM的Warp形状
  using WarpShape1 = typename WarpMmaOperator1::Shape;        // 第二个GEMM的Warp形状
  using InstructionShape = typename ArchMmaOperator::Shape;   // 指令级形状

  static int const kStages = B2bMma::kStages;                // 流水线阶段数
  static IteratorAlgorithm const kIteratorAlgorithm = B2bMma::IteratorA0::kIteratorAlgorithm;  // 迭代器算法 
 
  /// Warp count (concept: GemmShape)
  /// Warp计数（概念：GemmShape）
  using WarpCount0 = typename B2bMma::WarpCount0;             // 第一个GEMM的Warp数量
  static int const kThreadCount = 32 * WarpCount0::kCount;    // 总线程数 = 32 * Warp数量

  using TensorRefA0 = typename B2bMma::IteratorA0::TensorRef;         // 第一个GEMM的A张量引用
  using TensorRefB0 = typename B2bMma::IteratorB0::TensorRef;         // 第一个GEMM的B张量引用
  using TensorRefScaleBias0 = typename B2bMma::IteratorAccumulatorScaleBias::TensorRef;  // 缩放/偏置张量引用
  using TensorRefB1 = typename B2bMma::IteratorB1::TensorRef;         // 第二个GEMM的B张量引用
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;           // C张量引用

  /// Check iterator A and B convolution dimension are the same and
  /// set device::B2bImplicitGemmConvolution::kConvDim
  /// 检查迭代器A和B的卷积维度是否相同，并设置设备级B2bImplicitGemmConvolution::kConvDim
  static_assert(B2bMma::IteratorA0::kConvDim == B2bMma::IteratorB0::kConvDim,
    "Convolution on different dimensions is not supported");  // 不支持不同维度的卷积
  static int const kConvDim = B2bMma::IteratorA0::kConvDim;  // 卷积维度常量

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  /// 卷积维度和问题规模结构（Conv2d或Conv3d）
  using ConvProblemSize = ConvProblemSize_;  // 卷积问题规模类型

  /// Wgrad C stride idx for implicit gemm algorithm
  /// 权重梯度C矩阵步长索引（用于隐式GEMM算法）
  // Conv2d row-major matrix C (KxRSC)   // Conv2d行主序矩阵C（KxRSC）
  // Conv3d row-major matrix C (KxTRSC)  // Conv3d行主序矩阵C（KxTRSC）
  static int const kWgradCStrideIdx =
    cutlass::platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;  // NHWC布局用2，否则用3

  /// This chooses the appropriate stride element of the C tensor.
  /// 选择C张量的合适步长元素
  static int const kTensorCStrideIdx =
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);  // 权重梯度用kWgradCStrideIdx，否则用0

  //
  //
  //
  // 卷积输出迭代器参数类型
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,                                          // C矩阵布局
    typename Epilogue::OutputTileIterator::Layout,   // 输出tile迭代器布局
    TensorRefC,                                       // C张量引用
    ConvOperator,                                     // 卷积操作类型
    ConvProblemSize                                   // 卷积问题规模
    >;

  /// Argument structure
  /// 参数结构体
  struct Arguments {

    //
    // Data members  // 数据成员
    //

    ConvProblemSize problem_size_0;                    // 第一个卷积的问题规模
    ConvProblemSize problem_size_1;                    // 第二个卷积的问题规模
    TensorRefA0 ref_A0;                                // 第一个卷积的输入张量引用
    TensorRefB0 ref_B0;                                // 第一个卷积的滤波器张量引用
    TensorRefC ref_C0;                                 // 第一个卷积的输出张量引用
    TensorRefScaleBias0 ref_Scale0;                    // 第一个卷积的缩放因子引用
    TensorRefScaleBias0 ref_Bias0;                     // 第一个卷积的偏置引用
    TensorRefB1 ref_B1;                                // 第二个卷积的滤波器张量引用
    TensorRefC ref_C1;                                 // 第二个卷积的输入张量引用（源累加器）
    TensorRefC ref_D1;                                 // 第二个卷积的输出张量引用（目标）
    typename EpilogueOutputOp0::Params output_op_0;    // 第一个卷积的输出操作参数
    typename EpilogueOutputOp1::Params output_op_1;    // 第二个卷积的输出操作参数
    SplitKMode split_k_mode;                           // Split-K模式

    //
    // Methods  // 方法
    //

    /// Default ctor
    /// 默认构造函数
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      ConvProblemSize const & problem_size_0,
      ConvProblemSize const & problem_size_1
    ):
      problem_size_0(problem_size_0),
      problem_size_1(problem_size_1) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size_0,
      ConvProblemSize const & problem_size_1,
      TensorRefA0 const & ref_A0,
      TensorRefB0 const & ref_B0,
      TensorRefC const & ref_C0,
      TensorRefScaleBias0 const & ref_Scale0,
      TensorRefScaleBias0 const & ref_Bias0,
      TensorRefB1 const & ref_B1,
      TensorRefC const & ref_C1,
      TensorRefC const & ref_D1,
      typename EpilogueOutputOp0::Params const & output_op_0,
      typename EpilogueOutputOp1::Params const & output_op_1,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size_0(problem_size_0),
      problem_size_1(problem_size_1),
      ref_A0(ref_A0),
      ref_B0(ref_B0),
      ref_C0(ref_C0),
      ref_Scale0(ref_Scale0),
      ref_Bias0(ref_Bias0),
      ref_B1(ref_B1),
      ref_C1(ref_C1),
      ref_D1(ref_D1),
      output_op_0(output_op_0),
      output_op_1(output_op_1),
      split_k_mode(split_k_mode)
    {

    }

  };

  /// Parameters structure
  /// 参数结构体（设备端使用）
  struct Params {
    ConvProblemSize problem_size_0;                    // 第一个卷积问题规模
    ConvProblemSize problem_size_1;                    // 第二个卷积问题规模
    cutlass::gemm::GemmCoord grid_tiled_shape;         // 网格tile形状
    gemm::GemmCoord implicit_gemm_problem_size_0;      // 第一个隐式GEMM问题规模
    gemm::GemmCoord implicit_gemm_problem_size_1;      // 第二个隐式GEMM问题规模
    int swizzle_log_tile;                              // 调度tile的对数
    int gemm_k_iterations_0;                           // 第一个GEMM的K维迭代次数
    int gemm_k_iterations_1;                           // 第二个GEMM的K维迭代次数
    typename B2bMma::IteratorA0::Params iterator_A0;                         // 第一个GEMM的A迭代器参数
    typename B2bMma::IteratorA0::Element const *ptr_A0;                      // 第一个GEMM的A数据指针
    typename B2bMma::IteratorB0::Params iterator_B0;                         // 第一个GEMM的B迭代器参数
    typename B2bMma::IteratorB0::Element const *ptr_B0;                      // 第一个GEMM的B数据指针
    typename Epilogue::OutputTileIterator::Params iterator_C0;               // 第一个GEMM的C迭代器参数
    typename Epilogue::OutputTileIterator::Element *ptr_C0;                  // 第一个GEMM的C数据指针
    typename B2bMma::IteratorAccumulatorScaleBias::Element *ptr_Scale0;      // 缩放因子数据指针
    typename B2bMma::IteratorAccumulatorScaleBias::Element *ptr_Bias0;       // 偏置数据指针
    typename B2bMma::IteratorB1::Params iterator_B1;                         // 第二个GEMM的B迭代器参数
    typename B2bMma::IteratorB1::Element const *ptr_B1;                      // 第二个GEMM的B数据指针
    typename Epilogue::OutputTileIterator::Params iterator_C1;               // 第二个GEMM的C迭代器参数（源）
    typename Epilogue::OutputTileIterator::Element *ptr_C1;                  // 第二个GEMM的C数据指针（源）
    typename Epilogue::OutputTileIterator::Params iterator_D1;               // 第二个GEMM的D迭代器参数（目标）
    typename Epilogue::OutputTileIterator::Element *ptr_D1;                  // 第二个GEMM的D数据指针（目标）
    typename EpilogueOutputOp0::Params output_op_0;                          // 第一个输出操作参数
    typename EpilogueOutputOp1::Params output_op_1;                          // 第二个输出操作参数
    int *semaphore;                                                          // 信号量指针（用于同步）
    SplitKMode split_k_mode;                                                 // Split-K模式

    //
    // Methods  // 方法
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), gemm_k_iterations_0(0), gemm_k_iterations_1(0) { }  // 默认构造函数

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size_0(args.problem_size_0),
      problem_size_1(args.problem_size_1),
      implicit_gemm_problem_size_0(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_0)),
      implicit_gemm_problem_size_1(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_1)),
      iterator_A0(B2bMma::IteratorA0::getParams(args.problem_size_0, args.ref_A0.layout())),
      ptr_A0(args.ref_A0.data()),
      iterator_B0(args.problem_size_0, args.ref_B0.layout()),
      ptr_B0(args.ref_B0.data()),
      iterator_C0(ConvOutputIteratorParameter::layout(args.ref_C0)),
      ptr_C0(args.ref_C0.data()),
      ptr_Scale0(args.ref_Scale0.data()),
      ptr_Bias0(args.ref_Bias0.data()),
      iterator_B1(args.problem_size_1, args.ref_B1.layout()),
      ptr_B1(args.ref_B1.data()),
      iterator_C1(ConvOutputIteratorParameter::layout(args.ref_C1)),
      ptr_C1(args.ref_C1.data()),
      iterator_D1(ConvOutputIteratorParameter::layout(args.ref_D1)),
      ptr_D1(args.ref_D1.data()),
      output_op_0(args.output_op_0),
      output_op_1(args.output_op_1),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      // 计算K维迭代次数
      gemm_k_iterations_0 = implicit_gemm_k_iterations(kConvolutionalOperator, ThreadblockShape0::kK, args.problem_size_0);
      gemm_k_iterations_1 = implicit_gemm_k_iterations(kConvolutionalOperator, ThreadblockShape1::kK, args.problem_size_1);

      ThreadblockSwizzle threadblock_swizzle;  // 创建线程块调度器

      // 获取网格tile形状
      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size_0,
        {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
        args.problem_size_0.split_k_slices);

      // 计算调度tile的对数
      swizzle_log_tile = ThreadblockSwizzle().get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  /// 共享内存存储结构（使用union节省空间）
  union SharedStorage {
    typename B2bMma::B2bMmaSharedStorage main_loop;  // 主循环共享存储
    typename Epilogue::SharedStorage epilogue;       // Epilogue共享存储
  };

  //
  // Methods  // 方法
  //

  CUTLASS_HOST_DEVICE
  B2bImplicitGemmConvolution() { }  // 默认构造函数 

  /// Executes one ImplicitGEMM
  /// 执行一个隐式GEMM操作
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    // 计算线程块位置
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);  // 获取tile偏移

    // Early exit if CTA is out of range
    // 如果CTA超出范围则提前退出
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    // 计算线程在线程块内的位置
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    // 构造A和B操作数的迭代器
    typename B2bMma::IteratorA0 iterator_A0(
      params.iterator_A0,
      params.problem_size_0,
      params.ptr_A0,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * B2bMma::Shape0::kM,
        threadblock_tile_idx.k() * B2bMma::Shape0::kK
      )
    );
    
    typename B2bMma::IteratorB0 iterator_B0(
      params.iterator_B0,
      params.problem_size_0,
      params.ptr_B0,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * B2bMma::Shape0::kK,
        threadblock_tile_idx.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorB1 iterator_B1(
      params.iterator_B1,
      params.problem_size_1,
      params.ptr_B1,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * B2bMma::Shape1::kK,
        threadblock_tile_idx.n() * B2bMma::Shape1::kN
      )
    );


    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    // 将lane 0计算的warp_id广播给所有线程，确保相关代码编译为warp统一的
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);  // 计算warp索引
    int lane_idx = threadIdx.x % 32;                               // 计算lane索引

    // Construct iterators to accumulator scale/bias vector
    // 构造累加器缩放/偏置向量的迭代器
    typename B2bMma::IteratorAccumulatorScaleBias iterator_Scale0(
      params.ptr_Scale0,
      {1, params.problem_size_0.K},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_idx.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorAccumulatorScaleBias iterator_Bias0(
      params.ptr_Bias0,
      {1, params.problem_size_0.K},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_idx.n() * B2bMma::Shape0::kN
      )
    );


    //
    // Main loop  // 主循环
    //

    EpilogueOutputOp0 output_op_0(params.output_op_0);  // 第一个输出操作

    // Construct thread-scoped matrix multiply
    // 构造线程级矩阵乘法
    B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename B2bMma::FragmentC0 src_accum;     // 第一个GEMM的累加器片段
    typename B2bMma::FragmentC1 accumulators;  // 第二个GEMM的累加器片段

    src_accum.clear();     // 清零第一个累加器
    accumulators.clear();  // 清零第二个累加器

    // Compute threadblock-scoped matrix multiply-add
    // 计算线程块级矩阵乘加
    b2bMma(params.gemm_k_iterations_0, accumulators, iterator_A0, iterator_B0,
        iterator_Scale0, iterator_Bias0, iterator_B1, src_accum, output_op_0);

    //
    // Epilogue  // Epilogue阶段
    //

    EpilogueOutputOp1 output_op_1(params.output_op_1);  // 第二个输出操作

    // Construct the semaphore.
    // 构造信号量
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();  // 计算块索引

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);  // 创建信号量对象
    
    // Compute logical position within grid
    // 计算网格内的逻辑位置
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // If performing a reduction via split-K, fetch the initial synchronization
    // 如果通过split-K执行归约，获取初始同步
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      // 初始获取同步锁但不阻塞
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      // 指示输出操作符当前正在更新串行归约中的哪个位置
      output_op_1.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    // 计算线程块偏移
    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * B2bMma::Shape1::kM,  // M维偏移
      threadblock_tile_idx.n() * B2bMma::Shape1::kN   // N维偏移
    );

    // Tile iterator writing to destination tensor
    // 写入目标张量的tile迭代器
    typename Epilogue::OutputTileIterator iterator_D1(
      params.iterator_D1,
      params.ptr_D1,
      ConvOutputIteratorParameter::extent(params.problem_size_1),
      thread_idx,
      threadblock_offset
    );
    
    // Tile iterator reading from source accumulator tensor
    // 从源累加器张量读取的tile迭代器
    typename Epilogue::OutputTileIterator iterator_C1(
      params.iterator_C1,
      params.ptr_C1,
      ConvOutputIteratorParameter::extent(params.problem_size_1),
      thread_idx,
      threadblock_offset
    );


    // Construct the epilogue
    // 构造Epilogue
    Epilogue epilogue(
      shared_storage.epilogue,   // 共享存储
      thread_idx,                // 线程索引
      warp_idx,                  // Warp索引
      lane_idx);                 // Lane索引

    // Wait on the semaphore - this latency may have been covered by iterator construction
    // 等待信号量 - 这个延迟可能已被迭代器构造所覆盖
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      // 对于后续线程块，源矩阵保存在'D'张量中
      if (threadblock_tile_idx.k()) {
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_idx.k());  // 等待信号量

      __threadfence();  // 内存栅栏，确保全局内存一致性
    }
    // Each split-k-slice writes to a unique tensor location
    // 每个split-k切片写入唯一的张量位置
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D1.add_pointer_offset(threadblock_tile_idx.k() *
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size_1));  // 添加指针偏移
    }

    // Run efficient epilogue
    // 运行高效的Epilogue
    epilogue(output_op_1, iterator_D1, accumulators, iterator_C1);
  
    //
    // Release the semaphore  // 释放信号量
    //

    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        // 最后一个线程块重置信号量，为后续网格做准备
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        // 否则，信号量递增
        lock = threadblock_tile_idx.k() + 1;
      }

      semaphore.release(lock);  // 释放信号量
    }
  } 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

