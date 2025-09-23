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
    \brief
    Default kernel-level implicit GEMM convolution definitions combine threadblock-scoped
      matrix multiply-add with the appropriate threadblock-scoped epilogue.

    默认内核级隐式GEMM卷积定义，将线程块级矩阵乘加与相应的线程块级epilogue结合。
*/

#pragma once

#include "cutlass/cutlass.h"                                                      // CUTLASS核心头文件
#include "cutlass/conv/kernel/default_conv2d.h"                                  // 默认2D卷积内核

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"  // 激活tile访问迭代器（解析版）
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"     // 滤波器tile访问迭代器（解析版）
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h" // 激活tile访问迭代器（优化版）
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"    // 滤波器tile访问迭代器（优化版）

#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"     // 带谓词的向量访问迭代器
#include "cutlass/transform/threadblock/vector_iterator.h"                        // 向量迭代器
#include "cutlass/transform/warp/vector_fragment_iterator.h"                       // Warp级向量片段迭代器

#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"                   // MMA Tensor Core片段迭代器

#include "kernel/b2b_implicit_gemm_convolution.h"                                 // B2B隐式GEMM卷积内核
#include "threadblock/b2b_implicit_gemm_pipelined.h"                              // B2B隐式GEMM流水线版本
#include "threadblock/b2b_implicit_gemm_multistage.h"                             // B2B隐式GEMM多阶段版本

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop
/// 定义Conv2d前向传播的内核
template <
  typename ElementA,                  // 输入激活的元素类型
  typename LayoutA,                   // 输入激活的布局
  typename ElementB,                  // 滤波器权重的元素类型
  typename LayoutB,                   // 滤波器权重的布局
  typename ElementC,                  // 输出的元素类型
  typename LayoutC,                   // 输出的布局
  typename ElementAccumulator,        // 累加器的元素类型
  typename OperatorClass,             // 操作符类别（如TensorOp）
  typename ArchTag,                   // 架构标签（如Sm75, Sm80）
  typename ThreadblockShape0,         // 第一个卷积的线程块形状
  typename ThreadblockShape1,         // 第二个卷积的线程块形状
  typename WarpShape0,                // 第一个卷积的Warp形状
  typename WarpShape1,                // 第二个卷积的Warp形状
  typename InstructionShape,          // 指令级形状
  typename EpilogueOutputOp0,         // 第一个卷积的Epilogue输出操作
  typename EpilogueOutputOp1,         // 第二个卷积的Epilogue输出操作
  typename ThreadblockSwizzle,        // 线程块调度策略
  int Stages,                         // 流水线阶段数
  typename MathOperatorTag,           // 数学操作符标签
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kAnalytic,  // 迭代器算法（解析或优化）
  bool SmemAccumulator = false        // 是否使用共享内存累加器
> struct DefaultB2bConv2dFprop;        // B2B Conv2d前向传播默认模板

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
