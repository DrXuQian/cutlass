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
    \brief Scheduler for grouped B2b GEMMs
    \brief 分组B2B GEMM的调度器
*/

#pragma once

#include "cutlass/cutlass.h"                                      // CUTLASS核心头文件
#include "cutlass/gemm/gemm.h"                                   // GEMM相关定义
#include "cutlass/matrix_coord.h"                                // 矩阵坐标定义
#include "cutlass/gemm/kernel/grouped_problem_visitor.h"        // 分组问题访问器基类
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"   // GEMM分组问题访问器

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles
/// 访问器类，用于抽象化tile迭代算法
template <typename ThreadblockShape,         // 线程块形状
          GroupScheduleMode GroupScheduleMode_,  // 分组调度模式
          int PrefetchTileCount,              // 预取tile数量
          int ThreadCount,                    // 线程数量
          bool Transposed = false>            // 是否转置（默认false）
struct B2bGemmGroupedProblemVisitor : public GroupedProblemVisitor<
                                            detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>,
                                            ThreadblockShape,
                                            GroupScheduleMode_,
                                            PrefetchTileCount,
                                            ThreadCount> {

  // 类型定义
  using ProblemSizeHelper = detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>;  // 问题规模辅助类
  using Base = GroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape, GroupScheduleMode_, PrefetchTileCount, ThreadCount>;  // 基类
  using BaseParams = typename Base::Params;     // 基类参数类型
  using SharedStorage = typename Base::SharedStorage;  // 共享存储类型
  static bool const kTransposed = Transposed;   // 转置标志常量

  // 数据成员：两个GEMM的问题规模数组指针
  cutlass::gemm::GemmCoord const *problem_sizes0;  // 第一个GEMM的问题规模数组
  cutlass::gemm::GemmCoord const *problem_sizes1;  // 第二个GEMM的问题规模数组

  // 参数结构体
  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes0;  // 第一个GEMM的问题规模数组指针
    cutlass::gemm::GemmCoord const *problem_sizes1;  // 第二个GEMM的问题规模数组指针
    int32_t                         problem_count;   // 问题总数
    void const                     *workspace;       // 工作空间指针（用于临时存储）
    int32_t                         tile_count;      // tile总数

    //
    // Methods  // 方法
    //

    /// Ctor
    /// 默认构造函数
    CUTLASS_HOST_DEVICE
    Params(): problem_sizes0(nullptr), problem_sizes1(nullptr),
              problem_count(0), workspace(nullptr), tile_count(0) { }

    /// Ctor
    /// 参数化构造函数
    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const *problem_sizes0,  // 第一个GEMM问题规模数组
      cutlass::gemm::GemmCoord const *problem_sizes1,  // 第二个GEMM问题规模数组
      int32_t                         problem_count,   // 问题数量
      void const                     *workspace = nullptr,  // 可选的工作空间
      int32_t                         tile_count = 0   // 可选的tile计数
    ):
      problem_sizes0(problem_sizes0),  // 初始化第一个GEMM规模
      problem_sizes1(problem_sizes1),  // 初始化第二个GEMM规模
      problem_count(problem_count),    // 初始化问题计数
      workspace(workspace),            // 初始化工作空间
      tile_count(tile_count)          // 初始化tile计数
    {}

    /// Convert the B2b-GEMM-specific parameters to those used by the base class
    /// 将B2B GEMM特定的参数转换为基类使用的参数
    CUTLASS_HOST_DEVICE
    BaseParams to_base() const {
        return BaseParams(// Set problem_sizes as problem_sizes0 because these determine
                          // shape of the grid used in the non-grouped B2b GEMM
                          // 使用problem_sizes0作为问题规模，因为它决定了非分组B2B GEMM的网格形状
                          problem_sizes0,
                          problem_count,   // 问题数量
                          workspace,       // 工作空间
                          tile_count);     // tile计数
    }

  };

  //
  // Methods  // 方法
  //
  // 设备端构造函数
  CUTLASS_DEVICE
  B2bGemmGroupedProblemVisitor(
    Params const &params_,              // 参数结构体引用
    SharedStorage &shared_storage_,     // 共享存储引用
    int32_t block_idx                   // 线程块索引
  ): Base (                             // 调用基类构造函数
        params_.to_base(),              // 转换后的基类参数
        shared_storage_, block_idx),    // 共享存储和块索引
     problem_sizes0(params_.problem_sizes0),  // 初始化第一个GEMM规模
     problem_sizes1(params_.problem_sizes1)   // 初始化第二个GEMM规模
  {}

  /// Returns the problem size 0 for the current problem
  /// 返回当前问题的第一个GEMM规模
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size0() const {
    GemmCoord problem = problem_sizes0[this->problem_idx];  // 从数组中获取当前问题规模
    ProblemSizeHelper::possibly_transpose_problem(problem); // 根据转置标志可能转置问题
    return problem;                                        // 返回问题规模
  }

  /// Returns the problem size 1 for the current problem
  /// 返回当前问题的第二个GEMM规模
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size1() const {
    GemmCoord problem = problem_sizes1[this->problem_idx];  // 从数组中获取当前问题规模
    ProblemSizeHelper::possibly_transpose_problem(problem); // 根据转置标志可能转置问题
    return problem;                                        // 返回问题规模
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
