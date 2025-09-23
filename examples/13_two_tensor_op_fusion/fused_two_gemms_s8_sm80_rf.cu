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
#include <iostream>  // C++标准输入输出流

#include "cutlass/cutlass.h"                               // CUTLASS核心头文件
#include "cutlass/gemm/device/gemm.h"                      // 设备端GEMM操作

#include "cutlass/util/host_tensor.h"                      // 主机端张量容器
#include "cutlass/util/tensor_view_io.h"                   // 张量视图I/O
#include "cutlass/util/reference/host/tensor_fill.h"       // 张量填充工具
#include "cutlass/util/reference/host/tensor_copy.h"       // 张量复制工具
#include "cutlass/util/reference/host/tensor_compare.h"    // 张量比较工具
#include "cutlass/util/reference/host/gemm.h"              // 参考GEMM实现

#include "device/b2b_gemm.h"                               // B2B GEMM设备端实现
#include "b2b_interleaved_gemm_run.h"                      // B2B交错GEMM运行器
#include "test_run.h"                                      // 测试运行辅助函数

////////////////////////////////////////////////////////////////////////////////

// 定义两个INT8 GEMM的问题规模（SM80架构）
cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0(128*640, 64, 576);  // 第一个GEMM: [81920, 64, 576]
cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_1(128*640, 128, 64); // 第二个GEMM: [81920, 128, 64]

// 运行非融合的INT8 GEMM（SM80架构）
bool run_nonfused_gemm_s8_sm80() {

  using ElementOutput = int8_t;        // 输出元素类型：8位有符号整数
  using ElementAccumulator = int32_t;  // 累加器元素类型：32位整数（避免溢出）
  using ElementCompute = float;        // 计算元素类型：单精度浮点数

  ElementCompute alpha0 = ElementCompute(1);  // 第一个GEMM的alpha系数
  ElementCompute beta0 = ElementCompute(1);   //beta=1 for bias  // beta=1用于偏置加法
  ElementCompute alpha1 = ElementCompute(1);  // 第二个GEMM的alpha系数
  ElementCompute beta1 = ElementCompute(1);   //beta=1 for bias  // beta=1用于偏置加法

  // 定义线程块和Warp形状（适用于INT8计算）
  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 64>;   // 第一个GEMM线程块：128x64x64
  using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 64>;           // 第一个GEMM Warp：64x64x64
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 64>;  // 第二个GEMM线程块：128x128x64
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 64>;           // 第二个GEMM Warp：64x64x64
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;      // INT8 Tensor Core指令形状：16x8x32

  // 定义第一个INT8 GEMM操作
  using Gemm0 = cutlass::gemm::device::Gemm<
    int8_t,                                           // A矩阵元素类型：INT8
    cutlass::layout::ColumnMajorInterleaved<32>,     // A矩阵布局：列主序交错（32元素交错）
    int8_t,                                           // B矩阵元素类型：INT8
    cutlass::layout::RowMajorInterleaved<32>,        // B矩阵布局：行主序交错（32元素交错）
    ElementOutput,                                    // C/D矩阵元素类型
    cutlass::layout::ColumnMajorInterleaved<32>,     // C/D矩阵布局：列主序交错
    ElementAccumulator,                               // 累加器类型：INT32
    cutlass::arch::OpClassTensorOp,                  // 使用Tensor Core
    cutlass::arch::Sm80,                             // 目标架构：SM80
    ThreadblockShape0,                               // 线程块形状
    WarpShape0,                                      // Warp形状
    InstructionShape,                                // 指令形状
    cutlass::epilogue::thread::LinearCombinationRelu<  // Epilogue操作：线性组合+ReLU
      ElementOutput,                                 // 输出元素类型
      64 / cutlass::sizeof_bits<ElementOutput>::value,  // 向量化宽度
      ElementAccumulator,                            // 累加器类型
      ElementCompute,                               // 计算类型
      cutlass::epilogue::thread::ScaleType::NoBetaScaling  // 不进行beta缩放
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // 线程块调度策略
    3,                                               // 流水线阶段数
    16,                                              // A矩阵对齐要求
    16,                                              // B矩阵对齐要求
    false,                                           // 不使用split-K
    cutlass::arch::OpMultiplyAddSaturate            // 使用饱和乘加操作
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedNonFusedGemmRun<Gemm0, Gemm1, 32> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back INT8 NT interleaved GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_s8_sm80_problem_size_0, gemm_s8_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}


bool run_fused_gemm_s8_sm80_rf_res() {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 64>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      8 * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  const bool SmemAccumulator = false;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    SmemAccumulator,
    16,
    16,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedFusedGemmRun<B2bGemm, 32> fusedGemm;

  std::cout << "Running Fused back-to-back INT8 NT interleaved GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(
    gemm_s8_sm80_problem_size_0,
    gemm_s8_sm80_problem_size_1,
    alpha0,
    beta0,
    alpha1,
    beta1
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

bool run_fused_gemm_s8_sm80_rf_res_batch() {


  cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0(256, 64, 128);
  cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_1(256, 128, 64);

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 64>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 64>;

  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      8 * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  const bool SmemAccumulator = false;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    SmemAccumulator,
    16,
    16,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedFusedGemmRun<B2bGemm, 32> fusedGemm;

  int batch_count = 2;
  int64_t batch_stride_A0 = gemm_s8_sm80_problem_size_0.m() * gemm_s8_sm80_problem_size_0.k();
  int64_t batch_stride_B0 = gemm_s8_sm80_problem_size_1.k() * gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_C0 = gemm_s8_sm80_problem_size_0.m() * gemm_s8_sm80_problem_size_0.n();
  int64_t batch_stride_B1 = gemm_s8_sm80_problem_size_1.k() * gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_C1 = gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_D1 = gemm_s8_sm80_problem_size_1.m() * gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_Bias0 = gemm_s8_sm80_problem_size_0.n();
  int64_t batch_stride_Scale0 = 0;

  std::cout << "Running Fused back-to-back INT8 NT interleaved Batched GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(
    gemm_s8_sm80_problem_size_0,
    gemm_s8_sm80_problem_size_1,
    alpha0,
    beta0,
    alpha1,
    beta1,
    cutlass::gemm::GemmUniversalMode::kBatched,
    batch_count,
    batch_stride_A0,
    batch_stride_B0,
    batch_stride_C0,
    batch_stride_B1,
    batch_stride_C1,
    batch_stride_D1,
    batch_stride_Bias0,
    batch_stride_Scale0
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_s8_sm80,
    &run_fused_gemm_s8_sm80_rf_res,
    &run_fused_gemm_s8_sm80_rf_res_batch
  };

  return testRun(80, funcs, "gemm int8 RF residency");
}

////////////////////////////////////////////////////////////////////////////////
