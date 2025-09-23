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
#include "cutlass/gemm/device/gemm.h"                      // CUTLASS设备端GEMM操作

#include "cutlass/util/host_tensor.h"                      // 主机端张量容器
#include "cutlass/util/tensor_view_io.h"                   // 张量视图I/O工具
#include "cutlass/util/reference/host/tensor_fill.h"       // 张量填充工具
#include "cutlass/util/reference/host/tensor_copy.h"       // 张量复制工具
#include "cutlass/util/reference/host/tensor_compare.h"    // 张量比较工具
#include "cutlass/util/reference/host/gemm.h"              // 参考实现的GEMM

#include "device/b2b_gemm.h"                               // B2B（back-to-back）GEMM设备端实现
#include "b2b_gemm_run.h"                                  // B2B GEMM运行辅助函数
#include "test_run.h"                                      // 测试运行辅助函数

////////////////////////////////////////////////////////////////////////////////

// 定义两个GEMM操作的问题规模
// 第一个GEMM: [M=81920, N=64, K=576]
cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_0(128*640, 64, 576);  // M = 128*640 = 81920
// 第二个GEMM: [M=81920, N=128, K=64] - 注意K维度与第一个GEMM的N维度匹配
cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_1(128*640, 128, 64);

bool run_nonfused_gemm_f16_sm80() {  // 运行非融合的FP16 GEMM（基准版本）

  using ElementOutput = cutlass::half_t;       // 输出元素类型：半精度浮点数
  using ElementAccumulator = cutlass::half_t;  // 累加器元素类型：半精度浮点数
  using ElementCompute = cutlass::half_t;      // 计算元素类型：半精度浮点数

  ElementCompute alpha0 = ElementCompute(1);  // 第一个GEMM的alpha系数
  ElementCompute beta0 = ElementCompute(1); //beta=1 for bias  // 第一个GEMM的beta系数，设为1用于偏置加法
  ElementCompute alpha1 = ElementCompute(1);  // 第二个GEMM的alpha系数
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias  // 第二个GEMM的beta系数，设为1用于偏置加法

  // 定义第一个GEMM的线程块形状和warp形状
  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 32>;  // 线程块处理的瓦片大小：128x64x32
  using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;          // Warp处理的瓦片大小：64x64x32
  // 定义第二个GEMM的线程块形状和warp形状
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>; // 线程块处理的瓦片大小：128x128x32
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;          // Warp处理的瓦片大小：64x64x32
  // Tensor Core指令形状（适用于SM80架构）
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     // MMA指令形状：16x8x16

  // 定义第一个GEMM操作
  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                        // A矩阵元素类型
    cutlass::layout::RowMajor,              // A矩阵布局：行主序
    cutlass::half_t,                        // B矩阵元素类型
    cutlass::layout::ColumnMajor,           // B矩阵布局：列主序（TN GEMM中的T）
    ElementOutput,                          // C/D矩阵元素类型
    cutlass::layout::RowMajor,              // C/D矩阵布局：行主序
    ElementAccumulator,                     // 累加器元素类型
    cutlass::arch::OpClassTensorOp,         // 使用Tensor Core操作
    cutlass::arch::Sm80,                    // 目标架构：SM80（Ampere）
    ThreadblockShape0,                      // 线程块瓦片形状
    WarpShape0,                             // Warp瓦片形状
    InstructionShape,                       // MMA指令形状
    cutlass::epilogue::thread::LinearCombinationRelu<  // Epilogue操作：线性组合+ReLU
      ElementOutput,                        // 输出元素类型
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // 向量化存储宽度
      ElementAccumulator,                   // 累加器类型
      ElementCompute,                       // 计算类型
      cutlass::epilogue::thread::ScaleType::NoBetaScaling  // 不进行beta缩放
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,  // 线程块调度策略
    3                                       // 流水线阶段数
  >;
  // 定义第二个GEMM操作（配置与第一个类似，但使用不同的线程块形状）
  using Gemm1 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                        // A矩阵元素类型
    cutlass::layout::RowMajor,              // A矩阵布局：行主序
    cutlass::half_t,                        // B矩阵元素类型
    cutlass::layout::ColumnMajor,           // B矩阵布局：列主序
    ElementOutput,                          // C/D矩阵元素类型
    cutlass::layout::RowMajor,              // C/D矩阵布局：行主序
    ElementAccumulator,                     // 累加器元素类型
    cutlass::arch::OpClassTensorOp,         // 使用Tensor Core操作
    cutlass::arch::Sm80,                    // 目标架构：SM80
    ThreadblockShape1,                      // 使用更大的线程块形状（128x128x32）
    WarpShape1,                             // Warp瓦片形状
    InstructionShape,                       // MMA指令形状
    cutlass::epilogue::thread::LinearCombinationRelu<  // Epilogue操作：线性组合+ReLU
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,  // 线程块调度策略
    3                                       // 流水线阶段数
  >;

  // 创建非融合的B2B GEMM运行器
  B2bNonFusedGemmRun<Gemm0, Gemm1> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back FP16 TN GEMMs...\n";  // 运行非融合的背靠背FP16 TN GEMM
  // 运行两个独立的GEMM操作
  bool pass = nonFusedGemm.run(gemm_f16_sm80_problem_size_0, gemm_f16_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";  // 测试通过
  else
    std::cout << "Fail\n";  // 测试失败

  return pass;
}

bool run_fused_gemm_f16_sm80_rf_res() {  // 运行融合的FP16 GEMM（寄存器文件驻留版本）

  using ElementOutput = cutlass::half_t;       // 输出元素类型：半精度浮点数
  using ElementAccumulator = cutlass::half_t;  // 累加器元素类型：半精度浮点数
  using ElementCompute = cutlass::half_t;      // 计算元素类型：半精度浮点数

  ElementCompute alpha0 = ElementCompute(1);  // 第一个GEMM的alpha系数
  //Fused kernel has built-in bias, setting beta=0  // 融合内核有内置偏置，设置beta=0
  ElementCompute beta0 = ElementCompute(0);   // 融合版本中第一个GEMM的beta设为0（内置偏置处理）
  ElementCompute alpha1 = ElementCompute(1);  // 第二个GEMM的alpha系数
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias  // 第二个GEMM的beta系数，设为1用于偏置加法

  // 融合版本使用不同的瓦片形状配置（优化寄存器使用）
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;   // 第一个GEMM线程块：64x64x32（较小）
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 32>;          // 第一个GEMM Warp：16x64x32
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;  // 第二个GEMM线程块：64x128x32
  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 32>;         // 第二个GEMM Warp：16x128x32
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     // Tensor Core MMA指令形状

  // 第一个GEMM的epilogue操作（输出处理）
  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,                                      // 输出元素类型
      InstructionShape::kM * InstructionShape::kN / 32,   // 向量化存储宽度（基于指令形状计算）
      ElementAccumulator,                                 // 累加器类型
      ElementCompute,                                     // 计算类型
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling  // 仅进行alpha缩放（无beta）
    >;

  // 第二个GEMM的epilogue操作
  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,                                      // 输出元素类型
      128 / cutlass::sizeof_bits<ElementOutput>::value,   // 向量化存储宽度（128位/元素位数）
      ElementAccumulator,                                 // 累加器类型
      ElementCompute,                                     // 计算类型
      cutlass::epilogue::thread::ScaleType::NoBetaScaling // 不进行beta缩放
    >;

  // 定义融合的B2B GEMM操作
  using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,                        // A矩阵元素类型
    cutlass::layout::RowMajor,              // A矩阵布局：行主序
    cutlass::half_t,                        // B矩阵元素类型
    cutlass::layout::ColumnMajor,           // B矩阵布局：列主序
    ElementOutput,                          // 输出元素类型
    cutlass::layout::RowMajor,              // 输出布局：行主序
    ElementAccumulator,                     // 累加器类型
    cutlass::arch::OpClassTensorOp,         // 使用Tensor Core
    cutlass::arch::Sm80,                    // 目标架构：SM80
    ThreadblockShape0,                      // 第一个GEMM的线程块形状
    ThreadblockShape1,                      // 第二个GEMM的线程块形状
    WarpShape0,                             // 第一个GEMM的Warp形状
    WarpShape1,                             // 第二个GEMM的Warp形状
    InstructionShape,                       // MMA指令形状
    EpilogueOutputOp0,                      // 第一个GEMM的epilogue操作
    EpilogueOutputOp1,                      // 第二个GEMM的epilogue操作
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,  // 线程块调度策略
    3                                       // 流水线阶段数
  >;

  // 创建融合的B2B GEMM运行器
  B2bFusedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN GEMMs with RF residency...\n";  // 运行具有寄存器文件驻留的融合B2B FP16 TN GEMM
  // 运行融合的GEMM操作（中间结果保留在寄存器中）
  bool passed = fusedGemm.run(gemm_f16_sm80_problem_size_0, gemm_f16_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";  // 测试通过
  else
    std::cout << "Fail\n";  // 测试失败

  return passed;

}

int main() {  // 主函数入口

  // 创建函数指针数组，包含所有要测试的函数
  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_f16_sm80,      // 非融合版本（基准）
    &run_fused_gemm_f16_sm80_rf_res   // 融合版本（寄存器文件驻留）
  };

  // 运行测试，需要SM80（Ampere）架构，测试名称为"gemm f16 RF residency"
  return testRun(80, funcs, "gemm f16 RF residency");


}




////////////////////////////////////////////////////////////////////////////////
