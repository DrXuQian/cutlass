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

#include <iostream>                                       // C++标准输入输出流

#include "cutlass/cutlass.h"                            // CUTLASS核心头文件

#include "cutlass/conv/kernel/default_conv2d_fprop.h"     // 默认2D卷积前向传播内核
#include "cutlass/conv/device/implicit_gemm_convolution.h"  // 隐式GEMM卷积设备端实现

#include "device/b2b_implicit_gemm_convolution.h"        // B2B隐式GEMM卷积
#include "b2b_interleaved_conv2d_run.h"                  // B2B交错卷积运行器
#include "test_run.h"                                    // 测试运行辅助函数

////////////////////////////////////////////////////////////////////////////////

// 定义两个INT8卷积的问题规模（SM80架构，寄存器文件驻留）
// 第一个卷积：3x3卷积核
cutlass::conv::Conv2dProblemSize conv2d_s8_sm80_problem_size_0 (
    {32, 56, 56, 64},    // input size (NHWC)    // 输入尺寸：批次=32，高=56，宽=56，通道=64
    {64, 3, 3, 64},      // filter size (KRSC)   // 滤波器尺寸：输出通道=64，高=3，宽=3，输入通道=64
    {1, 1, 1, 1},        // padding (pad_h, _, pad_w, _)  // 填充：上=1，下=1，左=1，右=1
    {1, 1},              // stride (stride_h, stride_w)    // 步长：垂直=1，水平=1
    {1, 1},              // dilation (dilation_h, dilation_w)  // 扩张：垂直=1，水平=1
    {32, 56, 56, 64}     // output size (NPQK)    // 输出尺寸：批次=32，高=56，宽=56，通道=64
  );
// 第二个卷积：1x1卷积核（点卷积）
cutlass::conv::Conv2dProblemSize conv2d_s8_sm80_problem_size_1 (
    {32, 56, 56, 64},    // input size (NHWC)    // 输入尺寸（使用第一个卷积的输出）
    {128, 1, 1, 64},     // filter size (KRSC)   // 滤波器尺寸：输出通道=128，1x1卷积核
    {0, 0, 0, 0},        // padding (pad_h, _, pad_w, _)  // 无填充
    {1, 1},              // stride (stride_h, stride_w)    // 步长：1x1
    {1, 1},              // dilation (dilation_h, dilation_w)  // 无扩张
    {32, 56, 56, 128}    // output size (NPQK)    // 输出尺寸：通道数增加到128
  );

// 运行非融合的优化INT8 2D卷积前向传播（SM80架构）
bool run_nonfused_conv2d_fprop_optimized_s8_sm80() {

  using ElementA           = int8_t;   // 输入元素类型：INT8
  using ElementB           = int8_t;   // 滤波器元素类型：INT8
  using ElementC           = int8_t;   // 输出元素类型：INT8
  using ElementAccumulator = int32_t;  // 累加器元素类型：INT32（避免溢出）
  using ElementCompute = float;        // 计算元素类型：单精度浮点数

  ElementCompute alpha0 = ElementCompute(1);  // 第一个卷积的alpha系数
  ElementCompute beta0 = ElementCompute(1);   // beta=1 for bias  // beta=1用于偏置加法
  ElementCompute alpha1 = ElementCompute(1);  // 第二个卷积的alpha系数
  ElementCompute beta1 = ElementCompute(1);   // beta=1 for bias  // beta=1用于偏置加法

  // 定义线程块和Warp形状（适用于INT8卷积）
  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 64>;   // 第一个卷积线程块：128x64x64
  using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 64>;           // 第一个卷积 Warp：64x64x64
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 64>;  // 第二个卷积线程块：128x128x64
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 64>;           // 第二个卷积 Warp：64x64x64
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;      // INT8 Tensor Core指令形状：16x8x32

  // 定义第一个INT8卷积内核（使用交错布局）
  using Conv2dFpropKernel0 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,        // 输入布局：NCxHWx，x=32交错
    ElementB, cutlass::layout::TensorCxRSKx<32>,        // 滤波器布局：CxRSKx，x=32交错
    ElementC, cutlass::layout::TensorNCxHWx<32>,        // 输出布局：NCxHWx，x=32交错
    ElementAccumulator,                                 // 累加器类型
    cutlass::arch::OpClassTensorOp,                     // 使用Tensor Core
    cutlass::arch::Sm80,                                // 目标架构：SM80
    ThreadblockShape0,                                  // 线程块形状
    WarpShape0,                                         // Warp形状
    InstructionShape,                                   // 指令形状
    cutlass::epilogue::thread::LinearCombinationRelu<   // Epilogue：线性组合+ReLU
      ElementC,                                         // 输出类型
      64 / cutlass::sizeof_bits<ElementC>::value,      // 向量化宽度
      ElementAccumulator,                               // 累加器类型
      ElementCompute,                                  // 计算类型
      cutlass::epilogue::thread::ScaleType::NoBetaScaling  // 无beta缩放
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,  // 线程块调度
    3,                                                  // 流水线阶段数
    cutlass::arch::OpMultiplyAddSaturate,              // 使用饱和乘加操作
    cutlass::conv::IteratorAlgorithm::kOptimized       // 使用优化迭代器算法
  >::Kernel;

  using Conv2dFprop0 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel0>;

  using Conv2dFpropKernel1 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop1 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel1>;

  B2bInterleavedNonFusedConv2dRun<Conv2dFprop0, Conv2dFprop1, 32> nonFusedConv2d;

  std::cout << "Running Non-fused back-to-back INT8 interleaved Optimized Convolution Fprops...\n";
  bool pass = nonFusedConv2d.run(conv2d_s8_sm80_problem_size_0, conv2d_s8_sm80_problem_size_1, cutlass::conv::SplitKMode::kSerial,
      alpha0, beta0, alpha1, beta1);

  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}


bool run_fused_conv2d_fprop_optimized_s8_sm80_rf_res() {

  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int8_t;
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
      ElementC,
      8 * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementC,
      64 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;



  using B2bConv2dFpropKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<
    ElementA, cutlass::layout::TensorNCxHWx<32>,
    ElementB, cutlass::layout::TensorCxRSKx<32>,
    ElementC, cutlass::layout::TensorNCxHWx<32>,
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
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;
  
  using B2bConv2dFprop = cutlass::conv::device::B2bImplicitGemmConvolution<B2bConv2dFpropKernel>;

  B2bInterleavedFusedConv2dRun<B2bConv2dFprop, 32> fusedConv2d;

  std::cout << "Running Fused back-to-back INT8 interleaved Optimized Convolution Fprops with RF residency...\n";
  bool pass = fusedConv2d.run(conv2d_s8_sm80_problem_size_0, conv2d_s8_sm80_problem_size_1, cutlass::conv::SplitKMode::kSerial,
      alpha0, beta0, alpha1, beta1);

  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_conv2d_fprop_optimized_s8_sm80,
    &run_fused_conv2d_fprop_optimized_s8_sm80_rf_res
  };

  return testRun(80, funcs, "conv int8 RF residency");


}


////////////////////////////////////////////////////////////////////////////////

