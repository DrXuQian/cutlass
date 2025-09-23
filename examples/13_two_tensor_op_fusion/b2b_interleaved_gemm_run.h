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
#pragma once

#include <iostream>      // C++标准输入输出流
#include <fstream>       // 文件流操作
#include <sstream>       // 字符串流操作

#include "cutlass/util/host_tensor.h"                      // 主机端张量容器
#include "cutlass/util/tensor_view_io.h"                   // 张量视图I/O
#include "cutlass/util/distribution.h"                     // 数据分布类型
#include "cutlass/util/reference/host/tensor_fill.h"       // 张量填充工具
#include "cutlass/util/reference/host/tensor_copy.h"       // 张量复制工具
#include "cutlass/util/reference/host/tensor_compare.h"    // 张量比较工具
#include "cutlass/util/reference/host/tensor_norm.h"       // 张量范数计算
#include "cutlass/util/host_reorder.h"                     // 主机端数据重排
#include "cutlass/util/reference/device/gemm.h"            // 设备端GEMM参考实现
#include "cutlass/util/reference/device/gemm_complex.h"    // 复数GEMM参考实现
#include "cutlass/util/reference/device/tensor_relu.h"     // ReLU激活函数

#include "reference/device/tensor_scale_bias.h"            // 张量缩放和偏置
#include "helper.h"                                        // 辅助函数

// 检查val1是否大于val2的宏定义
#define CHECK_GT(val1, val2) \
    if((val1) <= (val2)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_GT failed\n";
// 检查条件是否为真的宏定义
#define CHECK_TRUE(val) \
    if(!(val)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_TRUE failed\n";

// B2B交错布局非融合GEMM运行器模板
template <typename Gemm0_, typename Gemm1_, int InterleavedK_>  // InterleavedK_: 交错大小
struct B2bInterleavedNonFusedGemmRun
{

  using Gemm0 = Gemm0_;                                              // 第一个GEMM操作类型
  using Gemm1 = Gemm1_;                                              // 第二个GEMM操作类型
  using ElementAccumulator = typename Gemm0::ElementAccumulator;    // 累加器元素类型
  using ElementCompute = typename Gemm0::GemmKernel::Epilogue::OutputOp::ElementCompute;  // 计算元素类型

  /// Initialization  // 初始化参数
  cutlass::Distribution::Kind init_A;      // A矩阵的初始化分布类型
  cutlass::Distribution::Kind init_B;      // B矩阵的初始化分布类型
  cutlass::Distribution::Kind init_C;      // C矩阵的初始化分布类型
  cutlass::Distribution::Kind init_Bias;   // 偏置的初始化分布类型
  uint64_t seed;                            // 随机数种子

  //
  // Methods
  //

  // 构造函数，设置默认的初始化分布为均匀分布
  B2bInterleavedNonFusedGemmRun(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,      // A矩阵默认均匀分布
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,      // B矩阵默认均匀分布
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,      // C矩阵默认均匀分布
    cutlass::Distribution::Kind init_Bias_ = cutlass::Distribution::Uniform,   // 偏置默认均匀分布
    uint64_t seed_ = 2080                                                      // 默认随机种子
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), init_Bias(init_Bias_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, 2, -2, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view, Element(0));
    }
    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }
    else {
      std::cerr << "Not implemented\n";
      return false;
    }

    return true;
  }




  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size_0,
    cutlass::gemm::GemmCoord problem_size_1,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    //
    // Allocate the GEMM workspace
    //

    cutlass::HostTensor<
      typename Gemm0::ElementA,
      typename Gemm0::LayoutA> tensor_A0(problem_size_0.mk());

    cutlass::HostTensor<
      typename Gemm0::ElementB,
      typename Gemm0::LayoutB> tensor_B0(problem_size_0.kn());

    cutlass::HostTensor<
      typename Gemm0::ElementB,
      typename Gemm0::LayoutB> tensor_B0_reordered(problem_size_0.kn());

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> tensor_C0(problem_size_0.mn());

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> tensor_Bias0({1, problem_size_0.n()});

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> tensor_D0(problem_size_0.mn());

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> reference_D0(problem_size_0.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementB,
      typename Gemm1::LayoutB> tensor_B1(problem_size_1.kn());

    cutlass::HostTensor<
      typename Gemm1::ElementB,
      typename Gemm1::LayoutB> tensor_B1_reordered(problem_size_1.kn());

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> tensor_C1(problem_size_1.mn());

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm1::LayoutC> tensor_Bias1({1, problem_size_1.n()});

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> tensor_D1(problem_size_1.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> reference_D1(problem_size_1.mn());

    // 初始化所有张量的数据（使用不同的随机种子确保数据唯一性）
    CHECK_TRUE(initialize_tensor(tensor_A0.host_view(), init_A, seed + 2019));      // 初始化第一个GEMM的A矩阵
    CHECK_TRUE(initialize_tensor(tensor_B0.host_view(), init_B, seed + 2018));      // 初始化第一个GEMM的B矩阵
    CHECK_TRUE(initialize_tensor(tensor_C0.host_view(), init_C, seed + 2017));      // 初始化第一个GEMM的C矩阵
    CHECK_TRUE(initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed + 2014)); // 初始化第一个GEMM的偏置
    CHECK_TRUE(initialize_tensor(tensor_B1.host_view(), init_B, seed + 2016));      // 初始化第二个GEMM的B矩阵
    CHECK_TRUE(initialize_tensor(tensor_C1.host_view(), init_C, seed + 2015));      // 初始化第二个GEMM的C矩阵
    CHECK_TRUE(initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed + 2013)); // 初始化第二个GEMM的偏置

    // 对B0和B1矩阵进行重排序以适应交错布局
    // Reorder B0 and B1
    cutlass::reorder_column<InterleavedK_>(                                         // 使用InterleavedK参数重排列
        tensor_B0_reordered.host_ref(), tensor_B0.host_ref(), problem_size_0);      // 重排序第一个GEMM的B矩阵
    cutlass::reorder_column<InterleavedK_>(                                         // 使用相同的交错参数
        tensor_B1_reordered.host_ref(), tensor_B1.host_ref(), problem_size_1);      // 重排序第二个GEMM的B矩阵

    // 填充输出张量（初始化为默认值）
    cutlass::reference::host::TensorFill(                                           // 使用TensorFill初始化
      tensor_D0.host_view());                                                       // 第一个GEMM的输出张量
    cutlass::reference::host::TensorFill(                                           // 使用TensorFill初始化
      tensor_D1.host_view());                                                       // 第二个GEMM的输出张量
    cutlass::reference::host::TensorFill(                                           // 使用TensorFill初始化
      reference_D0.host_view());                                                    // 第一个GEMM的参考输出
    cutlass::reference::host::TensorFill(                                           // 使用TensorFill初始化
      reference_D1.host_view());                                                    // 第二个GEMM的参考输出

    // 将所有张量从主机内存同步到设备内存
    tensor_A0.sync_device();           // 同步第一个GEMM的A矩阵到GPU
    tensor_B0.sync_device();           // 同步第一个GEMM的原始B矩阵到GPU
    tensor_B0_reordered.sync_device(); // 同步第一个GEMM的重排序B矩阵到GPU
    tensor_C0.sync_device();           // 同步第一个GEMM的C矩阵到GPU
    tensor_Bias0.sync_device();        // 同步第一个GEMM的偏置到GPU
    tensor_D0.sync_device();           // 同步第一个GEMM的输出到GPU
    tensor_B1.sync_device();           // 同步第二个GEMM的原始B矩阵到GPU
    tensor_B1_reordered.sync_device(); // 同步第二个GEMM的重排序B矩阵到GPU
    tensor_C1.sync_device();           // 同步第二个GEMM的C矩阵到GPU
    tensor_Bias1.sync_device();        // 同步第二个GEMM的偏置到GPU
    tensor_D1.sync_device();           // 同步第二个GEMM的输出到GPU
    reference_D0.sync_device();        // 同步参考实现的第一个输出到GPU
    reference_D1.sync_device();        // 同步参考实现的第二个输出到GPU

    //
    // Initialize the GEMM operator  // 初始化GEMM操作符
    //

    // 构造第一个GEMM的参数
    typename Gemm0::Arguments arguments_0{
      problem_size_0,                                                      // GEMM问题规模
      tensor_A0.device_ref(),                                             // A矩阵的设备引用
      tensor_B0_reordered.device_ref(),                                   // 重排序后的B矩阵设备引用
      {tensor_Bias0.device_data(), typename Gemm0::LayoutC::Stride(0)},   // 偏置向量（stride为0表示广播）
      tensor_D0.device_ref(),                                             // 输出矩阵D的设备引用
      {alpha0, beta0}                                                     // alpha和beta缩放因子
    };

    // 构造第二个GEMM的参数（使用第一个GEMM的输出作为输入）
    typename Gemm1::Arguments arguments_1{
      problem_size_1,                                                      // GEMM问题规模
      tensor_D0.device_ref(),                                             // 使用第一个GEMM的输出作为A矩阵
      tensor_B1_reordered.device_ref(),                                   // 重排序后的B矩阵设备引用
      {tensor_Bias1.device_data(), typename Gemm1::LayoutC::Stride(0)},   // 偏置向量（stride为0表示广播）
      tensor_D1.device_ref(),                                             // 最终输出矩阵D的设备引用
      {alpha1, beta1}                                                     // alpha和beta缩放因子
    };


    // 创建GEMM操作对象
    Gemm0 gemm_op_0;                                                      // 第一个GEMM操作符实例
    Gemm1 gemm_op_1;                                                      // 第二个GEMM操作符实例

    // 初始化第一个GEMM操作符
    cutlass::Status status = gemm_op_0.initialize(arguments_0);          // 传入参数并初始化

    CUTLASS_CHECK(status);                                                // 检查初始化状态

    // 初始化第二个GEMM操作符
    status = gemm_op_1.initialize(arguments_1);                          // 传入参数并初始化

    CUTLASS_CHECK(status);                                                // 检查初始化状态

    // 预热运行（确保GPU达到稳定状态）
    for(int i = 0; i < warm_ups; i++) {                                  // 执行warm_ups次预热
        status = gemm_op_0();                                             // 运行第一个GEMM
        CUTLASS_CHECK(status);                                            // 检查执行状态
        status = gemm_op_1();                                             // 运行第二个GEMM
        CUTLASS_CHECK(status);                                            // 检查执行状态
    }

    //
    // Run the GEMM  // 运行GEMM
    //
    // 创建CUDA事件用于性能测量
    cudaEvent_t start, stop1, stop2;                                     // 定义三个事件：开始、第一个GEMM结束、第二个GEMM结束
    cudaEventCreate(&start);                                             // 创建开始事件
    cudaEventCreate(&stop1);                                             // 创建第一个停止事件
    cudaEventCreate(&stop2);                                             // 创建第二个停止事件

    cudaEventRecord(start);                                              // 记录开始时间

    // 运行第一个GEMM多次并测量性能
    for(int i = 0; i < runs; i++) {                                      // 执行runs次以获得平均性能
        status = gemm_op_0();                                             // 执行第一个GEMM

        CUTLASS_CHECK(status);                                            // 检查执行状态
    }
    cudaEventRecord(stop1);                                              // 记录第一个GEMM结束时间
    // 运行第二个GEMM多次并测量性能
    for(int i = 0; i < runs; i++) {                                      // 执行runs次以获得平均性能
        status = gemm_op_1();                                             // 执行第二个GEMM

        CUTLASS_CHECK(status);                                            // 检查执行状态
    }

    cudaEventRecord(stop2);                                              // 记录所有操作结束时间
    cudaDeviceSynchronize();                                             // 等待所有GPU操作完成
    // 计算各阶段耗时
    float gemm0Time, gemm1Time, totalTime;                               // 定义时间变量
    cudaEventElapsedTime(&gemm0Time, start, stop1);                     // 计算第一个GEMM耗时
    cudaEventElapsedTime(&gemm1Time, stop1, stop2);                     // 计算第二个GEMM耗时
    cudaEventElapsedTime(&totalTime, start, stop2);                     // 计算总耗时
    std::cout << "gemm 0 time " << gemm0Time / (float)runs << " ms\n";  // 输出第一个GEMM平均耗时
    std::cout << "gemm 1 time " << gemm1Time / (float)runs << " ms\n";  // 输出第二个GEMM平均耗时
    std::cout << "Non-fusion time " << totalTime / (float)runs << " ms\n"; // 输出非融合总耗时

    // 将结果从设备内存同步到主机内存
    tensor_D0.sync_host();                                               // 同步第一个GEMM输出到主机
    tensor_D1.sync_host();                                               // 同步第二个GEMM输出到主机

    //
    // Verify  // 验证结果正确性
    //
    // 创建第一个GEMM的参考实现
    cutlass::reference::device::Gemm<
        typename Gemm0::ElementA, typename Gemm0::LayoutA,               // A矩阵类型和布局
        typename Gemm0::ElementB, typename Gemm0::LayoutB,               // B矩阵类型和布局
        typename Gemm0::ElementC, typename Gemm0::LayoutC, ElementCompute, // C矩阵类型、布局和计算类型
        ElementAccumulator, typename Gemm0::Operator>                    // 累加器类型和操作符
        reference_gemm_0;                                                // 参考GEMM实例

    // 创建第二个GEMM的参考实现
    cutlass::reference::device::Gemm<
        typename Gemm1::ElementA, typename Gemm1::LayoutA,               // A矩阵类型和布局
        typename Gemm1::ElementB, typename Gemm1::LayoutB,               // B矩阵类型和布局
        typename Gemm1::ElementC, typename Gemm1::LayoutC, ElementCompute, // C矩阵类型、布局和计算类型
        ElementAccumulator, typename Gemm1::Operator>                    // 累加器类型和操作符
        reference_gemm_1;                                                // 参考GEMM实例

    reference_gemm_0(
      problem_size_0,
      alpha0,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      beta0,
      {tensor_Bias0.device_data(), typename Gemm0::LayoutC::Stride(0)},
      reference_D0.device_ref()
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D0.device_view());
    }

    reference_gemm_1(
      problem_size_1,
      alpha1,
      reference_D0.device_ref(),
      tensor_B1.device_ref(),
      beta1,
      {tensor_Bias1.device_data(), typename Gemm1::LayoutC::Stride(0)},
      reference_D1.device_ref()
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D1.device_view());
    }

    // Wait for kernels to finish
    cudaDeviceSynchronize();
    reference_D0.sync_host();
    reference_D1.sync_host();

    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D1.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(),
      tensor_D1.host_view());

    CHECK_TRUE(passed);
    if (!passed) {

      std::stringstream fname;

      fname << "error_B2bGemm_device_interleaved_nonfused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file
        << "A0 =\n" << tensor_A0.host_view()
        << "\nB0 =\n" << tensor_B0.host_view()
        << "\nB0_reordered =\n" << tensor_B0_reordered.host_view()
        << "\nC0 =\n" << tensor_C0.host_view()
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nD0 =\n" << tensor_D0.host_view()
        << "\nB1 =\n" << tensor_B1.host_view()
        << "\nB1_reordered =\n" << tensor_B1_reordered.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\n\nReference =\n" << reference_D1.host_view()
        << "\nComputed =\n" << tensor_D1.host_view();
    }
    return passed;
  }
};

template <typename B2bGemm_, int InterleavedK_>
struct B2bInterleavedFusedGemmRun
{

  using B2bGemm = B2bGemm_;
  using ElementAccumulator = typename B2bGemm::ElementAccumulator;
  using ElementCompute = typename B2bGemm::B2bGemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_Scale;
  cutlass::Distribution::Kind init_Bias;
  uint64_t seed;

  //
  // Methods
  //

  B2bInterleavedFusedGemmRun(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_),
    init_Scale(init_Scale_), init_Bias(init_Bias_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, 2, -2, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view, Element(0));
    }
    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }
    else {
      std::cerr << "Not implemented\n";
      return false;
    }

    return true;
  }




  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size_0,
    cutlass::gemm::GemmCoord problem_size_1,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm,

    // batch_count is used as split-k when mode is kGemm according
    // to the GemmUniversal interface

    int batch_count = 1,

    int64_t batch_stride_A0 = 0,
    int64_t batch_stride_B0 = 0,
    int64_t batch_stride_C0 = 0,
    int64_t batch_stride_B1 = 0,
    int64_t batch_stride_C1 = 0,
    int64_t batch_stride_D1 = 0,
    int64_t batch_stride_Bias0 = 0,
    int64_t batch_stride_Scale0 = 0,
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    //
    // Allocate the GEMM workspace
    //

    cutlass::gemm::GemmCoord CoordA0(problem_size_0.m(), problem_size_0.n(), batch_count * problem_size_0.k());
    cutlass::gemm::GemmCoord CoordB0(problem_size_0.m(), problem_size_0.n(), batch_count * problem_size_0.k());
    cutlass::gemm::GemmCoord CoordC0(problem_size_0.m(), batch_count * problem_size_0.n(), problem_size_0.k());
    cutlass::gemm::GemmCoord CoordB1(problem_size_1.m(), problem_size_1.n(), batch_count * problem_size_1.k());
    cutlass::gemm::GemmCoord CoordC1(problem_size_1.m(), batch_count * problem_size_1.n(), problem_size_1.k());

    cutlass::HostTensor<
      typename B2bGemm::ElementA,
      typename B2bGemm::LayoutA> tensor_A0(CoordA0.mk());

    cutlass::HostTensor<
      typename B2bGemm::ElementB,
      typename B2bGemm::LayoutB> tensor_B0(CoordB0.kn());

    cutlass::HostTensor<
      typename B2bGemm::ElementB,
      typename B2bGemm::LayoutB> tensor_B0_reordered(CoordB0.kn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> tensor_C0(CoordC0.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementScaleBias,
      typename B2bGemm::LayoutScaleBias> tensor_Scale0;

    if(alpha0 == ElementCompute(0)) //per-channel scale
        tensor_Scale0.resize({1, batch_count * problem_size_0.n()});

    cutlass::HostTensor<
      typename B2bGemm::ElementScaleBias,
      typename B2bGemm::LayoutScaleBias> tensor_Bias0({1, batch_count * problem_size_0.n()});

    cutlass::HostTensor<
      ElementAccumulator,
      typename B2bGemm::LayoutC> reference_Z0(CoordC0.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> reference_D0(CoordC0.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementB,
      typename B2bGemm::LayoutB> tensor_B1(CoordB1.kn());

    cutlass::HostTensor<
      typename B2bGemm::ElementB,
      typename B2bGemm::LayoutB> tensor_B1_reordered(CoordB1.kn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> tensor_C1(CoordC1.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutScaleBias> tensor_Bias1({1, batch_count * problem_size_1.n()});

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> tensor_D1(CoordC1.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> reference_D1(CoordC1.mn());


    CHECK_TRUE(initialize_tensor(tensor_A0.host_view(), init_A, seed + 2019));
    CHECK_TRUE(initialize_tensor(tensor_B0.host_view(), init_B, seed + 2018));
    CHECK_TRUE(initialize_tensor(tensor_C0.host_view(), init_C, seed + 2017));
    if(alpha0 == ElementCompute(0)) //per-channel scale
      CHECK_TRUE(initialize_tensor(tensor_Scale0.host_view(), init_Scale, seed + 2014));
    CHECK_TRUE(initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed + 2013));
    CHECK_TRUE(initialize_tensor(tensor_B1.host_view(), init_B, seed + 2016));
    CHECK_TRUE(initialize_tensor(tensor_C1.host_view(), init_C, seed + 2015));
    CHECK_TRUE(initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed + 2012));

    //Reorder B0
    cutlass::reorder_column<16>(
        tensor_B0_reordered.host_ref(), tensor_B0.host_ref(), CoordB0);
    cutlass::reorder_column<InterleavedK_>(
        tensor_B1_reordered.host_ref(), tensor_B1.host_ref(), CoordB1);

    cutlass::reference::host::TensorFill(
      tensor_D1.host_view());
    cutlass::reference::host::TensorFill(
      reference_D0.host_view());
    cutlass::reference::host::TensorFill(
      reference_D1.host_view());

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_B0_reordered.sync_device();
    tensor_C0.sync_device();
    if(alpha0 == ElementCompute(0)) //per-channel scale
        tensor_Scale0.sync_device();
    tensor_Bias0.sync_device();
    tensor_B1.sync_device();
    tensor_B1_reordered.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D1.sync_device();
    reference_D0.sync_device();
    reference_D1.sync_device();
    // tensor_Bias0_batched.sync_device();

    //
    // Initialize the GEMM operator
    //

    typename B2bGemm::Arguments arguments{
      mode,
      problem_size_0,
      problem_size_1,
      tensor_A0.device_ref(),
      tensor_B0_reordered.device_ref(),
      tensor_C0.device_ref(),
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref(),
      tensor_B1_reordered.device_ref(),
      {tensor_Bias1.device_data(), typename B2bGemm::LayoutC::Stride(0)},
      tensor_D1.device_ref(),
      batch_stride_A0,
      batch_stride_B0,
      batch_stride_B1,
      batch_stride_C1,
      batch_stride_D1,
      batch_stride_Bias0,
      batch_stride_Scale0,
      {alpha0, beta0},
      {alpha1, beta1},
      batch_count,
    };

    B2bGemm b2b_gemm_op;

    cutlass::Status status = b2b_gemm_op.can_implement(arguments);

    if(status != cutlass::Status::kSuccess) {
        std::cout << "Problem sizes not supported.\n"
                << "Requirments:\n"
                << "    problem_size_0.M = problem_size_1.M\n"
                << "    problem_size_0.N = problem_size_1.K\n"
                << "    ThreadblockShape0::kN = problem_size_0.N\n"
                << "    ThreadblockShape1::kN = problem_size_1.N" << std::endl;
    }

    status = b2b_gemm_op.initialize(arguments);

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = b2b_gemm_op();
        CUTLASS_CHECK(status);
    }

    //
    // Run the GEMM
    //

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++) {
        status = b2b_gemm_op();

        CUTLASS_CHECK(status);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float gemmTime;
    cudaEventElapsedTime(&gemmTime, start, stop);
    std::cout << "Fusion time " << gemmTime / (float)runs << " ms\n";

    tensor_D1.sync_host();

    //
    // Verify
    //

    cutlass::reference::device::GemmComplex<
      typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
      typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
      ElementAccumulator, typename B2bGemm::LayoutC,
      ElementAccumulator, ElementAccumulator
    >(
      problem_size_0,
      ElementAccumulator(1), //intermediate alpha=1
      tensor_A0.device_ref(),
      cutlass::ComplexTransform::kNone,
      tensor_B0.device_ref(),
      cutlass::ComplexTransform::kNone,
      ElementAccumulator(0), //beta = 0
      reference_Z0.device_ref(),
      reference_Z0.device_ref(),
      ElementAccumulator(0),
      int(batch_count),
      batch_stride_A0,
      batch_stride_B0,
      batch_stride_C0,
      batch_stride_C0
    );

    cutlass::reference::device::TensorScaleBiasGemmBatched<
      ElementAccumulator, typename B2bGemm::ElementC, typename B2bGemm::LayoutC,
      ElementCompute, typename B2bGemm::LayoutScaleBias
    > (
      problem_size_0,
      reference_Z0.device_ref(),
      reference_D0.device_ref(),
      alpha0,
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref(),
      int(batch_count),
      batch_stride_C0,
      batch_stride_C0,
      batch_stride_Scale0,
      batch_stride_Bias0
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D0.device_view());
    }

    cutlass::reference::device::GemmComplex<
      typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
      typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
      typename B2bGemm::ElementC, typename B2bGemm::LayoutC,
      ElementCompute, ElementAccumulator
    >(
      problem_size_1,
      alpha1, //intermediate alpha=1
      reference_D0.device_ref(),
      cutlass::ComplexTransform::kNone,
      tensor_B1.device_ref(),
      cutlass::ComplexTransform::kNone,
      beta1, //beta = 0
      {tensor_Bias1.device_data(), typename B2bGemm::LayoutC::Stride(0)},
      reference_D1.device_ref(),
      ElementAccumulator(0),
      int(batch_count),
      batch_stride_C0,
      batch_stride_B1,
      batch_stride_C1,
      batch_stride_D1
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D1.device_view());
    }

    cudaDeviceSynchronize();
    reference_D0.sync_host();
    reference_D1.sync_host();

    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D1.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(),
      tensor_D1.host_view());

    CHECK_TRUE(passed);
    if (!passed)
    {

      std::stringstream fname;

      fname << "error_B2bGemm_device_interleaved_fused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file
        << "A0 =\n" << tensor_A0.host_view()
        << "\nB0 =\n" << tensor_B0.host_view()
        << "\nB0_reordered =\n" << tensor_B0_reordered.host_view()
        << "\nC0 =\n" << tensor_C0.host_view()
        << "\nScale0:\n" << tensor_Scale0.host_view() << "\n"
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nB1 =\n" << tensor_B1.host_view()
        << "\nB1_reordered =\n" << tensor_B1_reordered.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\n\nReference =\n" << reference_D1.host_view()
        << "\nComputed =\n" << tensor_D1.host_view();
    }
    return passed;
  }

};

////////////////////////////////////////////////////////////////////////////////
