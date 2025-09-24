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

/**
 * CUTLASS Example 23: GEMM 与操作数归约融合
 *
 * 本示例演示了一种高级融合技术，将 GEMM 计算与沿 K 维度的操作数归约相结合。
 * 这同时产生标准 GEMM 输出 C = alpha * A * B + beta * C 和一个归约向量（Mx1 或 1xN）。
 *
 * 核心特性：
 * =========
 * - 将归约操作与 GEMM 计算融合，避免额外的内核启动
 * - 在 GEMM 执行期间沿 K 维度归约 A 或 B 操作数
 * - 针对 Ampere 架构的 16x8x16 Tensor Core 操作优化
 * - 支持串行和并行 split-K 策略
 *
 * 性能优势：
 * =========
 * - 消除单独的归约内核启动开销
 * - 通过避免额外的内存读取降低带宽需求
 * - 归约在 GEMM 计算过程中进行，提供更好的数据局部性
 *
 * 应用场景：
 * =========
 * - 在执行矩阵乘法时计算行/列和
 * - 批归一化（Batch Normalization）计算
 * - 神经网络中的统计操作
 * - 注意力机制中的 softmax 归一化
 *
 * 实现细节：
 * =========
 * - 归约主要在 warp 级别进行（gemm/warp/mma_with_reduction_tensor_op.h）
 * - 最终归约在后处理阶段完成（epilogue/threadblock/epilogue_gemm_k_reduction.h）
 * - 使用 FP16/BF16 数据类型，在 Ampere SM80 架构上运行
 * - 通过双缓冲技术隐藏内存延迟
 */

// 标准 C++ 库
#include <iostream>
#include <fstream>
#include <sstream>

// CUTLASS 核心库
#include "cutlass/cutlass.h"
// 带 K 维归约的 GEMM 设备接口
#include "cutlass/gemm/device/gemm_with_k_reduction.h"
// 带 K 维归约的默认 GEMM 内核
#include "cutlass/gemm/kernel/default_gemm_with_k_reduction.h"
// Split-K 归约设备接口
#include "cutlass/reduction/device/reduce_split_k.h"
// Split-K 归约内核
#include "cutlass/reduction/kernel/reduce_split_k.h"
// 线程级归约操作符
#include "cutlass/reduction/thread/reduction_operators.h"
// 矩阵坐标辅助类
#include "cutlass/matrix_coord.h"

// CUTLASS 实用工具
#include "cutlass/util/command_line.h"        // 命令行参数解析
#include "cutlass/util/host_tensor.h"         // 主机端张量容器
#include "cutlass/util/tensor_view_io.h"      // 张量 I/O 操作
#include "cutlass/util/reference/device/gemm.h"  // 设备端参考 GEMM 实现
#include "cutlass/util/reference/host/tensor_compare.h"  // 张量比较工具
#include "cutlass/util/reference/host/tensor_copy.h"     // 张量复制工具
#include "cutlass/util/reference/host/tensor_fill.h"     // 张量填充工具
#include "cutlass/util/reference/device/convolution.h"   // 设备端卷积参考实现

// 辅助函数头文件
#include "helper.h"

// =====================================================================
// 数据类型配置
// =====================================================================
// 定义输入、输出张量和计算过程中使用的数据类型
using ElementAccumulator = float;                  // 累加器数据类型 - 使用 float 以获得更高精度
using ElementComputeEpilogue = ElementAccumulator; // 后处理计算数据类型 - 与累加器一致
using ElementInputA = cutlass::bfloat16_t;         // A 矩阵元素数据类型 - BF16 提高效率
using ElementInputB = cutlass::bfloat16_t;         // B 矩阵元素数据类型 - BF16 提高效率
using ElementOutput = cutlass::bfloat16_t;         // 输出矩阵元素数据类型 - BF16 与输入匹配

// =====================================================================
// 内存布局配置
// =====================================================================
// 定义矩阵在内存中的存储方式
using LayoutInputA = cutlass::layout::ColumnMajor;  // A 矩阵列主序存储（FORTRAN 风格）
using LayoutInputB = cutlass::layout::RowMajor;     // B 矩阵行主序存储（C 风格）
using LayoutOutput = cutlass::layout::ColumnMajor;  // 输出 C 矩阵列主序存储
// 归约操作输出向量的布局
using LayoutGemmKReduction = cutlass::layout::PitchLinear;  // 归约向量使用线性布局

// =====================================================================
// 硬件架构配置
// =====================================================================
// 选择使用 Tensor Core 还是常规 SIMT 核心
using MMAOp = cutlass::arch::OpClassTensorOp;  // 使用 Tensor Core 进行矩阵乘法

// 指定 CUDA SM 架构版本
using SmArch = cutlass::arch::Sm80;  // 目标 Ampere 架构（计算能力 8.0）

// =====================================================================
// 分层计算的 Tile 配置
// =====================================================================
// 定义线程块（Thread Block）计算的 tile 大小
// 形状：<M, N, K> = <128, 128, 32>
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // 每个线程块计算 128x128 的输出 tile

// 定义 Warp 计算的 tile 大小
// 形状：<M, N, K> = <64, 64, 32>
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;  // 每个 warp 计算 64x64 的输出 tile

// 定义 MMA 操作的大小
// 形状：<M, N, K> = <16, 8, 16> - Ampere 的 mma.sync 指令形状
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // 单个 Tensor Core 指令计算 16x8 的输出

// =====================================================================
// 性能优化配置
// =====================================================================
// 定义线程块在 GPU 上的调度方式
// Swizzle 通过改变线程块调度模式来提高 L2 缓存局部性
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

// 流水线级数配置
// 更多的级数可以更好地隐藏延迟，但会使用更多共享内存
constexpr int NumStages = 4;  // 4 级流水线，为 Ampere 架构优化

// =====================================================================
// 归约和对齐配置
// =====================================================================
// 选择沿 K 维度归约 A 还是 B 操作数
// true: 归约 A 产生 Mx1 向量（A 的行和）
// false: 归约 B 产生 1xN 向量（B 的列和）
constexpr bool ReduceKForA = true;

// 向量化加载的内存对齐要求
// 8 个元素 = 128 位（BF16 每个元素 16 位）
constexpr int AlignmentA = 8;  // A 矩阵必须对齐到 8 个 BF16 元素
constexpr int AlignmentB = 8;  // B 矩阵必须对齐到 8 个 BF16 元素

// =====================================================================
// 后处理（Epilogue）配置
// =====================================================================
// 定义内核的后处理部分
// LinearCombination 执行：D = alpha * accumulator + beta * C
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // 输出矩阵数据类型（BF16）
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // 向量宽度 = 128 位 / 16 位 = 8 个元素
                                                          // 决定每次内存事务的元素数
    ElementAccumulator,                                   // 累加器数据类型（float）
    ElementComputeEpilogue>;                             // 后处理计算数据类型（float）

// =====================================================================
// 带 K 维归约的主 GEMM 内核
// =====================================================================
// 这个特殊的 GEMM 内核在执行矩阵乘法的同时
// 沿 K 维度归约一个操作数
using Gemm = typename cutlass::gemm::device::GemmWithKReduction<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  ReduceKForA,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  AlignmentA,
  AlignmentB,
  cutlass::arch::OpMultiplyAdd,
  cutlass::ComplexTransform::kNone,
  cutlass::ComplexTransform::kNone
>;

// =====================================================================
// Split-K 归约配置
// =====================================================================
// 并行 split-k 情况下使用的归约内核
// Shape<4, 64> 表示每个归约 tile 包含 4 行和 64 列
using ReduceGemmSplitKShape = cutlass::MatrixShape<4, 64>;

// 定义归约操作为加法
// 将多个累加器值相加得到最终输出
using ReduceOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator,  // 输入数据类型（累加器）
    ElementOutput,       // 输出数据类型
    EpilogueOp::kCount   // 向量元素个数
  >;

// Split-K GEMM 归约内核
// 用于合并多个 K 分片的部分结果
using ReduceGemmSplitKKernel = cutlass::reduction::kernel::ReduceSplitK<
    ReduceGemmSplitKShape,  // 归约 tile 形状
    EpilogueOp,             // 后处理操作
    ReduceOp                // 归约操作（加法）
  >;

// Split-K GEMM 归约设备接口
using ReduceGemmSplitK = cutlass::reduction::device::ReduceSplitK<ReduceGemmSplitKKernel>;

// Split-K 向量归约的 tile 形状
// Shape<1, 256> 表示归约 256 个元素的向量
using ReduceVectorSplitKShape = cutlass::MatrixShape<1, 256>;

// 定义向量归约的虚拟后处理操作
// 用于 Split-K 向量归约，不执行额外的缩放
using DummyEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // 输出矩阵数据类型
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // 向量化访问的元素数量
                                                          // 128位/16位 = 8个元素
                                                          // 这也决定了后处理中数学指令的向量宽度
    ElementAccumulator,                                   // 累加器数据类型
    ElementComputeEpilogue,                              // 后处理计算数据类型
    cutlass::epilogue::thread::ScaleType::Nothing>;      // 不进行缩放

// Split-K 向量归约内核
// 用于合并多个 K 分片的向量结果
using ReduceVectorSplitKKernel = cutlass::reduction::kernel::ReduceSplitK<
    ReduceVectorSplitKShape,  // 向量归约 tile 形状
    DummyEpilogueOp,          // 虚拟后处理操作
    ReduceOp                  // 归约操作（加法）
  >;

// Split-K 向量归约设备接口
using ReduceVectorSplitK = cutlass::reduction::device::ReduceSplitK<ReduceVectorSplitKKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// =====================================================================
// 命令行接口
// =====================================================================
// 命令行选项解析结构体
struct Options {

  bool help;                                  // 显示帮助信息
  cutlass::gemm::GemmCoord problem_size;      // GEMM 问题大小 (M, N, K)
  int split_k_slices;                         // K 维度分片数量
  bool parallel_split_k;                      // 是否使用并行 split-K
  bool reference_check;                       // 是否进行正确性检查
  bool measure_performance;                   // 是否测量性能
  int iterations;                             // 性能测试迭代次数
  bool save_workspace;                        // 是否保存工作空间到文件
  ElementComputeEpilogue alpha;               // GEMM 缩放因子 alpha
  ElementComputeEpilogue beta;                // GEMM 缩放因子 beta
  bool benchmark;                             // 是否运行基准测试
  std::string tag;                            // 结果标签

  // 构造函数：设置默认参数值
  Options():
    help(false),
    problem_size(1024, 1024, 1024),  // 默认 1024x1024x1024 矩阵
    split_k_slices(1),                // 默认不分片
    parallel_split_k(false),          // 默认串行
    reference_check(true),            // 默认进行正确性检查
    measure_performance(false),       // 默认不测性能
    iterations(20),                   // 默认 20 次迭代
    save_workspace(false),            // 默认不保存工作空间
    alpha(-1),                        // 默认 alpha = -1
    beta(-1),                         // 默认 beta = -1
    benchmark(false) { }              // 默认不运行基准测试

  // 验证问题大小是否与 CUTLASS 实现兼容
  bool valid() {

    // CUTLASS 对 BF16 元素（每个 16 位）使用 128 位向量加载
    // 这要求所有维度必须能被 8 个元素整除（128/16 = 8）
    // 未对齐的访问会导致性能下降或错误
    int const kAlignment = 8;

    if ((problem_size.m() % kAlignment) ||
        (problem_size.n() % kAlignment) ||
        (problem_size.k() % kAlignment)) {

      // 张量未对齐
      return false;
    }

    return true;
  }

  /// 更新输入和过滤器大小
  void update(
    cutlass::gemm::GemmCoord problem_size,
    int split_k_slices,
    bool parallel_split_k) {

    this->problem_size = problem_size;
    this->split_k_slices = split_k_slices;
    this->parallel_split_k = parallel_split_k;
  }

  // 解析命令行参数
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("parallel-split-k")) {
      parallel_split_k = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("split-k-slices", split_k_slices);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);
  }

  /// 打印使用说明
  std::ostream & print_usage(std::ostream &out) const {

    out << "23_ampere_operand_gemm_reduction_fusion\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --m=<int>            GEMM M\n"
      << "  --n=<int>            GEMM N\n"
      << "  --k=<int>            GEMM K\n"
      << "  --split-k-slices=<int> Split K Slices\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --parallel-split-k   If set (true), use parallel split K\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several problem sizes.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion  --m=1024 --n=1024 --k=1024 \n\n";

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// 结果结构体：保存运行结果和状态
struct Result {
  double runtime_ms;                   // 运行时间（毫秒）
  cutlass::Status status;               // CUTLASS 执行状态
  cutlass::Status reference_check;      // 正确性验证状态
  cudaError_t error;                   // CUDA 错误代码

  // 构造函数：初始化所有成员
  Result():
    runtime_ms(0),                                     // 初始运行时间为 0
    status(cutlass::Status::kSuccess),                 // 初始状态为成功
    reference_check(cutlass::Status::kInvalid),        // 初始验证状态为无效
    error(cudaSuccess) { }                             // 初始 CUDA 错误为成功

  // 打印结果表头
  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";  // 如果有标签，添加名称列
    }

    // 打印各列标题
    out << "ID,M,N,K,SplitK-Slices,Parallel-SplitK,Runtime";

    return out;
  }

  // 打印结果数据行
  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";  // 输出标签
    }

    // 输出测试编号和参数
    out
      << "gemm_" << idx << ","
      << options.problem_size.m() << ","
      << options.problem_size.n() << ","
      << options.problem_size.k() << ","
      << options.split_k_slices << ","
      << options.parallel_split_k << ","
      << runtime_ms ;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// 运行一个基准测试
Result profile(Options const &options) {

  Result result;

  // 使用 CUTLASS 辅助函数初始化张量
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.problem_size.kn());


  // 创建张量 C，维度为 M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.problem_size.mn());

  // 创建张量 D 用于存储 CUTLASS 内核的输出
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.problem_size.mn());
  // 创建矩阵 D，维度为 M x N，用于存储参考内核的输出
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.problem_size.mn());

  // 根据 ReduceKForA 配置确定归约向量的长度
  // true: 归约 A 矩阵，得到 M 长度的向量（每行的和）
  // false: 归约 B 矩阵，得到 N 长度的向量（每列的和）
  int reduce_vector_length = ReduceKForA ? options.problem_size.m() : options.problem_size.n();

  // 创建归约结果向量（大小为 reduce_vector_length x 1）
  cutlass::HostTensor<ElementOutput, LayoutGemmKReduction> tensor_reduction({reduce_vector_length, 1});
  // 创建参考归约结果向量，用于验证
  cutlass::HostTensor<ElementOutput, LayoutGemmKReduction> tensor_ref_reduction({reduce_vector_length, 1});

  // 使用 CUTLASS 辅助函数在主机上填充输入和输出矩阵
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1997,
      ElementInputA(1),
      ElementInputA(-1),
      0);  // <- 在主机上用均匀分布的随机数据填充张量 A

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      2003,
      ElementInputB(1),
      ElementInputB(-1),
      0);  // <- 在主机上用均匀分布的随机数据填充张量 B

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      2017,
      ElementOutput(1),
      ElementOutput(-1),
      0);  // <- 在主机上用均匀分布的随机数据填充矩阵 C
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- 在主机上用零填充矩阵 D
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- 在主机上用零填充参考矩阵 D

  cutlass::reference::host::TensorFill(
      tensor_reduction.host_view());  // <- 在主机上用零填充归约向量
  cutlass::reference::host::TensorFill(
      tensor_ref_reduction.host_view());  // <- 在主机上用零填充参考归约向量

  // 将数据从主机复制到 GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();
  tensor_reduction.sync_device();

  // 根据 split-K 模式设置 alpha 和 beta 值
  // 并行 split-K: alpha=1, beta=0（第一次累加不需要额外缩放）
  // 串行模式: 使用用户指定的 alpha 和 beta
  ElementComputeEpilogue alpha = options.parallel_split_k ? ElementComputeEpilogue(1)
                                                          : ElementComputeEpilogue(options.alpha);
  ElementComputeEpilogue beta = options.parallel_split_k ? ElementComputeEpilogue(0)
                                                         : ElementComputeEpilogue(options.beta);

  // 设置 GEMM 运行模式
  // kGemmSplitKParallel: 并行 split-K，多个线程块处理同一个输出 tile
  // kGemm: 标准 GEMM 模式
  cutlass::gemm::GemmUniversalMode mode = options.parallel_split_k ?
                     cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel :
                     cutlass::gemm::GemmUniversalMode::kGemm;

  // 设置批处理数量（即 K 维度的分片数）
  int batch_count = options.split_k_slices;

  // 创建 GEMM 内核参数元组
  // 这将作为参数传递给实例化的 CUTLASS 内核
  typename Gemm::Arguments arguments(
    mode,                                            // GEMM 运行模式
    options.problem_size,                            // 问题大小 (M, N, K)
    batch_count,                                     // K 维分片数
    {alpha, beta},                                   // 缩放因子
    tensor_a.device_ref().data(),              // <- 设备上张量 A 的引用
    tensor_b.device_ref().data(),              // <- 设备上张量 B 的引用
    tensor_c.device_ref().data(),              // <- 设备上矩阵 C 的引用
    tensor_d.device_ref().data(),              // <- 设备上矩阵 D 的引用
    tensor_reduction.device_ref().data(),      // <- 设备上归约张量的引用
    options.problem_size.m() * options.problem_size.k(),  // A 矩阵元素总数
    options.problem_size.n() * options.problem_size.k(),  // B 矩阵元素总数
    options.problem_size.m() * options.problem_size.n(),  // C 矩阵元素总数
    options.problem_size.m() * options.problem_size.n(),  // D 矩阵元素总数
    reduce_vector_length,                                 // 归约向量长度
    tensor_a.layout().stride(0),                         // A 矩阵 leading dimension
    tensor_b.layout().stride(0),                         // B 矩阵 leading dimension
    tensor_c.layout().stride(0),                         // C 矩阵 leading dimension
    tensor_d.layout().stride(0),                         // D 矩阵 leading dimension
    tensor_reduction.layout().stride(0));                // 归约向量 stride

  // 根据模板实例化 CUTLASS 内核
  Gemm gemm_op;

  // 使用参数查询矩阵乘法计算所需的额外工作空间
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // 分配工作空间内存
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // 检查问题大小是否受支持
  result.status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  // 使用参数和工作空间指针初始化 CUTLASS 内核
  result.status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  // 启动已初始化的 CUTLASS 内核
  result.status = gemm_op();

  CUTLASS_CHECK(result.status);

  // 如果启用了并行 split-K 且分片数大于 1，需要执行归约
  if (options.parallel_split_k && batch_count > 1) {
    // 归约 GEMM 结果（合并多个 K 分片的部分结果）

    // 使用用户指定的 alpha 和 beta 进行最终归约
    ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
    ElementComputeEpilogue beta = ElementComputeEpilogue(options.beta);

    // 设置 split-K GEMM 的 stride（列主序存储）
    int splitk_gemm_stride = options.problem_size.m();

    // 创建行主序布局，用于 split-K 归约
    cutlass::layout::RowMajor splitk_gemm_layout(splitk_gemm_stride);

    // 获取工作空间指针，用于存储中间结果
    void * workspace_gemm_ptr = workspace.get();
    // 创建工作空间的张量引用
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> workspace_gemm_tensorref(static_cast<ElementOutput *>(workspace_gemm_ptr), splitk_gemm_layout);

    // 创建输出矩阵 D 的张量引用
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_d_tensorref(tensor_d.device_ref().data(), splitk_gemm_layout);

    // 创建输入矩阵 C 的张量引用
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_c_tensorref(tensor_c.device_ref().data(), splitk_gemm_layout);

    // 创建 split-K GEMM 归约参数
    typename ReduceGemmSplitK::Arguments reduce_gemm_splitk_arguments{
      cutlass::MatrixCoord(options.problem_size.n(), options.problem_size.m()),  // 输出矩阵大小（转置）
      batch_count,                                                              // K 分片数
      size_t(options.problem_size.m() * options.problem_size.n()),              // 单个矩阵元素数
      workspace_gemm_tensorref,                                                 // 中间结果输入
      tensor_d_tensorref,                                                      // 最终输出
      tensor_c_tensorref,                                                      // C 矩阵输入
      {alpha, beta}                                                            // 缩放因子
    };

    // 创建并执行 split-K GEMM 归约操作
    ReduceGemmSplitK reduce_gemm_splitk_op;

    // 初始化归约操作
    result.status = reduce_gemm_splitk_op.initialize(reduce_gemm_splitk_arguments);
    CUTLASS_CHECK(result.status);

    // 执行归约
    result.status = reduce_gemm_splitk_op();
    CUTLASS_CHECK(result.status);

    // 归约 K 维向量（合并多个分片的向量结果）
    cutlass::layout::RowMajor splitk_vector_layout(reduce_vector_length);
   
    // 计算向量工作空间的起始位置（在 GEMM 工作空间之后）
    ElementOutput *workspace_vector_ptr = static_cast<ElementOutput *>(workspace_gemm_ptr) + batch_count * options.problem_size.m() * options.problem_size.n();
    // 创建向量工作空间的张量引用
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> workspace_vector_tensorref(workspace_vector_ptr, splitk_vector_layout);

    // 创建归约输出向量的张量引用
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_reduction_tensorref(tensor_reduction.device_ref().data(), splitk_vector_layout);

    // 创建空引用（不需要 C 向量输入）
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_nullptr_tensorref(nullptr, splitk_vector_layout);

    // 创建 split-K 向量归约参数
    typename ReduceVectorSplitK::Arguments reduce_vector_splitk_arguments(
      cutlass::MatrixCoord(1, reduce_vector_length),   // 向量大小 (1 x length)
      batch_count,                                     // K 分片数
      size_t(reduce_vector_length),                    // 单个向量元素数
      workspace_vector_tensorref,                      // 中间结果输入
      tensor_reduction_tensorref,                      // 最终输出
      tensor_nullptr_tensorref,                        // 无需 C 向量
      {1.0f, 0.0f});                                   // alpha=1, beta=0（直接累加）

    // 创建并执行 split-K 向量归约操作
    ReduceVectorSplitK reduce_vector_splitk_op;

    // 初始化归约操作
    result.status = reduce_vector_splitk_op.initialize(reduce_vector_splitk_arguments);
    CUTLASS_CHECK(result.status);

    // 执行归约
    result.status = reduce_vector_splitk_op();
    CUTLASS_CHECK(result.status);
  }

  //
  // 创建设备参考内核的实例，用于验证结果正确性
  //
  if (options.reference_check) {
    // 启动设备参考内核来严格计算乘积 A * B
    // 这是一个标准的 GEMM 实现，用于对比验证
    cutlass::reference::device::Gemm<
        ElementInputA,                // A 矩阵元素类型
        LayoutInputA,                 // A 矩阵布局
        ElementInputB,                // B 矩阵元素类型
        LayoutInputB,                 // B 矩阵布局
        ElementOutput,                // 输出元素类型
        LayoutOutput,                 // 输出布局
        ElementComputeEpilogue,       // 后处理计算类型
        ElementAccumulator> gemm_device;  // 累加器类型
  
    // 执行参考 GEMM: D = alpha * A * B + beta * C
    gemm_device
      (
        options.problem_size,                    // 问题大小 (M, N, K)
        ElementComputeEpilogue(options.alpha),   // 缩放因子 alpha
        tensor_a.device_ref(),                   // A 矩阵
        tensor_b.device_ref(),                   // B 矩阵
        ElementComputeEpilogue(options.beta),    // 缩放因子 beta
        tensor_c.device_ref(),                   // C 矩阵
        tensor_ref_d.device_ref()                // 输出 D 矩阵
      );
  
    // 等待内核完成
    cudaDeviceSynchronize();
  
    // 将 CUTLASS 和参考内核的输出数据复制到主机进行比较
    tensor_d.sync_host();
    tensor_ref_d.sync_host();
  
    tensor_reduction.sync_host();
  
    // 在主机代码中执行 K 维归约，作为参考结果
    if (ReduceKForA) {
      // 归约 A 矩阵：计算每一行的和
      for (int m = 0; m < options.problem_size.m(); ++m) {
        for (int k = 0; k < options.problem_size.k(); ++k) {
          tensor_ref_reduction.at({m, 0}) +=     // 累加到第 m 个元素
            tensor_a.at(cutlass::MatrixCoord(m, k));  // A[m][k]
        }
      }
    } else {
      // 归约 B 矩阵：计算每一列的和
      for (int k = 0; k < options.problem_size.k(); ++k) {
        for (int n = 0; n < options.problem_size.n(); ++n) {
          tensor_ref_reduction.at({n, 0}) +=     // 累加到第 n 个元素
            tensor_b.at(cutlass::MatrixCoord(k, n));  // B[k][n]
        }
      }
    }
  
    // 检查 CUTLASS 内核和参考内核的输出是否相等
    // 首先比较 GEMM 结果
    bool pass = cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                       tensor_ref_d.host_view());

    // 然后比较归约向量结果
    pass &= cutlass::reference::host::TensorEquals(tensor_ref_reduction.host_view(),
                                                   tensor_reduction.host_view());

    if (!pass) {
      // 结果不匹配，设置错误状态
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    } else {
      // 结果匹配，验证通过
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  } else {
    // 未进行参考检查
    result.reference_check = cutlass::Status::kInvalid;
  }

  // 如果需要保存工作空间数据到文件
  if (options.save_workspace) {

    std::stringstream ss;

    // 构建输出文件名，包含问题大小信息
    ss << "23_ampere_gemm_operand_reduction_fusion"
      << options.problem_size.m() << "x" << options.problem_size.n() << "x" << options.problem_size.k()
      << ".dat";

    std::ofstream output_workspace(ss.str());

    // 输出输入矩阵 A 和 B
    output_workspace
      << "A = \n" << tensor_a.host_view() << "\n\n"
      << "B = \n" << tensor_b.host_view() << "\n\n";

    // 如果进行了参考检查，输出参考结果
    if (options.reference_check) {
      output_workspace << "Reference D = \n" << tensor_ref_d.host_view() << "\n\n";
      output_workspace << "Reference reduction vector = \n" << tensor_ref_reduction.host_view() << "\n\n";
    }

    // 输出计算结果
    output_workspace << "Computed D = \n" << tensor_d.host_view() << std::endl;
    output_workspace << "Computed reduction vector = \n" << tensor_reduction.host_view() << std::endl;

    std::cout << "Results written to '" << ss.str() << "'." << std::endl;
  }
 
  //
  // 性能测量
  //

  // 如果需要测量性能
  if (options.measure_performance) {

    // 创建 CUDA 事件用于计时
    cudaEvent_t events[2];
    
    // 创建两个事件：开始和结束
    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }
    }

    // 在执行 GEMM 操作之前记录开始事件
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // 重复执行 GEMM 操作以获得平均性能
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result.status = gemm_op();  // 执行一次 GEMM 操作
      CUTLASS_CHECK(result.status);
    }

    // 在所有 GEMM 操作启动后记录结束事件
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // 等待所有 GPU 工作完成
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // 计算两个事件之间的经过时间（毫秒）
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // 计算平均运行时间（总时间 / 迭代次数）
    result.runtime_ms = double(runtime_ms) / double(options.iterations);

    // 销毁 CUDA 事件，释放资源
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
  }

  return result;
}

// 主函数：程序入口点
int main(int argc, char const **args) {

  bool notSupported = false;  // 标记是否支持当前硬件

  // Ampere Tensor Core 操作通过 mma.sync 暴露，首次在 CUDA 11.0 中可用
  //
  // CUTLASS 必须使用 CUDA 11 工具包编译才能运行 Conv2dFprop 示例
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  // 获取 GPU 设备属性
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));  // 获取设备 0 的属性

  // 检查计算能力是否满足 Ampere 架构要求（SM 8.0 或更高）
  if (!(props.major >= 8)) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // 如果不支持，返回 0（避免测试失败）
    return 0;
  }

  // 创建选项对象并解析命令行参数
  Options options;

  options.parse(argc, args);

  // 如果请求帮助，显示使用说明
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.benchmark) {
    // 运行基准测试：测试多个预定义的问题大小

    // 定义基准测试的问题大小和配置
    struct Benchmark {
      int m, n, k, split_k_slices, parallel_split_k;
    } problem_sizes[] = {
      {4096, 6144, 4096, 1, false},  // 大型矩阵，无 split-K
    };

    // 打印结果表头
    Result::print_header(std::cout, options) << "\n";

    int idx = 1;

    // 遍历所有问题大小并运行测试
    for (auto const &problem_size : problem_sizes) {
      // 更新选项为当前问题大小
      options.update({problem_size.m, problem_size.n, problem_size.k},
                     problem_size.split_k_slices, problem_size.parallel_split_k);

      // 运行性能分析
      Result result = profile(options);
      // 打印结果
      result.print(std::cout, idx, options) << "\n";

      ++idx;
    }
  } else {

    // 执行单个问题大小测试
    if (!options.valid()) {
      // 检查问题大小是否有效（对齐要求等）
      std::cerr << "Invalid problem." << "\n";
      return -1;
    }

    // 运行性能分析
    Result result = profile(options);

    // 打印结果
    Result::print_header(std::cout, options) << "\n";
    result.print(std::cout, 1, options) << "\n";
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
