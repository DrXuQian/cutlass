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
 * CUTLASS Example 19: Tensor Core 规范化布局操作示例
 *
 * 本示例展示了如何使用 CUTLASS 的 Warp 级 Tensor Core API 进行矩阵乘法。
 * 这是理解 CUTLASS 底层 Tensor Core 编程模型的基础示例。
 *
 * 核心技术要点：
 * ============
 * 1. Tensor Core 操作：利用专用硬件单元加速矩阵运算
 * 2. 规范化布局（Canonical Layout）：优化的数据排列方式，减少 bank 冲突
 * 3. Warp 级编程：32 个线程协同执行矩阵运算
 * 4. Fragment 概念：每个线程持有的矩阵数据片段
 * 5. 双缓冲技术：计算与数据加载重叠
 *
 * 性能优化策略：
 * ============
 * - 使用 Tensor Core 指令（mma.sync）实现高吞吐量
 * - 数据在寄存器和共享内存间高效传输
 * - 通过 Fragment 分布减少线程间通信
 * - 利用流水线隐藏内存延迟
 *
 * 本示例需要 NVIDIA Ampere GPU 或更新架构。
 */

// 标准库头文件
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS 核心头文件
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

// CUTLASS 工具头文件
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm_complex.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// 定义 Warp 级问题的矩阵维度
// M: 输出矩阵行数, N: 输出矩阵列数, K: 矩阵乘法的归约维度
int const kM = 27;
int const kN = 31;
int const kK = 17;

///////////////////////////////////////////////////////////////////////////////////////////////////

// 定义 Warp 级 GEMM 操作符
//
// 这个模板封装了矩阵乘法操作和后处理逻辑，提供了类似 GEMM 的接口，
// 可以在设备代码中实例化。该操作符协调 Warp 内所有线程共同完成矩阵乘法。

namespace cutlass {
namespace gemm {
namespace warp {

template <
  typename Shape,
  typename InstructionShape,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementScalar
>
class GemmTensorOp {
public:
  // 计算对齐后的 Warp 形状，确保维度是指令形状的整数倍
  using WarpShape = GemmShape<
    ((Shape::kM + InstructionShape::kM - 1) / InstructionShape::kM) * InstructionShape::kM,
    ((Shape::kN + InstructionShape::kN - 1) / InstructionShape::kN) * InstructionShape::kN,
    InstructionShape::kK
  >;

  // 定义 Warp 级矩阵乘法操作，使用 Tensor Core 指令
  using MmaWarp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    InstructionShape,
    double,                             // A 矩阵元素数据类型
    cutlass::layout::RowMajor,          // A 矩阵布局（行主序）
    double,                             // B 矩阵元素数据类型
    cutlass::layout::ColumnMajor,       // B 矩阵布局（列主序）
    double,                             // C 矩阵元素数据类型
    cutlass::layout::RowMajor           // C 矩阵布局（行主序）
  >::Type;
 
  // K 维度分组数：将 K 维度分割成多个指令大小的块 
  int const kKgroups = (Shape::kK + InstructionShape::kK - 1) / InstructionShape::kK;

  // Fragment 迭代器：用于遍历累加器的片段
  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    typename MmaWarp::Shape,
    InstructionShape,
    double,
    typename MmaWarp::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor
  >;

  // 后处理 Tile 迭代器：用于遍历共享内存中的元素片段
  using AccumulatorTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
    typename MmaWarp::Shape,
    InstructionShape,
    double,
    cutlass::layout::RowMajor
  >;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  using TensorRefC = typename AccumulatorTileIterator::TensorRef;

public:
  CUTLASS_HOST_DEVICE
  GemmTensorOp() { }

  CUTLASS_DEVICE
  void operator()(
    ElementScalar alpha, 
    TensorRefA ref_A, 
    TensorRefB ref_B, 
    ElementScalar beta,
    TensorRefC ref_C,
    TensorRefC ref_D,
    int lane_id) const {
  
    // 创建指向共享内存中 A 和 B 矩阵片段的迭代器
    typename MmaWarp::IteratorA iter_A(ref_A, {Shape::kM, Shape::kK}, lane_id);
    typename MmaWarp::IteratorB iter_B(ref_B, {Shape::kK, Shape::kN}, lane_id);

    // 创建并清零用于保存 C 矩阵的累加器 tile
    typename MmaWarp::FragmentC accum;
    accum.clear();
  
    // 实例化 Warp 级矩阵乘法操作符
    MmaWarp mma_op;

    // 创建用于保存每个 Warp 持有的矩阵片段的 Fragment
    // 使用双缓冲技术（[2]）实现计算与数据加载的重叠
    typename MmaWarp::FragmentA frag_A[2];
    typename MmaWarp::FragmentB frag_B[2];
      
    // 预加载第一个 K 分组的数据片段
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);

    // 移动到下一个 K 分组
    ++iter_A;
    ++iter_B;

    // 主循环：遍历所有 K 分组进行矩阵乘法
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {

      // 预加载下一个 K 分组的数据（双缓冲）
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);

      ++iter_A;
      ++iter_B;

      // 执行当前 K 分组的矩阵乘法，累加到 accum
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
  
    // 创建后处理阶段的迭代器
    FragmentIterator accum_frag_it(accum);
    AccumulatorTileIterator source_tile_it(ref_C, {Shape::kM, Shape::kN}, lane_id);
    AccumulatorTileIterator dest_tile_it(ref_D, {Shape::kM, Shape::kN}, lane_id);

    // 定义线性缩放操作的函数对象
    cutlass::multiplies<typename FragmentIterator::Fragment> mul_source;
    cutlass::multiply_add<typename FragmentIterator::Fragment> mul_add_accumulator;

    // 遍历后处理组件，应用 alpha 和 beta 缩放
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < FragmentIterator::kIterations; ++idx) {

      // 定义累加器片段的存储空间
      typename FragmentIterator::Fragment accum_fragment;
      typename FragmentIterator::Fragment source_fragment;

      // 从累加器 tile 中选择一个片段
      accum_frag_it.load(accum_fragment);
      ++accum_frag_it;

      // 从共享内存加载对应的源数据片段（C 矩阵）
      source_tile_it.load(source_fragment);
      ++source_tile_it;

      // 计算线性组合：D = alpha * AB + beta * C
      source_fragment = mul_source(beta, source_fragment);
      accum_fragment = mul_add_accumulator(alpha, accum_fragment, source_fragment);

      // 将结果存储回共享内存
      dest_tile_it.store(accum_fragment);
      ++dest_tile_it;
    }
  }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

// 示例核函数：演示 Warp 对共享内存中矩阵的协同 GEMM 操作
// 这个核函数展示了完整的 Tensor Core GEMM 流程：
// 1. 从全局内存加载数据到共享内存
// 2. 执行 Warp 级矩阵乘法
// 3. 将结果写回全局内存
__global__ void kernel(
  double *D_gmem, 
  double alpha, 
  double const *A_gmem, 
  double const *B_gmem, 
  double beta,
  double const *C_gmem) {

  // 在共享内存中定义矩阵
  // 注意：这些矩阵的布局已经针对 Tensor Core 操作优化
  __shared__ double A[kM][kK];
  __shared__ double B[kN][kK];
  __shared__ double C[kM][kN];

  // 将数据从全局内存复制到共享内存
  // 只有线程 0 执行复制操作，避免冲突
  if (threadIdx.x == 0) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      for (int k = 0; k < kK; ++k) {
        A[m][k] = A_gmem[m * kK + k];
      }
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        B[n][k] = B_gmem[n * kK + k];
      }
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < kN; ++n) {
        C[m][n] = C_gmem[m * kN + n];
      }
    }
  }

  __syncthreads();
  
  // 实例化 Warp 级矩阵乘法操作符
  // 参数说明：
  // - 指令形状 (8x8x4)：Tensor Core 的基本操作单元
  // - 整体形状 (kM, kN, kK)：完整的矩阵维度
  // - 数据类型：所有矩阵使用 double 精度
  // - 布局：A 行主序，B 列主序，C 行主序

  using GemmTensorOp = cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<kM, kN, kK>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    double,                             // A 矩阵元素类型
    cutlass::layout::RowMajor,          // A 矩阵布局（行主序）
    double,                             // B 矩阵元素类型
    cutlass::layout::ColumnMajor,       // B 矩阵布局（列主序）
    double,                             // C 矩阵元素类型
    cutlass::layout::RowMajor,          // C 矩阵布局（行主序）
    double                              // alpha 和 beta 标量类型
  >;

  // 实例化 GEMM 操作符
  GemmTensorOp gemm;

  // 执行 Warp 级 GEMM 操作
  // D = alpha * A * B + beta * C
  // threadIdx.x 作为 lane_id 传入，用于确定每个线程的职责
  gemm(
    alpha, 
    {&A[0][0], kK},
    {&B[0][0], kK},
    beta,
    {&C[0][0], kN},
    {&C[0][0], kN},
    threadIdx.x);

  __syncthreads();

  // 将结果从共享内存复制回全局内存
  // 只有线程 0 执行复制操作
  if (threadIdx.x == 0) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < kN; ++n) {
        D_gmem[m * kN + n] = C[m][n];
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 程序入口点：规范化 Warp 级 GEMM 操作演示
int main(int argc, const char *arg[]) {

  bool notSupported = false;

  // 检查 CUDA 版本：CUTLASS 需要 CUDA 11 工具包来运行这些示例
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "NVIDIA Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "This example requires compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // 在不支持的平台上返回 0，使测试通过
    return 0;
  }

  cutlass::HostTensor<double, cutlass::layout::RowMajor> A({kM, kK});
  cutlass::HostTensor<double, cutlass::layout::ColumnMajor> B({kK, kN});
  cutlass::HostTensor<double, cutlass::layout::RowMajor> C({kM, kN});
  cutlass::HostTensor<double, cutlass::layout::RowMajor> D({kM, kN});

  uint64_t seed = 2020;
  double max = 8;
  double min = -8;

  cutlass::reference::host::TensorFillRandomUniform(
    A.host_view(),
    seed,
    max,
    min,
    0
  );

  cutlass::reference::host::TensorFillRandomUniform(
    B.host_view(),
    seed + 17,
    max,
    min,
    0
  );

  cutlass::reference::host::TensorFillRandomUniform(
    C.host_view(),
    seed + 31,
    max,
    min,
    0
  );

  A.sync_device();
  B.sync_device();
  C.sync_device();
  D.sync_device();

  dim3 grid(1,1);
  dim3 block(32, 1, 1);

  double alpha = 2.25;
  double beta = 1.24;

  kernel<<< grid, block >>>(
    D.device_data(),
    alpha,
    A.device_data(),
    B.device_data(),
    beta,
    C.device_data()
  );

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Failed to synchronize device after kernel launch." << std::endl;
    return -1;
  }

  D.sync_host();
  
  // 在主机端计算参考结果，用于验证 GPU 计算的正确性
  cutlass::HostTensor<double, cutlass::layout::RowMajor> D_ref({kM, kN}, false);

  cutlass::reference::host::GemmComplex(
    {kM, kN, kK},
    alpha,
    A.host_ref(),
    cutlass::ComplexTransform::kNone,
    B.host_ref(),
    cutlass::ComplexTransform::kNone,
    beta,
    C.host_ref(),
    D_ref.host_ref(),
    double()
  );

  // 验证 GPU 计算结果是否与参考结果匹配
  if (!cutlass::reference::host::TensorEquals(
    D.host_view(),
    D_ref.host_view())) {

    std::cerr 
      << "A =\n" << A.host_view() 
      << "\n\nB = \n" << B.host_view() 
      << "\n\nC = " << C.host_view() 
      << "\n\nRef =\n" << D_ref.host_view()
      << "\n\nD =\n" << D.host_view() << "\n\n";

    std::cerr << "Error - device results mismatch host reference." << std::endl;

    return -1;
  }

  std::cout << "Passed" << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
