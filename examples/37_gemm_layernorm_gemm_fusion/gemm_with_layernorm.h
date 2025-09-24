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
    \brief A file contains all functioning classes needed by GemmLayernorm.
    \brief 包含GemmLayernorm所需的所有功能类的文件

    GemmLayernorm example =  GEMM0 with partial reduction fused in epilogue (EpilogueVisitorLayerNorm)
                          +  lightweight full reduction kernel (ApplyFinalReduction)
                          +  GEMM1 with elemenwise operations fused in mainloop (GemmLayernormMainloopFusion)

    GemmLayernorm示例 = GEMM0 在epilogue中融合部分规约（EpilogueVisitorLayerNorm）
                      + 轻量级完全规约核函数（ApplyFinalReduction）
                      + GEMM1 在主循环中融合逐元素操作（GemmLayernormMainloopFusion）

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_layernorm_mainloop_fusion.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_epilogue_visitor.h"
#include "helper.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// 完成LayerNorm的最终规约步骤，将部分和与部分平方和规约为最终的均值和方差
// 这个kernel执行跨线程块的规约，完成LayerNorm的统计量计算
template <
  typename ElementVariance_,           // 方差的数据类型
  typename ElementMean_,                // 均值的数据类型
  typename ElementLayernormCompute_,   // LayerNorm计算的数据类型（通常是float）
  typename ElementOutput,               // 输出的数据类型
  typename ThreadblockShape_,           // 线程块处理的矩阵块形状
  bool IsShiftedVariance_ = false      // 是否使用偏移方差（数值稳定性）
>
class ApplyFinalReduction {
public:

  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;
  using ThreadblockShape = ThreadblockShape_;

  // 预处理已确保布局等同于RowMajor
  // 注意：即使输入是ColumnMajor，也会在之前转置为RowMajor
  using Layout = cutlass::layout::RowMajor;

  using TensorVariance = TensorRef<ElementVariance, Layout>;
  using TensorMean = TensorRef<ElementMean, Layout>;

  // 是否使用偏移方差技术来提高数值稳定性
  // 偏移方差: Var(X-K) = E[(X-K)²] - E[X-K]²，其中K是偏移值
  static bool const kIsShiftedVariance = IsShiftedVariance_;

  //
  // Arguments
  //

  struct Arguments {

    MatrixCoord     extent;             ///< D和Layernorm矩阵的维度 [M, N]
    TensorVariance  ref_Variance;       ///< 平方和或方差张量（输入/输出），存储各行的平方和
    TensorMean      ref_Mean;           ///< 和或均值张量（输入/输出），存储各行的元素和
    ElementOutput   *ptr_Shifted_K;     ///< 偏移K张量指针，用于数值稳定的方差计算

    //
    // Methods
    //
    Arguments(){ }

    Arguments(
      MatrixCoord     extent_,
      TensorVariance  ref_Variance_,
      TensorMean      ref_Mean_,
      ElementOutput   *ptr_Shifted_K_
    ):
      extent(extent_),
      ref_Variance(ref_Variance_),
      ref_Mean(ref_Mean_),
      ptr_Shifted_K(ptr_Shifted_K_)
    {

    }
  };

  struct SharedStorage {


  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

private:

public:

  CUTLASS_DEVICE
  ApplyFinalReduction() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    apply(params, shared_storage);
  }

private:

  /// 执行部分规约，完成LayerNorm的统计量计算
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    // 计算需要多少个线程块来处理所有列（K维度被分块）
    int threadblock_num = (params.args.extent.column() + ThreadblockShape::kM - 1) / ThreadblockShape::kM;

    // 计算当前线程负责的行索引
    int block_n = blockIdx.x * blockDim.x;

    int thread_n = threadIdx.x;

    // 全局行索引（每个线程处理一行）
    int idx_n = block_n + thread_n;

    // 边界检查：超出行数的线程退出
    if (idx_n >= params.args.extent.row()) {
      return;
    }

    using ConvertVarianceOutput = cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput = cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    using ConvertVariance = cutlass::NumericConverter<ElementLayernormCompute, ElementVariance>;
    using ConvertMean = cutlass::NumericConverter<ElementLayernormCompute, ElementMean>;

    using ConvertShiftK = cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;

    ConvertVariance   convert_variance;
    ConvertMean  convert_mean;

    ConvertVarianceOutput   convert_variance_output;
    ConvertMeanOutput  convert_mean_output;

    ElementVariance *access_square = params.args.ref_Variance.data() + idx_n;
    ElementMean *access_mean = params.args.ref_Mean.data() + idx_n;

    ElementVariance *access_square_bak = access_square;
    ElementMean *access_mean_bak = access_mean;

    // 初始化累加器：平方和与元素和
    ElementLayernormCompute frag_square_sum = ElementLayernormCompute(0);
    ElementLayernormCompute frag_element_sum = ElementLayernormCompute(0);
    ElementVariance fetch_square;
    ElementMean fetch_mean;

    // 遍历所有线程块的部分结果，进行规约
    // 每个线程块计算了一部分列的统计量，这里将它们加起来
    CUTLASS_PRAGMA_UNROLL
    for (int idx_m = 0; idx_m < threadblock_num; idx_m++) {
      // 从全局内存加载部分平方和
      arch::global_load<ElementVariance, sizeof(ElementVariance)>(fetch_square, access_square, true);
      // 从全局内存加载部分元素和
      arch::global_load<ElementMean, sizeof(ElementMean)>(fetch_mean, access_mean, true);
      // 累加到总和中
      frag_element_sum += convert_mean(fetch_mean);
      frag_square_sum += convert_variance(fetch_square);
      // 移动到下一个线程块的结果（跨行存储）
      access_square += params.args.extent.row();
      access_mean += params.args.extent.row();
    }

    // 注意：这里的mean和square_mean还不是真正的均值，而是总和
    // 真正的均值计算在后续步骤中完成
    ElementLayernormCompute mean = frag_element_sum;
    ElementLayernormCompute square_mean = frag_square_sum;

    ElementLayernormCompute variance;

    // 计算方差的倒数（用于归一化）
    // 使用公式: Var(X) = E[X²] - E[X]²
    if (kIsShiftedVariance && params.args.ptr_Shifted_K != nullptr) {
      // 偏移方差版本：提高数值稳定性
      ElementOutput *access_shift_k = params.args.ptr_Shifted_K + idx_n;
      ElementOutput fetch_shift_k;
      ConvertShiftK convert_shift_k;
      arch::global_load<ElementOutput, sizeof(ElementOutput)>(fetch_shift_k, access_shift_k, true);
      ElementLayernormCompute shifted_mean =  mean - convert_shift_k(fetch_shift_k);
      // 方差的倒数 = 1 / sqrt(E[(X-K)²] - E[X-K]² + eps)
      variance = cutlass::constants::one<ElementLayernormCompute>() / cutlass::fast_sqrt(square_mean - shifted_mean * shifted_mean + ElementLayernormCompute(1e-6));
    }else{
      // 标准方差计算：1 / sqrt(E[X²] - E[X]² + eps)
      // 注意：这里没有使用Welford方法，而是传统的两遍扫描方法
      variance = cutlass::constants::one<ElementLayernormCompute>() / cutlass::fast_sqrt(square_mean - mean * mean + ElementLayernormCompute(1e-6));
    }

    // LayerNorm的变换是: (X - mean) * variance
    // 这里预计算 -mean * variance，后续直接使用
    mean = -mean * variance;

    // 恢复指针到原始位置
    access_square = access_square_bak;
    access_mean = access_mean_bak;

    // 将计算结果写回全局内存
    // 注意：这里存储的是方差的倒数（用于归一化）和预计算的-mean*variance
    access_square[0] = convert_variance_output(variance);
    access_mean[0] = convert_mean_output(mean);

  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

// Epilogue Visitor类用于在GEMM的epilogue阶段计算LayerNorm的部分统计量
// 这个类在每个线程块处理其输出tile时，同时计算该tile对应行的元素和与平方和
template <
  typename ThreadblockShape_,           // 线程块处理的矩阵块形状
  int ThreadCount,                      // 线程块中的线程数
  typename OutputTileIterator_,         // 输出tile的迭代器类型
  typename AccumulatorTile_,            // 累加器tile类型
  typename ElementAccumulator_,         // 累加器元素类型
  typename ElementVariance_,            // 方差元素类型
  typename ElementMean_,                // 均值元素类型
  typename ElementLayernormCompute_,    // LayerNorm计算类型（通常是float）
  typename ElementwiseFunctor_,         // 逐元素操作函数对象
  bool IsShiftedVariance_ = false      // 是否使用偏移方差
>
class EpilogueVisitorLayerNorm {
public:

  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;

  using AccumulatorTile = AccumulatorTile_;

  using ThreadblockShape   = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kRowIterations = OutputTileIterator::ThreadMap::Iterations::kRow;

  static int const kThreads = OutputTileIterator::ThreadMap::kThreads;

  // 是否使用偏移方差技术
  static bool const kIsShiftedVariance = IsShiftedVariance_;

  using ElementOutput = typename OutputTileIterator::Element;

  // 每个线程在行方向上的步进
  static int const kDeltaRow = OutputTileIterator::ThreadMap::Delta::kRow;

  /// 用于Shift-K Layernorm的数组类型
  // 每个线程需要访问的行数
  static int const kRowAccessCount = kIterations * kRowIterations;

  // 存储偏移值K的片段
  using ConvertedShiftFragment = Array<ElementLayernormCompute, kRowAccessCount>;

  // 对于列主序，在外部进行手动转置（已支持）
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulator_;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using LayernormFragment = Array<ElementLayernormCompute, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;
  using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::RowArrangement::Detail::kShapeWidth;
  static int const kThreadsInColumn = kThreads / kThreadsPerRow;
  static int const kHalfThreadsPerRow = (kThreadsPerRow >> 1);

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params   elementwise;
    TensorRefD                            ref_C;
    TensorRefD                            ref_D;
    ElementVariance                       *ptr_Variance;
    ElementMean                           *ptr_Mean;
    ElementOutput                         *ptr_Shifted_K;

    //
    // Methods
    //
    Arguments():
      ptr_Variance(nullptr),
      ptr_Mean(nullptr),
      ptr_Shifted_K(nullptr)
    {

    }

    Arguments(
      typename ElementwiseFunctor::Params   elementwise_,
      TensorRefD                            ref_C_,
      TensorRefD                            ref_D_,
      ElementVariance                       *ptr_Variance,
      ElementMean                           *ptr_Mean_,
      ElementOutput                         *ptr_Shifted_K_ = nullptr
    ):
      elementwise(elementwise_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ptr_Variance(ptr_Variance),
      ptr_Mean(ptr_Mean_),
      ptr_Shifted_K(ptr_Shifted_K_)
    {

    }
  };

  struct Params {

    typename ElementwiseFunctor::Params   elementwise;
    typename OutputTileIterator::Params   params_C;
    typename OutputTileIterator::Params   params_D;
    typename OutputTileIterator::Element *ptr_C;
    typename OutputTileIterator::Element *ptr_D;
    ElementVariance                       *ptr_Variance;
    ElementMean                           *ptr_Mean;
    ElementOutput                         *ptr_Shifted_K;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params():
      ptr_D(nullptr),
      ptr_Variance(nullptr),
      ptr_Mean(nullptr)
    {

    }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      elementwise(args.elementwise),
      params_C(args.ref_C.layout()),
      params_D(args.ref_D.layout()),
      ptr_C(args.ref_C.data()),
      ptr_D(args.ref_D.data()),
      ptr_Variance(args.ptr_Variance),
      ptr_Mean(args.ptr_Mean),
      ptr_Shifted_K(args.ptr_Shifted_K)
    {

    }
  };

  /// Shared storage
  struct SharedStorage {

  };

private:

  Params const &                        params_;
  SharedStorage &                       shared_storage_;
  MatrixCoord                           extent_;
  ElementwiseFunctor                    elementwise_;

  OutputTileIterator                    iterator_C_;
  OutputTileIterator                    iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator                    alpha_;
  ElementAccumulator                    beta_;
  ConvertedShiftFragment                shift_k_frag_;

  ElementLayernormCompute               accum_sum_square_;
  ElementLayernormCompute               accum_sum_element_;

  MatrixCoord                           thread_offset_;

public:

  CUTLASS_DEVICE
  EpilogueVisitorLayerNorm(
    Params const &params,                                         ///< Parameters routed to the epilogue
    SharedStorage &shared_storage,                                ///< Shared storage needed by the functors here
    MatrixCoord const &problem_size0,                              ///< Problem size of the output
    int thread_idx,                                               ///< Thread index within the threadblock
    int warp_idx,                                                 ///< Warp index within the threadblock
    int lane_idx,                                                 ///< Lane index within the warp
    MatrixCoord const &threadblock_offset = MatrixCoord(0, 0)
  ):
    params_(params),
    shared_storage_(shared_storage),
    extent_(problem_size0),
    elementwise_(params.elementwise),
    iterator_C_(params.params_C, params.ptr_C, problem_size0, thread_idx, threadblock_offset),
    iterator_D_(params.params_D, params.ptr_D, problem_size0, thread_idx, threadblock_offset)
  {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
    beta_ =  (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
    int split_k_index,                                            ///< Index of this threadblock within split-K partitioned scheme
    int split_k_slices) {                                         ///< Total number of split-K slices

  }

  /// 设置批次索引（用于批处理）
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {

  }

  /// 在epilogue开始时调用，在迭代累加器切片之前
  CUTLASS_DEVICE
  void begin_epilogue() {

    // 如果启用了shift-K特性，在epilogue开始时加载shift-k片段
    // Shift-K技术通过减去一个偏移值K来提高数值稳定性
    if (kIsShiftedVariance && params_.ptr_Shifted_K != nullptr) {
      shift_k_frag_.clear();
      int thread_offset_row_base = iterator_D_.thread_start_row();

      // 为每个线程加载所有需要的shift-k值
      CUTLASS_PRAGMA_UNROLL
      for (int iter_idx = 0; iter_idx < kIterations; ++iter_idx) {
        int step_offset = iter_idx * OutputTileIterator::Shape::kRow;
        CUTLASS_PRAGMA_UNROLL
        for (int rid = 0; rid < kRowIterations; ++rid) {
          int row_step_offset = rid * kDeltaRow;
          int row_offset = thread_offset_row_base + step_offset + row_step_offset;
          bool is_load = (row_offset < extent_.row());
          shift_k_frag_[iter_idx * kRowIterations + rid] = load_shift_k_(row_offset, is_load);
        }

      }

    }

  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();

    if (elementwise_.kScale != cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      fragment_C_.clear();
      iterator_C_.load(fragment_C_);
      ++iterator_C_;
    }
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {

  }

  /// 访问每个累加器向量后调用，这是计算LayerNorm部分统计量的核心函数
  CUTLASS_DEVICE
  void visit(
    int iter_idx,       // 迭代索引
    int row_idx,        // 行索引
    int column_idx,     // 列索引
    int frag_idx,       // 片段索引
    AccumulatorFragment const &accum) {  // 累加器片段

    using Mul = cutlass::multiplies<ElementLayernormCompute>;
    using Minus = cutlass::minus<ElementLayernormCompute>;
    using Exp   = cutlass::fast_exp_op<ElementLayernormCompute>;

    [[maybe_unused]] Minus minus;
    [[maybe_unused]] Mul   mul;
    [[maybe_unused]] Exp   exponential;

    LayernormFragment result;

    // 计算当前线程处理的全局坐标
    thread_offset_ =
      iterator_D_.thread_start() +
      OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    NumericArrayConverter<ElementLayernormCompute, ElementOutput, kElementsPerAccess> source_converter;
    OutputVector &source_vector = reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    // 列边界检查
    bool column_guard = (thread_offset_.column() < extent_.column());

    // 应用逐元素操作（如alpha缩放、beta加法等）
    if (elementwise_.kScale == cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      result = source_converter(elementwise_(accum));
    }else{
      result = source_converter(elementwise_(accum, source_vector));
    }


    // 计算1/N，用于求平均值
    ElementLayernormCompute inv_scalar = cutlass::constants::one<ElementLayernormCompute>() / ElementLayernormCompute(extent_.column());

    // 计算元素和（不需要列边界检查，因为越界的片段已被清零）
    accum_sum_element_ = element_sum_accumulator_(result);

    // 计算平方和（需要列边界检查）
    // 对于shift-k，越界列不应该计算，否则会错误地添加额外的k^2
    if (column_guard) {
      accum_sum_square_ = (kIsShiftedVariance) ? \
                        square_sum_accumulator_(result, shift_k_frag_[iter_idx * kRowIterations + row_idx]) : \
                        square_sum_accumulator_(result);
    }
    else {
      accum_sum_square_ = ElementLayernormCompute(0);
    }

    // 乘以1/N得到平均值
    accum_sum_element_ *= inv_scalar;
    accum_sum_square_ *= inv_scalar;

    // 线程内规约完成后，执行跨线程/warp内规约
    // 使用shuffle指令在warp内的线程间进行规约
    CUTLASS_PRAGMA_UNROLL
    for (int i = kHalfThreadsPerRow; i > 0; i >>= 1) {
      accum_sum_element_ += __shfl_xor_sync(0xFFFFFFFF, accum_sum_element_, i);
      accum_sum_square_ += __shfl_xor_sync(0xFFFFFFFF, accum_sum_square_, i);
    }

    // Convert to the output
    NumericArrayConverter<ElementOutput, ElementLayernormCompute, kElementsPerAccess> output_converter;
    OutputVector &output = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// 在行结束时调用，将统计量写入全局内存
  CUTLASS_DEVICE
  void end_row(int row_idx) {

    using ConvertVarianceOutput = cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput = cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    ConvertVarianceOutput   convert_variance_output;
    ConvertMeanOutput  convert_mean_output;

    // 只有每行的第一个线程负责写入（避免重复写入）
    bool is_write_thread = (thread_offset_.row() < extent_.row() && (threadIdx.x % kThreadsPerRow) == 0);
    int row_offset = thread_offset_.row() + blockIdx.y * extent_.row();

    // 计算全局内存中的写入地址
    ElementVariance *curr_ptr_sum_square = params_.ptr_Variance + row_offset;
    ElementMean *curr_ptr_element_sum = params_.ptr_Mean + row_offset;

    // 将平方和写入全局内存（部分规约结果）
    arch::global_store<ElementVariance, sizeof(ElementVariance)>(
              convert_variance_output(accum_sum_square_),
              (void *)curr_ptr_sum_square,
              is_write_thread);

    // 将元素和写入全局内存（部分规约结果）
    arch::global_store<ElementMean, sizeof(ElementMean)>(
              convert_mean_output(accum_sum_element_),
              (void *)curr_ptr_element_sum,
              is_write_thread);

  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {

  }

private:

  CUTLASS_DEVICE
  ElementLayernormCompute load_shift_k_(int row_offset, bool is_load) {
    using ConvertShiftK = cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;
    ConvertShiftK convert_shift_k;
    ElementOutput shift_k_val;

    // Computes the address to load shift_k element
    ElementOutput *curr_ptr_shift_k = params_.ptr_Shifted_K + row_offset;
    // Conditionally loads from global memory
    arch::global_load<ElementOutput, sizeof(ElementOutput)>(shift_k_val, (void *)curr_ptr_shift_k, is_load);
    // Converts data type to return
    ElementLayernormCompute converted_shift_k_val = convert_shift_k(shift_k_val);

    return converted_shift_k_val;
  }

  // 计算片段的平方和（标准版本）
  CUTLASS_DEVICE
  ElementLayernormCompute square_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i];
      sum_ += accum_ * accum_;  // 累加x²
    }

    return sum_;
  }

  // 计算片段的平方和（偏移版本，用于数值稳定性）
  CUTLASS_DEVICE
  ElementLayernormCompute square_sum_accumulator_(LayernormFragment const &accum, ElementLayernormCompute shift_k_val) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i] - shift_k_val;  // 减去偏移值K
      sum_ += accum_ * accum_;  // 累加(x-K)²
    }

    return sum_;
  }

  // 计算片段的元素和
  CUTLASS_DEVICE
  ElementLayernormCompute element_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      sum_ += accum[i];  // 累加x
    }

    return sum_;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GemmLayernorm类：实现Transformer中FFN层的融合操作
/// 融合了三个操作：
/// 1. GEMM0: 第一个矩阵乘法，并在epilogue中计算部分LayerNorm统计量
/// 2. LayerNorm: 完成统计量规约并应用归一化
/// 3. GEMM1: 第二个矩阵乘法，使用LayerNorm的结果作为输入
template <
  typename ElementInputA0_,       // GEMM0输入A的元素类型
  typename LayoutInputA0_,        // GEMM0输入A的布局
  typename ElementInputB0_,       // GEMM0输入B的元素类型
  typename LayoutInputB0_,        // GEMM0输入B的布局
  typename ElementOutput_,        // 输出元素类型
  typename LayoutOutput_,         // 输出布局
  typename ElementCompute_,       // 计算元素类型
  typename EpilogueFunctorOp_,    // Epilogue函数对象
  typename ThreadblockShape_,     // 线程块形状
  typename WarpShape_,            // Warp形状
  typename InstructionShape_,     // 指令形状（Tensor Core）
  int Stages0,                    // GEMM0的流水线级数
  int Stages1,                    // GEMM1的流水线级数
  bool IsShiftedVariance_ = false // 是否使用偏移方差
>
class GemmLayernorm {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  // 检查是否需要内部转置（当输出为列主序时）
  static bool const kInternalTranspose = cutlass::platform::is_same<LayoutOutput_, cutlass::layout::ColumnMajor>::value;
  static bool const kIsShiftedVariance = IsShiftedVariance_;

  // LayerNorm的scale和bias必须是行主序布局
  using LayoutInputScaleBias = cutlass::layout::RowMajor;

  // LayerNorm计算必须使用float以保证精度
  using ElementLayernormCompute = float;
  // Scale和Bias使用half类型
  using ElementInputScaleBias = cutlass::half_t;

  // 主循环融合需要的强制参数
  using OperatorClass       = cutlass::arch::OpClassTensorOp;  // 使用Tensor Core
  using ArchTag             = cutlass::arch::Sm80;             // Ampere架构

  // These are mandatory layouts and data types
  // that are inheritated from pre-defined params

  using LayoutSumSqr = LayoutInputScaleBias;
  using LayoutSum = LayoutInputScaleBias;

  using ElementMean = ElementInputScaleBias;
  using ElementVariance = ElementInputScaleBias;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using LayoutInputA0 = LayoutInputA0_;
  using LayoutInputB0 = LayoutInputB0_;
  using LayoutInputA1 = LayoutOutput_;
  using LayoutInputB1 = LayoutOutput_;
  using LayoutOutputC0 = LayoutOutput_;
  using LayoutOutputC1 = LayoutOutput_;

  using ElementInputA0 = ElementInputA0_;
  using ElementInputB0 = ElementInputB0_;
  using ElementOutputC0 = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementInputB1 = ElementInputB0_;

  using ElementInputA1 = ElementOutputC0;
  using ElementOutputC1 = ElementOutputC0;

  using EpilogueFunctorOp = EpilogueFunctorOp_;

  using TensorRefA = TensorRef<ElementInputA0, LayoutInputA0>;
  using TensorRefB = TensorRef<ElementInputB0, LayoutInputB0>;
  using TensorRefC = TensorRef<ElementOutputC0, LayoutOutputC0>;
  using TensorVariance = TensorRef<ElementVariance, LayoutSumSqr>;
  using TensorMean = TensorRef<ElementMean, LayoutSum>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  static int const kStages0 = Stages0;
  static int const kStages1 = Stages1;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using MapArguments = cutlass::gemm::kernel::detail::MapArguments<
    ElementInputA0,
    LayoutInputA0,
    cutlass::ComplexTransform::kNone,
    128 / cutlass::sizeof_bits<ElementInputA0>::value,
    ElementInputB0,
    LayoutInputB0,
    cutlass::ComplexTransform::kNone,
    128 / cutlass::sizeof_bits<ElementInputB0>::value,
    LayoutOutputC0,
    kInternalTranspose
  >;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementOutputC0,
    typename MapArguments::LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    SwizzleThreadBlock,
    kStages0,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementInputA0, ElementInputB0, ElementOutputC0, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = kernel::EpilogueVisitorLayerNorm<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    typename DefaultGemmKernel::Epilogue::AccumulatorFragmentIterator::AccumulatorTile,
    ElementCompute,
    ElementVariance,
    ElementMean,
    ElementLayernormCompute,
    EpilogueFunctorOp,
    kIsShiftedVariance
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmEpilogueFusion = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    SwizzleThreadBlock
  >;

  using ApplyFinalReductionKernel = kernel::ApplyFinalReduction<
    ElementVariance,
    ElementMean,
    ElementLayernormCompute,
    ElementOutputC0,
    ThreadblockShape,
    kIsShiftedVariance
  >;

using GemmMainloopFusion = typename cutlass::gemm::device::GemmLayernormMainloopFusion<
  ElementInputA1, LayoutInputA1,
  ElementInputB1, LayoutInputB1,
  ElementInputScaleBias, LayoutInputScaleBias,
  ElementOutputC1, LayoutOutputC1,
  ElementCompute,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueFunctorOp,
  SwizzleThreadBlock,
  kStages1
>;

public:

  /// 参数结构体，包含所有三个操作的参数
  struct Arguments {

    typename GemmEpilogueFusion::Arguments         gemm0;      // GEMM0的参数
    typename GemmMainloopFusion::Arguments         gemm1;      // GEMM1的参数
    typename ApplyFinalReductionKernel::Arguments reduction;   // LayerNorm规约的参数
    cutlass::gemm::GemmCoord extend;                          // 矩阵维度

    //
    // Methods
    //
    Arguments() { }

    // 构造函数：初始化所有参数
    Arguments(
      cutlass::gemm::GemmCoord problem_size0,  // GEMM0的问题规模 [M, N, K]
      cutlass::gemm::GemmCoord problem_size1,  // GEMM1的问题规模 [M, N, K]
      ElementInputA0 * ptr_A,                  // GEMM0输入A指针
      ElementInputB0 * ptr_B,                  // GEMM0输入B指针
      ElementOutputC0 * ptr_C,                 // GEMM0输入C指针（bias）
      ElementOutputC0 * ptr_D,                 // GEMM0输出指针
      ElementOutputC0 * ptr_E,                 // GEMM1输入B指针
      ElementOutputC0 * ptr_O,                 // 最终输出指针
      int64_t    ldm_A,                        // A的leading dimension
      int64_t    ldm_B,                        // B的leading dimension
      int64_t    ldm_C,                        // C的leading dimension
      int64_t    ldm_D,                        // D的leading dimension
      int64_t    ldm_E,                        // E的leading dimension
      int64_t    ldm_O,                        // O的leading dimension
      typename EpilogueFunctorOp::Params linear_scaling,  // 线性缩放参数(alpha, beta)
      TensorVariance ref_Variance_,            // 方差张量引用
      TensorMean ref_Mean_,                    // 均值张量引用
      TensorVariance ref_Gamma_,               // LayerNorm的Gamma参数（缩放）
      TensorMean ref_Beta_,                    // LayerNorm的Beta参数（偏移）
      ElementOutputC0 *ptr_Shifted_K = nullptr // 偏移K指针（可选）
    ):
      gemm0(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {kInternalTranspose ? problem_size0.n() : problem_size0.m(),\
         kInternalTranspose ? problem_size0.m() : problem_size0.n(),\
         problem_size0.k()},
        {kInternalTranspose ? ptr_B : ptr_A, \
        kInternalTranspose ? ldm_B : ldm_A},
        {kInternalTranspose ? ptr_A : ptr_B, \
        kInternalTranspose ? ldm_A : ldm_B},
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          {ptr_C, ldm_C},
          {ptr_D, ldm_D},
          ref_Variance_.data(),
          ref_Mean_.data(),
          ptr_Shifted_K
        )
      ),
      reduction(
        MatrixCoord(kInternalTranspose ? problem_size0.n() : problem_size0.m(),\
                    kInternalTranspose ? problem_size0.m() : problem_size0.n()),
        ref_Variance_,
        ref_Mean_,
        ptr_Shifted_K
      ),
      gemm1(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size1,
        1,
        linear_scaling,
        kInternalTranspose ? ptr_E : ptr_D,
        kInternalTranspose ? ptr_D : ptr_E,
        ref_Variance_.data(),
        ref_Mean_.data(),
        ref_Gamma_.data(),
        ref_Beta_.data(),
        ptr_O,
        ptr_O,
        problem_size1.m() * problem_size1.k(),
        problem_size1.n() * problem_size1.k(),
        problem_size1.n(),
        problem_size1.n(),
        problem_size1.k(),
        problem_size1.k(),
        problem_size1.m() * problem_size1.n(),
        problem_size1.m() * problem_size1.n(),
        kInternalTranspose ? ldm_E : ldm_D,
        kInternalTranspose ? ldm_D : ldm_D,
        ref_Variance_.layout().stride(0),
        ref_Mean_.layout().stride(0),
        ref_Gamma_.layout().stride(0),
        ref_Beta_.layout().stride(0),
        ldm_O,
        ldm_O
      ),
      extend(problem_size0)
    {

    }
  };

  struct Params {

    typename GemmEpilogueFusion::Params         gemm0;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm0(args.gemm0),
      reduction(args.reduction),
      extend(MatrixCoord(args.extend.m(), args.extend.n()))
    {

    }
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Params params_;
  GemmMainloopFusion gemm_fusion_op;

public:

  /// Ctor
  GemmLayernorm() {

  }

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);
    cutlass::Status status;
    size_t workspace_size = gemm_fusion_op.get_workspace_size(args.gemm1);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = gemm_fusion_op.can_implement(args.gemm1);
    CUTLASS_CHECK(status);

    status = gemm_fusion_op.initialize(args.gemm1, workspace.get());
    CUTLASS_CHECK(status);

    return cutlass::Status::kSuccess;
  }

  /// 执行融合操作
  Status run(cudaStream_t stream) {

    //
    // 第一步：启动GEMM0 + 部分LayerNorm规约的融合kernel
    //

    // 计算Grid维度
    dim3 gemm_grid = SwizzleThreadBlock().get_grid_shape(params_.gemm0.grid_tiled_shape);
    dim3 gemm_block(GemmEpilogueFusion::kThreadCount, 1, 1);

    // 计算共享内存大小
    int gemm_smem_size = int(sizeof(typename GemmEpilogueFusion::SharedStorage));

    // 启动GEMM0 kernel
    cutlass::Kernel<GemmEpilogueFusion><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm0);

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // 第二步：启动最终规约kernel，完成LayerNorm统计量计算
    //

    // 始终从主维度执行规约
    // 如果是列主序，需要交换行列维度
    int leading_dim_0 = kInternalTranspose ? params_.extend.row() : params_.extend.column();
    int leading_dim_1 = kInternalTranspose ? params_.extend.column() : params_.extend.row();

    // 动态选择线程块大小
    int thread_per_block = 128;
    int block_per_row = (leading_dim_1 + thread_per_block - 1) / thread_per_block;
    // 如果块数太少，减小线程块大小以提高并行度
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row = (leading_dim_1 + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_block(thread_per_block);
    dim3 final_reduction_grid(block_per_row);

    // 启动规约kernel
    Kernel<ApplyFinalReductionKernel><<<
      final_reduction_grid, final_reduction_block, sizeof(typename ApplyFinalReductionKernel::SharedStorage), stream
    >>>(params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // 第三步：启动GEMM1 + LayerNorm应用的融合kernel
    // 这个kernel在主循环中应用LayerNorm变换
    //

    cutlass::Status status = gemm_fusion_op();
    CUTLASS_CHECK(status);

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
