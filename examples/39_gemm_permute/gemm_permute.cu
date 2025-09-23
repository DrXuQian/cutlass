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
 * CUTLASS Example 39: GEMM with Tensor Permutation
 *
 * This example demonstrates GEMM operations with fused tensor permutation/transposition
 * applied to input and/or output matrices. This fusion is crucial for efficient implementation
 * of complex tensor operations commonly found in modern deep learning workloads.
 *
 * FUSION OVERVIEW:
 * ================
 * The kernel performs: D_permuted = permute(alpha * A_permuted @ B_permuted + beta * C)
 * where permutation operations are applied as part of the memory access pattern,
 * avoiding separate kernels for tensor reshaping and transposition.
 *
 * PERMUTATION PATTERNS:
 * ====================
 * This example supports several categories of tensor permutations:
 *
 * 1. Normal GEMM Permutations:
 *    - Tensor4DPermute0213: [X, Y] → [X/S1, S1, S2, Y/S2] → permute([0,2,1,3]) → [X*S2/S1, Y*S1/S2]
 *    - Tensor5DPermute20314: [X, Y] → [X/T1, T1, T2, T3, Y/T2/T3] → permute([2,0,3,1,4]) → [X*T2/T1, Y*T1/T2]
 *
 * 2. Batched GEMM Permutations:
 *    - Tensor4DPermuteBMM0213: [B, X, Y] → [B/D1, D1, X, Y] → permute([0,2,1,3]) → [B/D1, X, Y*D1]
 *
 * PERFORMANCE BENEFITS:
 * ====================
 * 1. Memory Bandwidth Efficiency: Eliminates separate permutation kernels
 * 2. Cache Optimization: Improved spatial and temporal locality
 * 3. Kernel Launch Overhead: Single kernel vs. separate GEMM + permute launches
 * 4. Memory Footprint: Reduced intermediate storage requirements
 *
 * KEY ARCHITECTURAL FEATURES:
 * ===========================
 * - Flexible layout plugin system for custom permutation patterns
 * - Tensor Core acceleration with optimized memory access patterns
 * - Support for both normal and batched GEMM operations
 * - Configurable alignment for optimal memory coalescing
 * - Runtime tensor shape validation and constraint checking
 *
 * COMMON USE CASES:
 * =================
 * 1. Transformer Attention: Query/Key/Value matrix transpositions
 * 2. Convolution as GEMM: Im2col transformations with reshaping
 * 3. Tensor Contractions: Multi-dimensional matrix multiplications
 * 4. Data Layout Conversions: NCHW ↔ NHWC transformations
 * 5. Batch Processing: Efficient batched operations with reordering
 *
 * LAYOUT PLUGIN ARCHITECTURE:
 * ===========================
 * The permutation system uses a plugin architecture defined in:
 * include/cutlass/layout/permute.h
 *
 * Key components:
 * - Address computation functions: compute(col, row, stride, batch_idx)
 * - Dimension tracking: {col_permute, row_permute, stride_permute}
 * - Memory alignment optimization for permuted access patterns
 *
 * IMPLEMENTATION CONSTRAINTS:
 * ===========================
 * 1. Batch Stride Configuration:
 *    - Set batch_stride = 0 for BMM permutations
 *    - Use GemmUniversalMode::kBatched (not kArray) for batched operations
 *
 * 2. Memory Alignment Requirements:
 *    - Alignment = 1 when contiguous dimension is permuted
 *    - Alignment = 8 (or higher) when unit stride dimension is preserved
 *    - Row-major: [0,2,3,1] permutation requires Alignment = 1
 *    - Column-major: [1,0,2,3] permutation requires Alignment = 1
 *
 * 3. Performance Optimization:
 *    - Avoid permuting the unit stride dimension for best performance
 *    - Larger alignment values improve memory throughput
 *    - Consider memory access patterns in permutation design
 *
 * NUMERICAL PROPERTIES:
 * =====================
 * - Maintains identical numerical results to unpermuted GEMM operations
 * - Permutation affects only memory layout, not computation precision
 * - Deterministic results for fixed input ordering
 *
 * USAGE EXAMPLES:
 * ===============
 *   # Run batched GEMM with 96 batches and default permutations
 *   $ ./39_gemm_permute --batch-count=96
 *
 *   # Run with custom dimensions and verbose output
 *   $ ./39_gemm_permute --batch-count=96 --k=1024 --verbose=true
 *
 *   # Profile with NSight Compute
 *   $ nv-nsight-cu-cli ./39_gemm_permute --m=256 --n=192 --k=256 --verbose=true --iterations=1 --reference-check=false
 *
 * COMPILE-TIME CONFIGURATION:
 * ===========================
 * Permutation parameters (S1, S2, D1, T1, T2, T3) are compile-time constants
 * defined below. Runtime specification is not currently supported due to
 * template instantiation requirements.
 */

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/permute.h"

#include "layouts.h"
#include "permute_info.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// COMPILE-TIME PERMUTATION CONFIGURATION
/// =======================================
/// These constants define the tensor reshaping and permutation patterns.
/// All values are compile-time constants and cannot be changed at runtime.
///
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Batched GEMM 4D Permutation: Tensor4DPermuteBMM0213
/// ====================================================
/// Pattern: [B, M, N] → [B/D1, D1, M, N] → permute([0, 2, 1, 3]) → [B/D1, M, D1, N]
/// Use case: Batched matrix multiplication with inter-batch dimension reordering
int constexpr D1 = 12;  // Batch subdivision factor

/// Normal GEMM 5D Permutation: Tensor5DPermute20314
/// ================================================
/// Pattern: [M, N] → [M/T1, T1, T2, T3, N/T2/T3] → permute([2, 0, 3, 1, 4]) → [T2, M/T1, T3, T1, N/T2/T3]
/// Use case: Complex tensor contractions and multi-dimensional reorderings
int constexpr T1 = 16;  // Primary dimension subdivision
int constexpr T2 = 3;   // Secondary dimension subdivision
int constexpr T3 = 8;   // Tertiary dimension subdivision

/// Normal GEMM 4D Permutation: Tensor4DPermute0213
/// ===============================================
/// Pattern: [M, N] → [M/S1, S1, S2, N/S2] → permute([0, 2, 1, 3]) → [M/S1, S2, S1, N/S2]
/// Use case: Standard matrix transposition with tiling for cache efficiency
int constexpr S1 = 8;   // Row tile size
int constexpr S2 = 4;   // Column tile size

/// Memory Alignment Configuration
/// ===============================
/// Alignment requirements balance memory bandwidth with permutation constraints
/// Higher alignment improves throughput when compatible with permutation patterns
int constexpr AlignmentA = 8;  // Matrix A alignment (8 elements = 16 bytes for FP16)
int constexpr AlignmentB = 8;  // Matrix B alignment (8 elements = 16 bytes for FP16)
int constexpr AlignmentC = 8;  // Matrix C/D alignment (8 elements = 16 bytes for FP16)

/// Data Type Configuration
/// =======================
/// Optimized for modern GPU architectures with mixed-precision support
using ElementInput = cutlass::half_t;     // Input matrices: FP16 for memory efficiency
using ElementOutput = cutlass::half_t;    // Output matrix: FP16 for downstream compatibility
using ElementAccumulator = float;         // Internal accumulation: FP32 for numerical accuracy

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Error Handling Macros
/// ======================
/// Convenience macros for consistent error checking throughout the example

// CUDA Runtime API error checking
#define CHECK_CUDA_CALL(call, handler) \
do { \
  cudaError_t __err = (call); \
  if (__err != cudaSuccess) { \
    std::cerr << #call " failed: " << cudaGetErrorString(__err) << std::endl; \
    handler; \
  } \
} while(0)

// CUTLASS API error checking
#define CHECK_CUTLASS_CALL(call, handler) \
do { \
  cutlass::Status __status = (call); \
  if (__status != cutlass::Status::kSuccess) { \
    std::cerr << #call " failed: " << cutlass::cutlassGetStatusString(__status) << std::endl; \
    handler; \
  } \
} while(0)

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Command Line Configuration
/// ===========================
/// Comprehensive options structure for controlling GEMM dimensions,
/// batching parameters, and execution settings
struct Options {

  bool help;
  bool error;
  bool reference_check;

  cutlass::gemm::GemmCoord problem_each;

  int batch_count;
  int iterations;
  int cuda_streams;
  bool verbose;
  float alpha;
  float beta;

  //
  // Methods
  // 

  Options():
    help(false),
    error(false),
    reference_check(true),    // Enable numerical verification by default
    batch_count(-1),          // Will be set to default in parse() if not specified
    iterations(20),           // Performance measurement iterations
    cuda_streams(0),          // Number of CUDA streams (0 = synchronous)
    verbose(false),           // Detailed output disabled by default
    alpha(1),                 // GEMM scaling factor
    beta()                    // Bias scaling factor (zero by default)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);    
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);

    int m, n, k;

    // Parse GEMM dimensions with reasonable defaults for demonstration
    cmd.get_cmd_line_argument("m", m, 384);           // M dimension (384 = multiple of tile sizes)
    cmd.get_cmd_line_argument("n", n, 192);           // N dimension (192 = multiple of tile sizes)
    cmd.get_cmd_line_argument("k", k, 384);           // K dimension (384 = multiple of tile sizes)
    cmd.get_cmd_line_argument("batch-count", batch_count, 96);  // 96 batches for meaningful statistics

    problem_each = cutlass::gemm::GemmCoord(m, n, k);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << 
      "39_gemm_permute\n"
      "\n"
      " This example tests and profiles the performance of normal GEMM and batched GEMM with different"
      " combinations of fused permutations of input and output tensors."
      "\n"
      " Permutations considered in this example:\n"
      "\n"
      " Normal GEMM:\n"
      " 1) Tensor4DPermute0213: matrix of shape [X, Y] is reshaped as [X/S1, S1, S2, Y/S2] and has its dimensions"
      " permuted as [0, 2, 1, 3], resulting in shape [X/S1, S2, S1, Y/S2] viewed as matrix of shape [X*S2/S1, Y*S1/S2].\n"
      " 2) Tensor5DPermute20314: matrix of shape [X, Y] is reshaped as [X/T1, T1, T2, T3, Y/T2/T3] and has its dimensions"
      " permuted as [2, 0, 3, 1, 4], resulting in shape [T2, X/T1, T3, T1, Y/T2/T3] viewed as matrix of shape [X*T2/T1, Y*T1/T2].\n"
       "\n"
      " Batched GEMM:\n"
      " 3) Tensor4DPermuteBMM0213: batched tensor of 3D shape [B, X, Y] is reshaped as 4D shape [B/D1, D1, X, Y]"
      " and has its dimensions permuted as [0, 2, 1, 3], resulting in shape [B/D1, X, D1, Y] viewed as"
      " a matrix of shape [B/D1, X, Y*D1] for batched GEMM purposes.\n"
      "\n"
      " Note: S1, S2, D1, D2, T1, T2, T3 are compile-time constants defined in gemm_permute.cu."
      " Runtime specification of these values is not supported."
      " These values along with alignment requirements place constraints on supported matrix sizes.\n"
      "\n"
      " Note: X, Y above may refer to M, N or K dimensions of GEMM problem, depending on the tensor considered (A, B or D)."
      " For the output tensor D the values correspond directly to dimensions of D, whereas for A and B the original dimensions"
      " X', Y' are inferred from the ones supplied to the GEMM, taking into account the permute operation.\n"
      "\n"
      "Options:\n"
      "\n"
      "  --help                      If specified, displays this usage statement.\n\n"
      "  --batch-count=<int>         Sets the number of batches in batched GEMM (batch number for BMM). (default: --batch-count=768)\n"
      "  --m=<int>                   Sets the M dimension for both batched GEMM and normal GEMM problems. (default: --m=128)\n"
      "  --n=<int>                   Sets the N dimension for both batched GEMM and normal GEMM problems. (default: --n=192)\n"
      "  --k=<int>                   Sets the K dimension for both batched GEMM and normal GEMM problems. (default: --k=384)\n"
      "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      "  --iterations=<int>          Number of profiling iterations to perform.\n"
      "  --reference-check=<bool>    If true, performs reference check.\n"
      "  --verbose=<bool>            If true, prints problem sizes and batching structure.\n"
      "\n"
      "Examples:\n"
      "\n"
      "# Runs a batched GEMM with 96 batches\n"
      "$ ./examples/39_gemm_permute/39_gemm_permute --batch-count=96\n"
      "\n"
      "# Runs a batched GEMM with 96 batches (with GEMM-K dimension equal to 1024)\n"
      "$ ./examples/39_gemm_permute/39_gemm_permute --batch-count=96 --k=1024 --verbose=true\n"
      "\n"
      "# Execute batched GEMM and profile with NSight\n"
      "$ nv-nsight-cu-cli ./examples/39_gemm_permute/39_gemm_permute --m=256 --n=192 --k=256 --verbose=true --iterations=1 --reference-check=false\n"
      "\n";

    return out;
  }

  /// Performance Calculation
  /// =======================
  /// Computes effective throughput in GFLOP/s for the permuted GEMM operations
  double gflops(double runtime_s, bool batched) const {

    // Calculate total multiply-add operations
    // Each GEMM contributes M*N*K multiply-adds
    int64_t fmas = problem_each.product() * (batched ? batch_count : 1);

    // Convert to floating-point operations (2 ops per multiply-add)
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace { // Anonymous namespace for implementation details

/// Recursive Host-Based Permutation Implementation
/// ===============================================
/// Template-based recursive function for applying arbitrary tensor permutations on CPU
/// Used for reference computation and verification purposes
template<int I, typename Element, typename Layout, typename PermuteOp, typename Coord>
void permute_host_impl(
    cutlass::TensorView<Element const, Layout> const & input,   // Source tensor view
    cutlass::TensorView<Element, Layout> const & output,        // Destination tensor view
    PermuteOp && permute,                                       // Permutation operation
    Coord & coord                                               // Current coordinate being processed
) {
  static_assert(Layout::kRank == Coord::kRank, "Layout and Coordinate ranks must match");

  if constexpr (I == Coord::kRank) {
    // Base case: copy element with permuted coordinates
    output.at(permute(coord)) = input.at(coord);
  } else {
    // Recursive case: iterate through dimension I
    for (coord[I] = 0; coord[I] < input.extent(I); ++coord[I]) {
      permute_host_impl<I+1>(input, output, std::forward<PermuteOp>(permute), coord);
    }
  }
}

} // namespace (anonymous)

/// Host-Based Reference Permutation
/// =================================
/// Performs tensor permutation on CPU for verification against GPU kernel results
/// Supports arbitrary permutation patterns defined by PermuteLayout template parameter
template<typename PermuteLayout, typename Element, typename Layout>
void permute_host(
    cutlass::TensorView<Element const, Layout> const &input,    // Input tensor (device memory)
    cutlass::TensorView<Element, Layout> const &output,         // Output tensor (device memory)
    int batch_count                                             // Number of batched tensors
) {
  // Extract tensor properties and allocate host memory
  Layout layout = input.layout();
  cutlass::MatrixCoord extent = input.extent();

  std::size_t num_elems = layout.capacity(extent) * batch_count;
  std::vector<Element> h_input(num_elems);   // Host input buffer
  std::vector<Element> h_output(num_elems);  // Host output buffer

  // Copy input data from device to host
  cutlass::device_memory::copy_to_host(h_input.data(), input.data(), num_elems);

  // Configure permutation using template parameter information
  using Info = PermuteInfo<PermuteLayout>;
  using TensorLayout = typename Info::Layout;

  // Calculate original and permuted tensor shapes
  auto shape_orig = Info::original_shape(extent, batch_count);
  auto shape_perm = Info::permute(shape_orig);

  // Create tensor views for the permutation operation
  cutlass::TensorView<Element const, TensorLayout> view_input(
    h_input.data(), TensorLayout::packed(shape_orig), shape_orig);
  cutlass::TensorView<Element, TensorLayout> view_output(
    h_output.data(), TensorLayout::packed(shape_perm), shape_perm);

  // Execute the permutation using recursive template implementation
  decltype(shape_orig) coord;
  permute_host_impl<0>(view_input, view_output, Info::permute, coord);

  // Copy permuted results back to device memory
  cutlass::device_memory::copy_to_device(output.data(), h_output.data(), num_elems);
}

/// Layout Information Helper
/// =========================
/// Template specializations for converting layout types to human-readable names
/// Used for verbose output and debugging information

template<typename Layout>
struct LayoutInfo;

template<>
struct LayoutInfo<cutlass::layout::RowMajor> {
  static std::string name() { return "RowMajor"; }
};

template<>
struct LayoutInfo<cutlass::layout::ColumnMajor> {
  static std::string name() { return "ColumnMajor"; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM Permutation Testbed
/// =========================
/// Comprehensive test harness for validating and benchmarking GEMM operations
/// with various tensor permutation patterns applied to inputs and outputs
template <typename ElementA, typename ElementB, typename ElementC>
class Testbed {
private:

  //
  // Internal State and Configuration
  // ================================
  //

  Options & options;  // Reference to command-line configuration

  // Random initialization parameters for reproducible testing
  cutlass::Distribution::Kind init_A;  // Distribution type for matrix A
  cutlass::Distribution::Kind init_B;  // Distribution type for matrix B
  cutlass::Distribution::Kind init_C;  // Distribution type for matrix C
  uint32_t seed;                       // Random seed for deterministic results

  // GPU memory allocations for input/output matrices
  cutlass::DeviceAllocation<ElementA> block_A;  // Input matrix A
  cutlass::DeviceAllocation<ElementB> block_B;  // Input matrix B
  cutlass::DeviceAllocation<ElementC> block_C;  // Input matrix C (bias)
  cutlass::DeviceAllocation<ElementC> block_D;  // Output matrix D

public:

  //
  // Methods
  //

  /// Constructor: Testbed Initialization
  /// ===================================
  /// Configures the test environment with specified data distributions and random seed
  Testbed(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,  // Matrix A: uniform distribution
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,  // Matrix B: uniform distribution
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,  // Matrix C: uniform distribution
    uint32_t seed_ = 3090                                                  // Default random seed
  ):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

private:

  /// Tensor Information Display
  /// ===========================
  /// Prints detailed information about tensor dimensions and permutation patterns
  /// for debugging and verification purposes
  template<typename PermuteLayout>
  void print_tensor_info(
      std::ostream & os,           // Output stream for information display
      std::string const &tensor_name,  // Human-readable tensor identifier
      int row_dim,                     // Row dimension index in problem coordinates
      int col_dim                      // Column dimension index in problem coordinates
  ) {

    // Extract tensor dimensions and permutation metadata
    cutlass::MatrixCoord extent(options.problem_each.at(row_dim), options.problem_each.at(col_dim));
    using Info = PermuteInfo<PermuteLayout>;

    // Display basic tensor information
    os << "Tensor " << tensor_name << ": " << Info::desc() << "\n";
    os << "    Extent: [" << extent.row() << ", " << extent.column() << "]";
    if (Info::kBatched) {
      os << ", Batch count: " << options.batch_count;
    }
    os << "\n";

    // Display permutation details for non-trivial cases
    if (!cutlass::layout::is_trivial_permute<PermuteLayout>) {
      auto shape_orig = Info::original_shape(extent, options.batch_count);
      auto shape_perm = Info::permute(shape_orig);
      os << "    Original shape: [" << shape_orig << "]\n";
      os << "    Permuted shape: [" << shape_perm << "]\n";
    }
  }

  /// Tensor Shape Validation
  /// ========================
  /// Validates that tensor dimensions are compatible with permutation requirements
  /// and memory alignment constraints
  template<typename Layout, typename PermuteLayout, int Alignment>
  bool check_tensor_shape(
      std::string const &tensor_name,  // Tensor identifier for error reporting
      int row_dim,                     // Row dimension index
      int col_dim                      // Column dimension index
  ) {

    // Extract tensor dimensions and permutation requirements
    cutlass::MatrixCoord extent(options.problem_each.at(row_dim), options.problem_each.at(col_dim));
    using Info = PermuteInfo<PermuteLayout>;

    // Calculate alignment requirements based on memory layout
    // Column-major: alignment applies to rows (leading dimension)
    // Row-major: alignment applies to columns (leading dimension)
    auto rowAlign = cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value ? Alignment : 1;
    auto colAlign = cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value ? Alignment : 1;

    // Combine permutation and alignment requirements
    auto rowFactor = Info::kRowFactor * rowAlign;
    auto colFactor = Info::kColumnFactor * colAlign;

    // Validate row dimension divisibility
    bool const valid_row = extent.row() % rowFactor == 0;
    if (!valid_row) {
      std::cerr << "ERROR: Tensor " << tensor_name << " row size (" << extent.row()
                << ") must be divisible by " << rowFactor
                << " (required by " << Info::name()
                << (rowAlign > 1 ? (" + alignment " + std::to_string(rowAlign)) : "")
                << ")" << std::endl;
    }

    // Validate column dimension divisibility
    bool const valid_col = extent.column() % colFactor == 0;
    if (!valid_col) {
      std::cerr << "ERROR: Tensor " << tensor_name << " column size (" << extent.column()
                << ") must be divisible by " << colFactor
                << " (required by " << Info::name()
                << (colAlign > 1 ? (" + alignment " + std::to_string(colAlign)) : "")
                << ")" << std::endl;
    }

    // Validate batch count divisibility for batched operations
    bool const valid_bsz = options.batch_count % Info::kBatchFactor == 0;
    if (!valid_bsz) {
      std::cerr << "ERROR: Batch count (" << options.batch_count
                << ") must be divisible by " << Info::kBatchFactor
                << " (required by " << Info::name() << ")" << std::endl;
    }

    return valid_row && valid_col && valid_bsz;
  }

  /// Tensor Data Initialization
  /// ===========================
  /// Fills tensor memory with values from specified probability distribution
  /// Range selection ensures numerical stability and meaningful verification
  template <typename Element>
  void initialize_tensor_(
      Element *ptr,                           // Device memory pointer
      size_t capacity,                        // Number of elements to initialize
      cutlass::Distribution::Kind dist_kind,  // Distribution type
      uint32_t seed                           // Random seed
  ) {

    if (dist_kind == cutlass::Distribution::Uniform) {
      // Determine value range based on element precision
      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<ElementC>::value;

      if (bits_input == 1) {
        // Binary values
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        // Low precision (INT8, etc.)
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        // Half precision output
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope_max = 5;   // Conservative range for FP16 accumulation
          scope_min = -5;
        } else {
          scope_max = 8;   // Wider range for FP32 accumulation
          scope_min = -8;
        }
      } else {
        // Full precision
        scope_max = 8;
        scope_min = -8;
      }

      // Fill with uniform random values in calculated range
      cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {
      // Gaussian distribution with mean=0, stddev=0.5
      cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(0), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      // Sequential values: 0, 1, 2, 3, ... (useful for debugging)
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element(0));
    }
    else {
      // Identity/constant fill: all elements = 1
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(0), Element(1));
    }
  }

  /// Memory Allocation and Data Initialization
  /// ===========================================
  /// Allocates GPU memory and fills tensors with test data
  void initialize(int batch_count) {

    // Set random seed for reproducible results
    srand(seed);

    // Calculate total memory requirements for all batches
    int64_t total_elements_A = options.problem_each.m() * options.problem_each.k() * batch_count;
    int64_t total_elements_B = options.problem_each.n() * options.problem_each.k() * batch_count;
    int64_t total_elements_C = options.problem_each.m() * options.problem_each.n() * batch_count;
    int64_t total_elements_D = options.problem_each.m() * options.problem_each.n() * batch_count;

    // Allocate GPU memory for all matrices
    block_A.reset(total_elements_A);  // Input matrix A
    block_B.reset(total_elements_B);  // Input matrix B
    block_C.reset(total_elements_C);  // Input matrix C (bias)
    block_D.reset(total_elements_D);  // Output matrix D

    // Initialize input tensors with specified distributions
    // Different seeds ensure uncorrelated random data
    initialize_tensor_(block_A.get(), total_elements_A, init_A, seed * 2021);
    initialize_tensor_(block_B.get(), total_elements_B, init_B, seed * 2022);
    initialize_tensor_(block_C.get(), total_elements_C, init_C, seed * 2023);

    // Initialize output tensor to zero (will be overwritten)
    cutlass::reference::device::BlockFillSequential(
      block_D.get(), total_elements_D, ElementC(0), ElementC(0));
  }


  /// Numerical Verification Against Reference Implementation
  /// =======================================================
  /// Validates GPU kernel results by comparing against CPU-based reference computation
  /// with separate host-side permutation operations
  template<typename Gemm>
  bool validate(Gemm const &gemm) {

    // Determine if this is a batched operation by checking permutation layouts
    bool constexpr kBatched = PermuteInfo<typename Gemm::PermuteALayout>::kBatched
                           || PermuteInfo<typename Gemm::PermuteBLayout>::kBatched
                           || PermuteInfo<typename Gemm::PermuteDLayout>::kBatched;

    int const batch_count = kBatched ? options.batch_count : 1;

    // Extract problem dimensions and create tensor layouts
    cutlass::gemm::GemmCoord problem = options.problem_each;
    cutlass::MatrixCoord extent_A{problem.m(), problem.k()};
    cutlass::MatrixCoord extent_B{problem.k(), problem.n()};
    cutlass::MatrixCoord extent_C{problem.m(), problem.n()};

    // Extract layout types from GEMM template
    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    // Create packed layouts for efficient memory access
    LayoutA layout_A(LayoutA::packed(extent_A));
    LayoutB layout_B(LayoutB::packed(extent_B));
    LayoutC layout_C(LayoutC::packed(extent_C));

    // Calculate total memory sizes including batching
    auto size_A = layout_A.capacity(extent_A) * batch_count;
    auto size_B = layout_B.capacity(extent_B) * batch_count;
    auto size_C = layout_C.capacity(extent_C) * batch_count;
    
    // Create tensor views for original data
    cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get(), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get(), layout_B, extent_B);
    cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get(), layout_C, extent_C);
    cutlass::TensorView<ElementC, LayoutC> view_D(block_D.get(), layout_C, extent_C);

    // Allocate temporary storage for permuted input matrices
    cutlass::DeviceAllocation<ElementA> block_A_perm(size_A);
    cutlass::DeviceAllocation<ElementA> block_B_perm(size_B);

    // Create tensor views for permuted data
    cutlass::TensorView<ElementA, LayoutA> view_A_perm(block_A_perm.get(), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B_perm(block_B_perm.get(), layout_B, extent_B);

    // Apply input permutations using host-based reference implementation
    permute_host<typename Gemm::PermuteALayout>(view_A.const_view(), view_A_perm, batch_count);
    permute_host<typename Gemm::PermuteBLayout>(view_B.const_view(), view_B_perm, batch_count);

    // Allocate storage for reference GEMM output
    cutlass::DeviceAllocation<ElementC> block_D_ref(size_C);
    cutlass::TensorView<ElementC, LayoutC> view_D_ref(block_D_ref.get(), layout_C, extent_C);

    // Extract epilogue configuration from GEMM template
    using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;

    // Compute reference GEMM using permuted inputs
    // This performs: D_ref = alpha * A_permuted @ B_permuted + beta * C
    cutlass::reference::device::GemmComplex<
        ElementA, LayoutA,                              // Input A configuration
        ElementB, LayoutB,                              // Input B configuration
        ElementC, LayoutC,                              // Output configuration
        typename EpilogueOutputOp::ElementCompute,      // Epilogue compute type
        typename Gemm::ElementAccumulator               // Accumulator type
    >(
      problem,                                        // GEMM dimensions
      options.alpha,                                  // Scaling factor alpha
      view_A_perm,                                    // Permuted matrix A
      Gemm::kTransformA,                              // Transform operation on A
      view_B_perm,                                    // Permuted matrix B
      Gemm::kTransformB,                              // Transform operation on B
      options.beta,                                   // Scaling factor beta
      view_C,                                         // Input matrix C
      view_D_ref,                                     // Reference output D
      ElementAccumulator(0),                          // Initial accumulator value
      batch_count,                                    // Number of batches
      options.problem_each.m() * options.problem_each.k(),  // Batch stride A
      options.problem_each.n() * options.problem_each.k(),  // Batch stride B
      options.problem_each.m() * options.problem_each.n(),  // Batch stride C
      options.problem_each.m() * options.problem_each.n()   // Batch stride D
    );

    // Apply output permutation to reference results
    cutlass::DeviceAllocation<ElementC> block_D_perm(size_C);
    cutlass::TensorView<ElementC, LayoutC> view_D_perm(block_D_perm.get(), layout_C, extent_C);
    permute_host<typename Gemm::PermuteDLayout>(view_D_ref.const_view(), view_D_perm, batch_count);

    // Compare permuted reference output against kernel output
    return cutlass::reference::device::BlockCompareEqual(view_D_perm.data(), view_D.data(), size_C);
}

public:

  /// GEMM Permutation Profiling and Validation
  /// ===========================================
  /// Complete workflow for testing a specific GEMM permutation configuration
  /// including validation, performance measurement, and detailed reporting
  template<typename Gemm>
  bool profile_GEMM_permute() {

    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    using PermuteALayout = typename Gemm::PermuteALayout;
    using PermuteBLayout = typename Gemm::PermuteBLayout;
    using PermuteDLayout = typename Gemm::PermuteDLayout;

    bool constexpr kBatched = PermuteInfo<PermuteALayout>::kBatched 
                           || PermuteInfo<PermuteBLayout>::kBatched 
                           || PermuteInfo<PermuteDLayout>::kBatched;

    // Display configuration header
    std::cout << "\n"
                 "====================================================\n"
                 << (kBatched ? "Batched" : "Normal") << " GEMM with Permutation:"
                 << "\n  Matrix A: " << LayoutInfo<LayoutA>::name() << " + " << PermuteInfo<PermuteALayout>::name()
                 << "\n  Matrix B: " << LayoutInfo<LayoutB>::name() << " + " << PermuteInfo<PermuteBLayout>::name()
                 << "\n  Matrix D: " << LayoutInfo<LayoutC>::name() << " + " << PermuteInfo<PermuteDLayout>::name()
                 << "\n"
                 "====================================================\n";

    // Display detailed tensor information if requested
    if (options.verbose) {
      print_tensor_info<PermuteALayout>(std::cout, "A", 0, 2);  // A: M x K
      print_tensor_info<PermuteBLayout>(std::cout, "B", 2, 1);  // B: K x N
      print_tensor_info<PermuteDLayout>(std::cout, "D", 0, 1);  // D: M x N
    }
    std::cout << std::endl;

    // Validate tensor shapes and alignment requirements
    bool valid = true;
    valid &= check_tensor_shape<LayoutA, PermuteALayout, Gemm::kAlignmentA>("A", 0, 2);
    valid &= check_tensor_shape<LayoutB, PermuteBLayout, Gemm::kAlignmentB>("B", 2, 1);
    valid &= check_tensor_shape<LayoutC, PermuteDLayout, Gemm::kAlignmentC>("D", 0, 1);
    if (!valid) {
      std::cout << "SKIPPED: Invalid tensor dimensions for this permutation pattern" << std::endl;
      return true;
    }

    // Determine effective batch count and initialize data
    int const batch_count = kBatched ? options.batch_count : 1;
    initialize(batch_count);

    // Configure epilogue operation (linear combination: alpha*AB + beta*C)
    using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Extract problem dimensions (uniform across all batches)
    auto problem = options.problem_each;
    cutlass::MatrixCoord extent_A{problem.m(), problem.k()};
    cutlass::MatrixCoord extent_B{problem.k(), problem.n()};
    cutlass::MatrixCoord extent_C{problem.m(), problem.n()};

    // Create optimized memory layouts
    LayoutA layout_A(LayoutA::packed(extent_A));
    LayoutB layout_B(LayoutB::packed(extent_B));
    LayoutC layout_C(LayoutC::packed(extent_C));

    // Configure comprehensive GEMM kernel arguments
    typename Gemm::Arguments arguments{
      kBatched ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
      problem,                                              // Problem dimensions
      batch_count,                                          // Number of batches
      epilogue_op,                                          // Linear combination parameters
      (void*)block_A.get(),                                 // Matrix A device pointer
      (void*)block_B.get(),                                 // Matrix B device pointer
      (void*)block_C.get(),                                 // Matrix C device pointer
      (void*)block_D.get(),                                 // Matrix D device pointer
      // Batch stride configuration (critical for permuted layouts)
      // Non-trivial permutations require batch_stride = 0
      cutlass::layout::is_trivial_permute<PermuteALayout> ? layout_A.capacity(extent_A) : 0,
      cutlass::layout::is_trivial_permute<PermuteBLayout> ? layout_B.capacity(extent_B) : 0,
      layout_C.capacity(extent_C),                         // Matrix C batch stride
      cutlass::layout::is_trivial_permute<PermuteDLayout> ? layout_C.capacity(extent_C) : 0,
      layout_A.stride(0),                                   // Matrix leading dimensions
      layout_B.stride(0),
      layout_C.stride(0),
      layout_C.stride(0),
    };

    //
    // Kernel Execution
    // ================
    //

    // Initialize GEMM kernel with configured arguments
    Gemm gemm_permute;
    CHECK_CUTLASS_CALL(gemm_permute.initialize(arguments, nullptr), return false);

    // Execute initial kernel run
    CHECK_CUTLASS_CALL(gemm_permute.run(), return false);

    // Synchronize to ensure completion before verification
    CHECK_CUDA_CALL(cudaDeviceSynchronize(), return false);

    //
    // Numerical Verification
    // ======================
    //
    if (options.reference_check) {
      if (validate(gemm_permute)) {
        std::cout << "\n✓ PASSED: Numerical verification successful\n" << std::endl;
      } else {
        std::cerr << "\n✗ FAILED: Numerical verification failed\n" << std::endl;
        std::cerr << "Kernel output does not match reference implementation.\n" << std::endl;
        return false;
      }
    }

    //
    // Performance Measurement
    // =======================
    //

    // Warm-up run to stabilize GPU clocks
    CHECK_CUTLASS_CALL(gemm_permute.run(), return false);

    // Create timing events
    cudaEvent_t events[2];
    for (auto & event : events) {
      CHECK_CUDA_CALL(cudaEventCreate(&event), return false);
    }

    // Start timing measurement
    CHECK_CUDA_CALL(cudaEventRecord(events[0]), return false);

    // Execute performance measurement loop
    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_permute();
    }

    // End timing measurement
    CHECK_CUDA_CALL(cudaEventRecord(events[1]), return false);

    // Wait for all operations to complete
    CHECK_CUDA_CALL(cudaEventSynchronize(events[1]), return false);

    // Calculate performance metrics
    float runtime_total_ms = 0;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&runtime_total_ms, events[0], events[1]), return false);

    double runtime_avg_ms = double(runtime_total_ms) / double(options.iterations);
    double gflops = options.gflops(runtime_avg_ms / 1000.0, kBatched);

    // Calculate effective bandwidth accounting for permutation overhead
    double total_bytes = double(sizeof(ElementInput)) *
                        (options.problem_each.product() * 2) * // A + B matrices
                        (kBatched ? options.batch_count : 1) +
                        double(sizeof(ElementOutput)) *
                        (options.problem_each.m() * options.problem_each.n()) * // D matrix
                        (kBatched ? options.batch_count : 1);
    double bandwidth_gbps = total_bytes / (runtime_avg_ms / 1000.0) / 1e9;

    // Cleanup timing resources
    for (auto event : events) {
      CHECK_CUDA_CALL(cudaEventDestroy(event), return false);
    }

    // Display performance results
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Average Runtime: " << runtime_avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOP/s" << std::endl;
    std::cout << "  Memory Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    return true;
  }
};

/// Shorthand alist for GEMM instantiations
template<typename LayoutA, typename PermuteALayout,
         typename LayoutB, typename PermuteBLayout,
         typename LayoutC, typename PermuteDLayout>
using GemmPermute = cutlass::gemm::device::GemmUniversal<
  ElementInput, LayoutA,
  ElementInput, LayoutB,
  ElementOutput, LayoutC,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 
    AlignmentC, //128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, 
    ElementAccumulator
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  4,              /*kStages*/
  AlignmentA,     /*AlignmentA*/
  AlignmentB,     /*AlignmentB*/
  cutlass::arch::OpMultiplyAdd,
  cutlass::ComplexTransform::kNone,
  cutlass::ComplexTransform::kNone,
  false,  /*GatherA*/
  false,  /*GatherB*/
  false,  /*ScatterD*/
  PermuteDLayout,  /*PermuteDLayout*/
  typename cutlass::layout::InversePermute<PermuteALayout>::type,  /*PermuteALayout*/
  typename cutlass::layout::InversePermute<PermuteBLayout>::type   /*PermuteBLayout*/
>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Main Entry Point
/// ================
/// Orchestrates the complete GEMM permutation demonstration including
/// hardware validation, configuration setup, and execution of all test cases
int main(int argc, char const **args) {

  //
  // Hardware and Software Requirements Validation
  // =============================================
  //

  cudaDeviceProp props;
  CHECK_CUDA_CALL(cudaGetDeviceProperties(&props, 0), return EXIT_FAILURE);

  // Verify Ampere architecture and CUDA 11+ for Tensor Core support
  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
    std::cout << "CUTLASS GEMM+Permutation example requires:\n"
                 "  - NVIDIA Ampere architecture (compute capability 8.0+)\n"
                 "  - CUDA Toolkit 11.0 or later\n"
                 "Current configuration is not supported.\n";
    return EXIT_SUCCESS;
  }

  //
  // Command Line Processing
  // =======================
  //

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return EXIT_SUCCESS;
  }

  if (options.error) {
    std::cerr << "ERROR: Invalid command line arguments." << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Define GEMM types to test
  //

  //
  // TTT (Row-major) GEMMs
  //

  using TTTGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteA = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using TTTGemmNormalPermuteB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using TTTGemmNormalPermuteD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using TTTGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  //
  // NNN (Col-major) GEMMs
  //

  using NNNGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteA = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using NNNGemmNormalPermuteB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using NNNGemmNormalPermuteD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using NNNGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  //
  // NNT (Col-major inputs, row-major output) GEMMs
  //

  using NNTGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteA = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using NNTGemmNormalPermuteB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using NNTGemmNormalPermuteD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using NNTGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  //
  // TTN (Row-major inputs, col-major output) GEMMs
  //

  using TTNGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteA = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using TTNGemmNormalPermuteB = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using TTNGemmNormalPermuteD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using TTNGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  //
  // TTT (Row-major) BMMs
  //

  using TTTGemmBatchedPermuteA = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmBatchedPermuteAD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmBatchedPermuteBD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteAB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteABD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  //
  // NNN (Col-major) BMMs
  //

  using NNNGemmBatchedPermuteA = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmBatchedPermuteAD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  using NNNGemmBatchedPermuteB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmBatchedPermuteBD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  using NNNGemmBatchedPermuteD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  using NNNGemmBatchedPermuteAB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmBatchedPermuteABD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  //
  // Test Execution: Comprehensive Permutation Pattern Evaluation
  // ============================================================
  //

  std::cout << "CUTLASS GEMM Permutation Example" << std::endl;
  std::cout << "================================" << std::endl;
  std::cout << "Testing various tensor permutation patterns with GEMM operations." << std::endl;
  std::cout << "Device: " << props.name << " (Compute Capability " << props.major << "." << props.minor << ")" << std::endl;
  std::cout << std::endl;

  Testbed<ElementInput, ElementInput, ElementOutput> testbed(options);

  bool result = true;

  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteA>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteAD>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteB>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteBD>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteD>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteAB>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteABD>();

  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteA>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteAD>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteB>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteBD>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteD>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteAB>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteABD>();

  //
  // Final Results Summary
  // ====================
  //
  std::cout << "\n"
               "====================================================\n"
               "GEMM Permutation Example: " << (result ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED") << "\n";
  if (result) {
    std::cout << "All permutation patterns executed successfully.\n"
                 "Performance results demonstrate efficient tensor\n"
                 "permutation fusion with GEMM operations.\n";
  } else {
    std::cout << "One or more permutation patterns failed validation.\n"
                 "Check tensor dimensions and alignment requirements.\n";
  }
  std::cout << "====================================================" << std::endl;

  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
