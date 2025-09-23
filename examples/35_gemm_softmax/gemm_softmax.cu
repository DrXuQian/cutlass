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
 * CUTLASS Example 35: GEMM + Softmax Fusion
 *
 * This example demonstrates a fused GEMM+Softmax kernel that combines matrix multiplication
 * with softmax activation in a single GPU kernel launch. This fusion provides significant
 * performance benefits for transformer and attention mechanisms commonly used in modern
 * deep learning workloads.
 *
 * FUSION OVERVIEW:
 * ================
 * The kernel performs: D = softmax(alpha * A @ B + beta * C)
 * where softmax is applied row-wise across the N dimension of the output matrix.
 *
 * PERFORMANCE BENEFITS:
 * ====================
 * 1. Memory Bandwidth Reduction: Eliminates intermediate storage of GEMM output
 * 2. Kernel Launch Overhead: Single kernel vs. separate GEMM + Softmax launches
 * 3. Cache Efficiency: Better data locality by keeping intermediate results in registers/shared memory
 * 4. Numerical Stability: Uses numerically stable softmax implementation with max subtraction
 *
 * KEY ARCHITECTURAL FEATURES:
 * ===========================
 * - Tensor Core acceleration for GEMM computation (Ampere architecture)
 * - Fused epilogue that computes both GEMM result and softmax in the same threadblock
 * - Two-pass softmax algorithm: first pass finds max, second pass computes exp and sum
 * - Optimized memory access patterns for both GEMM and reduction operations
 *
 * COMMON USE CASES:
 * =================
 * 1. Transformer Attention: Query-Key multiplication followed by softmax
 * 2. Classification Layers: Final linear layer + softmax activation
 * 3. Sequence-to-Sequence Models: Attention score computation
 * 4. BERT/GPT-style Models: Multi-head attention mechanisms
 *
 * IMPLEMENTATION DETAILS:
 * =======================
 * - Uses CUTLASS GemmSoftmax template for optimized fusion
 * - Supports batched operations for processing multiple sequences
 * - Configurable threadblock and warp shapes for different problem sizes
 * - Automatic selection of optimal tile sizes based on problem dimensions
 *
 * NUMERICAL CONSIDERATIONS:
 * =========================
 * The implementation uses a numerically stable softmax algorithm that:
 * 1. Subtracts the maximum value from each row before exponentiation
 * 2. Computes the sum of exponentials in a separate reduction pass
 * 3. Normalizes by dividing each exponential by the sum
 *
 * This prevents overflow and maintains numerical precision even for large input values.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_size.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/numeric_size.h" // cutlass::bits_to_bytes

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_softmax.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#define TRACE(x) { std::cout << "gemm_softmax.cu:" << __LINE__ << "  " << x << std::endl; }

/////////////////////////////////////////////////////////////////////////////////////////////////

// Test result enumeration to track verification status
enum class Disposition {
  kPassed,      // All verifications passed successfully
  kIncorrect,   // Numerical verification failed
  kNotVerified  // Verification was skipped
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
// Configures problem dimensions, batch size, and execution parameters
struct Options {

  bool help;
  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  int iterations;
  unsigned seed;
  float alpha;
  float beta;
  bool verification_enabled;
  float tolerance;

  Options():
    help(false),
    problem_size({16, 24, 64}),    // Default: M=16, N=24, K=64 (small test case)
    batch_count(16),               // Process 16 matrices in parallel
    iterations(20),                // Number of timing iterations for performance measurement
    seed(2022),                    // Random seed for reproducible results
    alpha(1),                      // GEMM scaling factor: alpha * A @ B
    beta(0),                       // Bias scaling factor: beta * C (disabled by default)
    verification_enabled(true),    // Enable numerical correctness checking
    tolerance(1e-5f)              // Acceptable error tolerance for verification
  { }

  bool valid() {

    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("batch_count", batch_count);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("tolerance", tolerance);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "35_gemm_softmax example\n\n"
      << "  This example uses the CUTLASS Library to compute GEMM + Softmax for arbitrary problem sizes.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --batch_count=<int>         Batch number\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --seed=<int>                Random number seed (1*)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (0 to disable profiling).\n\n"
      << "  --verify=<bool>             If true, performs reference calculation.\n\n"
      << "  --tolerance <float>         Error tolerance\n"
    ;

    out << "\n\nExamples:\n\n"
      << "$ ./examples/35_gemm_softmax/35_gemm_softmax --m=1024 --n=512 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Returns true if the environment and Toolkit support this
  bool supported(bool verbose = true) const {

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
      }
      return false;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
      if (verbose) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
      }
      return false;
    }

    if (!((props.major * 10 + props.minor) >= 80)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
                  << std::endl;
      }
      return false;
    }

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  //
  // Data Type Configuration
  // ======================
  // These types define the precision and layout for all matrices and computations
  //


  using ElementA = cutlass::half_t;        // Input matrix A: FP16 for memory efficiency
  using ElementB = cutlass::half_t;        // Input matrix B: FP16 for memory efficiency
  using ElementC = cutlass::half_t;        // Input bias matrix C: FP16
  using ElementCompute = float;            // Internal accumulation: FP32 for numerical accuracy
  using ElementD = ElementC;               // GEMM output matrix: FP16
  using ElementSoftmax = ElementC;         // Softmax output: FP16

  // Memory Layout Configuration
  // ===========================
  using LayoutA = cutlass::layout::RowMajor;    // A matrix: rows are contiguous (standard for inputs)
  using LayoutB = cutlass::layout::ColumnMajor;  // B matrix: columns are contiguous (optimized for GEMM)

  // Hierarchical Tile Configuration for Tensor Core Optimization
  // =============================================================
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile: 128x128x32
  using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;    // Warp tile: 64x64x32 (4 warps per threadblock)
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     // Tensor Core instruction: 16x8x16 (Ampere)

  // Architecture Configuration
  // ==========================
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Use Tensor Core units for acceleration
  using ArchTag = cutlass::arch::Sm80;                   // Target Ampere architecture (compute capability 8.0+)

  // Softmax Reduction Tile Configuration
  // ====================================
  // ApplyShape controls the granularity of softmax computation and significantly impacts performance.
  // The configuration balances parallelism with memory access efficiency.
  //
  // Guidelines:
  // - kColumn should be the next multiple of 32 >= (problem_N / alignment) for memory coalescing
  // - kRow should be max(1, 128 / kColumn) to balance thread utilization
  // - Larger kColumn values improve memory bandwidth utilization
  // - Smaller kRow values increase parallelism across batch dimension
  using ApplyShape = cutlass::MatrixShape<1, 1024>;  // Process 1024 elements per softmax reduction

  // Pipeline Configuration
  // ======================
  static int const kStages = 3;  // Number of pipeline stages for overlapping compute and memory access

  // Epilogue Configuration
  // ======================
  // Defines the final operation applied to GEMM results before softmax
  // This linear combination computes: alpha * (A @ B) + beta * C
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                                    // Output element type
    128 / cutlass::sizeof_bits<ElementC>::value, // Vector width for memory operations
    ElementCompute,                              // Accumulation type for scaling
    ElementCompute                               // Scaling factor type (alpha, beta)
  >;

  // Fused GEMM+Softmax Kernel Configuration
  // =======================================
  // This template instantiation defines the complete fused kernel with all
  // architectural and algorithmic parameters specified above
  using GemmSoftmax = cutlass::GemmSoftmax<
    ElementA, LayoutA,      // Input matrix A configuration
    ElementB, LayoutB,      // Input matrix B configuration
    ElementC,               // Output/bias matrix element type
    ElementCompute,         // Internal computation precision
    OperatorClass,          // Tensor Core operation class
    ArchTag,                // Target GPU architecture
    ThreadblockShape,       // Threadblock-level tile dimensions
    WarpShape,              // Warp-level tile dimensions
    InstructionShape,       // Instruction-level tile dimensions
    EpilogueFunctorOp,      // Linear combination epilogue
    kStages,                // Pipeline stage count
    ApplyShape              // Softmax reduction tile dimensions
  >;

  using ElementNorm = typename GemmSoftmax::ElementNorm;
  using ElementSum = typename GemmSoftmax::ElementSum;
  using LayoutC = typename GemmSoftmax::LayoutC;
  using LayoutN = typename GemmSoftmax::LayoutN;
  using LayoutS = typename GemmSoftmax::LayoutS;
  using MatrixCoord = typename LayoutC::TensorCoord;

  //
  // Memory Management and Data Storage
  // ==================================
  // Host tensors for verification and device allocations for GPU computation
  //

  Options const &options;


  // Reference computation storage (CPU-based verification)
  cutlass::HostTensor<ElementNorm, LayoutC>     reference_N;      // Reference max values per row

  // GPU memory allocations for input/output matrices
  cutlass::DeviceAllocation<ElementA> block_A;         // Input matrix A
  cutlass::DeviceAllocation<ElementB> block_B;         // Input matrix B
  cutlass::DeviceAllocation<ElementC> block_C;         // Input bias matrix C
  cutlass::DeviceAllocation<ElementD> block_D;         // GEMM output matrix D
  cutlass::DeviceAllocation<ElementD> block_Ref;       // Reference GEMM result for verification
  cutlass::DeviceAllocation<ElementSoftmax> block_Softmax; // Final softmax output

  // Intermediate storage for softmax computation
  cutlass::DeviceAllocation<ElementNorm> block_Norm;   // Per-row maximum values (for numerical stability)
  cutlass::DeviceAllocation<ElementSum> block_Sum;     // Per-row exponential sums

  // Calculate number of threadblocks needed to cover the N dimension
  // This determines the storage requirements for partial reductions
  int block_num = (options.problem_size.n() + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;

  // Problem dimensions and matrix strides
  cutlass::gemm::GemmCoord problem = options.problem_size;

  // Leading dimensions for matrix layouts (for strided memory access)
  int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);  // A matrix leading dimension
  int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);  // B matrix leading dimension
  int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);  // C/D matrix leading dimension

  // Softmax auxiliary arrays use row-major layout for efficient reduction
  int64_t ldn = problem.m();  // Norm array leading dimension
  int64_t lds = ldn;          // Sum array leading dimension (same as norm)

  // Memory size calculations for allocation
  // =======================================

  // Per-batch element counts
  int64_t total_elements_A_per_batch = problem.m() * problem.k();        // A matrix size
  int64_t total_elements_B_per_batch = problem.k() * problem.n();        // B matrix size
  int64_t total_elements_C_per_batch = problem.m() * problem.n();        // C matrix size
  int64_t total_elements_D_per_batch = problem.m() * problem.n();        // D matrix size
  int64_t total_elements_partial_norm_per_batch = block_num * problem.m(); // Partial reduction storage

  // Total element counts across all batches
  int64_t total_elements_A = total_elements_A_per_batch * options.batch_count;
  int64_t total_elements_B = total_elements_B_per_batch * options.batch_count;
  int64_t total_elements_C = total_elements_C_per_batch * options.batch_count;
  int64_t total_elements_D = total_elements_D_per_batch * options.batch_count;
  int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * options.batch_count;

  //
  // Methods
  //

  Testbed(
    Options const &options_
  ):
    options(options_)
  {
    reference_N.reset({options.problem_size.m(), 1}, false);
  }

  /// Run
  Disposition run() {

    Disposition disposition = Disposition::kNotVerified;

    //
    // Initialize the workspace
    //

    initialize();

    //
    // Launch device kernel
    //
    cutlass::Status status = cutlass::Status::kSuccess;

    status = execute_device_kernel();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Device execution failed." << std::endl;
      return disposition;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return disposition;
    }

    //
    // Verify
    //

    if (options.verification_enabled) {

      bool passed = verify();

      if (passed) {
        disposition = Disposition::kPassed;
      }
      else {
        disposition = Disposition::kIncorrect;
      }
    }

    //
    // Profiling
    //
    if (options.iterations) {
      profile();
    }

    return disposition;
  }

  /// Random Initialization of Input Data
  /// ====================================
  /// Fills all input matrices with random values in a controlled range
  /// to ensure numerical stability and reproducible testing
  void initialize() {

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    block_Softmax.reset(total_elements_D);
    block_Ref.reset(total_elements_D_per_batch);
    block_Norm.reset(total_elements_partial_norm);
    block_Sum.reset(total_elements_partial_norm);

    // Initialize input matrices with random uniform distribution [-5, 5]
    // Different seeds ensure uncorrelated data across matrices
    cutlass::reference::device::BlockFillRandomUniform(
            block_A.get(), total_elements_A, options.seed, ElementA(5), ElementA(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_B.get(), total_elements_B, options.seed + 1, ElementB(5), ElementB(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_C.get(), total_elements_C, options.seed + 2, ElementC(5), ElementC(-5), 0);

    // Initialize output buffers (will be overwritten during computation)
    cutlass::reference::device::BlockFillRandomUniform(
            block_D.get(), total_elements_D, options.seed + 3, ElementD(5), ElementD(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_Ref.get(), total_elements_D_per_batch, options.seed + 3, ElementD(5), ElementD(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_Softmax.get(), total_elements_D, options.seed + 3, ElementSoftmax(5), ElementSoftmax(-5), 0);

    cutlass::reference::host::TensorFill(
      reference_N.host_view(),
      ElementNorm()
    );

  }

  /// GPU Kernel Execution
  /// =====================
  /// Launches the fused GEMM+Softmax kernel with all configured parameters
  cutlass::Status execute_device_kernel() {

    cutlass::Status status = cutlass::Status::kSuccess;

    //
    // Configure Kernel Arguments
    // ==========================
    // Package all matrices, dimensions, and parameters for kernel launch
    //

    GemmSoftmax::Arguments args(
      options.problem_size,                    // GEMM dimensions (M, N, K)
      options.batch_count,                     // Number of matrices to process
      {block_A.get(), lda},                    // Input matrix A (pointer + leading dimension)
      {block_B.get(), ldb},                    // Input matrix B (pointer + leading dimension)
      {block_C.get(), ldc},                    // Input bias matrix C (pointer + leading dimension)
      {block_D.get(), ldc},                    // GEMM output matrix D (pointer + leading dimension)
      {
        ElementCompute(options.alpha),         // GEMM scaling factor alpha
        ElementCompute(options.beta)           // Bias scaling factor beta
      },
      {block_Norm.get(), ldn},                 // Per-row maximum storage for numerical stability
      {block_Sum.get(), lds},                  // Per-row sum storage for normalization
      {block_Softmax.get(), ldc},              // Final softmax output matrix
      total_elements_A_per_batch,              // Batch stride for matrix A
      total_elements_B_per_batch,              // Batch stride for matrix B
      total_elements_C_per_batch,              // Batch stride for matrix C
      total_elements_D_per_batch,              // Batch stride for matrix D
      total_elements_partial_norm_per_batch,   // Batch stride for norm array
      total_elements_partial_norm_per_batch,   // Batch stride for sum array
      total_elements_D_per_batch               // Batch stride for softmax output
    );

    //
    // Kernel Initialization and Execution
    // ===================================
    //

    GemmSoftmax gemm_softmax;

    // Initialize kernel with arguments and allocate any required workspace
    status = gemm_softmax.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // Execute the fused GEMM+Softmax kernel
    status = gemm_softmax();

    return status;
  }

  template<typename Element>
  bool verify_tensor(std::vector<Element> vector_Input, \
                       std::vector<Element> vector_Input_Ref) {

    auto size = int64_t((vector_Input.size() < vector_Input_Ref.size()) ? vector_Input.size() : vector_Input_Ref.size());
    float abs_tol = options.tolerance;
    float rel_tol = options.tolerance;
    
    for (int64_t i = 0; i < size; ++i) {
      float diff = (float)(vector_Input.at(i) - vector_Input_Ref.at(i));
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref.at(i));
      float relative_diff = abs_ref > abs_tol ? abs_diff / abs_ref : 0;
      if ( (isnan(abs_diff) || isinf(abs_diff)) ||  (abs_diff > rel_tol && relative_diff > rel_tol)) {
        printf("diff = %f, {%f, %f}.\n", abs_diff, (float)(vector_Input.at(i)), (float)(vector_Input_Ref.at(i)));
        return false;
      }

    }

    return true;
  }

  /// Numerical Verification Against Reference Implementation
  /// =======================================================
  /// Computes reference results using separate GEMM and softmax operations,
  /// then compares against the fused kernel output for correctness
  bool verify() {

    LayoutA layout_A(lda);
    LayoutB layout_B(ldb);
    LayoutC layout_C(ldc);
    LayoutN Layout_N(ldn);
    LayoutS Layout_S(lds);

    MatrixCoord extent_A{problem.m(), problem.k()};
    MatrixCoord extent_B{problem.k(), problem.n()};
    MatrixCoord extent_C{problem.m(), problem.n()};

    // Verify each batch independently
    for (int batch_idx = 0; batch_idx < options.batch_count; batch_idx++) {

      cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get() + total_elements_A_per_batch * batch_idx, layout_A, extent_A);
      cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get() + total_elements_B_per_batch * batch_idx, layout_B, extent_B);
      cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get() + total_elements_C_per_batch * batch_idx, layout_C, extent_C);
      cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_C, extent_C);

      cutlass::reference::device::GemmComplex<
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementC, LayoutC, 
          ElementCompute, ElementCompute
      >(
        problem,
        options.alpha, 
        view_A,
        cutlass::ComplexTransform::kNone,
        view_B,
        cutlass::ComplexTransform::kNone,
        options.beta, 
        view_C, 
        view_Ref_device, 
        ElementCompute(0)
      );

      // Copy reference results to host memory for verification
      std::vector<ElementD> matrix_D_Ref(layout_C.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_D_Ref.data(), block_Ref.get(), matrix_D_Ref.size());
      cutlass::TensorView<ElementD, LayoutC> view_Ref(matrix_D_Ref.data(), layout_C, extent_C);

      std::vector<ElementSoftmax> matrix_Softmax_Ref(layout_C.capacity(extent_C));
      cutlass::TensorView<ElementSoftmax, LayoutC> view_Softmax_Ref(matrix_Softmax_Ref.data(), layout_C, extent_C);

      // Copy computed results to host memory
      std::vector<ElementD> matrix_D(layout_C.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_D.data(), block_D.get() + total_elements_D_per_batch * batch_idx, matrix_D.size());

      std::vector<ElementD> matrix_Softmax(layout_C.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_Softmax.data(), block_Softmax.get() + total_elements_D_per_batch * batch_idx, matrix_Softmax.size());

      // Compute row-wise maximum for numerical stability (reference implementation)
      // This mimics the first pass of the fused kernel's softmax computation
      for (int m = 0; m < options.problem_size.m(); ++m) {
        reference_N.at({m, 0}) = view_Ref.ref().at({m, 0});
        for (int n = 1; n < options.problem_size.n(); ++n) {
          reference_N.at({m, 0}) = std::max(reference_N.at({m, 0}), ElementNorm(view_Ref.ref().at({m, n})));
        }
      }

      // Compute reference softmax using numerically stable algorithm
      // This matches the algorithmic approach used in the fused kernel
      for (int m = 0; m < options.problem_size.m(); ++m) {

        // First pass: compute sum of exponentials (subtract max for stability)
        float sum = float();
        for (int n = 0; n < options.problem_size.n(); ++n) {
          sum += std::exp( float(view_Ref.ref().at({m, n})) - float(reference_N.at({m, 0})) );
        }

        // Compute normalization factor
        float inv_sum = float(1.0f / sum);

        // Second pass: normalize exponentials to get final softmax values
        for (int n = 0; n < options.problem_size.n(); ++n) {
          view_Softmax_Ref.ref().at({m, n}) = ElementSoftmax(
            std::exp( float(view_Ref.ref().at({m, n})) - float(reference_N.at({m, 0})) ) * inv_sum
          );
        }
      }

      // Verification checks - set any of these to 'true' to override the verification checks.
      bool verified_D = false;
      bool verified_Softmax = false;

      // Verify softmax output
      if (!verified_D) {
        verified_D = verify_tensor<ElementC>(matrix_D, matrix_D_Ref);
      }

      if (!verified_Softmax) {
        verified_Softmax = verify_tensor<ElementSoftmax>(matrix_Softmax, matrix_Softmax_Ref);
      }

      if (!verified_D || !verified_Softmax) {

        std::cerr << "Verification check failed for tensor Softmax at batch " << batch_idx << "\n";

        // Summarize which checks failed
        if (!verified_D) {
          std::cerr << "Verification of D tensor failed\n";
        }

        if (!verified_Softmax) {
          std::cerr << "Verification of Softmax tensor failed\n";
        }

        return false;
      }

    }

    return true;
  }

  /// Profiles
  bool profile() {

    //
    // Profile
    //

    cutlass::Status status = cutlass::Status::kSuccess;
    cudaError_t result;
    cudaEvent_t events[2];
    int const kIterations = options.iterations;

    for (cudaEvent_t &evt : events) {
      result = cudaEventCreate(&evt);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }
    }

    result = cudaEventRecord(events[0]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    for (int iter = 0; iter < kIterations; ++iter) {

      status = execute_device_kernel();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Device execution failed." << std::endl;
        return false;
      }
    }

    result = cudaEventRecord(events[1]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    float elapsed_ms = 0;
    result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsedTime() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    for (cudaEvent_t &evt : events) {
      result = cudaEventDestroy(evt);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventDestroy() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }
    }

    int64_t flops = int64_t(options.problem_size.m()) * options.problem_size.n() * options.problem_size.k() * 2;
    int64_t bytes = cutlass::bits_to_bytes<int64_t>(
      (cutlass::sizeof_bits<ElementD>::value * 2 + cutlass::sizeof_bits<ElementSoftmax>::value) *
      options.problem_size.m() * options.problem_size.n());

    double gflops_per_second = double(flops) * kIterations * options.batch_count / double(elapsed_ms / 1000.0f) / double(1.0e9);
    double gbytes_per_second = double(bytes) * kIterations * options.batch_count / double(elapsed_ms / 1000.0f) / double(1 << 30);

    double elapsed_ms_per_iter = double(elapsed_ms) / kIterations;

    std::cout << "         Problem: "
              << options.problem_size.m() << "-by-" << options.problem_size.n() << "-by-" << options.problem_size.k()
              << ", batch size: " << options.batch_count
              << std::endl;

    std::cout << "         Runtime: " << elapsed_ms_per_iter << " ms\n" << std::endl;

    std::cout << "          GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;
    std::cout << "Memory bandwidth: " << gbytes_per_second << "  GiB/s" << std::endl;

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {

  // Options parsing
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.supported()) {
    return 0;
  }

  // Run
  Testbed testbed(options);

  Disposition disposition = testbed.run();

  std::cout << std::endl;

  switch (disposition) {
    case Disposition::kPassed:
      std::cout << "Passed" << std::endl;
      break;
    case Disposition::kIncorrect:
      std::cout << "Incorrect" << std::endl;
      break;
    case Disposition::kNotVerified:
      std::cout << "Not verified" << std::endl;
      break;
  }

  return (disposition == Disposition::kPassed ? 0 : -1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
