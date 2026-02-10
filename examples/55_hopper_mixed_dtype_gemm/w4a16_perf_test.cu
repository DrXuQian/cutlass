/*
 * W4A16 Performance Test for Hopper GPU
 *
 * Tests INT4 weights with BF16 activations (W4A16) GEMM performance.
 * Typical LLM inference scenario: M=1 (single token), N=2048, K=2048
 *
 * Based on CUTLASS 55_hopper_int4_bf16_gemm example.
 *
 * Usage:
 *   ./w4a16_perf_test --m=1 --n=2048 --k=2048 --g=128 --mode=1
 *   ./w4a16_perf_test --m=1 --n=2048 --k=2048 --g=2048 --mode=1  # per-channel scale
 *   ./w4a16_perf_test --sweep  # Run multiple sizes
 */

#include <iostream>
#include <vector>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations for W4A16
/////////////////////////////////////////////////////////////////////////////////////////////////

// Wide type (activation): BF16
using MmaType = cutlass::bfloat16_t;
// Narrow type (weight): INT4
using QuantType = cutlass::int4b_t;

constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A matrix: Activation (BF16, RowMajor)
using ElementA    = MmaType;
using LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix: Weight (INT4, ColumnMajor for better TMA performance)
using ElementB    = QuantType;
using LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Transposed layouts for the swap trick
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

// Value shuffle for efficient INT4->BF16 conversion: [0,2,4,6,1,3,5,7]
using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>;
int constexpr NumShuffleAtoms = 1;
using MmaAtomShape = Layout<Shape<_1,Int<NumShuffleAtoms>>>;
using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,int>, StrideB>{}));

// Scale type (same as activation type)
using ElementScale = MmaType;
using LayoutScale = cutlass::layout::RowMajor;

// Output: FP16
using ElementC    = cutlass::half_t;
using LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD    = ElementC;
using LayoutD     = LayoutC;
constexpr int AlignmentD = AlignmentC;

// Accumulator and compute types
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

// Tile configuration - smaller tiles for better occupancy with mixed dtypes
using TileShape    = Shape<_128, _128, cute::Int<TileShapeK>>;
using ClusterShape = Shape<_1, _1, _1>;

using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// Epilogue (with swap + transpose)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

// Mainloop with scale (shuffled for better performance)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScale>, LayoutB_Reordered, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using StrideS = typename CollectiveMainloop::StrideScale;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Test runner
/////////////////////////////////////////////////////////////////////////////////////////////////

struct PerfResult {
  int m, n, k, g;
  double avg_time_ms;
  double gflops;
  double memory_gb_s;  // Memory bandwidth (GB/s)
  bool passed;
};

class W4A16PerfTest {
public:
  // Strides
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideS stride_S;
  LayoutB_Reordered layout_B_reordered;

  // Device memory
  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementScale> block_scale;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementD> block_D;

  void initialize(int m, int n, int k, int g, int l = 1) {
    auto shape_B = cute::make_shape(n, k, l);
    int scale_k = cutlass::ceil_div(k, g);

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, l));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, l));
    stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, l));

    // Allocate memory
    block_A.reset(m * k * l);
    block_B.reset(n * k * l);
    block_scale.reset(n * scale_k * l);
    block_C.reset(m * n * l);
    block_D.reset(m * n * l);

    // Initialize with random data
    cutlass::reference::device::BlockFillRandomUniform(
        block_A.get(), block_A.size(), 2023, ElementA(2), ElementA(-2));
    cutlass::reference::device::BlockFillRandomUniform(
        block_B.get(), block_B.size(), 2024, ElementB(7), ElementB(-8));
    cutlass::reference::device::BlockFillRandomUniform(
        block_scale.get(), block_scale.size(), 2025, ElementScale(2), ElementScale(0.1f));
    cutlass::reference::device::BlockFillRandomUniform(
        block_C.get(), block_C.size(), 2026, ElementC(1), ElementC(-1));

    // Apply offline reordering for better performance
    auto layout_B = make_layout(shape_B, stride_B);
    layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(block_B.get(), layout_B, layout_B_reordered);
  }

  PerfResult run(int m, int n, int k, int g, int warmup = 10, int iterations = 100) {
    initialize(m, n, k, g);

    PerfResult result;
    result.m = m;
    result.n = n;
    result.k = k;
    result.g = g;

    // Create GEMM arguments
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, 1},  // Problem shape (swapped)
      {block_B.get(), layout_B_reordered, block_A.get(), stride_A,
       block_scale.get(), stride_S, g},
      {{1.0f, 0.0f}, block_C.get(), stride_C, block_D.get(), stride_D}
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "GEMM cannot implement for size " << m << "x" << n << "x" << k << std::endl;
      result.passed = false;
      return result;
    }

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
      CUTLASS_CHECK(gemm.run());
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
      CUTLASS_CHECK(gemm.run());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.avg_time_ms = total_ms / iterations;

    // Compute GFLOPS (2 ops per multiply-add)
    double flops = 2.0 * m * n * k;
    result.gflops = flops / (result.avg_time_ms * 1e6);  // GFLOPS

    // Compute memory bandwidth
    // A: m*k elements * 2 bytes (BF16)
    // B: n*k elements * 0.5 bytes (INT4)
    // Scale: n*(k/g) elements * 2 bytes
    // D: m*n elements * 2 bytes (FP16)
    double bytes_read = m * k * 2.0 + n * k * 0.5 + n * (k/g) * 2.0;
    double bytes_written = m * n * 2.0;
    double total_bytes = bytes_read + bytes_written;
    result.memory_gb_s = total_bytes / (result.avg_time_ms * 1e6);  // GB/s

    result.passed = true;
    return result;
  }
};

void print_header() {
  std::cout << std::setw(6) << "M"
            << std::setw(8) << "N"
            << std::setw(8) << "K"
            << std::setw(8) << "G"
            << std::setw(12) << "Time(ms)"
            << std::setw(12) << "GFLOPS"
            << std::setw(14) << "BW(GB/s)"
            << std::setw(10) << "Status"
            << std::endl;
  std::cout << std::string(78, '-') << std::endl;
}

void print_result(const PerfResult& r) {
  std::cout << std::setw(6) << r.m
            << std::setw(8) << r.n
            << std::setw(8) << r.k
            << std::setw(8) << r.g
            << std::setw(12) << std::fixed << std::setprecision(4) << r.avg_time_ms
            << std::setw(12) << std::fixed << std::setprecision(2) << r.gflops
            << std::setw(14) << std::fixed << std::setprecision(2) << r.memory_gb_s
            << std::setw(10) << (r.passed ? "PASS" : "FAIL")
            << std::endl;
}

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

  if (props.major != 9 || props.minor != 0) {
    std::cerr << "This example requires Hopper GPU (SM90).\n";
    return 0;
  }

  std::cout << "Device: " << props.name << std::endl;
  std::cout << "W4A16 (INT4 Weight, BF16 Activation) GEMM Performance Test\n" << std::endl;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  cutlass::CommandLine cmd(argc, args);

  bool help = cmd.check_cmd_line_flag("help");

  if (help) {
    std::cout << "Usage: ./w4a16_perf_test [options]\n"
              << "Options:\n"
              << "  --m=<int>        M dimension (default: 1)\n"
              << "  --n=<int>        N dimension (default: 2048)\n"
              << "  --k=<int>        K dimension (default: 2048)\n"
              << "  --g=<int>        Group size for scales (default: 128)\n"
              << "  --warmup=<int>   Warmup iterations (default: 10)\n"
              << "  --iterations=<int> Timing iterations (default: 100)\n"
              << "  --sweep          Run sweep over multiple sizes\n"
              << std::endl;
    return 0;
  }

  bool sweep = cmd.check_cmd_line_flag("sweep");

  W4A16PerfTest tester;

  if (sweep) {
    // Sweep over typical LLM inference sizes
    std::vector<std::tuple<int, int, int, int>> sizes = {
      // M, N, K, G (group size)
      // Single token decode (M=1)
      {1, 2048, 2048, 128},
      {1, 2048, 2048, 2048},  // per-channel
      {1, 4096, 4096, 128},
      {1, 4096, 4096, 4096},
      {1, 8192, 8192, 128},
      {1, 11008, 4096, 128},  // LLaMA-7B FFN
      {1, 4096, 11008, 128},  // LLaMA-7B FFN

      // Small batch decode
      {4, 2048, 2048, 128},
      {8, 4096, 4096, 128},
      {16, 4096, 4096, 128},
      {32, 4096, 4096, 128},

      // Prefill (larger M)
      {128, 4096, 4096, 128},
      {256, 4096, 4096, 128},
      {512, 4096, 4096, 128},
      {1024, 4096, 4096, 128},
    };

    print_header();
    for (auto& [m, n, k, g] : sizes) {
      auto result = tester.run(m, n, k, g);
      print_result(result);
    }
  } else {
    // Single test
    int m = 1, n = 2048, k = 2048, g = 128;
    int warmup = 10, iterations = 100;

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);

    std::cout << "Testing M=" << m << " N=" << n << " K=" << k << " G=" << g << std::endl;
    std::cout << "Warmup: " << warmup << ", Iterations: " << iterations << std::endl;
    std::cout << std::endl;

    print_header();
    auto result = tester.run(m, n, k, g, warmup, iterations);
    print_result(result);

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Average Time: " << result.avg_time_ms << " ms" << std::endl;
    std::cout << "  Performance:  " << result.gflops << " GFLOPS" << std::endl;
    std::cout << "  Memory BW:    " << result.memory_gb_s << " GB/s" << std::endl;
  }

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

  return 0;
}
