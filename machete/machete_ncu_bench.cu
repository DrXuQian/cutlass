/***************************************************************************************************
 * NCU-friendly Machete Benchmark
 *
 * A simplified benchmark for NCU profiling - runs kernel exactly once without iterations.
 * Use with: ncu --set full ./machete_ncu_bench --m=1 --n=2048 --k=2048 --g=128 --mode=1
 *
 * Compare with vLLM Python benchmark using same problem size.
 **************************************************************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>

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
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "helper.h"
#include "mixed_dtype_utils.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (same as machete_gemm_example.cu)
/////////////////////////////////////////////////////////////////////////////////////////////////
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::int4b_t;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

using ElementA    = MmaType;
using LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB    = QuantType;
using LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>;
int constexpr NumShuffleAtoms = 1;
using MmaAtomShape = Layout<Shape<_1,Int<NumShuffleAtoms>>>;
using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,int>, StrideB>{}));

using ElementScale = MmaType;
using ElementZero = ElementScale;
using LayoutScale = cutlass::layout::RowMajor;

using ElementC    = cutlass::half_t;
using LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD    = ElementC;
using LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator  = float;
using ElementCompute      = float;
using ArchTag             = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
using TileShape           = Shape<_128,_128,cute::Int<TileShapeK>>;
using ClusterShape        = Shape<_1,_1,_1>;
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

// Scale-only with shuffle (most common for W4A16)
using CollectiveMainloopScaleOnlyShuffled = typename cutlass::gemm::collective::CollectiveBuilder<
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

using GemmKernelScaleOnlyShuffled = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloopScaleOnlyShuffled,
    CollectiveEpilogue
>;

using GemmScaleOnlyShuffled = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnlyShuffled>;

using StrideC = typename GemmKernelScaleOnlyShuffled::StrideC;
using StrideD = typename GemmKernelScaleOnlyShuffled::StrideD;
using StrideS = typename CollectiveMainloopScaleOnlyShuffled::StrideScale;

// Global data
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
StrideS stride_S;
LayoutB_Reordered layout_B_reordered;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementD> block_D;

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Command line options
/////////////////////////////////////////////////////////////////////////////////////////////////
struct Options {
  int m = 1;
  int n = 2048;
  int k = 2048;
  int g = 128;  // group size
  int mode = 1; // 1 = scale only
  float alpha = 1.0f;
  float beta = 0.0f;
  bool help = false;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("help", help);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
  }

  void print_usage(std::ostream &out) const {
    out << "machete_ncu_bench - NCU-friendly W4A16 GEMM benchmark\n\n"
        << "Options:\n"
        << "  --help         Show this help message\n"
        << "  --m=<int>      M dimension (default: 1)\n"
        << "  --n=<int>      N dimension (default: 2048)\n"
        << "  --k=<int>      K dimension (default: 2048)\n"
        << "  --g=<int>      Group size for scales (default: 128)\n"
        << "  --mode=<int>   0=convert, 1=scale, 2=scale+zero (default: 1)\n"
        << "  --alpha=<f32>  Epilogue alpha (default: 1.0)\n"
        << "  --beta=<f32>   Epilogue beta (default: 0.0)\n\n"
        << "Example NCU usage:\n"
        << "  ncu --set full ./machete_ncu_bench --m=1 --n=2048 --k=2048 --g=128\n";
  }
};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

void initialize(Options const& options) {
  uint64_t seed = 2024;

  auto shape_B = cute::make_shape(options.n, options.k, 1);
  int const scale_k = cutlass::ceil_div(options.k, options.g);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, 1));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.n, options.m, 1));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.n, options.m, 1));
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(options.n, scale_k, 1));

  auto layout_B = make_layout(shape_B, stride_B);

  // Allocate memory
  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_scale.reset(scale_k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);

  // Initialize with random data
  initialize_tensor(block_A, seed + 1);
  initialize_tensor(block_B, seed + 2);

  // Initialize scales to 1.0 for simplicity
  std::vector<ElementScale> host_scale(scale_k * options.n);
  for (auto& s : host_scale) s = ElementScale(1.0f);
  cudaMemcpy(block_scale.get(), host_scale.data(), host_scale.size() * sizeof(ElementScale), cudaMemcpyHostToDevice);

  // Shuffle B for efficient loads
  layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
  cutlass::reorder_tensor(block_B.get(), layout_B, layout_B_reordered);
}

int run_gemm(Options &options) {
  initialize(options);

  using Gemm = GemmScaleOnlyShuffled;
  Gemm gemm;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.n, options.m, options.k, 1},
    {block_B.get(), layout_B_reordered, block_A.get(), stride_A, block_scale.get(), stride_S, options.g},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Print problem configuration
  std::cout << "=== Machete NCU Benchmark ===" << std::endl;
  std::cout << "Problem: M=" << options.m << ", N=" << options.n
            << ", K=" << options.k << ", G=" << options.g << std::endl;
  std::cout << "Running single kernel launch for NCU profiling..." << std::endl;

  // Synchronize before kernel launch
  cudaDeviceSynchronize();

  // Single kernel launch for NCU profiling
  CUTLASS_CHECK(gemm.run());

  // Synchronize after kernel
  cudaDeviceSynchronize();

  std::cout << "Kernel completed successfully." << std::endl;
  return 0;
}

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

int main(int argc, char const **args) {
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 9 || props.minor != 0) {
    std::cerr << "This example requires NVIDIA Hopper (SM90).\n";
    return 0;
  }

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  return run_gemm(options);
#else
  std::cerr << "CUTLASS_ARCH_MMA_SM90_SUPPORTED not defined.\n";
  return 0;
#endif
}
