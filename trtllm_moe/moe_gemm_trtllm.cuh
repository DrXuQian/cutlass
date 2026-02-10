/*
 * Standalone TRT-LLM MoE Grouped GEMM Kernel for Hopper (SM90)
 *
 * This is extracted from TensorRT-LLM's moe_gemm_tma_ws_launcher.inl
 * with dependencies removed for standalone benchmarking.
 *
 * Original TRT-LLM code: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <array>

// CUTLASS 3.x headers
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================================
// Error handling macros (replacing TRT-LLM's TLLM_CHECK/TLLM_THROW)
// ============================================================================
#define MOE_CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "MOE_CHECK failed: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while(0)

#define MOE_CHECK_WITH_INFO(cond, ...) \
    do { \
        if (!(cond)) { \
            char buf[512]; \
            snprintf(buf, sizeof(buf), __VA_ARGS__); \
            std::cerr << "MOE_CHECK failed: " << buf << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while(0)

#define MOE_THROW(msg) \
    do { \
        std::cerr << "MOE Error: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    } while(0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::abort(); \
        } \
    } while(0)

namespace moe_gemm {

// ============================================================================
// TmaWarpSpecializedGroupedGemmInput - Core input structure from TRT-LLM
// ============================================================================
struct TmaWarpSpecializedGroupedGemmInput {
    template <class Tag>
    using TransposeLayoutTag = std::conditional_t<std::is_same_v<Tag, cutlass::layout::RowMajor>,
        cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

    // These are always the layout of A & B matrices
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using LayoutC_T = TransposeLayoutTag<LayoutC>;
    using LayoutD_T = TransposeLayoutTag<LayoutD>;

    using StrideA = std::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
    using StrideB = std::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
    using StrideC = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutC*>>;
    using StrideD = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutD*>>;
    using StrideC_T = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutC_T*>>;
    using StrideD_T = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutD_T*>>;

    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int64_t, int64_t, int64_t>>;

    bool swap_ab = false;
    ProblemShape shape_info{};
    void* stride_act = nullptr;
    void* stride_weight = nullptr;

    void const** ptr_act = nullptr;
    void const** ptr_weight = nullptr;

    void* stride_c = nullptr;
    void const** ptr_c = nullptr;

    void* stride_d = nullptr;
    void** ptr_d = nullptr;

    float const** alpha_scale_ptr_array = nullptr;

    uint8_t* gemm_workspace = nullptr;
    size_t gemm_workspace_size = 0;

    bool isValid() const {
        return stride_act != nullptr && ptr_act != nullptr;
    }
};

// ============================================================================
// Type conversion helper (simplified from TRT-LLM)
// ============================================================================
template <typename T>
struct TllmToCutlassTypeAdapter {
    using type = T;
};

template <>
struct TllmToCutlassTypeAdapter<half> {
    using type = cutlass::half_t;
};

#ifdef ENABLE_BF16
template <>
struct TllmToCutlassTypeAdapter<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};
#endif

// ============================================================================
// MoE Grouped GEMM Kernel Definition - From TRT-LLM
// ============================================================================
template <typename ArchTag, typename T, typename WeightType, typename OutputType,
          typename MmaTileShape, typename ClusterShape, bool SwapAB = false>
struct MoeGroupedGemmKernel {
    static constexpr bool IsHopper = ArchTag::kMinComputeCapability == 90;
    static_assert(IsHopper, "This kernel is optimized for Hopper (SM90)");

    using ElementType = typename TllmToCutlassTypeAdapter<T>::type;
    using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;
    using ElementD = typename TllmToCutlassTypeAdapter<OutputType>::type;

    using ElementAct = ElementType;
    using ElementWeight = CutlassWeightType;
    using ElementC = void;  // No bias
    using ElementCSafe = ElementD;
    using ElementAccumulator = float;

    // Alignment settings
    static constexpr int AlignmentAct = 128 / cutlass::sizeof_bits<ElementAct>::value;
    static constexpr int AlignmentWeight = 128 / cutlass::sizeof_bits<ElementWeight>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementType>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Layouts
    using LayoutA = TmaWarpSpecializedGroupedGemmInput::LayoutA;
    using LayoutB = TmaWarpSpecializedGroupedGemmInput::LayoutB;
    using LayoutC = std::conditional_t<SwapAB, TmaWarpSpecializedGroupedGemmInput::LayoutC_T,
                                       TmaWarpSpecializedGroupedGemmInput::LayoutC>;
    using LayoutD = std::conditional_t<SwapAB, TmaWarpSpecializedGroupedGemmInput::LayoutD_T,
                                       TmaWarpSpecializedGroupedGemmInput::LayoutD>;
    using StrideA = typename TmaWarpSpecializedGroupedGemmInput::StrideA;
    using StrideB = typename TmaWarpSpecializedGroupedGemmInput::StrideB;
    using StrideC = std::conditional_t<SwapAB, TmaWarpSpecializedGroupedGemmInput::StrideC_T,
                                       TmaWarpSpecializedGroupedGemmInput::StrideC>;
    using StrideD = std::conditional_t<SwapAB, TmaWarpSpecializedGroupedGemmInput::StrideD_T,
                                       TmaWarpSpecializedGroupedGemmInput::StrideD>;

    // Swapped element types for mainloop
    using SwappedMainloopElementA = std::conditional_t<SwapAB, ElementWeight, ElementAct>;
    using SwappedMainloopElementB = std::conditional_t<SwapAB, ElementAct, ElementWeight>;
    static constexpr auto SwappedAlignmentA = SwapAB ? AlignmentWeight : AlignmentAct;
    static constexpr auto SwappedAlignmentB = SwapAB ? AlignmentAct : AlignmentWeight;

    // Epilogue schedule for SM90
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

    // Kernel schedule for SM90
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;

    // Epilogue operation
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

    // Collective Epilogue
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD,
        EpilogueSchedule
    >::CollectiveOp;

    // Stage count with epilogue carveout
    using StageCountAutoCarveout = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

    // Collective Mainloop
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        SwappedMainloopElementA, LayoutA*, SwappedAlignmentA,
        SwappedMainloopElementB, LayoutB*, SwappedAlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        StageCountAutoCarveout, KernelSchedule
    >::CollectiveOp;

    // GEMM Kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        TmaWarpSpecializedGroupedGemmInput::ProblemShape,
        CollectiveMainloop, CollectiveEpilogue, void, void>;

    // Device adapter
    using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ============================================================================
// MoE Grouped GEMM Launcher - machete-style simple interface
// ============================================================================
template <typename T, typename WeightType = T, typename OutputType = T,
          typename TileShape = Shape<_128, _128, _64>,
          typename ClusterShape = Shape<_1, _2, _1>,
          bool SwapAB = false>
class MoeGroupedGemmLauncher {
public:
    using ArchTag = cutlass::arch::Sm90;
    using KernelDef = MoeGroupedGemmKernel<ArchTag, T, WeightType, OutputType, TileShape, ClusterShape, SwapAB>;
    using GemmGrouped = typename KernelDef::GemmGrouped;
    using GemmKernel = typename KernelDef::GemmKernel;
    using StrideA = typename TmaWarpSpecializedGroupedGemmInput::StrideA;
    using StrideB = typename TmaWarpSpecializedGroupedGemmInput::StrideB;
    using StrideD = typename KernelDef::StrideD;
    using ElementD = typename KernelDef::ElementD;
    using ElementAct = typename KernelDef::ElementAct;
    using ElementWeight = typename KernelDef::ElementWeight;
    using ElementAccumulator = typename KernelDef::ElementAccumulator;

    MoeGroupedGemmLauncher(int multi_processor_count = 0)
        : multi_processor_count_(multi_processor_count) {
        if (multi_processor_count_ == 0) {
            int device_id = 0;
            cudaDeviceProp props;
            CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
            multi_processor_count_ = props.multiProcessorCount;
        }
    }

    // Get workspace size
    size_t get_workspace_size(int num_experts) {
        GemmGrouped gemm;

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = multi_processor_count_;

        typename TmaWarpSpecializedGroupedGemmInput::ProblemShape shape_info{num_experts, nullptr, nullptr};
        typename GemmKernel::TileScheduler::Arguments scheduler_args{
            1, GemmKernel::TileScheduler::RasterOrderOptions::AlongN};

        const typename GemmGrouped::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped, shape_info, {}, {}, hw_info, scheduler_args};

        return gemm.get_workspace_size(args);
    }

    // Run grouped GEMM
    void run(TmaWarpSpecializedGroupedGemmInput& input,
             int num_experts,
             cudaStream_t stream = 0) {

        MOE_CHECK(input.isValid());
        MOE_CHECK_WITH_INFO(SwapAB == input.swap_ab, "SwapAB must match runtime swap_ab");

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = multi_processor_count_;

        GemmGrouped gemm;

        using MainloopArguments = typename KernelDef::CollectiveMainloop::Arguments;

        MainloopArguments mainloop_args;
        if constexpr (SwapAB) {
            mainloop_args = MainloopArguments{
                reinterpret_cast<ElementWeight const**>(input.ptr_weight),
                reinterpret_cast<StrideA*>(input.stride_weight),
                reinterpret_cast<ElementAct const**>(input.ptr_act),
                reinterpret_cast<StrideB*>(input.stride_act)
            };
        } else {
            mainloop_args = MainloopArguments{
                reinterpret_cast<ElementAct const**>(input.ptr_act),
                reinterpret_cast<StrideA*>(input.stride_act),
                reinterpret_cast<ElementWeight const**>(input.ptr_weight),
                reinterpret_cast<StrideB*>(input.stride_weight)
            };
        }

        using EpilogueArguments = typename KernelDef::CollectiveEpilogue::Arguments;
        using EpilogueScalars = decltype(EpilogueArguments{}.thread);

        // TRT-LLM original logic for epilogue scalars
        // Check if EpilogueScalars supports simple (alpha, beta) construction
        constexpr bool IsSimpleAlphaBeta = std::is_constructible_v<EpilogueScalars, ElementAccumulator, ElementAccumulator>;

        EpilogueScalars epilogue_scalars = [&]() {
            if constexpr (IsSimpleAlphaBeta) {
                if (input.alpha_scale_ptr_array) {
                    return EpilogueScalars{input.alpha_scale_ptr_array};
                } else {
                    return EpilogueScalars{
                        ElementAccumulator(1.f),
                        input.ptr_c ? ElementAccumulator(1.f) : ElementAccumulator(0.f)
                    };
                }
            } else {
                // For non-simple epilogue, use the more complex construction
                return EpilogueScalars{
                    ElementAccumulator(1.f),
                    input.ptr_c ? ElementAccumulator(1.f) : ElementAccumulator(0.f),
                    nullptr, nullptr,
                    input.alpha_scale_ptr_array, nullptr,
                    cute::Shape<_0, _0, int64_t>{cute::_0{}, cute::_0{}, (input.alpha_scale_ptr_array != nullptr) ? 1 : 0},
                    cute::Shape<_0, _0, int64_t>{cute::_0{}, cute::_0{}, 0}
                };
            }
        }();

        EpilogueArguments epilogue_args{
            epilogue_scalars,
            nullptr, nullptr,
            reinterpret_cast<ElementD**>(input.ptr_d),
            reinterpret_cast<StrideD*>(input.stride_d)
        };

        typename GemmKernel::TileScheduler::Arguments scheduler_args{
            1, GemmKernel::TileScheduler::RasterOrderOptions::AlongN};

        const typename GemmGrouped::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            input.shape_info, mainloop_args, epilogue_args, hw_info, scheduler_args};

        size_t calculated_ws_size = gemm.get_workspace_size(args);
        MOE_CHECK_WITH_INFO(calculated_ws_size <= input.gemm_workspace_size,
            "Workspace is size %zu but only %zu were allocated",
            calculated_ws_size, input.gemm_workspace_size);

        auto can_implement = gemm.can_implement(args);
        MOE_CHECK_WITH_INFO(can_implement == cutlass::Status::kSuccess,
            "Grouped GEMM kernel will fail for params. Error: %s",
            cutlass::cutlassGetStatusString(can_implement));

        auto init_status = gemm.initialize(args, input.gemm_workspace);
        MOE_CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
            "Failed to initialize cutlass TMA WS grouped gemm. Error: %s",
            cutlass::cutlassGetStatusString(init_status));

        auto run_status = gemm.run(stream);
        MOE_CHECK_WITH_INFO(run_status == cutlass::Status::kSuccess,
            "Failed to run cutlass TMA WS grouped gemm. Error: %s",
            cutlass::cutlassGetStatusString(run_status));
    }

private:
    int multi_processor_count_;
};

// ============================================================================
// Helper to create strides for experts
// ============================================================================
template <typename StrideType>
void fill_strides(StrideType* strides, int num_experts, int leading_dim) {
    for (int i = 0; i < num_experts; i++) {
        strides[i] = cutlass::make_cute_packed_stride(StrideType{}, cute::make_shape(1, leading_dim, 1));
    }
}

// ============================================================================
// Simplified MoE GEMM runner - machete-style interface
// ============================================================================
template <typename T>
class SimpleMoeGemmRunner {
public:
    using StrideA = typename TmaWarpSpecializedGroupedGemmInput::StrideA;
    using StrideB = typename TmaWarpSpecializedGroupedGemmInput::StrideB;
    using StrideD = typename TmaWarpSpecializedGroupedGemmInput::StrideD;
    using ProblemShape = typename TmaWarpSpecializedGroupedGemmInput::ProblemShape;

    SimpleMoeGemmRunner() {
        int device_id = 0;
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
        multi_processor_count_ = props.multiProcessorCount;
    }

    // Get workspace size needed
    size_t get_workspace_size(int num_experts) {
        using Launcher = MoeGroupedGemmLauncher<T>;
        Launcher launcher(multi_processor_count_);
        return launcher.get_workspace_size(num_experts);
    }

    // Run MoE grouped GEMM
    // A: [num_experts, M, K] - activations (each expert has M tokens, K hidden)
    // B: [num_experts, K, N] - weights (stored column-major, so K x N per expert)
    // C: [num_experts, M, N] - outputs
    void run(
        int num_experts,
        int64_t M,  // tokens per expert
        int64_t N,  // output dimension
        int64_t K,  // hidden dimension
        T const** ptr_A,  // activation pointers [num_experts]
        T const** ptr_B,  // weight pointers [num_experts]
        T** ptr_C,        // output pointers [num_experts]
        void* workspace,
        size_t workspace_size,
        cudaStream_t stream = 0
    ) {
        // Allocate strides on device
        StrideA* d_stride_a;
        StrideB* d_stride_b;
        StrideD* d_stride_d;

        CUDA_CHECK(cudaMalloc(&d_stride_a, num_experts * sizeof(StrideA)));
        CUDA_CHECK(cudaMalloc(&d_stride_b, num_experts * sizeof(StrideB)));
        CUDA_CHECK(cudaMalloc(&d_stride_d, num_experts * sizeof(StrideD)));

        // Fill strides on host then copy
        std::vector<StrideA> h_stride_a(num_experts);
        std::vector<StrideB> h_stride_b(num_experts);
        std::vector<StrideD> h_stride_d(num_experts);

        for (int i = 0; i < num_experts; i++) {
            // A is RowMajor: stride = (K, 1, 0) for shape (M, K)
            h_stride_a[i] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(int(M), int(K), int(1)));
            // B is ColumnMajor: stride = (1, K, 0) for shape (K, N)
            h_stride_b[i] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(int(K), int(N), int(1)));
            // D is RowMajor: stride = (N, 1, 0) for shape (M, N)
            h_stride_d[i] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(int(M), int(N), int(1)));
        }

        CUDA_CHECK(cudaMemcpy(d_stride_a, h_stride_a.data(), num_experts * sizeof(StrideA), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_stride_b, h_stride_b.data(), num_experts * sizeof(StrideB), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_stride_d, h_stride_d.data(), num_experts * sizeof(StrideD), cudaMemcpyHostToDevice));

        // Create problem shapes
        std::vector<cute::Shape<int64_t, int64_t, int64_t>> h_problem_sizes(num_experts);
        for (int i = 0; i < num_experts; i++) {
            h_problem_sizes[i] = cute::make_shape(M, N, K);
        }

        cute::Shape<int64_t, int64_t, int64_t>* d_problem_sizes;
        CUDA_CHECK(cudaMalloc(&d_problem_sizes, num_experts * sizeof(cute::Shape<int64_t, int64_t, int64_t>)));
        CUDA_CHECK(cudaMemcpy(d_problem_sizes, h_problem_sizes.data(),
                   num_experts * sizeof(cute::Shape<int64_t, int64_t, int64_t>), cudaMemcpyHostToDevice));

        // Setup input structure
        TmaWarpSpecializedGroupedGemmInput input;
        input.swap_ab = false;
        input.shape_info = ProblemShape{num_experts, d_problem_sizes, nullptr};
        input.stride_act = d_stride_a;
        input.stride_weight = d_stride_b;
        input.ptr_act = reinterpret_cast<void const**>(ptr_A);
        input.ptr_weight = reinterpret_cast<void const**>(ptr_B);
        input.stride_d = d_stride_d;
        input.ptr_d = reinterpret_cast<void**>(ptr_C);
        input.ptr_c = nullptr;
        input.stride_c = nullptr;
        input.alpha_scale_ptr_array = nullptr;
        input.gemm_workspace = reinterpret_cast<uint8_t*>(workspace);
        input.gemm_workspace_size = workspace_size;

        // Run kernel
        using Launcher = MoeGroupedGemmLauncher<T>;
        Launcher launcher(multi_processor_count_);
        launcher.run(input, num_experts, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Cleanup temporary allocations
        CUDA_CHECK(cudaFree(d_stride_a));
        CUDA_CHECK(cudaFree(d_stride_b));
        CUDA_CHECK(cudaFree(d_stride_d));
        CUDA_CHECK(cudaFree(d_problem_sizes));
    }

private:
    int multi_processor_count_;
};

} // namespace moe_gemm
