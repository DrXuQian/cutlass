// CuTe Hopper GEMM with Warp Specialization (SM90)
// Based on cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
//
// This example demonstrates:
// 1. Warp specialization: Producer warp (TMA) + Consumer warps (MMA)
// 2. TMA (Tensor Memory Accelerator) for async global->shared loads
// 3. WGMMA (Warp Group Matrix Multiply Accumulate) for Hopper tensor cores
// 4. Cooperative kernel with dedicated producer/consumer roles
//
// Key difference from non-cooperative version:
// - In non-cooperative: all warps do both TMA and MMA (interleaved)
// - In cooperative (this file): warp 0 is dedicated producer (TMA only),
//   remaining warps are dedicated consumers (MMA only)
//
// This improves performance by:
// - Eliminating synchronization overhead within warps
// - Better instruction-level parallelism
// - More efficient use of TMA and tensor core units

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

using namespace cute;

// ============================================================================
// Shared memory storage with pipeline barriers
// ============================================================================
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB, int NumStages>
struct SharedStorageWarpSpecialized {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

    // Pipeline barriers
    alignas(8) uint64_t tma_barrier[NumStages];   // Producer (TMA) signals completion
    alignas(8) uint64_t mma_barrier[NumStages];   // Consumer (MMA) signals consumption done
};

// ============================================================================
// Warp-Specialized GEMM Kernel
// ============================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta,
          int NumStages>
__global__ static
__launch_bounds__(256)  // 2 warpgroups: 1 producer + 1 consumer (each 128 threads)
void gemm_warp_specialized_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    TC* C, CStride dC, TiledMma mma,
    Alpha alpha, Beta beta)
{
    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    // ========================================================================
    // Setup tensors
    // ========================================================================
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    // CTA tile
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    // Shared memory
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageWarpSpecialized<TA, TB, SmemLayoutA, SmemLayoutB, NumStages>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

    // ========================================================================
    // TMA partition
    // ========================================================================
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sB), group_modes<0, 2>(gB));

    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                        + sizeof(make_tensor_like(tensor<0>(tBsB)));

    // ========================================================================
    // Warp specialization: determine role
    // ========================================================================
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    int lane_predicate = cute::elect_one_sync();

    // Warp group 0 = Producer (TMA), Warp group 1 = Consumer (MMA)
    // Each warp group has 4 warps (128 threads)
    constexpr int ProducerWarpGroup = 0;
    constexpr int ConsumerWarpGroup = 1;

    bool is_producer = (warp_group_idx == ProducerWarpGroup);
    bool is_consumer = (warp_group_idx == ConsumerWarpGroup);

    // ========================================================================
    // Initialize barriers
    // ========================================================================
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    // Only one thread initializes barriers
    if (threadIdx.x == 0) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < NumStages; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);  // Consumer warp group size
        }
    }
    // Ensure barrier init is complete
    cluster_sync();

    // ========================================================================
    // Pipeline state
    // ========================================================================
    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    auto write_state = cutlass::PipelineState<NumStages>();
    auto read_state = cutlass::PipelineState<NumStages>();

    // ========================================================================
    // PRODUCER WARP GROUP: TMA loads only
    // ========================================================================
    if (is_producer) {
        // Prologue: fill pipeline
        CUTE_UNROLL
        for (int pipe = 0; pipe < NumStages; ++pipe) {
            if ((warp_idx == 0) && lane_predicate) {
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            }
            ++k_tile;
        }

        // Mainloop: produce tiles as consumers consume them
        CUTE_NO_UNROLL
        while (k_tile < k_tile_count) {
            int pipe = write_state.index();

            // Wait for consumer to finish with this stage
            if ((warp_idx == 0) && lane_predicate) {
                ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());

                // Issue next TMA load
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            }

            ++write_state;
            ++k_tile;
        }

        // Producer is done - exit early
        return;
    }

    // ========================================================================
    // CONSUMER WARP GROUP: MMA compute only
    // ========================================================================
    if (is_consumer) {
        // MMA setup - use thread index within consumer warp group
        int consumer_thread_idx = threadIdx.x - 128;  // Offset by producer warp group

        ThrMMA thr_mma = mma.get_thread_slice(consumer_thread_idx);
        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);
        Tensor tCgC = thr_mma.partition_C(gC);

        // Allocate accumulators
        Tensor tCrC = thr_mma.make_fragment_C(tCgC);
        clear(tCrC);

        // WGMMA descriptors (views of smem)
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        // Mainloop: consume all K tiles
        int tiles_to_consume = k_tile_count;

        CUTE_NO_UNROLL
        while (tiles_to_consume > 0) {
            int read_pipe = read_state.index();

            // Wait for producer to complete this tile
            ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

            // Execute WGMMA
            warpgroup_arrive();
            gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
            warpgroup_commit_batch();

            // Wait for MMA to complete
            warpgroup_wait<0>();

            // Signal that we're done consuming this stage
            ConsumerBarType::arrive(&consumer_mbar[read_pipe]);

            ++read_state;
            --tiles_to_consume;
        }

        // Epilogue: write output
        axpby(alpha, tCrC, beta, tCgC);
    }
}

// ============================================================================
// Host wrapper for TN GEMM (row-major A, row-major B)
// ============================================================================
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn_warp_specialized(int m, int n, int k,
                               Alpha alpha,
                               TA const* A, int ldA,
                               TB const* B, int ldB,
                               Beta beta,
                               TC* C, int ldC,
                               cudaStream_t stream = 0)
{
    using namespace cute;

    // Problem shape
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // TN strides: A is (M,K) with stride (ldA,1), B is (N,K) with stride (ldB,1)
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // CTA tile sizes
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    constexpr int NumStages = 4;  // More stages for better latency hiding
    auto bP = Int<NumStages>{};

    // SMEM layouts with 128B swizzle for WGMMA
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // TiledMMA: WGMMA 64x64x16 for FP16
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    // Create TMA atoms
    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    // Launch params
    // 256 threads = 2 warp groups (producer + consumer)
    dim3 dimBlock(256);
    dim3 dimCluster(1, 1, 1);  // Single CTA per cluster for simplicity
    dim3 dimGrid(ceil_div(M, int(bM)), ceil_div(N, int(bN)));

    int smem_size = sizeof(SharedStorageWarpSpecialized<TA, TB, decltype(sA), decltype(sB), NumStages>);

    auto* kernel_ptr = &gemm_warp_specialized_device<
        decltype(prob_shape), decltype(cta_tiler),
        TA, decltype(sA), decltype(tmaA),
        TB, decltype(sB), decltype(tmaB),
        TC, decltype(dC), decltype(tiled_mma),
        decltype(alpha), decltype(beta), NumStages>;

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size));

    // Launch with cluster
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const*)kernel_ptr,
        prob_shape, cta_tiler,
        A, tmaA,
        B, tmaB,
        C, dC, tiled_mma,
        alpha, beta);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Kernel launch failed" << std::endl;
    }
}

// ============================================================================
// Host wrapper for NT GEMM (column-major A, column-major B)
// ============================================================================
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt_warp_specialized(int m, int n, int k,
                               Alpha alpha,
                               TA const* A, int ldA,
                               TB const* B, int ldB,
                               Beta beta,
                               TC* C, int ldC,
                               cudaStream_t stream = 0)
{
    using namespace cute;

    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // NT strides
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    constexpr int NumStages = 4;
    auto bP = Int<NumStages>{};

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    dim3 dimBlock(256);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(ceil_div(M, int(bM)), ceil_div(N, int(bN)));

    int smem_size = sizeof(SharedStorageWarpSpecialized<TA, TB, decltype(sA), decltype(sB), NumStages>);

    auto* kernel_ptr = &gemm_warp_specialized_device<
        decltype(prob_shape), decltype(cta_tiler),
        TA, decltype(sA), decltype(tmaA),
        TB, decltype(sB), decltype(tmaB),
        TC, decltype(dC), decltype(tiled_mma),
        decltype(alpha), decltype(beta), NumStages>;

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size));

    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const*)kernel_ptr,
        prob_shape, cta_tiler,
        A, tmaA,
        B, tmaB,
        C, dC, tiled_mma,
        alpha, beta);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Kernel launch failed" << std::endl;
    }
}

// ============================================================================
// Main entry point
// ============================================================================
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k,
          Alpha alpha,
          TA const* A, int ldA,
          TB const* B, int ldB,
          Beta beta,
          TC* C, int ldC,
          cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T') {
        return gemm_nt_warp_specialized(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    } else if (transA == 'T' && transB == 'N') {
        return gemm_tn_warp_specialized(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not implemented");
}

int main(int argc, char** argv)
{
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);

    if (props.major != 9) {
        std::cout << "This example requires NVIDIA Hopper Architecture (SM90)\n";
        return 0;
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    printf("Warp-Specialized Hopper GEMM Example\n");
    printf("Device: %s (SM%d%d)\n", props.name, props.major, props.minor);
    printf("Architecture: Hopper with TMA + WGMMA\n");
    printf("Design: Producer warp group (TMA) + Consumer warp group (MMA)\n\n");

    int m = 5120;
    int n = 5120;
    int k = 4096;

    if (argc >= 2) sscanf(argv[1], "%d", &m);
    if (argc >= 3) sscanf(argv[2], "%d", &n);
    if (argc >= 4) sscanf(argv[3], "%d", &k);

    char transA = 'T';
    char transB = 'N';

    if (argc >= 5) sscanf(argv[4], "%c", &transA);
    if (argc >= 6) sscanf(argv[5], "%c", &transB);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;

    TI alpha = TI(1.0f);
    TI beta = TI(0.0f);

    printf("M = %d, N = %d, K = %d\n", m, n, k);
    printf("C = A^%c * B^%c\n\n", transA, transB);

    // Allocate
    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    // Initialize
    for (int i = 0; i < m * k; ++i) h_A[i] = TA((rand() % 2) ? 1.0f : -1.0f);
    for (int i = 0; i < n * k; ++i) h_B[i] = TB((rand() % 2) ? 1.0f : -1.0f);
    for (int i = 0; i < m * n; ++i) h_C[i] = TC(0.0f);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = 2.0 * m * n * k * 1e-9;

    int ldA = (transA == 'N') ? m : k;
    int ldB = (transB == 'N') ? k : n;
    int ldC = m;

    // Warmup
    d_C = h_C;
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    // Benchmark
    const int timing_iterations = 100;
    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k,
             alpha,
             d_A.data().get(), ldA,
             d_B.data().get(), ldB,
             beta,
             d_C.data().get(), ldC);
    }
    double time_ms = timer.seconds() * 1000.0 / timing_iterations;
    CUTE_CHECK_LAST();

    printf("Performance: [%6.1f] GFlop/s  (%.4f ms)\n", gflops / (time_ms * 1e-3), time_ms);

#else
    std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled. Test is waived.\n";
#endif

    return 0;
}
