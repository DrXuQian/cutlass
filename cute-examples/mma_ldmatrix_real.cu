// CuTe MMA Example with Real ldmatrix (SM80+)
// Based on cutlass/examples/cute/tutorial/sgemm_sm80.cu
//
// This example demonstrates:
// 1. cp.async for Global -> Shared memory transfer
// 2. ldmatrix (SM75_U32x4_LDSM_N) for Shared -> Register transfer
// 3. Proper swizzle layout for ldmatrix compatibility
// 4. TiledMMA with SM80_16x8x16_F16F16F16F16_TN
//
// Key insight: ldmatrix requires specific smem layout (swizzled) to work correctly.
// The swizzle pattern ensures 128-bit aligned access for ldmatrix.

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

// ============================================================================
// Debug print helpers
// ============================================================================
template <class Tensor>
__device__ void print_tensor_info(const char* name, const Tensor& t) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("%s: ", name);
        print(t);
        printf("\n");
    }
}

#define PRINT_TENSOR(t) print_tensor_info(#t, t)

// ============================================================================
// Shared memory storage
// ============================================================================
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

// ============================================================================
// GEMM Kernel with ldmatrix
// ============================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RCopyB,
          class TC, class CStride, class TiledMma>
__global__ __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_ldmatrix_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA g2s_copy_a, S2RCopyA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB g2s_copy_b, S2RCopyB s2r_atom_b,
    TC* C, CStride dC, TiledMma mma,
    bool print_debug = false)
{
    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size(g2s_copy_a) == size(mma));
    CUTE_STATIC_ASSERT_V(size(g2s_copy_b) == size(mma));

    // Full tensors in global memory
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M,N)

    // Get CTA tile
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory
    extern __shared__ char shared_memory[];
    using SharedStorageT = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorageT& smem = *reinterpret_cast<SharedStorageT*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);  // (BLK_N,BLK_K,PIPE)

    // ========================================================================
    // G2S Copy: Global -> Shared using cp.async
    // ========================================================================
    ThrCopy thr_g2s_a = g2s_copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_g2s_a.partition_S(gA);  // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_g2s_a.partition_D(sA);  // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_g2s_b = g2s_copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_g2s_b.partition_S(gB);  // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_g2s_b.partition_D(sB);  // (CPY,CPY_N,CPY_K,PIPE)

    // ========================================================================
    // MMA partitions
    // ========================================================================
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)

    // Register fragments
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));  // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));  // (MMA,MMA_N,MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);           // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    // ========================================================================
    // S2R Copy using ldmatrix: make_tiled_copy_A/B + retile_D
    // ========================================================================
    // make_tiled_copy_A creates a TiledCopy that matches the MMA's A operand layout
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy thr_s2r_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);     // (CPY,MMA_M,MMA_K,PIPE)
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);      // (CPY,MMA_M,MMA_K) - retile MMA fragment for copy

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy thr_s2r_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = thr_s2r_b.partition_S(sB);     // (CPY,MMA_N,MMA_K,PIPE)
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);      // (CPY,MMA_N,MMA_K)

    // Debug prints (controlled by print_debug flag)
    if (print_debug && thread0()) {
        printf("\n=== Layout Debug Info ===\n");

        printf("\n--- Global Memory ---\n");
        printf("mA: "); print(mA); printf("\n");
        printf("gA: "); print(gA); printf("\n");

        printf("\n--- Shared Memory ---\n");
        printf("sA: "); print(sA); printf("\n");
        printf("sB: "); print(sB); printf("\n");

        printf("\n--- G2S Copy (cp.async) ---\n");
        printf("tAgA (G2S src): "); print(tAgA); printf("\n");
        printf("tAsA (G2S dst): "); print(tAsA); printf("\n");

        printf("\n--- S2R Copy (ldmatrix) ---\n");
        printf("s2r_copy_a: "); print(s2r_copy_a); printf("\n");
        printf("tXsA (S2R src): "); print(tXsA); printf("\n");
        printf("tXrA (S2R dst): "); print(tXrA); printf("\n");

        printf("\n--- MMA Fragments ---\n");
        printf("tCrA: "); print(tCrA); printf("\n");
        printf("tCrB: "); print(tCrB); printf("\n");
        printf("tCrC: "); print(tCrC); printf("\n");

        printf("\n--- MMA Info ---\n");
        printf("mma: "); print(mma); printf("\n");

        printf("\n=========================\n\n");
    }

    // ========================================================================
    // Pipelined main loop (simplified: 2-stage pipeline)
    // ========================================================================
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_count = size<3>(tAgA);
    int k_tile_next = 0;

    // Prefetch: load first K_PIPE_MAX-1 tiles
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(g2s_copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy(g2s_copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) ++k_tile_next;
    }

    // Current pipe indices
    int smem_pipe_read = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    // Pipe slice
    Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
    Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

    // K blocks per tile (register pipeline depth)
    auto K_BLOCK_MAX = size<2>(tCrA);

    // Prefetch first register block
    if (K_BLOCK_MAX > 1) {
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        // Load first k-block from smem to regs using ldmatrix
        copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
        copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
    }

    // Main loop
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice next smem pipe
                tXsA_p = tXsA(_,_,_,smem_pipe_read);
                tXsB_p = tXsB(_,_,_,smem_pipe_read);

                // Wait for smem to be ready
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Prefetch next k-block using ldmatrix
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
            copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

            // Issue next G2S copy
            if (k_block == 0) {
                copy(g2s_copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                copy(g2s_copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                cp_async_fence();

                --k_tile_count;
                if (k_tile_count > 0) ++k_tile_next;

                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
            }

            // Compute MMA for current k-block
            gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
    }

    // ========================================================================
    // Epilogue: write output
    // ========================================================================
    copy(tCrC, tCgC);
}

// ============================================================================
// Host wrapper
// ============================================================================
void gemm_ldmatrix(int m, int n, int k, bool print_debug = false) {
    using namespace cute;

    using TA = half_t;
    using TB = half_t;
    using TC = half_t;

    // Problem shape
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // TN layout: A is row-major (ldA=K), B is row-major (ldB=K), C is col-major (ldC=M)
    // A: (M,K) with stride (K,1) -> K-major
    // B: (N,K) with stride (K,1) -> K-major
    // C: (M,N) with stride (1,M) -> M-major (column-major)
    auto dA = make_stride(K, Int<1>{});  // A is (M,K), stride (K,1)
    auto dB = make_stride(K, Int<1>{});  // B is (N,K), stride (K,1)
    auto dC = make_stride(Int<1>{}, M);  // C is (M,N), stride (1,M)

    // CTA tile sizes
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};  // Pipeline depth

    // ========================================================================
    // Smem layout with swizzle for ldmatrix compatibility
    // ========================================================================
    // This swizzle pattern is CRITICAL for ldmatrix to work!
    // Swizzle<3,3,3> creates 8x8 bank-conflict-free access pattern
    // The atom layout (8,(8,8)):(8,(1,64)) arranges data for 128-bit loads
    auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                    Layout<Shape <_8, Shape <_8, _8>>,
                                           Stride<_8, Stride<_1,_64>>>{});

    auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));

    // ========================================================================
    // G2S Copy: cp.async with 128-bit vectorized loads
    // ========================================================================
    // Thread layout: 16x8 threads in k-major order
    // Value layout: 1x8 (each thread loads 8 half_t = 128 bits)
    TiledCopy g2s_copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // 16x8 threads, k-major
        Layout<Shape<_1, _8>>{});                   // 1x8 values per thread

    TiledCopy g2s_copy_b = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});

    // ========================================================================
    // TiledMMA: SM80_16x8x16 with 2x2 warp arrangement
    // ========================================================================
    TiledMMA mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        Layout<Shape<_2, _2>>{},      // 2x2 MMA atoms -> 4 warps (128 threads)
        Tile<_32, _32, _16>{});       // 32x32x16 per warp tile (for ldmatrix)

    // ========================================================================
    // S2R Copy: ldmatrix
    // ========================================================================
    // SM75_U32x4_LDSM_N: Non-transposed ldmatrix.x4
    // Each thread loads 4x32bit = 128 bits = 8 half_t
    Copy_Atom<SM75_U32x4_LDSM_N, TA> s2r_atom_a;
    Copy_Atom<SM75_U32x4_LDSM_N, TB> s2r_atom_b;

    // ========================================================================
    // Launch kernel
    // ========================================================================
    int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
    dim3 block(size(mma));
    dim3 grid(ceil_div(M, int(bM)), ceil_div(N, int(bN)));

    printf("GEMM with ldmatrix (SM80)\n");
    printf("  M=%d, N=%d, K=%d\n", m, n, k);
    printf("  Tile: %dx%dx%d, Pipeline: %d\n", int(bM), int(bN), int(bK), int(bP));
    printf("  Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
    printf("  Shared memory: %d bytes\n", smem_size);

    // Allocate memory
    thrust::host_vector<TA> h_A(M * K);
    thrust::host_vector<TB> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N);

    // Initialize
    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = TA(2.0f * rand() / RAND_MAX - 1.0f);
    for (int i = 0; i < N * K; ++i) h_B[i] = TB(2.0f * rand() / RAND_MAX - 1.0f);
    for (int i = 0; i < M * N; ++i) h_C[i] = TC(-1.0f);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    // Set smem size
    auto kernel_ptr = gemm_ldmatrix_kernel<
        decltype(prob_shape), decltype(cta_tiler),
        TA, decltype(dA), decltype(sA), decltype(g2s_copy_a), decltype(s2r_atom_a),
        TB, decltype(dB), decltype(sB), decltype(g2s_copy_b), decltype(s2r_atom_b),
        TC, decltype(dC), decltype(mma)>;

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // Launch
    kernel_ptr<<<grid, block, smem_size>>>(
        prob_shape, cta_tiler,
        d_A.data().get(), dA, sA, g2s_copy_a, s2r_atom_a,
        d_B.data().get(), dB, sB, g2s_copy_b, s2r_atom_b,
        d_C.data().get(), dC, mma,
        print_debug);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        kernel_ptr<<<grid, block, smem_size>>>(
            prob_shape, cta_tiler,
            d_A.data().get(), dA, sA, g2s_copy_a, s2r_atom_a,
            d_B.data().get(), dB, sB, g2s_copy_b, s2r_atom_b,
            d_C.data().get(), dC, mma,
            false);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iterations;

    double gflops = 2.0 * M * N * K * 1e-9;
    printf("  Time: %.3f ms, TFLOPS: %.2f\n", ms, gflops / (ms * 1e-3));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify (for small sizes)
    if (M <= 256 && N <= 256) {
        thrust::host_vector<TC> result = d_C;

        // Reference computation: C = A * B^T (TN layout)
        // A: (M,K), B: (N,K), C: (M,N) col-major
        std::vector<float> ref(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < K; ++kk) {
                    sum += float(h_A[i * K + kk]) * float(h_B[j * K + kk]);
                }
                ref[i + j * M] = sum;  // Column-major storage
            }
        }

        float max_diff = 0.0f;
        int max_i = 0, max_j = 0;
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                float diff = fabs(float(result[i + j * M]) - ref[i + j * M]);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        printf("  Max error: %e at [%d,%d] (GPU=%.4f, CPU=%.4f)\n",
               max_diff, max_i, max_j,
               float(result[max_i + max_j * M]),
               ref[max_i + max_j * M]);
        printf("  Verification: %s\n", max_diff < 0.1f ? "PASSED" : "FAILED");
    }
}

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 8) {
        printf("This example requires SM80+\n");
        return 0;
    }

    int m = 5120, n = 5120, k = 4096;
    bool print_debug = false;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    if (argc >= 5) {
        print_debug = (argv[4][0] == '1');
    }

    gemm_ldmatrix(m, n, k, print_debug);

    // Small test with debug output
    printf("\n--- Small test with layout debug ---\n");
    gemm_ldmatrix(128, 128, 64, true);

    return 0;
}
