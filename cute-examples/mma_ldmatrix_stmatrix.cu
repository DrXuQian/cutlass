// CuTe MMA with ldmatrix (load A/B) and stmatrix (store C)
// Target: SM90 (Hopper) - stmatrix requires SM90+
// MMA: SM80_16x8x16_F16F16F16F16_TN

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

// ============================================================================
// Kernel with ldmatrix for A/B, stmatrix for C
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kStages = 1>
__global__ void mma_ldmatrix_stmatrix_kernel(
    __half* __restrict__ Cptr,
    const __half* __restrict__ Aptr,
    const __half* __restrict__ Bptr,
    int m, int n, int k)
{
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // ========================================================================
    // 1. Create global memory tensors
    // ========================================================================
    // A: (M, K) row-major
    Tensor gA_full = make_tensor(make_gmem_ptr(Aptr),
                                 make_shape(m, k),
                                 make_stride(k, Int<1>{}));
    // B: (N, K) row-major (will be transposed in MMA)
    Tensor gB_full = make_tensor(make_gmem_ptr(Bptr),
                                 make_shape(n, k),
                                 make_stride(k, Int<1>{}));
    // C: (M, N) row-major
    Tensor gC_full = make_tensor(make_gmem_ptr(Cptr),
                                 make_shape(m, n),
                                 make_stride(n, Int<1>{}));

    // Extract this CTA's tile
    Tensor gA = local_tile(gA_full, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                           make_coord(by, _));  // (kTileM, kTileK, num_k_tiles)
    Tensor gB = local_tile(gB_full, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                           make_coord(bx, _));  // (kTileN, kTileK, num_k_tiles)
    Tensor gC = local_tile(gC_full, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                           make_coord(by, bx)); // (kTileM, kTileN)

    // ========================================================================
    // 2. Define shared memory layouts with swizzle for bank conflict free
    // ========================================================================
    // For ldmatrix: need 128-bit (8 half) aligned rows
    // Swizzle<B, M, S>: bits [B, B+M) are XORed with bits [B+M, B+M+S)
    // For FP16 with 16 elements per row: Swizzle<3,3,3> works well

    // A: (kTileM, kTileK) with swizzle
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<3, 3, 3>{},
                    make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtomA{},
                      make_shape(Int<kTileM>{}, Int<kTileK>{})));

    // B: (kTileN, kTileK) with swizzle
    using SmemLayoutAtomB = decltype(
        composition(Swizzle<3, 3, 3>{},
                    make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtomB{},
                      make_shape(Int<kTileN>{}, Int<kTileK>{})));

    // C: (kTileM, kTileN) with swizzle for stmatrix
    using SmemLayoutAtomC = decltype(
        composition(Swizzle<3, 3, 3>{},
                    make_layout(make_shape(Int<8>{}, Int<kTileN>{}),
                                make_stride(Int<kTileN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(
        tile_to_shape(SmemLayoutAtomC{},
                      make_shape(Int<kTileM>{}, Int<kTileN>{})));

    // Allocate shared memory
    extern __shared__ char smem_buf[];
    __half* smem_a = reinterpret_cast<__half*>(smem_buf);
    __half* smem_b = smem_a + cosize(SmemLayoutA{});
    __half* smem_c = smem_b + cosize(SmemLayoutB{});

    Tensor sA = make_tensor(make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_b), SmemLayoutB{});
    Tensor sC = make_tensor(make_smem_ptr(smem_c), SmemLayoutC{});

    // ========================================================================
    // 3. Define TiledMMA
    // ========================================================================
    using MmaAtom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;

    // 2x2 warp tiling for 32x32 tile, partition tile as 32x32x16
    using TiledMma = TiledMMA<
        MmaAtom,
        Layout<Shape<_2, _2, _1>>,      // 2x2 warp arrangement
        Tile<Int<kTileM>, Int<kTileN>, Int<kTileK>>>;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // ========================================================================
    // 4. Define TiledCopy for G2S (global to shared)
    // ========================================================================
    // Use simple copy for G2S (could use cp.async for better perf)
    using G2SCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, __half>;

    // For A: vectorized copy along K dimension
    auto g2s_tiled_copy_a = make_tiled_copy(
        G2SCopyAtom{},
        Layout<Shape<_32, _1>>{},      // 32 threads
        Layout<Shape<_1, _8>>{}        // 8 values per thread (128-bit)
    );

    // For B: vectorized copy along K dimension
    auto g2s_tiled_copy_b = make_tiled_copy(
        G2SCopyAtom{},
        Layout<Shape<_32, _1>>{},
        Layout<Shape<_1, _8>>{}
    );

    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);

    // ========================================================================
    // 5. Define TiledCopy for S2R using ldmatrix
    // ========================================================================
    // ldmatrix.x4.trans for A (16x16 tile, transposed load)
    using S2RCopyAtomA = Copy_Atom<SM75_U16x8_LDSM_T, __half>;
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);

    // ldmatrix.x4.trans for B
    using S2RCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, __half>;
    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);

    // ========================================================================
    // 6. Define TiledCopy for R2S using stmatrix (SM90+)
    // ========================================================================
    // stmatrix.x4.trans for C
    using R2SCopyAtomC = Copy_Atom<SM90_U16x8_STSM_T, __half>;
    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);

    // ========================================================================
    // 7. Define TiledCopy for S2G (shared to global)
    // ========================================================================
    using S2GCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, __half>;
    auto s2g_tiled_copy_c = make_tiled_copy(
        S2GCopyAtom{},
        Layout<Shape<_32, _1>>{},
        Layout<Shape<_1, _8>>{}
    );
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(tid);

    // ========================================================================
    // 8. Partition tensors for copies
    // ========================================================================
    // G2S partitions
    Tensor tAgA = g2s_thr_copy_a.partition_S(gA(_, _, 0));  // (CPY, CPY_M, CPY_K)
    Tensor tAsA = g2s_thr_copy_a.partition_D(sA);           // (CPY, CPY_M, CPY_K)

    Tensor tBgB = g2s_thr_copy_b.partition_S(gB(_, _, 0));
    Tensor tBsB = g2s_thr_copy_b.partition_D(sB);

    // S2R partitions for ldmatrix
    Tensor tAsA_ldm = s2r_thr_copy_a.partition_S(sA);       // (CPY, CPY_M, CPY_K)
    Tensor tArA = s2r_thr_copy_a.retile_D(thr_mma.partition_fragment_A(sA));

    Tensor tBsB_ldm = s2r_thr_copy_b.partition_S(sB);
    Tensor tBrB = s2r_thr_copy_b.retile_D(thr_mma.partition_fragment_B(sB));

    // R2S partitions for stmatrix
    Tensor tCrC = thr_mma.partition_fragment_C(gC);         // Accumulator
    Tensor tCrC_stm = r2s_thr_copy_c.retile_S(tCrC);        // Retiled for stmatrix
    Tensor tCsC_stm = r2s_thr_copy_c.partition_D(sC);       // Dest in smem

    // S2G partitions
    Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
    Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gC);

    // ========================================================================
    // 9. Initialize accumulator
    // ========================================================================
    clear(tCrC);

    // ========================================================================
    // 10. Main loop over K dimension
    // ========================================================================
    int num_k_tiles = size<2>(gA);

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // --- Load A and B from global to shared ---
        // Update source pointers for this k_tile
        Tensor tAgA_k = g2s_thr_copy_a.partition_S(gA(_, _, k_tile));
        Tensor tBgB_k = g2s_thr_copy_b.partition_S(gB(_, _, k_tile));

        cute::copy(g2s_tiled_copy_a, tAgA_k, tAsA);
        cute::copy(g2s_tiled_copy_b, tBgB_k, tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // --- Load A and B from shared to registers using ldmatrix ---
        cute::copy(s2r_tiled_copy_a, tAsA_ldm, tArA);
        cute::copy(s2r_tiled_copy_b, tBsB_ldm, tBrB);

        // --- Compute MMA ---
        cute::gemm(tiled_mma, tArA, tBrB, tCrC);

        __syncthreads();
    }

    // ========================================================================
    // 11. Store C from registers to shared using stmatrix
    // ========================================================================
    cute::copy(r2s_tiled_copy_c, tCrC_stm, tCsC_stm);
    __syncthreads();

    // ========================================================================
    // 12. Store C from shared to global
    // ========================================================================
    cute::copy(s2g_tiled_copy_c, tCsC_s2g, tCgC_s2g);
}

// ============================================================================
// Host code
// ============================================================================
void run_mma_test(int m, int n, int k) {
    constexpr int kTileM = 32;
    constexpr int kTileN = 32;
    constexpr int kTileK = 16;

    // Calculate shared memory size
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<3, 3, 3>{},
                    make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));

    using SmemLayoutAtomB = decltype(
        composition(Swizzle<3, 3, 3>{},
                    make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<kTileN>{}, Int<kTileK>{})));

    using SmemLayoutAtomC = decltype(
        composition(Swizzle<3, 3, 3>{},
                    make_layout(make_shape(Int<8>{}, Int<kTileN>{}),
                                make_stride(Int<kTileN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(
        tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kTileM>{}, Int<kTileN>{})));

    size_t smem_size = (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{}) +
                        cosize(SmemLayoutC{})) * sizeof(__half);

    printf("MMA ldmatrix/stmatrix Test\n");
    printf("  M=%d, N=%d, K=%d\n", m, n, k);
    printf("  TileM=%d, TileN=%d, TileK=%d\n", kTileM, kTileN, kTileK);
    printf("  Shared memory: %zu bytes\n", smem_size);

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(__half));
    cudaMalloc(&d_B, n * k * sizeof(__half));
    cudaMalloc(&d_C, m * n * sizeof(__half));

    // Initialize with random data
    std::vector<__half> h_A(m * k), h_B(n * k), h_C(m * n, __float2half(0.0f));

    for (int i = 0; i < m * k; ++i) {
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < n * k; ++i) {
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }

    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), m * n * sizeof(__half), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
    dim3 block(128);  // 4 warps = 128 threads for 2x2 warp tiling

    printf("  Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);

    // Set shared memory size
    cudaFuncSetAttribute(
        mma_ldmatrix_stmatrix_kernel<kTileM, kTileN, kTileK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    mma_ldmatrix_stmatrix_kernel<kTileM, kTileN, kTileK>
        <<<grid, block, smem_size>>>(d_C, d_A, d_B, m, n, k);

    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        mma_ldmatrix_stmatrix_kernel<kTileM, kTileN, kTileK>
            <<<grid, block, smem_size>>>(d_C, d_A, d_B, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;

    double flops = 2.0 * m * n * k;
    double tflops = flops / (ms * 1e9);

    printf("  Time: %.3f ms, TFLOPS: %.2f\n", ms, tflops);

    // Verify result (simple check)
    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(__half), cudaMemcpyDeviceToHost);

    // Check for NaN
    bool has_nan = false;
    for (int i = 0; i < m * n && i < 100; ++i) {
        if (isnan(__half2float(h_C[i]))) {
            has_nan = true;
            break;
        }
    }
    printf("  Result check: %s\n", has_nan ? "FAILED (NaN detected)" : "PASSED");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    int m = 256, n = 256, k = 256;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    // Check CUDA device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("Warning: stmatrix requires SM90+. This kernel may not work on SM%d%d.\n",
               prop.major, prop.minor);
        printf("Consider using the SM80 version without stmatrix.\n");
    }

    run_mma_test(m, n, k);

    return 0;
}
