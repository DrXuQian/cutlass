// CuTe MMA Example (SM80 Ampere compatible)
// Demonstrates TiledMMA with TiledCopy
// MMA: SM80_16x8x16_F16F16F16F16_TN

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

// ============================================================================
// Debug helper to print layouts
// ============================================================================
template <class Layout>
__device__ void print_layout_info(const char* name, Layout const& layout) {
    printf("  %s: ", name);
    print(layout);
    printf(" (size=%d, cosize=%d)\n", (int)size(layout), (int)cosize(layout));
}

template <class Tensor>
__device__ void print_tensor_info(const char* name, Tensor const& tensor) {
    printf("  %s shape: ", name);
    print(shape(tensor));
    printf(", layout: ");
    print(layout(tensor));
    printf("\n");
}

// ============================================================================
// Kernel
// ============================================================================
template <int kTileM, int kTileN, int kTileK, bool kDebugPrint = false>
__global__ void mma_kernel(
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
    Tensor gA_full = make_tensor(make_gmem_ptr(Aptr),
                                 make_shape(m, k),
                                 make_stride(k, Int<1>{}));
    Tensor gB_full = make_tensor(make_gmem_ptr(Bptr),
                                 make_shape(n, k),
                                 make_stride(k, Int<1>{}));
    Tensor gC_full = make_tensor(make_gmem_ptr(Cptr),
                                 make_shape(m, n),
                                 make_stride(n, Int<1>{}));

    // Extract this CTA's tile
    Tensor gA = local_tile(gA_full, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                           make_coord(by, _));
    Tensor gB = local_tile(gB_full, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                           make_coord(bx, _));
    Tensor gC = local_tile(gC_full, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                           make_coord(by, bx));

    // ========================================================================
    // 2. Define TiledMMA
    // ========================================================================
    using MmaAtom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;

    // 2x2 warp tiling for 32x32 output tile (4 warps = 128 threads)
    using TiledMma = TiledMMA<
        MmaAtom,
        Layout<Shape<_2, _2, _1>>,
        Tile<Int<kTileM>, Int<kTileN>, Int<kTileK>>>;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // ========================================================================
    // 3. Define shared memory layouts
    // ========================================================================
    // Simple row-major layouts without swizzle (for clarity)
    using SmemLayoutA = Layout<Shape<Int<kTileM>, Int<kTileK>>,
                               Stride<Int<kTileK>, Int<1>>>;
    using SmemLayoutB = Layout<Shape<Int<kTileN>, Int<kTileK>>,
                               Stride<Int<kTileK>, Int<1>>>;

    // Allocate shared memory
    extern __shared__ char smem_buf[];
    __half* smem_a = reinterpret_cast<__half*>(smem_buf);
    __half* smem_b = smem_a + cosize(SmemLayoutA{});

    Tensor sA = make_tensor(make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_b), SmemLayoutB{});

    // ========================================================================
    // 4. Define TiledCopy for G2S using cp.async
    // ========================================================================
    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, __half>;

    // 64 threads: (32 along M) × (2 along K), each thread copies 8 fp16
    // This covers 32×16 = 512 elements perfectly
    auto g2s_tiled_copy_a = make_tiled_copy(
        G2SCopyAtom{},
        Layout<Shape<_32, _2>, Stride<_1, _32>>{},
        Layout<Shape<_1, _8>>{}
    );

    auto g2s_tiled_copy_b = make_tiled_copy(
        G2SCopyAtom{},
        Layout<Shape<_32, _2>, Stride<_1, _32>>{},
        Layout<Shape<_1, _8>>{}
    );

    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid % 64);
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid % 64);

    // ========================================================================
    // 5. Define TiledCopy for S2R (smem to register)
    // ========================================================================
    // Use simple copy that matches MMA fragment layout
    using S2RCopyAtom = Copy_Atom<DefaultCopy, __half>;
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);

    // ========================================================================
    // 6. Create partitions
    // ========================================================================
    // G2S partitions
    Tensor tAgA = g2s_thr_copy_a.partition_S(gA(_, _, 0));
    Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA);

    Tensor tBgB = g2s_thr_copy_b.partition_S(gB(_, _, 0));
    Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB);

    // S2R partitions
    Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA);
    Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB);

    // MMA fragment partitions
    Tensor tArA = thr_mma.partition_fragment_A(sA);
    Tensor tBrB = thr_mma.partition_fragment_B(sB);

    // Retile for S2R copy destination
    Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tArA);
    Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tBrB);

    // Accumulator
    Tensor tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    // Output partition
    Tensor tCgC = thr_mma.partition_C(gC);

    // ========================================================================
    // Debug print (thread 0, block 0 only)
    // ========================================================================
    if constexpr (kDebugPrint) {
        if (tid == 0 && bx == 0 && by == 0) {
            printf("\n========== MMA Layout Debug ==========\n");
            printf("\n--- Shared Memory Layouts ---\n");
            print_layout_info("SmemLayoutA", SmemLayoutA{});
            print_layout_info("SmemLayoutB", SmemLayoutB{});

            printf("\n--- G2S Copy Partitions ---\n");
            print_tensor_info("tAgA (G2S src)", tAgA);
            print_tensor_info("tAsA_g2s (G2S dst)", tAsA_g2s);
            print_tensor_info("tBgB (G2S src)", tBgB);
            print_tensor_info("tBsB_g2s (G2S dst)", tBsB_g2s);

            printf("\n--- S2R Copy Partitions ---\n");
            print_tensor_info("tAsA_s2r (S2R src)", tAsA_s2r);
            print_tensor_info("tArA_s2r (S2R dst)", tArA_s2r);
            print_tensor_info("tBsB_s2r (S2R src)", tBsB_s2r);
            print_tensor_info("tBrB_s2r (S2R dst)", tBrB_s2r);

            printf("\n--- MMA Fragment Partitions ---\n");
            print_tensor_info("tArA (MMA A frag)", tArA);
            print_tensor_info("tBrB (MMA B frag)", tBrB);
            print_tensor_info("tCrC (MMA C frag)", tCrC);

            printf("\n--- TiledCopy Info ---\n");
            printf("  G2S Copy A TiledLayout_TV: ");
            print(typename decltype(g2s_tiled_copy_a)::TiledLayout_TV{});
            printf("\n");
            printf("  G2S Copy A Tiler_MN: ");
            print(typename decltype(g2s_tiled_copy_a)::Tiler_MN{});
            printf("\n");

            printf("  S2R Copy A TiledLayout_TV: ");
            print(typename decltype(s2r_tiled_copy_a)::TiledLayout_TV{});
            printf("\n");
            printf("  S2R Copy A Tiler_MN: ");
            print(typename decltype(s2r_tiled_copy_a)::Tiler_MN{});
            printf("\n");

            printf("\n--- MMA Info ---\n");
            printf("  MMA layoutA_TV: ");
            print(tiled_mma.get_layoutA_TV());
            printf("\n");
            printf("  MMA layoutB_TV: ");
            print(tiled_mma.get_layoutB_TV());
            printf("\n");
            printf("  MMA layoutC_TV: ");
            print(tiled_mma.get_layoutC_TV());
            printf("\n");

            printf("\n=======================================\n\n");
        }
    }

    // ========================================================================
    // 7. Main loop over K dimension
    // ========================================================================
    int num_k_tiles = size<2>(gA);

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // --- Copy A and B from global to shared ---
        Tensor tAgA_k = g2s_thr_copy_a.partition_S(gA(_, _, k_tile));
        Tensor tBgB_k = g2s_thr_copy_b.partition_S(gB(_, _, k_tile));

        cute::copy(g2s_tiled_copy_a, tAgA_k, tAsA_g2s);
        cute::copy(g2s_tiled_copy_b, tBgB_k, tBsB_g2s);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // --- Load from shared to registers ---
        cute::copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
        cute::copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

        // --- Compute MMA ---
        cute::gemm(tiled_mma, tArA, tBrB, tCrC);

        __syncthreads();
    }

    // ========================================================================
    // 8. Store C directly to global memory
    // ========================================================================
    cute::copy(tCrC, tCgC);
}

// ============================================================================
// Host code
// ============================================================================
void run_test(int m, int n, int k, bool debug_print = false) {
    constexpr int kTileM = 32;
    constexpr int kTileN = 32;
    constexpr int kTileK = 16;

    using SmemLayoutA = Layout<Shape<Int<kTileM>, Int<kTileK>>,
                               Stride<Int<kTileK>, Int<1>>>;
    using SmemLayoutB = Layout<Shape<Int<kTileN>, Int<kTileK>>,
                               Stride<Int<kTileK>, Int<1>>>;

    size_t smem_size = (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) * sizeof(__half);

    printf("MMA Test (SM80)\n");
    printf("  M=%d, N=%d, K=%d\n", m, n, k);
    printf("  TileM=%d, TileN=%d, TileK=%d\n", kTileM, kTileN, kTileK);
    printf("  Shared memory: %zu bytes\n", smem_size);

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(__half));
    cudaMalloc(&d_B, n * k * sizeof(__half));
    cudaMalloc(&d_C, m * n * sizeof(__half));

    std::vector<__half> h_A(m * k), h_B(n * k);
    std::vector<__half> h_C(m * n, __float2half(0.0f));

    srand(42);
    for (int i = 0; i < m * k; ++i)
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX - 0.5f);
    for (int i = 0; i < n * k; ++i)
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX - 0.5f);

    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(__half));

    dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
    dim3 block(128);

    printf("  Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch with debug print if requested
    if (debug_print) {
        mma_kernel<kTileM, kTileN, kTileK, true>
            <<<grid, block, smem_size>>>(d_C, d_A, d_B, m, n, k);
    } else {
        mma_kernel<kTileM, kTileN, kTileK, false>
            <<<grid, block, smem_size>>>(d_C, d_A, d_B, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return;
    }

    // Benchmark (without debug print)
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        mma_kernel<kTileM, kTileN, kTileK, false>
            <<<grid, block, smem_size>>>(d_C, d_A, d_B, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;

    printf("  Time: %.3f ms, TFLOPS: %.2f\n", ms, 2.0 * m * n * k / (ms * 1e9));

    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(__half), cudaMemcpyDeviceToHost);

    if (m <= 64 && n <= 64 && k <= 64) {
        std::vector<float> ref_C(m * n, 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; ++kk)
                    sum += __half2float(h_A[i * k + kk]) * __half2float(h_B[j * k + kk]);
                ref_C[i * n + j] = sum;
            }
        }

        float max_diff = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < m * n; ++i) {
            float diff = fabs(__half2float(h_C[i]) - ref_C[i]);
            if (diff > max_diff) { max_diff = diff; max_idx = i; }
        }
        printf("  Max diff: %e at idx %d (GPU=%f, CPU=%f)\n",
               max_diff, max_idx, __half2float(h_C[max_idx]), ref_C[max_idx]);
        printf("  Verification: %s\n", max_diff < 0.1f ? "PASSED" : "FAILED");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    int m = 256, n = 256, k = 256;
    bool debug_print = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug_print = true;
        } else if (i + 2 < argc) {
            m = atoi(argv[i]);
            n = atoi(argv[i + 1]);
            k = atoi(argv[i + 2]);
            i += 2;
        }
    }

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 8) {
        printf("Error: Requires SM80+\n");
        return 1;
    }

    run_test(m, n, k, debug_print);

    printf("\n--- Small size verification (with debug) ---\n");
    run_test(32, 32, 16, true);

    return 0;
}
