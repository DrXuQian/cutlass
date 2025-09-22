/*
 * Simplified B2B GEMM with RF (Register File) Residency
 * This is a simplified version for understanding the core concepts
 * Only supports SM80 FP16
 */

#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"

using namespace nvcuda;

// Simple B2B GEMM kernel with RF residency
// The key idea: keep intermediate results in registers between two GEMMs
template<int M_TILES, int N_TILES, int K_TILES>
__global__ void b2b_gemm_rf_kernel(
    half const* __restrict__ A,    // M x K
    half const* __restrict__ B0,   // K x N
    half const* __restrict__ B1,   // N x P
    half* __restrict__ D,          // M x P
    int M, int N, int K, int P
) {
    // Tensor Core tile size for SM80: 16x8x16
    const int WMMA_M = 16;
    const int WMMA_N = 8;
    const int WMMA_K = 16;

    // Calculate warp and lane IDs
    int warpId = (threadIdx.x / 32);
    int laneId = threadIdx.x % 32;

    // Each warp computes one 16x8 tile
    int warpM = blockIdx.x * M_TILES + (warpId / (N_TILES/WMMA_N)) * WMMA_M;
    int warpN = blockIdx.y * N_TILES + (warpId % (N_TILES/WMMA_N)) * WMMA_N;

    // Bounds check
    if (warpM >= M || warpN >= N) return;

    // Declare fragments for Tensor Core operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;

    // ========== First GEMM: C = A * B0 ==========
    // Initialize accumulator for first GEMM
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Loop over K dimension for first GEMM
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A matrix tile
        int aRow = warpM;
        int aCol = k;
        if (aRow < M && aCol + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }

        // Load B0 matrix tile
        int bRow = k;
        int bCol = warpN;
        if (bRow + WMMA_K <= K && bCol < N) {
            // B0 is in column major for Tensor Core
            wmma::load_matrix_sync(b_frag, B0 + bRow + bCol * K, K);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }

        // Perform matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // ========== RF Residency: Keep c_frag in registers ==========
    // The intermediate result c_frag stays in registers (RF)
    // No store to global memory here!

    // ========== Second GEMM: D = C * B1 ==========
    // Now c_frag contains the result of first GEMM
    // Use it as input for second GEMM

    // Initialize accumulator for second GEMM
    wmma::fill_fragment(d_frag, __float2half(0.0f));

    // For second GEMM, we need different tile indexing
    int warpP = blockIdx.y * N_TILES + (warpId % (N_TILES/WMMA_N)) * WMMA_N;
    if (warpP >= P) return;

    // Loop over N dimension for second GEMM (C is M x N, B1 is N x P)
    for (int n = 0; n < N; n += WMMA_K) {
        // Here we would need to reconstruct matrix tiles from c_frag
        // This is simplified - in real implementation, this requires
        // careful fragment manipulation and potentially shared memory

        // Load B1 matrix tile
        int b1Row = n;
        int b1Col = warpP;
        if (b1Row + WMMA_K <= N && b1Col < P) {
            wmma::load_matrix_sync(b_frag, B1 + b1Row + b1Col * N, N);

            // Simplified: Use c_frag directly as a_frag for second GEMM
            // In practice, this needs proper tile reformatting
            wmma::mma_sync(d_frag, c_frag, b_frag, d_frag);
        }
    }

    // Store final result D
    if (warpM < M && warpP < P) {
        wmma::store_matrix_sync(D + warpM * P + warpP, d_frag, P, wmma::mem_row_major);
    }
}

// Simplified launcher for B2B GEMM with RF residency
class SimplifiedB2bGemmRF {
public:
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // Run the B2B GEMM
    bool run(int M, int N, int K, int P) {
        std::cout << "\n=== Simplified B2B GEMM with RF Residency ===\n";
        std::cout << "Problem: [" << M << "," << K << "] * [" << K << "," << N << "] = ["
                  << M << "," << N << "]\n";
        std::cout << "        [" << M << "," << N << "] * [" << N << "," << P << "] = ["
                  << M << "," << P << "]\n\n";

        // Allocate host tensors
        cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
        cutlass::HostTensor<ElementB, LayoutB> tensor_B0({K, N});
        cutlass::HostTensor<ElementB, LayoutB> tensor_B1({N, P});
        cutlass::HostTensor<ElementD, LayoutD> tensor_D({M, P});
        cutlass::HostTensor<ElementD, LayoutD> tensor_D_ref({M, P});

        // Initialize tensors with random values
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_A.host_view(), 1, ElementA(1), ElementA(-1), 0);
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_B0.host_view(), 1, ElementB(1), ElementB(-1), 1);
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_B1.host_view(), 1, ElementB(1), ElementB(-1), 2);

        // Copy to device
        tensor_A.sync_device();
        tensor_B0.sync_device();
        tensor_B1.sync_device();

        // Launch kernel with simplified configuration
        const int M_TILES = 64;  // Tile size in M dimension
        const int N_TILES = 64;  // Tile size in N dimension
        const int K_TILES = 32;  // Tile size in K dimension

        dim3 gridDim((M + M_TILES - 1) / M_TILES, (std::max(N, P) + N_TILES - 1) / N_TILES);
        dim3 blockDim(128);  // 4 warps per block

        std::cout << "Launching kernel with grid(" << gridDim.x << "," << gridDim.y
                  << ") block(" << blockDim.x << ")\n";

        // Launch RF-resident kernel
        b2b_gemm_rf_kernel<M_TILES, N_TILES, K_TILES><<<gridDim, blockDim>>>(
            (half const*)tensor_A.device_data(),
            (half const*)tensor_B0.device_data(),
            (half const*)tensor_B1.device_data(),
            (half*)tensor_D.device_data(),
            M, N, K, P
        );

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << "\n";
            return false;
        }

        // Copy result back
        tensor_D.sync_host();

        // Compute reference on CPU for verification
        std::cout << "Computing reference on CPU...\n";

        // Intermediate result C
        cutlass::HostTensor<ElementC, LayoutC> tensor_C_ref({M, N});

        // First GEMM: C = A * B0
        cutlass::reference::host::Gemm<
            ElementA, LayoutA,
            ElementB, LayoutB,
            ElementC, LayoutC,
            float, float
        > gemm_op;

        gemm_op(
            {M, N, K},
            float(1),
            tensor_A.host_view(),
            tensor_B0.host_view(),
            float(0),
            tensor_C_ref.host_view()
        );

        // Second GEMM: D = C * B1
        gemm_op(
            {M, P, N},
            float(1),
            tensor_C_ref.host_view(),
            tensor_B1.host_view(),
            float(0),
            tensor_D_ref.host_view()
        );

        // Compare results (simplified comparison)
        bool passed = true;
        float max_error = 0.0f;
        for (int i = 0; i < M * P; ++i) {
            float diff = std::abs(float(tensor_D.host_data()[i]) -
                                 float(tensor_D_ref.host_data()[i]));
            max_error = std::max(max_error, diff);
            if (diff > 0.1f) {  // Relaxed tolerance for simplified kernel
                passed = false;
            }
        }

        std::cout << "Max error: " << max_error << "\n";

        if (passed) {
            std::cout << "*** PASSED ***\n";
        } else {
            std::cout << "*** FAILED ***\n";

            // Print first few elements for debugging
            std::cout << "\nFirst 4x4 elements:\n";
            std::cout << "GPU result:\n";
            for (int i = 0; i < std::min(4, M); ++i) {
                for (int j = 0; j < std::min(4, P); ++j) {
                    std::cout << float(tensor_D.at({i, j})) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\nCPU reference:\n";
            for (int i = 0; i < std::min(4, M); ++i) {
                for (int j = 0; j < std::min(4, P); ++j) {
                    std::cout << float(tensor_D_ref.at({i, j})) << " ";
                }
                std::cout << "\n";
            }
        }

        return passed;
    }
};

int main() {
    // Check GPU
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "Running on: " << props.name << " (SM" << props.major << props.minor << ")\n";

    if (props.major < 8) {
        std::cerr << "This example requires SM80 or newer for FP16 Tensor Cores\n";
        return -1;
    }

    SimplifiedB2bGemmRF b2b_gemm;

    // Test with small sizes
    bool passed = b2b_gemm.run(64, 64, 64, 32);

    return passed ? 0 : -1;
}