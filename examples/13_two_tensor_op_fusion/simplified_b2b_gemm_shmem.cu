/*
 * Simplified B2B GEMM with Shared Memory Residency
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

// Shared memory structure for intermediate results
template<int TILE_M, int TILE_N>
struct SharedMemory {
    half intermediate[TILE_M][TILE_N];  // Store intermediate C matrix
    half tileA[TILE_M][TILE_N];         // Tile for matrix A
    half tileB[TILE_N][TILE_N];         // Tile for matrix B
};

// Simple B2B GEMM kernel with Shared Memory residency
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void b2b_gemm_shmem_kernel(
    half const* __restrict__ A,    // M x K
    half const* __restrict__ B0,   // K x N
    half const* __restrict__ B1,   // N x P
    half* __restrict__ D,          // M x P
    int M, int N, int K, int P
) {
    // Allocate shared memory
    __shared__ SharedMemory<TILE_M, TILE_N> smem;

    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for this thread
    int row = bx * TILE_M + ty;
    int col = by * TILE_N + tx;

    // ========== First GEMM: C = A * B0 ==========
    float accumulator = 0.0f;

    // Loop over tiles in K dimension
    for (int tileIdx = 0; tileIdx < (K + TILE_K - 1) / TILE_K; ++tileIdx) {
        // Collaborative loading of A tile into shared memory
        int aRow = bx * TILE_M + ty;
        int aCol = tileIdx * TILE_K + tx;
        if (aRow < M && aCol < K && ty < TILE_M && tx < TILE_K) {
            smem.tileA[ty][tx] = A[aRow * K + aCol];
        } else {
            smem.tileA[ty][tx] = __float2half(0.0f);
        }

        // Collaborative loading of B0 tile into shared memory
        int bRow = tileIdx * TILE_K + ty;
        int bCol = by * TILE_N + tx;
        if (bRow < K && bCol < N && ty < TILE_K && tx < TILE_N) {
            // Note: B0 is in column major, so we transpose during load
            smem.tileB[ty][tx] = B0[bRow + bCol * K];
        } else {
            smem.tileB[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

        // Compute partial dot product
        if (ty < TILE_M && tx < TILE_N) {
            for (int k = 0; k < TILE_K && k < (K - tileIdx * TILE_K); ++k) {
                accumulator += __half2float(smem.tileA[ty][k]) *
                              __half2float(smem.tileB[k][tx]);
            }
        }

        __syncthreads();
    }

    // ========== Store intermediate result C in shared memory ==========
    if (row < M && col < N && ty < TILE_M && tx < TILE_N) {
        smem.intermediate[ty][tx] = __float2half(accumulator);
    }
    __syncthreads();

    // ========== Second GEMM: D = C * B1 ==========
    // Now smem.intermediate contains the result of first GEMM
    // Use it for second GEMM

    // Reset accumulator for second GEMM
    accumulator = 0.0f;

    // Different column index for second GEMM output
    int colP = by * TILE_N + tx;  // Column in P dimension

    if (colP < P) {
        // Loop over tiles in N dimension
        for (int tileIdx = 0; tileIdx < (N + TILE_K - 1) / TILE_K; ++tileIdx) {
            // Load C tile from shared memory (already there for first tile)
            // For subsequent tiles, we would need to load from global memory
            // This is simplified - showing only the first tile

            if (tileIdx == 0) {
                // C tile is already in smem.intermediate
                // Just need to load corresponding B1 tile
            } else {
                // In full implementation, load next C tile
                // For simplification, we skip this
                break;
            }

            // Load B1 tile into shared memory
            int b1Row = tileIdx * TILE_K + ty;
            int b1Col = colP;
            if (b1Row < N && b1Col < P && ty < TILE_K && tx < 1) {
                smem.tileB[ty][0] = B1[b1Row + b1Col * N];
            } else if (ty < TILE_K) {
                smem.tileB[ty][0] = __float2half(0.0f);
            }

            __syncthreads();

            // Compute partial dot product for second GEMM
            if (ty < TILE_M && tx < 1 && row < M && colP < P) {
                for (int n = 0; n < TILE_N && n < N; ++n) {
                    accumulator += __half2float(smem.intermediate[ty][n]) *
                                  __half2float(smem.tileB[n][0]);
                }
            }

            __syncthreads();
        }
    }

    // Store final result D
    if (row < M && colP < P && ty < TILE_M && tx < 1) {
        D[row * P + colP] = __float2half(accumulator);
    }
}

// Full implementation using CUTLASS-style shared memory B2B GEMM
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void b2b_gemm_shmem_full_kernel(
    half const* __restrict__ A,
    half const* __restrict__ B0,
    half const* __restrict__ B1,
    half* __restrict__ D,
    int M, int N, int K, int P
) {
    // Shared memory for tiles and intermediate results
    extern __shared__ half shared_mem[];

    half* shmem_A = shared_mem;
    half* shmem_B = shmem_A + BLOCK_M * BLOCK_K;
    half* shmem_C = shmem_B + BLOCK_K * BLOCK_N;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    // ========== First GEMM: C = A * B0 ==========

    // Initialize C tile in shared memory
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x * blockDim.y) {
        shmem_C[i] = __float2half(0.0f);
    }
    __syncthreads();

    // Loop over K dimension
    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; ++k_tile) {
        // Load A tile collaboratively
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x * blockDim.y) {
            int local_row = i / BLOCK_K;
            int local_col = i % BLOCK_K;
            int global_row = block_row * BLOCK_M + local_row;
            int global_col = k_tile * BLOCK_K + local_col;

            if (global_row < M && global_col < K) {
                shmem_A[i] = A[global_row * K + global_col];
            } else {
                shmem_A[i] = __float2half(0.0f);
            }
        }

        // Load B0 tile collaboratively
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x * blockDim.y) {
            int local_row = i / BLOCK_N;
            int local_col = i % BLOCK_N;
            int global_row = k_tile * BLOCK_K + local_row;
            int global_col = block_col * BLOCK_N + local_col;

            if (global_row < K && global_col < N) {
                shmem_B[i] = B0[global_row + global_col * K];
            } else {
                shmem_B[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute tile matrix multiply
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x * blockDim.y) {
            int tile_row = i / BLOCK_N;
            int tile_col = i % BLOCK_N;

            float sum = __half2float(shmem_C[i]);
            for (int k = 0; k < BLOCK_K; ++k) {
                sum += __half2float(shmem_A[tile_row * BLOCK_K + k]) *
                       __half2float(shmem_B[k * BLOCK_N + tile_col]);
            }
            shmem_C[i] = __float2half(sum);
        }

        __syncthreads();
    }

    // ========== Shared Memory Residency ==========
    // shmem_C now contains intermediate result C
    // Keep it in shared memory for second GEMM

    // ========== Second GEMM: D = C * B1 ==========

    // Reuse shmem_A for output D
    half* shmem_D = shmem_A;

    // Initialize D tile
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x * blockDim.y) {
        shmem_D[i] = __float2half(0.0f);
    }
    __syncthreads();

    // For second GEMM, block_col now indexes into P dimension
    if (block_col * BLOCK_N < P) {
        // Loop over N dimension
        for (int n_tile = 0; n_tile < (N + BLOCK_K - 1) / BLOCK_K; ++n_tile) {
            // Load B1 tile
            for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x * blockDim.y) {
                int local_row = i / BLOCK_N;
                int local_col = i % BLOCK_N;
                int global_row = n_tile * BLOCK_K + local_row;
                int global_col = block_col * BLOCK_N + local_col;

                if (global_row < N && global_col < P) {
                    shmem_B[i] = B1[global_row + global_col * N];
                } else {
                    shmem_B[i] = __float2half(0.0f);
                }
            }

            __syncthreads();

            // Matrix multiply for second GEMM
            // Note: C is BLOCK_M x N, we need to use appropriate tiles
            for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x * blockDim.y) {
                int tile_row = i / BLOCK_N;
                int tile_col = i % BLOCK_N;

                if (block_col * BLOCK_N + tile_col < P) {
                    float sum = __half2float(shmem_D[i]);

                    // Use only the first BLOCK_K columns of C for this simplified version
                    int k_limit = min(BLOCK_K, N - n_tile * BLOCK_K);
                    for (int k = 0; k < k_limit; ++k) {
                        if (n_tile == 0) {  // Simplified: only use first tile of C
                            sum += __half2float(shmem_C[tile_row * BLOCK_N + k]) *
                                   __half2float(shmem_B[k * BLOCK_N + tile_col]);
                        }
                    }
                    shmem_D[i] = __float2half(sum);
                }
            }

            __syncthreads();
        }

        // Store D tile to global memory
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x * blockDim.y) {
            int tile_row = i / BLOCK_N;
            int tile_col = i % BLOCK_N;
            int global_row = block_row * BLOCK_M + tile_row;
            int global_col = block_col * BLOCK_N + tile_col;

            if (global_row < M && global_col < P) {
                D[global_row * P + global_col] = shmem_D[i];
            }
        }
    }
}

// Simplified launcher for B2B GEMM with Shared Memory residency
class SimplifiedB2bGemmShmem {
public:
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    bool run(int M, int N, int K, int P) {
        std::cout << "\n=== Simplified B2B GEMM with Shared Memory Residency ===\n";
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

        // Initialize tensors
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_A.host_view(), 1, ElementA(1), ElementA(-1), 0);
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_B0.host_view(), 1, ElementB(1), ElementB(-1), 1);
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_B1.host_view(), 1, ElementB(1), ElementB(-1), 2);
        cutlass::reference::host::TensorFill(
            tensor_D.host_view(), ElementD(0));

        // Copy to device
        tensor_A.sync_device();
        tensor_B0.sync_device();
        tensor_B1.sync_device();
        tensor_D.sync_device();

        // Launch configuration
        const int BLOCK_M = 32;
        const int BLOCK_N = 32;
        const int BLOCK_K = 16;

        dim3 gridDim(
            (M + BLOCK_M - 1) / BLOCK_M,
            (std::max(N, P) + BLOCK_N - 1) / BLOCK_N
        );
        dim3 blockDim(16, 16);  // 256 threads per block

        // Calculate shared memory size
        size_t shmem_size = sizeof(half) * (
            BLOCK_M * BLOCK_K +     // A tile
            BLOCK_K * BLOCK_N +     // B tile
            BLOCK_M * BLOCK_N       // C/D tile
        );

        std::cout << "Launching kernel with:\n";
        std::cout << "  Grid: (" << gridDim.x << ", " << gridDim.y << ")\n";
        std::cout << "  Block: (" << blockDim.x << ", " << blockDim.y << ")\n";
        std::cout << "  Shared memory: " << shmem_size << " bytes\n";

        // Set shared memory config
        cudaFuncSetAttribute(
            b2b_gemm_shmem_full_kernel<BLOCK_M, BLOCK_N, BLOCK_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );

        // Launch kernel
        b2b_gemm_shmem_full_kernel<BLOCK_M, BLOCK_N, BLOCK_K>
            <<<gridDim, blockDim, shmem_size>>>(
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

        // Compute reference on CPU
        std::cout << "Computing reference on CPU...\n";

        cutlass::HostTensor<ElementC, LayoutC> tensor_C_ref({M, N});

        // First GEMM
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

        // Second GEMM
        gemm_op(
            {M, P, N},
            float(1),
            tensor_C_ref.host_view(),
            tensor_B1.host_view(),
            float(0),
            tensor_D_ref.host_view()
        );

        // Compare results
        bool passed = true;
        float max_error = 0.0f;
        for (int i = 0; i < M * P; ++i) {
            float diff = std::abs(float(tensor_D.host_data()[i]) -
                                 float(tensor_D_ref.host_data()[i]));
            max_error = std::max(max_error, diff);
            if (diff > 0.1f) {
                passed = false;
            }
        }

        std::cout << "Max error: " << max_error << "\n";

        if (passed) {
            std::cout << "*** PASSED ***\n";
        } else {
            std::cout << "*** FAILED ***\n";

            // Print comparison
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
        std::cerr << "This example requires SM80 or newer for FP16 support\n";
        return -1;
    }

    SimplifiedB2bGemmShmem b2b_gemm;

    // Test with small problem sizes
    bool passed = b2b_gemm.run(64, 32, 48, 32);

    return passed ? 0 : -1;
}