#include <iostream>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include "../common/matrix_utils.h"
#include "../common/cpu_gemm.h"
#include "../common/cuda_timer.h"
#include "../common/cuda_utils.h"

// Simple kernel demonstrating Split-K concept
__global__ void manual_splitk_gemm(
    int M, int N, int K, int K_split,
    float alpha, const float* A, const float* B,
    float beta, float* C, float* workspace) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int split_id = blockIdx.z;  // Which K slice this block handles

    if (row < M && col < N && split_id < K_split) {
        // Calculate K range for this split
        int k_per_split = K / K_split;
        int k_start = split_id * k_per_split;
        int k_end = (split_id == K_split - 1) ? K : k_start + k_per_split;

        // Compute partial sum for this K slice
        float sum = 0.0f;
        for (int k = k_start; k < k_end; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Store partial result in workspace
        workspace[split_id * M * N + row * N + col] = sum;
    }
}

// Reduction kernel to sum partial results
__global__ void reduce_splitk_results(
    int M, int N, int K_split,
    float alpha, float beta,
    const float* workspace, const float* C_in, float* C_out) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Sum all K-split partial results
        for (int split = 0; split < K_split; ++split) {
            sum += workspace[split * M * N + row * N + col];
        }

        // Apply alpha/beta and write final result
        C_out[row * N + col] = alpha * sum + beta * C_in[row * N + col];
    }
}

int main() {
    // Matrix dimensions
    const int M = 512;
    const int N = 512;
    const int K = 4096;  // Large K for Split-K benefit
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Split-K configurations to test
    std::vector<int> split_configs = {1, 2, 4, 8};

    std::cout << "Split-K GEMM Demonstration" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "\n=== Split-K Concept ===" << std::endl;
    std::cout << "Instead of each thread block computing the full K reduction," << std::endl;
    std::cout << "Split-K divides K into chunks processed by different blocks," << std::endl;
    std::cout << "then reduces the partial results.\n" << std::endl;

    // Allocate host memory
    float *h_A = cuda_utils::allocateHostMatrix<float>(M, K);
    float *h_B = cuda_utils::allocateHostMatrix<float>(K, N);
    float *h_C = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D_ref = cuda_utils::allocateHostMatrix<float>(M, N);

    // Initialize matrices
    cuda_utils::initializeRandomMatrix(h_A, M, K, -1.0f, 1.0f, 42);
    cuda_utils::initializeRandomMatrix(h_B, K, N, -1.0f, 1.0f, 43);
    cuda_utils::initializeConstantMatrix(h_C, M, N, 0.0f);
    cuda_utils::initializeConstantMatrix(h_D_ref, M, N, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_A, M, K));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_B, K, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_C, M, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_D, M, N));

    // Copy to device
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_A, h_A, M, K));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_B, h_B, K, N));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_C, h_C, M, N));

    // CPU reference
    cpu_gemm::gemm_cpu(M, N, K, alpha, h_A, K, h_B, N, beta, h_D_ref, N);

    // Test different Split-K configurations
    cuda_utils::CudaTimer timer;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    std::cout << "K_split | Time (ms) | Speedup | Verified" << std::endl;
    std::cout << "--------|-----------|---------|----------" << std::endl;

    float baseline_time = 0;

    for (int K_split : split_configs) {
        // Allocate workspace for partial results
        float* d_workspace;
        CHECK_CUDA(cudaMalloc(&d_workspace, sizeof(float) * K_split * M * N));

        // Set grid for Split-K (using Z dimension for K splits)
        dim3 splitk_grid(grid.x, grid.y, K_split);

        // Warmup
        manual_splitk_gemm<<<splitk_grid, block>>>(
            M, N, K, K_split, alpha, d_A, d_B, beta, d_C, d_workspace);
        reduce_splitk_results<<<grid, block>>>(
            M, N, K_split, alpha, beta, d_workspace, d_C, d_D);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark
        const int iterations = 100;
        timer.reset();
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            manual_splitk_gemm<<<splitk_grid, block>>>(
                M, N, K, K_split, alpha, d_A, d_B, beta, d_C, d_workspace);
            reduce_splitk_results<<<grid, block>>>(
                M, N, K_split, alpha, beta, d_workspace, d_C, d_D);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        float time = timer.getElapsedTime() / iterations;
        if (K_split == 1) baseline_time = time;
        float speedup = baseline_time / time;

        // Verify
        CHECK_CUDA(cuda_utils::copyDeviceToHost(h_D, d_D, M, N));
        bool passed = cuda_utils::compareMatrices(h_D, h_D_ref, M, N, 1e-3f, false);

        std::cout << std::setw(7) << K_split << " | "
                  << std::setw(9) << std::setprecision(4) << time << " | "
                  << std::setw(7) << std::setprecision(3) << speedup << "x | "
                  << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;

        cudaFree(d_workspace);
    }

    // CUTLASS comparison
    std::cout << "\n=== CUTLASS GEMM (for comparison) ===" << std::endl;

    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, RowMajor,
        float, RowMajor,
        float, RowMajor
    >;

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {M, N, K},
        {d_A, K}, {d_B, N}, {d_C, N}, {d_D, N},
        {alpha, beta}
    );

    // Warmup
    gemm_op(args);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    timer.reset();
    timer.start();
    for (int i = 0; i < 100; ++i) {
        gemm_op(args);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float cutlass_time = timer.getElapsedTime() / 100;
    double cutlass_gflops = cuda_utils::computeGFLOPS(M, N, K, cutlass_time);

    std::cout << "CUTLASS GEMM: " << cutlass_time << " ms ("
              << cutlass_gflops << " GFLOPS)" << std::endl;

    // Analysis
    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "1. Split-K reduces memory pressure by processing smaller K chunks" << std::endl;
    std::cout << "2. Multiple blocks can work on the same output tile" << std::endl;
    std::cout << "3. Reduction overhead must be considered" << std::endl;
    std::cout << "4. Most beneficial when K >> M, N" << std::endl;

    std::cout << "\nKey insights:" << std::endl;
    std::cout << "- K=" << K << " is " << (K > M ? "larger" : "smaller") << " than M=" << M << std::endl;
    std::cout << "- Each K-split processes " << K/8 << " elements (for K_split=8)" << std::endl;
    std::cout << "- Workspace size: " << 8 * M * N * sizeof(float) / (1024.0 * 1024.0)
              << " MB for K_split=8" << std::endl;

    // Cleanup
    cuda_utils::freeHostMatrix(h_A);
    cuda_utils::freeHostMatrix(h_B);
    cuda_utils::freeHostMatrix(h_C);
    cuda_utils::freeHostMatrix(h_D);
    cuda_utils::freeHostMatrix(h_D_ref);
    cuda_utils::freeDeviceMatrix(d_A);
    cuda_utils::freeDeviceMatrix(d_B);
    cuda_utils::freeDeviceMatrix(d_C);
    cuda_utils::freeDeviceMatrix(d_D);

    return 0;
}