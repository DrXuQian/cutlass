#include <iostream>
#include <cuda_runtime.h>
#include "matrix_utils.h"
#include "cpu_gemm.h"
#include "cuda_timer.h"
#include "cuda_utils.h"

// Simple CUDA kernel for GEMM
__global__ void simple_gemm_kernel(int M, int N, int K,
                                   float alpha, const float* A,
                                   const float* B, float beta, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

int main() {
    // Matrix dimensions
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::cout << "Testing CUDA common utilities" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Print device info
    cuda_utils::printDeviceInfo();

    // Allocate host matrices
    float *h_A = cuda_utils::allocateHostMatrix<float>(M, K);
    float *h_B = cuda_utils::allocateHostMatrix<float>(K, N);
    float *h_C = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_C_ref = cuda_utils::allocateHostMatrix<float>(M, N);

    // Initialize matrices with random values
    cuda_utils::initializeRandomMatrix(h_A, M, K, -1.0f, 1.0f);
    cuda_utils::initializeRandomMatrix(h_B, K, N, -1.0f, 1.0f);
    cuda_utils::initializeConstantMatrix(h_C, M, N, 0.0f);
    cuda_utils::initializeConstantMatrix(h_C_ref, M, N, 0.0f);

    // Print sample of input matrices
    cuda_utils::printMatrix(h_A, M, K, "Matrix A (sample)", 5, 5);
    cuda_utils::printMatrix(h_B, K, N, "Matrix B (sample)", 5, 5);

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_A, M, K));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_B, K, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_C, M, N));

    // Copy matrices to device
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_A, h_A, M, K));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_B, h_B, K, N));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_C, h_C, M, N));

    // Configure kernel launch parameters
    dim3 block_size = cuda_utils::calculate2DBlockSize(N, M);
    dim3 grid_size = cuda_utils::calculate2DGridSize(N, M, block_size);

    std::cout << "\nKernel configuration:" << std::endl;
    std::cout << "  Block size: (" << block_size.x << ", " << block_size.y << ")" << std::endl;
    std::cout << "  Grid size: (" << grid_size.x << ", " << grid_size.y << ")" << std::endl;

    // Create timer
    cuda_utils::CudaTimer timer;

    // Warm up
    simple_gemm_kernel<<<grid_size, block_size>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cuda_utils::syncAndCheck();

    // Benchmark GPU kernel
    const int num_iterations = 10;
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        simple_gemm_kernel<<<grid_size, block_size>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cuda_utils::syncAndCheck();
    timer.stop();

    float gpu_time = timer.getElapsedTime() / num_iterations;
    double gpu_gflops = cuda_utils::computeGFLOPS(M, N, K, gpu_time);

    std::cout << "\nGPU Performance:" << std::endl;
    std::cout << "  Time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << gpu_gflops << " GFLOPS" << std::endl;

    // Copy result back to host
    CHECK_CUDA(cuda_utils::copyDeviceToHost(h_C, d_C, M, N));

    // CPU verification
    std::cout << "\nRunning CPU verification..." << std::endl;
    double cpu_time = cpu_gemm::benchmark_cpu_gemm(M, N, K, alpha, h_A, M,
                                                   h_B, N, beta, h_C_ref, N, 1);

    // Verify results
    bool passed = cpu_gemm::verify_gemm(M, N, K, alpha, h_A, M,
                                        h_B, N, beta, h_C, N, 1e-3f, true);

    if (passed) {
        std::cout << "\nVerification PASSED!" << std::endl;
    } else {
        std::cout << "\nVerification FAILED!" << std::endl;
    }

    // Print speedup
    double speedup = cpu_time / gpu_time;
    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

    // Print memory usage
    std::cout << "\nMemory usage:" << std::endl;
    cuda_utils::printMemoryUsage();

    // Clean up
    cuda_utils::freeHostMatrix(h_A);
    cuda_utils::freeHostMatrix(h_B);
    cuda_utils::freeHostMatrix(h_C);
    cuda_utils::freeHostMatrix(h_C_ref);
    cuda_utils::freeDeviceMatrix(d_A);
    cuda_utils::freeDeviceMatrix(d_B);
    cuda_utils::freeDeviceMatrix(d_C);

    return passed ? 0 : 1;
}