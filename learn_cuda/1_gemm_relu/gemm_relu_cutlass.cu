#include <iostream>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>

#include "../common/matrix_utils.h"
#include "../common/cpu_gemm.h"
#include "../common/cuda_timer.h"
#include "../common/cuda_utils.h"

// CPU reference for GEMM + ReLU
void cpu_gemm_relu_ref(int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc) {
    cpu_gemm::gemm_cpu(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    for (int i = 0; i < M * N; ++i) {
        C[i] = fmaxf(0.0f, C[i]);
    }
}

int main() {
    // Matrix dimensions
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::cout << "CUTLASS GEMM + ReLU Fusion Example" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Operation: C = ReLU(alpha * A * B + beta * C)" << std::endl;
    std::cout << "alpha=" << alpha << ", beta=" << beta << "\n" << std::endl;

    // Allocate host memory
    float *h_A = cuda_utils::allocateHostMatrix<float>(M, K);
    float *h_B = cuda_utils::allocateHostMatrix<float>(K, N);
    float *h_C = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D_ref = cuda_utils::allocateHostMatrix<float>(M, N);

    // Initialize matrices
    cuda_utils::initializeRandomMatrix(h_A, M, K, -2.0f, 2.0f, 42);
    cuda_utils::initializeRandomMatrix(h_B, K, N, -2.0f, 2.0f, 43);
    cuda_utils::initializeConstantMatrix(h_C, M, N, 0.0f);

    // Copy C for reference
    for (int i = 0; i < M * N; ++i) {
        h_D_ref[i] = h_C[i];
    }

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

    // Define CUTLASS GEMM with ReLU epilogue
    using RowMajor = cutlass::layout::RowMajor;

    // GEMM with ReLU fusion in epilogue
    using CutlassGemmRelu = cutlass::gemm::device::Gemm<
        float,                                          // ElementA
        RowMajor,                                       // LayoutA
        float,                                          // ElementB
        RowMajor,                                       // LayoutB
        float,                                          // ElementC
        RowMajor,                                       // LayoutC
        float,                                          // ElementAccumulator
        cutlass::arch::OpClassSimt,                    // OpClass
        cutlass::arch::Sm80,                           // ArchTag
        cutlass::gemm::GemmShape<128, 128, 8>,        // ThreadblockShape
        cutlass::gemm::GemmShape<32, 64, 8>,          // WarpShape
        cutlass::gemm::GemmShape<1, 1, 1>,            // InstructionShape
        cutlass::epilogue::thread::LinearCombinationRelu< // Epilogue with ReLU
            float,                                      // ElementOutput
            1,                                          // ElementsPerAccess
            float,                                      // ElementAccumulator
            float                                       // ElementCompute
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
        3                                              // Stages
    >;

    CutlassGemmRelu gemm_relu_op;

    // Setup arguments
    CutlassGemmRelu::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D, N},
        {alpha, beta}
    );

    // Execute CUTLASS GEMM + ReLU
    std::cout << "Running CUTLASS GEMM with Fused ReLU..." << std::endl;

    cuda_utils::CudaTimer timer;
    const int num_iterations = 10;

    // Warmup
    cutlass::Status status = gemm_relu_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM+ReLU failed!" << std::endl;
        return -1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        status = gemm_relu_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM+ReLU failed!" << std::endl;
            return -1;
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float gpu_time = timer.getElapsedTime() / num_iterations;
    double gpu_gflops = cuda_utils::computeGFLOPS(M, N, K, gpu_time);

    std::cout << "CUTLASS Performance:" << std::endl;
    std::cout << "  Time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << gpu_gflops << " GFLOPS" << std::endl;

    // Copy result back
    CHECK_CUDA(cuda_utils::copyDeviceToHost(h_D, d_D, M, N));

    // CPU reference
    std::cout << "\nRunning CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm_relu_ref(M, N, K, alpha, h_A, K, h_B, N, beta, h_D_ref, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;
    double cpu_time = cpu_diff.count() * 1000.0;
    double cpu_gflops = cuda_utils::computeGFLOPS(M, N, K, cpu_time);

    std::cout << "CPU Performance:" << std::endl;
    std::cout << "  Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << cpu_gflops << " GFLOPS" << std::endl;

    // Verification
    std::cout << "\nVerifying results..." << std::endl;
    bool passed = cuda_utils::compareMatrices(h_D, h_D_ref, M, N, 1e-3f, true);

    if (passed) {
        std::cout << "✓ Verification PASSED!" << std::endl;
    } else {
        std::cout << "✗ Verification FAILED!" << std::endl;
    }

    // Count ReLU activations
    int relu_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (h_D_ref[i] > 0) relu_count++;
    }

    std::cout << "\nReLU activation rate: " << (100.0f * relu_count / (M * N))
              << "% (" << relu_count << "/" << (M * N) << " values > 0)" << std::endl;

    std::cout << "\nSpeedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Print output sample
    cuda_utils::printMatrix(h_D, M, N, "\nOutput Matrix (after ReLU, sample)", 5, 5);

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

    return passed ? 0 : -1;
}