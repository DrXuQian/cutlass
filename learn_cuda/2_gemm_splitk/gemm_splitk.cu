#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/reduction/device/reduce_split_k.h>
#include <cutlass/reduction/thread/reduction_operators.h>

#include "../common/matrix_utils.h"
#include "../common/cpu_gemm.h"
#include "../common/cuda_timer.h"
#include "../common/cuda_utils.h"

// Helper function to print Split-K explanation
void printSplitKExplanation() {
    std::cout << "\n=== Split-K GEMM Explanation ===" << std::endl;
    std::cout << "Split-K divides the K dimension into multiple chunks:" << std::endl;
    std::cout << "- Regular GEMM: C = A(M×K) × B(K×N)" << std::endl;
    std::cout << "- Split-K GEMM: K is split into P parts" << std::endl;
    std::cout << "  - Each part computes: C_p = A(M×K/P) × B(K/P×N)" << std::endl;
    std::cout << "  - Final result: C = sum(C_p) for p=0 to P-1" << std::endl;
    std::cout << "Benefits:" << std::endl;
    std::cout << "  - Better parallelism for large K" << std::endl;
    std::cout << "  - Improved load balancing" << std::endl;
    std::cout << "  - Reduced shared memory pressure" << std::endl;
    std::cout << "================================\n" << std::endl;
}

int main(int argc, char** argv) {
    // Matrix dimensions - using large K for Split-K benefit
    int M = 1024;
    int N = 1024;
    int K = 8192;  // Large K dimension
    float alpha = 1.0f;
    float beta = 0.0f;

    // Split-K configuration
    int split_k_slices = 8;  // Number of K partitions

    // Allow command line arguments
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        split_k_slices = std::atoi(argv[4]);
    }

    std::cout << "CUTLASS Split-K GEMM Example" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Split-K slices: " << split_k_slices << std::endl;
    std::cout << "Each slice processes K/" << split_k_slices << " = "
              << K/split_k_slices << " elements" << std::endl;

    printSplitKExplanation();

    // Print device info
    cuda_utils::printDeviceInfo();

    // Allocate host memory
    float *h_A = cuda_utils::allocateHostMatrix<float>(M, K);
    float *h_B = cuda_utils::allocateHostMatrix<float>(K, N);
    float *h_C = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D_regular = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D_splitk = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D_ref = cuda_utils::allocateHostMatrix<float>(M, N);

    // Initialize matrices
    cuda_utils::initializeRandomMatrix(h_A, M, K, -1.0f, 1.0f, 42);
    cuda_utils::initializeRandomMatrix(h_B, K, N, -1.0f, 1.0f, 43);
    cuda_utils::initializeConstantMatrix(h_C, M, N, 0.0f);

    // Copy for reference
    for (int i = 0; i < M * N; ++i) {
        h_D_ref[i] = h_C[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D_regular, *d_D_splitk;
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_A, M, K));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_B, K, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_C, M, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_D_regular, M, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_D_splitk, M, N));

    // Copy to device
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_A, h_A, M, K));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_B, h_B, K, N));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_C, h_C, M, N));

    // Define CUTLASS GEMM types
    using RowMajor = cutlass::layout::RowMajor;

    // Regular GEMM (no Split-K)
    using CutlassGemmRegular = cutlass::gemm::device::Gemm<
        float, RowMajor,
        float, RowMajor,
        float, RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 8>,   // ThreadblockShape
        cutlass::gemm::GemmShape<32, 64, 8>,     // WarpShape
        cutlass::gemm::GemmShape<1, 1, 1>,       // InstructionShape
        cutlass::epilogue::thread::LinearCombination<
            float, 1, float, float
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3  // Stages
    >;

    // Split-K GEMM
    using CutlassGemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
        float, RowMajor,
        float, RowMajor,
        float, RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 8>,   // ThreadblockShape - K must be 8 for Sm80
        cutlass::gemm::GemmShape<32, 64, 8>,     // WarpShape
        cutlass::gemm::GemmShape<1, 1, 1>,       // InstructionShape
        cutlass::epilogue::thread::LinearCombination<
            float, 1, float, float
        >
    >;

    cuda_utils::CudaTimer timer;
    const int num_iterations = 10;

    // ========================================
    // 1. Regular GEMM (baseline)
    // ========================================
    std::cout << "\n1. Regular GEMM (baseline):" << std::endl;

    CutlassGemmRegular gemm_regular;
    CutlassGemmRegular::Arguments args_regular(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D_regular, N},
        {alpha, beta}
    );

    // Warmup
    cutlass::Status status = gemm_regular(args_regular);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Regular GEMM failed!" << std::endl;
        return -1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        status = gemm_regular(args_regular);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float regular_time = timer.getElapsedTime() / num_iterations;
    double regular_gflops = cuda_utils::computeGFLOPS(M, N, K, regular_time);

    std::cout << "  Time: " << regular_time << " ms" << std::endl;
    std::cout << "  Performance: " << regular_gflops << " GFLOPS" << std::endl;

    // ========================================
    // 2. Split-K GEMM
    // ========================================
    std::cout << "\n2. Split-K GEMM (" << split_k_slices << " slices):" << std::endl;

    CutlassGemmSplitK gemm_splitk;
    CutlassGemmSplitK::Arguments args_splitk(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D_splitk, N},
        {alpha, beta},
        split_k_slices  // Split-K slices
    );

    // Warmup
    status = gemm_splitk(args_splitk);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Split-K GEMM failed!" << std::endl;
        return -1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    timer.reset();
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        status = gemm_splitk(args_splitk);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float splitk_time = timer.getElapsedTime() / num_iterations;
    double splitk_gflops = cuda_utils::computeGFLOPS(M, N, K, splitk_time);

    std::cout << "  Time: " << splitk_time << " ms" << std::endl;
    std::cout << "  Performance: " << splitk_gflops << " GFLOPS" << std::endl;
    std::cout << "  Speedup over regular: " << regular_time / splitk_time << "x" << std::endl;

    // ========================================
    // 3. Try different Split-K configurations
    // ========================================
    std::cout << "\n3. Performance with different Split-K slices:" << std::endl;
    std::cout << "Slices | Time (ms) | GFLOPS | Speedup" << std::endl;
    std::cout << "-------|-----------|--------|--------" << std::endl;

    std::vector<int> slice_counts = {1, 2, 4, 8, 16, 32};
    float best_time = regular_time;
    int best_slices = 1;

    for (int slices : slice_counts) {
        if (K % slices != 0) continue;  // Skip if K is not divisible

        CutlassGemmSplitK::Arguments args_test(
            {M, N, K},
            {d_A, K}, {d_B, N}, {d_C, N}, {d_D_splitk, N},
            {alpha, beta},
            slices
        );

        // Warmup
        gemm_splitk(args_test);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark
        timer.reset();
        timer.start();
        for (int i = 0; i < num_iterations; ++i) {
            gemm_splitk(args_test);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        float time = timer.getElapsedTime() / num_iterations;
        double gflops = cuda_utils::computeGFLOPS(M, N, K, time);
        float speedup = regular_time / time;

        std::cout << std::setw(6) << slices << " | "
                  << std::setw(9) << std::setprecision(4) << time << " | "
                  << std::setw(6) << std::setprecision(0) << gflops << " | "
                  << std::setw(6) << std::setprecision(3) << speedup << "x" << std::endl;

        if (time < best_time) {
            best_time = time;
            best_slices = slices;
        }
    }

    std::cout << "\nBest configuration: " << best_slices << " slices" << std::endl;

    // ========================================
    // 4. CPU Reference
    // ========================================
    std::cout << "\n4. CPU Reference:" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm::gemm_cpu(M, N, K, alpha, h_A, K, h_B, N, beta, h_D_ref, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;
    double cpu_time = cpu_diff.count() * 1000.0;
    double cpu_gflops = cuda_utils::computeGFLOPS(M, N, K, cpu_time);

    std::cout << "  Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << cpu_gflops << " GFLOPS" << std::endl;

    // ========================================
    // 5. Verification
    // ========================================
    std::cout << "\n=== Verification ===" << std::endl;

    // Copy results back
    CHECK_CUDA(cuda_utils::copyDeviceToHost(h_D_regular, d_D_regular, M, N));
    CHECK_CUDA(cuda_utils::copyDeviceToHost(h_D_splitk, d_D_splitk, M, N));

    bool regular_passed = cuda_utils::compareMatrices(h_D_regular, h_D_ref, M, N, 1e-3f, false);
    bool splitk_passed = cuda_utils::compareMatrices(h_D_splitk, h_D_ref, M, N, 1e-3f, false);

    std::cout << "Regular GEMM: " << (regular_passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    std::cout << "Split-K GEMM: " << (splitk_passed ? "✓ PASSED" : "✗ FAILED") << std::endl;

    // ========================================
    // 6. Performance Summary
    // ========================================
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Configuration          | Time (ms) | GFLOPS  | vs CPU" << std::endl;
    std::cout << "-----------------------|-----------|---------|--------" << std::endl;
    std::cout << "CPU Reference          | " << std::setw(9) << cpu_time
              << " | " << std::setw(7) << std::setprecision(1) << std::fixed << cpu_gflops
              << " | 1.00x" << std::endl;
    std::cout << "Regular GEMM           | " << std::setw(9) << std::setprecision(3) << regular_time
              << " | " << std::setw(7) << std::setprecision(1) << regular_gflops
              << " | " << std::setprecision(1) << cpu_time/regular_time << "x" << std::endl;
    std::cout << "Split-K GEMM (best)    | " << std::setw(9) << std::setprecision(3) << best_time
              << " | " << std::setw(7) << std::setprecision(1) << cuda_utils::computeGFLOPS(M, N, K, best_time)
              << " | " << std::setprecision(1) << cpu_time/best_time << "x" << std::endl;

    // Split-K analysis
    std::cout << "\n=== Split-K Analysis ===" << std::endl;
    std::cout << "K dimension: " << K << std::endl;
    std::cout << "Optimal slices: " << best_slices << std::endl;
    std::cout << "Elements per slice: " << K/best_slices << std::endl;
    std::cout << "Split-K speedup: " << std::setprecision(2) << regular_time/best_time << "x over regular GEMM" << std::endl;

    if (best_slices > 1) {
        std::cout << "\n✓ Split-K provides performance benefit for this configuration!" << std::endl;
        std::cout << "  Large K dimension (" << K << ") benefits from parallel reduction." << std::endl;
    } else {
        std::cout << "\n✗ Regular GEMM is optimal for this configuration." << std::endl;
        std::cout << "  Consider increasing K dimension for Split-K benefits." << std::endl;
    }

    // Cleanup
    cuda_utils::freeHostMatrix(h_A);
    cuda_utils::freeHostMatrix(h_B);
    cuda_utils::freeHostMatrix(h_C);
    cuda_utils::freeHostMatrix(h_D_regular);
    cuda_utils::freeHostMatrix(h_D_splitk);
    cuda_utils::freeHostMatrix(h_D_ref);
    cuda_utils::freeDeviceMatrix(d_A);
    cuda_utils::freeDeviceMatrix(d_B);
    cuda_utils::freeDeviceMatrix(d_C);
    cuda_utils::freeDeviceMatrix(d_D_regular);
    cuda_utils::freeDeviceMatrix(d_D_splitk);

    return (regular_passed && splitk_passed) ? 0 : -1;
}