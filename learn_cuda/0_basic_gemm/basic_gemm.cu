#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include "../common/matrix_utils.h"
#include "../common/cpu_gemm.h"
#include "../common/cuda_timer.h"
#include "../common/cuda_utils.h"

int main() {
    // 定义矩阵尺寸
    int M = 512, N = 512, K = 512;
    float alpha = 1.0f, beta = 0.0f;

    std::cout << "CUTLASS Basic GEMM Example" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "alpha=" << alpha << ", beta=" << beta << std::endl;

    // 打印设备信息
    cuda_utils::printDeviceInfo();

    // 分配主机内存
    float *h_A = cuda_utils::allocateHostMatrix<float>(M, K);
    float *h_B = cuda_utils::allocateHostMatrix<float>(K, N);
    float *h_C = cuda_utils::allocateHostMatrix<float>(M, N);
    float *h_D = cuda_utils::allocateHostMatrix<float>(M, N);  // CUTLASS output
    float *h_D_ref = cuda_utils::allocateHostMatrix<float>(M, N);  // CPU reference

    // 初始化矩阵
    cuda_utils::initializeRandomMatrix(h_A, M, K, -1.0f, 1.0f, 42);
    cuda_utils::initializeRandomMatrix(h_B, K, N, -1.0f, 1.0f, 43);
    cuda_utils::initializeRandomMatrix(h_C, M, N, -1.0f, 1.0f, 44);

    // 复制C到D_ref用于CPU计算
    for (int i = 0; i < M * N; ++i) {
        h_D_ref[i] = h_C[i];
    }

    // 打印输入矩阵样本
    cuda_utils::printMatrix(h_A, M, K, "Matrix A (sample)", 5, 5);
    cuda_utils::printMatrix(h_B, K, N, "Matrix B (sample)", 5, 5);
    cuda_utils::printMatrix(h_C, M, N, "Matrix C (sample)", 5, 5);

    // 分配设备内存
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_A, M, K));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_B, K, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_C, M, N));
    CHECK_CUDA(cuda_utils::allocateDeviceMatrix(&d_D, M, N));

    // 复制数据到设备
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_A, h_A, M, K));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_B, h_B, K, N));
    CHECK_CUDA(cuda_utils::copyHostToDevice(d_C, h_C, M, N));

    // 使用列主序布局定义GEMM类型
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;

    // 注意：我们的数据是行主序，所以需要使用RowMajor布局
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float,        // A矩阵数据类型
        RowMajor,     // A矩阵布局
        float,        // B矩阵数据类型
        RowMajor,     // B矩阵布局
        float,        // C矩阵数据类型
        RowMajor      // C矩阵布局
    >;

    // 创建GEMM操作符
    CutlassGemm gemm_operator;

    // 构造CUTLASS GEMM参数对象
    CutlassGemm::Arguments args(
        {M, N, K},          // GEMM问题维度
        {d_A, K},           // A矩阵张量引用 (pointer, leading dimension)
        {d_B, N},           // B矩阵张量引用
        {d_C, N},           // C矩阵张量引用
        {d_D, N},           // 目标矩阵D
        {alpha, beta}       // 标量参数
    );

    // 创建计时器
    cuda_utils::CudaTimer timer;

    // 预热
    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed during warmup!" << std::endl;
        return -1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 基准测试
    const int num_iterations = 10;
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        status = gemm_operator(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed!" << std::endl;
            return -1;
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float gpu_time = timer.getElapsedTime() / num_iterations;
    double gpu_gflops = cuda_utils::computeGFLOPS(M, N, K, gpu_time);

    std::cout << "\nCUTLASS Performance:" << std::endl;
    std::cout << "  Time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << gpu_gflops << " GFLOPS" << std::endl;

    // 复制结果回主机
    CHECK_CUDA(cuda_utils::copyDeviceToHost(h_D, d_D, M, N));

    // CPU验证
    std::cout << "\nRunning CPU verification..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm::gemm_cpu(M, N, K, alpha, h_A, K, h_B, N, beta, h_D_ref, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;
    double cpu_time = cpu_diff.count() * 1000.0;  // Convert to ms
    double cpu_gflops = cuda_utils::computeGFLOPS(M, N, K, cpu_time);

    std::cout << "CPU Performance:" << std::endl;
    std::cout << "  Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << cpu_gflops << " GFLOPS" << std::endl;

    // 验证结果
    bool passed = cuda_utils::compareMatrices(h_D, h_D_ref, M, N, 1e-3f, true);

    if (passed) {
        std::cout << "\n✓ Verification PASSED!" << std::endl;
    } else {
        std::cout << "\n✗ Verification FAILED!" << std::endl;
    }

    // 打印加速比
    double speedup = cpu_time / gpu_time;
    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

    // 打印输出矩阵样本
    cuda_utils::printMatrix(h_D, M, N, "Matrix D = alpha*A*B + beta*C (sample)", 5, 5);

    // 打印内存使用情况
    std::cout << "\nMemory usage:" << std::endl;
    cuda_utils::printMemoryUsage();

    // 清理内存
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