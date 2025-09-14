#include <iostream>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/activation.h>

#include "../common/matrix_utils.h"
#include "../common/cpu_gemm.h"
#include "../common/cuda_timer.h"
#include "../common/cuda_utils.h"

// 自定义 ReLU 激活函数
template <typename T>
struct CustomReLU {
    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return value > T(0) ? value : T(0);
    }
};

// 自定义 Epilogue Functor
template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = ElementOutput_
>
class CustomLinearCombinationRelu {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementC = ElementOutput_;

    static int const kCount = Count;

    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
    using FragmentC = cutlass::Array<ElementC, kCount>;

    struct Params {
        ElementCompute alpha;
        ElementCompute beta;

        CUTLASS_HOST_DEVICE
        Params() : alpha(1), beta(0) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha, ElementCompute beta)
            : alpha(alpha), beta(beta) {}
    };

private:
    ElementCompute alpha_;
    ElementCompute beta_;

public:
    CUTLASS_HOST_DEVICE
    CustomLinearCombinationRelu(Params const &params)
        : alpha_(params.alpha), beta_(params.beta) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const {
        return beta_ != ElementCompute(0);
    }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count) {}

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const &accumulator,
        FragmentC const &source) const {

        FragmentOutput output;
        CustomReLU<ElementCompute> relu_op;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            // 计算 alpha * accumulator + beta * source
            ElementCompute compute_result =
                alpha_ * ElementCompute(accumulator[i]) +
                beta_ * ElementCompute(source[i]);

            // 应用 ReLU: max(0, x)
            output[i] = ElementOutput(relu_op(compute_result));
        }

        return output;
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
        FragmentOutput output;
        CustomReLU<ElementCompute> relu_op;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            // 计算 alpha * accumulator
            ElementCompute compute_result = alpha_ * ElementCompute(accumulator[i]);

            // 应用 ReLU: max(0, x)
            output[i] = ElementOutput(relu_op(compute_result));
        }

        return output;
    }
};

// CPU reference
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
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::cout << "CUTLASS GEMM with Custom ReLU Epilogue" << std::endl;
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

    // Define CUTLASS GEMM with custom epilogue
    using RowMajor = cutlass::layout::RowMajor;

    // 使用自定义的 epilogue functor
    using CutlassGemmCustom = cutlass::gemm::device::Gemm<
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
        CustomLinearCombinationRelu<                   // 自定义 Epilogue
            float,                                      // ElementOutput
            1,                                          // ElementsPerAccess
            float,                                      // ElementAccumulator
            float                                       // ElementCompute
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3                                              // Stages
    >;

    CutlassGemmCustom gemm_op;

    // Setup arguments
    CutlassGemmCustom::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D, N},
        {alpha, beta}
    );

    std::cout << "Running CUTLASS GEMM with Custom ReLU Epilogue..." << std::endl;

    cuda_utils::CudaTimer timer;
    const int num_iterations = 10;

    // Warmup
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed! Error: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed!" << std::endl;
            return -1;
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float gpu_time = timer.getElapsedTime() / num_iterations;
    double gpu_gflops = cuda_utils::computeGFLOPS(M, N, K, gpu_time);

    std::cout << "CUTLASS with Custom Epilogue Performance:" << std::endl;
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

    std::cout << "CPU Performance:" << std::endl;
    std::cout << "  Time: " << cpu_time << " ms" << std::endl;

    // Verification
    std::cout << "\nVerifying results..." << std::endl;
    bool passed = cuda_utils::compareMatrices(h_D, h_D_ref, M, N, 1e-3f, true);

    if (passed) {
        std::cout << "✓ Verification PASSED!" << std::endl;
    } else {
        std::cout << "✗ Verification FAILED!" << std::endl;
    }

    // Show custom epilogue details
    std::cout << "\n=== Custom Epilogue Implementation ===" << std::endl;
    std::cout << "自定义 Epilogue 实现了以下操作：" << std::endl;
    std::cout << "1. 计算线性组合: result = alpha * accumulator + beta * source" << std::endl;
    std::cout << "2. 应用 ReLU 激活: output = max(0, result)" << std::endl;
    std::cout << "3. 所有操作在一个融合的 epilogue 中完成，避免额外内存访问" << std::endl;

    // Count ReLU activations
    int relu_count = 0;
    int zero_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (h_D_ref[i] > 0) {
            relu_count++;
        } else {
            zero_count++;
        }
    }

    std::cout << "\nReLU Statistics:" << std::endl;
    std::cout << "  Active values (> 0): " << relu_count << " ("
              << (100.0f * relu_count / (M * N)) << "%)" << std::endl;
    std::cout << "  Zeroed values (≤ 0): " << zero_count << " ("
              << (100.0f * zero_count / (M * N)) << "%)" << std::endl;

    std::cout << "\nSpeedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Print sample output
    std::cout << "\nOutput sample (first 5x5):" << std::endl;
    for (int i = 0; i < std::min(5, M); ++i) {
        for (int j = 0; j < std::min(5, N); ++j) {
            std::cout << std::setw(8) << std::setprecision(2) << h_D[i * N + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

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