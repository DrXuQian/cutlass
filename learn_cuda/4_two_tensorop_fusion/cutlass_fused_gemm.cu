#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    const int M = 256, N = 256, K = 256;

    // Define CUTLASS GEMM with ReLU fusion using TensorCore
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    // GEMM with fused ReLU epilogue
    using GemmReLU = cutlass::gemm::device::Gemm<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombinationRelu<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2
    >;

    // Allocate host matrices
    std::vector<cutlass::half_t> h_A(M * K);
    std::vector<cutlass::half_t> h_B(K * N);
    std::vector<cutlass::half_t> h_C(M * N, cutlass::half_t(0));
    std::vector<cutlass::half_t> h_D(M * N, cutlass::half_t(0));

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = cutlass::half_t(2.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = cutlass::half_t(1.0f);
    }

    // Allocate device memory
    cutlass::half_t *d_A, *d_B, *d_C, *d_D;
    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B = K * N * sizeof(cutlass::half_t);
    size_t size_C = M * N * sizeof(cutlass::half_t);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMalloc(&d_D, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    // Create GEMM+ReLU instance
    GemmReLU gemm_relu_op;

    // Configure arguments
    GemmReLU::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D, N},
        {ElementCompute(1.0f), ElementCompute(0.0f)}
    );

    // Run GEMM+ReLU
    cutlass::Status status = gemm_relu_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM+ReLU execution failed" << std::endl;
        return -1;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, size_C, cudaMemcpyDeviceToHost));

    // Verify result (should be max(2*K, 0) = 512)
    bool correct = true;
    float expected = 2.0f * K;
    for (int i = 0; i < 10; ++i) {
        float val = float(h_D[i]);
        if (std::abs(val - expected) > 1.0f) {
            correct = false;
            break;
        }
    }

    std::cout << "CUTLASS TensorCore GEMM+ReLU: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Expected: " << expected << ", Got: " << float(h_D[0]) << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));

    return 0;
}