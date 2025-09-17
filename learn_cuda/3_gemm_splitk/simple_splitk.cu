#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    const int M = 512, N = 512, K = 2048;
    const int split_k_slices = 4;

    // Define CUTLASS GEMM with Split-K and TensorCore
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using Gemm = cutlass::gemm::device::GemmSplitKParallel<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
        >
    >;

    // Allocate host matrices
    std::vector<cutlass::half_t> h_A(M * K);
    std::vector<cutlass::half_t> h_B(K * N);
    std::vector<cutlass::half_t> h_C(M * N, cutlass::half_t(0));
    std::vector<cutlass::half_t> h_D(M * N, cutlass::half_t(0));

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = cutlass::half_t(1.0f);
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

    // Create GEMM instance
    Gemm gemm_op;

    // Configure problem and arguments
    Gemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D, N},
        {ElementCompute(1.0f), ElementCompute(0.0f)},
        split_k_slices
    );

    // Query workspace size
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    // Initialize and run
    cutlass::Status status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM initialization failed" << std::endl;
        return -1;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM execution failed" << std::endl;
        return -1;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, size_C, cudaMemcpyDeviceToHost));

    // Verify result
    bool correct = true;
    float expected = float(K);
    for (int i = 0; i < M * N; ++i) {
        float val = float(h_D[i]);
        if (std::abs(val - expected) > 1.0f) {
            correct = false;
            break;
        }
    }

    std::cout << "CUTLASS Split-K TensorCore GEMM: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Expected: " << expected << ", Got: " << float(h_D[0]) << std::endl;
    std::cout << "Split-K slices: " << split_k_slices << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    if (workspace) {
        CHECK_CUDA(cudaFree(workspace));
    }

    return 0;
}