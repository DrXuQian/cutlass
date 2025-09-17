#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/reduction/device/reduce_split_k.h>
#include <cutlass/reduction/thread/reduction_operators.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Custom epilogue for GEMM with reduction
template <typename ElementC>
__global__ void reduceRows(
    ElementC const* matrix,
    ElementC* row_sums,
    int M, int N) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        ElementC sum = ElementC(0);
        for (int col = 0; col < N; ++col) {
            sum += matrix[row * N + col];
        }
        row_sums[row] = sum;
    }
}

int main() {
    const int M = 128, N = 128, K = 128;

    // Define CUTLASS GEMM with TensorCore
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using Gemm = cutlass::gemm::device::Gemm<
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
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2
    >;

    // Allocate host matrices
    std::vector<cutlass::half_t> h_A(M * K);
    std::vector<cutlass::half_t> h_B(K * N);
    std::vector<cutlass::half_t> h_C(M * N, cutlass::half_t(0));
    std::vector<cutlass::half_t> h_D(M * N, cutlass::half_t(0));
    std::vector<cutlass::half_t> h_row_sums(M, cutlass::half_t(0));

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = cutlass::half_t(1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = cutlass::half_t(1.0f);
    }

    // Allocate device memory
    cutlass::half_t *d_A, *d_B, *d_C, *d_D, *d_row_sums;
    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B = K * N * sizeof(cutlass::half_t);
    size_t size_C = M * N * sizeof(cutlass::half_t);
    size_t size_row_sums = M * sizeof(cutlass::half_t);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMalloc(&d_D, size_C));
    CHECK_CUDA(cudaMalloc(&d_row_sums, size_row_sums));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    // Phase 1: Run TensorCore GEMM
    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D, N},
        {ElementCompute(1.0f), ElementCompute(0.0f)}
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM execution failed" << std::endl;
        return -1;
    }

    // Phase 2: Fused row reduction
    dim3 blockDim(256);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x);
    reduceRows<<<gridDim, blockDim>>>(d_D, d_row_sums, M, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_row_sums.data(), d_row_sums, size_row_sums, cudaMemcpyDeviceToHost));

    // Verify GEMM result
    bool gemm_correct = true;
    float expected_gemm = float(K);
    for (int i = 0; i < 10; ++i) {
        float val = float(h_D[i]);
        if (std::abs(val - expected_gemm) > 1.0f) {
            gemm_correct = false;
            break;
        }
    }

    // Verify row sums
    bool sum_correct = true;
    float expected_sum = N * K;
    for (int i = 0; i < M; ++i) {
        float val = float(h_row_sums[i]);
        if (std::abs(val - expected_sum) > 10.0f) {
            sum_correct = false;
            break;
        }
    }

    std::cout << "CUTLASS TensorCore GEMM: " << (gemm_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "GEMM expected: " << expected_gemm << ", Got: " << float(h_D[0]) << std::endl;

    std::cout << "Row Reduction: " << (sum_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Row sum expected: " << expected_sum << ", Got: " << float(h_row_sums[0]) << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_row_sums));

    return 0;
}