#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Simple softmax kernel
template <typename Element>
__global__ void applySoftmax(
    Element* matrix,
    int M, int N) {

    int row = blockIdx.x;
    if (row >= M) return;

    // Find max in row
    float max_val = -INFINITY;
    for (int col = 0; col < N; ++col) {
        float val = float(matrix[row * N + col]);
        max_val = fmaxf(max_val, val);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        float val = float(matrix[row * N + col]);
        float exp_val = expf(val - max_val);
        matrix[row * N + col] = Element(exp_val);
        sum += exp_val;
    }

    // Normalize
    for (int col = 0; col < N; ++col) {
        float val = float(matrix[row * N + col]);
        matrix[row * N + col] = Element(val / sum);
    }
}

int main() {
    const int M = 64, N = 64, K = 64;

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

    // Initialize matrices with small values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = cutlass::half_t((rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = cutlass::half_t((rand() % 100) / 100.0f);
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

    // Phase 1: TensorCore GEMM
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

    // Phase 2: Apply softmax
    applySoftmax<<<M, 1>>>(d_D, M, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, size_C, cudaMemcpyDeviceToHost));

    // Verify softmax properties
    bool correct = true;
    for (int i = 0; i < M; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float val = float(h_D[i * N + j]);
            row_sum += val;
            // Check values are in [0, 1]
            if (val < 0.0f || val > 1.0f) {
                correct = false;
                break;
            }
        }
        // Check row sums to 1
        if (std::abs(row_sum - 1.0f) > 0.01f) {
            correct = false;
            std::cout << "Row " << i << " sum: " << row_sum << std::endl;
            break;
        }
    }

    std::cout << "CUTLASS TensorCore GEMM + Softmax: " << (correct ? "PASS" : "FAIL") << std::endl;

    float first_row_sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        first_row_sum += float(h_D[j]);
    }
    std::cout << "First row sum: " << first_row_sum << " (should be ~1.0)" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));

    return 0;
}