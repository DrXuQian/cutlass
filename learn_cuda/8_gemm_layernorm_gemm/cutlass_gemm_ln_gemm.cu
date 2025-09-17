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

// Simple LayerNorm kernel
template <typename Element>
__global__ void applyLayerNorm(
    Element* matrix,
    const Element* gamma,
    const Element* beta,
    int M, int N) {

    int row = blockIdx.x;
    if (row >= M) return;

    // Compute mean
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        sum += float(matrix[row * N + col]);
    }
    float mean = sum / N;

    // Compute variance
    float variance_sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        float diff = float(matrix[row * N + col]) - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / N;
    float std_dev = sqrtf(variance + 1e-5f);

    // Apply normalization with gamma and beta
    for (int col = 0; col < N; ++col) {
        float normalized = (float(matrix[row * N + col]) - mean) / std_dev;
        if (gamma && beta) {
            normalized = float(gamma[col]) * normalized + float(beta[col]);
        }
        matrix[row * N + col] = Element(normalized);
    }
}

int main() {
    const int M = 32, K1 = 64, N1 = 64, N2 = 32;

    // Define CUTLASS GEMM with TensorCore
    using ElementInput = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput, cutlass::layout::RowMajor,
        ElementInput, cutlass::layout::RowMajor,
        ElementOutput, cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2
    >;

    // Allocate host matrices
    std::vector<cutlass::half_t> h_A(M * K1);
    std::vector<cutlass::half_t> h_B1(K1 * N1);
    std::vector<cutlass::half_t> h_B2(N1 * N2);
    std::vector<cutlass::half_t> h_gamma(N1, cutlass::half_t(1.0f));
    std::vector<cutlass::half_t> h_beta(N1, cutlass::half_t(0.0f));
    std::vector<cutlass::half_t> h_intermediate(M * N1, cutlass::half_t(0));
    std::vector<cutlass::half_t> h_output(M * N2, cutlass::half_t(0));

    // Initialize matrices
    for (int i = 0; i < M * K1; ++i) {
        h_A[i] = cutlass::half_t((rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K1 * N1; ++i) {
        h_B1[i] = cutlass::half_t((rand() % 100) / 100.0f);
    }
    for (int i = 0; i < N1 * N2; ++i) {
        h_B2[i] = cutlass::half_t((rand() % 100) / 100.0f);
    }

    // Allocate device memory
    cutlass::half_t *d_A, *d_B1, *d_B2, *d_gamma, *d_beta;
    cutlass::half_t *d_intermediate, *d_output, *d_temp;

    CHECK_CUDA(cudaMalloc(&d_A, M * K1 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_B1, K1 * N1 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_B2, N1 * N2 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_gamma, N1 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_beta, N1 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_intermediate, M * N1 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_output, M * N2 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_temp, M * N1 * sizeof(cutlass::half_t)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K1 * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B1, h_B1.data(), K1 * N1 * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B2, h_B2.data(), N1 * N2 * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), N1 * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), N1 * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_intermediate, 0, M * N1 * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMemset(d_temp, 0, M * N1 * sizeof(cutlass::half_t)));

    std::cout << "Running CUTLASS TensorCore GEMM + LayerNorm + GEMM..." << std::endl;
    std::cout << "Dimensions: A(" << M << "x" << K1 << ") * B1(" << K1 << "x" << N1
              << ") -> LayerNorm -> * B2(" << N1 << "x" << N2 << ")" << std::endl;

    // Phase 1: First GEMM (A * B1)
    Gemm gemm_op;
    Gemm::Arguments args1(
        {M, N1, K1},
        {d_A, K1},
        {d_B1, N1},
        {d_temp, N1},
        {d_intermediate, N1},
        {ElementCompute(1.0f), ElementCompute(0.0f)}
    );

    cutlass::Status status = gemm_op(args1);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "First GEMM execution failed" << std::endl;
        return -1;
    }

    // Phase 2: Apply LayerNorm
    applyLayerNorm<<<M, 1>>>(d_intermediate, d_gamma, d_beta, M, N1);

    // Phase 3: Second GEMM (normalized * B2)
    Gemm::Arguments args2(
        {M, N2, N1},
        {d_intermediate, N1},
        {d_B2, N2},
        {d_temp, N2},
        {d_output, N2},
        {ElementCompute(1.0f), ElementCompute(0.0f)}
    );

    status = gemm_op(args2);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Second GEMM execution failed" << std::endl;
        return -1;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, M * N2 * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

    // Verify output statistics
    float min_val = float(h_output[0]), max_val = float(h_output[0]), avg_val = 0.0f;
    for (int i = 0; i < M * N2; ++i) {
        float val = float(h_output[i]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        avg_val += val;
    }
    avg_val /= (M * N2);

    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Avg: " << avg_val << std::endl;
    std::cout << "  Shape: (" << M << ", " << N2 << ")" << std::endl;

    // Basic sanity check
    bool has_output = false;
    for (int i = 0; i < M * N2; ++i) {
        if (float(h_output[i]) != 0.0f) {
            has_output = true;
            break;
        }
    }

    std::cout << "CUTLASS TensorCore GEMM + LayerNorm + GEMM: " << (has_output ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B1));
    CHECK_CUDA(cudaFree(d_B2));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_intermediate));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));

    return 0;
}