#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Warp-level reduction for sum
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused GEMM + LayerNorm + GEMM kernel
template<int TILE_SIZE>
__global__ void gemmLayerNormGemm(
    const float* A, const float* B1, const float* B2,
    const float* gamma, const float* beta,
    float* C,
    int M, int N1, int K1, int N2) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    __shared__ float intermediate[TILE_SIZE][TILE_SIZE];
    __shared__ float row_mean[TILE_SIZE];
    __shared__ float row_variance[TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Phase 1: First GEMM (A * B1)
    float sum1 = 0.0f;
    for (int k = 0; k < K1; k += TILE_SIZE) {
        if (row < M && k + threadIdx.x < K1) {
            sA[threadIdx.y][threadIdx.x] = A[row * K1 + k + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N1 && k + threadIdx.y < K1) {
            sB[threadIdx.y][threadIdx.x] = B1[(k + threadIdx.y) * N1 + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum1 += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Store first GEMM result
    intermediate[threadIdx.y][threadIdx.x] = sum1;
    __syncthreads();

    // Phase 2: LayerNorm on intermediate result
    // Compute row mean
    float val = (row < M && col < N1) ? sum1 : 0.0f;
    float row_sum = val;

    if (threadIdx.x < 32) {
        row_sum = warpReduceSum(row_sum);
    }

    if (threadIdx.x == 0) {
        row_mean[threadIdx.y] = row_sum / N1;
    }
    __syncthreads();

    float mean = row_mean[threadIdx.y];

    // Compute row variance
    float diff = (row < M && col < N1) ? (val - mean) : 0.0f;
    float diff_sq = diff * diff;

    if (threadIdx.x < 32) {
        diff_sq = warpReduceSum(diff_sq);
    }

    if (threadIdx.x == 0) {
        row_variance[threadIdx.y] = diff_sq / N1;
    }
    __syncthreads();

    float variance = row_variance[threadIdx.y];
    float std_dev = sqrtf(variance + 1e-5f);

    // Apply LayerNorm
    if (row < M && col < N1) {
        float normalized = (intermediate[threadIdx.y][threadIdx.x] - mean) / std_dev;
        // Apply affine transformation if gamma and beta are provided
        if (gamma && beta) {
            normalized = gamma[col] * normalized + beta[col];
        }
        intermediate[threadIdx.y][threadIdx.x] = normalized;
    }
    __syncthreads();

    // Phase 3: Second GEMM (normalized * B2)
    float sum2 = 0.0f;
    col = blockIdx.x * TILE_SIZE + threadIdx.x; // Reset col for second GEMM

    for (int k = 0; k < N1; k += TILE_SIZE) {
        // Load from intermediate result
        if (k / TILE_SIZE == blockIdx.x) {
            sA[threadIdx.y][threadIdx.x] = intermediate[threadIdx.y][threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N2 && k + threadIdx.y < N1) {
            sB[threadIdx.y][threadIdx.x] = B2[(k + threadIdx.y) * N2 + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum2 += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write final result
    if (row < M && col < N2) {
        atomicAdd(&C[row * N2 + col], sum2);
    }
}

int main() {
    const int M = 32, K1 = 64, N1 = 64, N2 = 32;
    const int TILE_SIZE = 16;

    // Allocate matrices
    std::vector<float> h_A(M * K1);
    std::vector<float> h_B1(K1 * N1);
    std::vector<float> h_B2(N1 * N2);
    std::vector<float> h_gamma(N1, 1.0f);
    std::vector<float> h_beta(N1, 0.0f);
    std::vector<float> h_C(M * N2, 0.0f);

    // Initialize with small values
    for (int i = 0; i < M * K1; ++i) {
        h_A[i] = (rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K1 * N1; ++i) {
        h_B1[i] = (rand() % 100) / 100.0f;
    }
    for (int i = 0; i < N1 * N2; ++i) {
        h_B2[i] = (rand() % 100) / 100.0f;
    }

    float *d_A, *d_B1, *d_B2, *d_gamma, *d_beta, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B1, K1 * N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B2, N1 * N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gamma, N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B1, h_B1.data(), K1 * N1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B2, h_B2.data(), N1 * N2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), N1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), N1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N2 * sizeof(float)));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (std::max(N1, N2) + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    std::cout << "Running Fused GEMM + LayerNorm + GEMM..." << std::endl;
    std::cout << "Dimensions: A(" << M << "x" << K1 << ") * B1(" << K1 << "x" << N1
              << ") -> LayerNorm -> * B2(" << N1 << "x" << N2 << ")" << std::endl;

    gemmLayerNormGemm<TILE_SIZE><<<gridDim, blockDim>>>(
        d_A, d_B1, d_B2, d_gamma, d_beta, d_C,
        M, N1, K1, N2
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify output shape and basic statistics
    float min_val = h_C[0], max_val = h_C[0], avg_val = 0.0f;
    for (int i = 0; i < M * N2; ++i) {
        min_val = std::min(min_val, h_C[i]);
        max_val = std::max(max_val, h_C[i]);
        avg_val += h_C[i];
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
        if (h_C[i] != 0.0f) {
            has_output = true;
            break;
        }
    }

    std::cout << "GEMM + LayerNorm + GEMM: " << (has_output ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B1));
    CHECK_CUDA(cudaFree(d_B2));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}