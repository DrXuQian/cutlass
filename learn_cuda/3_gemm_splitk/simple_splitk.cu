#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Simple Split-K GEMM kernel
template<int TILE_SIZE, int K_SPLIT>
__global__ void splitKGemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int split_id = blockIdx.z;

    // Calculate K range for this split
    int k_per_split = (K + K_SPLIT - 1) / K_SPLIT;
    int k_start = split_id * k_per_split;
    int k_end = min(k_start + k_per_split, K);

    float sum = 0.0f;

    // Process K tiles for this split
    for (int k = k_start; k < k_end; k += TILE_SIZE) {
        // Load tiles into shared memory
        if (row < M && k + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && k + threadIdx.y < K) {
            sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Atomic add to accumulate results from different splits
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

int main() {
    const int M = 512, N = 512, K = 2048;
    const int K_SPLIT = 4;
    const int TILE_SIZE = 16;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate and initialize matrices
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        K_SPLIT
    );

    splitKGemm<TILE_SIZE, K_SPLIT><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, M, N, K
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // Verify result (all elements should be K)
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_C[i] - K) > 1e-3) {
            correct = false;
            break;
        }
    }

    std::cout << "Split-K GEMM: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Expected: " << K << ", Got: " << h_C[0] << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}