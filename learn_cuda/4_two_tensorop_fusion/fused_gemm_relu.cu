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

// Fused GEMM + ReLU kernel
template<int TILE_SIZE>
__global__ void gemmReLU(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Compute GEMM
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load tiles
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

        // Compute dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Apply scaling and bias, then ReLU activation
    if (row < M && col < N) {
        float result = alpha * sum + beta * C[row * N + col];
        // Fused ReLU activation
        C[row * N + col] = fmaxf(result, 0.0f);
    }
}

// Fused GEMM + GEMM kernel (two consecutive GEMMs)
template<int TILE_SIZE>
__global__ void fusedDoubleGemm(
    const float* A, const float* B1, const float* B2, float* C,
    int M, int N, int K1, int K2) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    __shared__ float intermediate[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_x = threadIdx.x;

    // First GEMM: A * B1 -> intermediate
    float sum1 = 0.0f;
    for (int k = 0; k < K1; k += TILE_SIZE) {
        if (row < M && k + tid_x < K1) {
            sA[tid_y][tid_x] = A[row * K1 + k + tid_x];
        } else {
            sA[tid_y][tid_x] = 0.0f;
        }

        if (col < K2 && k + tid_y < K1) {
            sB[tid_y][tid_x] = B1[(k + tid_y) * K2 + col];
        } else {
            sB[tid_y][tid_x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum1 += sA[tid_y][i] * sB[i][tid_x];
        }
        __syncthreads();
    }

    // Store intermediate result in shared memory
    intermediate[tid_y][tid_x] = sum1;
    __syncthreads();

    // Second GEMM: intermediate * B2 -> C
    float sum2 = 0.0f;
    for (int k = 0; k < K2; k += TILE_SIZE) {
        // Reuse intermediate result
        if (k == 0) {
            sA[tid_y][tid_x] = intermediate[tid_y][tid_x];
        } else {
            sA[tid_y][tid_x] = 0.0f;
        }

        if (col < N && k + tid_y < K2) {
            sB[tid_y][tid_x] = B2[(k + tid_y) * N + col];
        } else {
            sB[tid_y][tid_x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum2 += sA[tid_y][i] * sB[i][tid_x];
        }
        __syncthreads();
    }

    // Write final result
    if (row < M && col < N) {
        C[row * N + col] = sum2;
    }
}

int main() {
    const int M = 256, N = 256, K = 256;
    const int TILE_SIZE = 16;

    size_t size = M * N * sizeof(float);

    // Allocate matrices
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, size));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    // Test GEMM + ReLU fusion
    std::cout << "Testing Fused GEMM + ReLU..." << std::endl;
    gemmReLU<TILE_SIZE><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, M, N, K, 1.0f, 0.0f
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify (should be max(2*K, 0) = 512)
    bool correct = true;
    float expected = 2.0f * K;
    for (int i = 0; i < 10; ++i) {
        if (std::abs(h_C[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    std::cout << "GEMM + ReLU: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Expected: " << expected << ", Got: " << h_C[0] << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}