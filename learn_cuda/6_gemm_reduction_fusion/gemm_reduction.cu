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

// Fused GEMM + Row Reduction kernel
template<int TILE_SIZE>
__global__ void gemmRowReduction(
    const float* A, const float* B,
    float* C, float* row_sums,
    int M, int N, int K) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_row_sums[TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Compute GEMM
    for (int k = 0; k < K; k += TILE_SIZE) {
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

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Store GEMM result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

    // Initialize shared memory for row reduction
    if (threadIdx.x == 0) {
        tile_row_sums[threadIdx.y] = 0.0f;
    }
    __syncthreads();

    // Accumulate row sum within tile
    if (row < M && col < N) {
        atomicAdd(&tile_row_sums[threadIdx.y], sum);
    }
    __syncthreads();

    // Write row sums (one thread per row in tile)
    if (threadIdx.x == 0 && row < M) {
        atomicAdd(&row_sums[row], tile_row_sums[threadIdx.y]);
    }
}

// Fused GEMM + Column Maximum kernel
template<int TILE_SIZE>
__global__ void gemmColMax(
    const float* A, const float* B,
    float* C, float* col_max,
    int M, int N, int K) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_col_max[TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Compute GEMM
    for (int k = 0; k < K; k += TILE_SIZE) {
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

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Store GEMM result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

    // Initialize shared memory for column max
    if (threadIdx.y == 0) {
        tile_col_max[threadIdx.x] = -INFINITY;
    }
    __syncthreads();

    // Find column maximum within tile
    if (row < M && col < N) {
        atomicMax(reinterpret_cast<int*>(&tile_col_max[threadIdx.x]),
                  __float_as_int(sum));
    }
    __syncthreads();

    // Write column max (one thread per column in tile)
    if (threadIdx.y == 0 && col < N) {
        atomicMax(reinterpret_cast<int*>(&col_max[col]),
                  __float_as_int(tile_col_max[threadIdx.x]));
    }
}

int main() {
    const int M = 128, N = 128, K = 128;
    const int TILE_SIZE = 16;

    // Allocate matrices
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_row_sums(M, 0.0f);
    std::vector<float> h_col_max(N, -INFINITY);

    float *d_A, *d_B, *d_C, *d_row_sums, *d_col_max;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_sums, M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_col_max, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_row_sums, 0, M * sizeof(float)));

    // Initialize col_max with -infinity
    std::vector<float> neg_inf(N, -INFINITY);
    CHECK_CUDA(cudaMemcpy(d_col_max, neg_inf.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    // Test GEMM + Row Reduction
    std::cout << "Testing GEMM + Row Reduction..." << std::endl;
    gemmRowReduction<TILE_SIZE><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, d_row_sums, M, N, K
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_row_sums.data(), d_row_sums, M * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify row sums (each row sum should be N * K)
    bool row_correct = true;
    float expected_row_sum = N * K;
    for (int i = 0; i < M; ++i) {
        if (std::abs(h_row_sums[i] - expected_row_sum) > 1e-3) {
            row_correct = false;
            break;
        }
    }

    std::cout << "GEMM + Row Reduction: " << (row_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Expected row sum: " << expected_row_sum << ", Got: " << h_row_sums[0] << std::endl;

    // Test GEMM + Column Max
    std::cout << "\nTesting GEMM + Column Max..." << std::endl;
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    gemmColMax<TILE_SIZE><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, d_col_max, M, N, K
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_col_max.data(), d_col_max, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify column max (each should be K)
    bool col_correct = true;
    float expected_col_max = K;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_col_max[i] - expected_col_max) > 1e-3) {
            col_correct = false;
            break;
        }
    }

    std::cout << "GEMM + Column Max: " << (col_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Expected column max: " << expected_col_max << ", Got: " << h_col_max[0] << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_row_sums));
    CHECK_CUDA(cudaFree(d_col_max));

    return 0;
}