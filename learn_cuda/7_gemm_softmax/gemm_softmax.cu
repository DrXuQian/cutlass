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

// Warp-level reduction for max
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused GEMM + Softmax kernel
template<int TILE_SIZE>
__global__ void gemmSoftmax(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    __shared__ float row_max[TILE_SIZE];
    __shared__ float row_sum[TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Phase 1: Compute GEMM
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

    // Store intermediate GEMM result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

    // Phase 2: Compute row-wise softmax
    // Step 1: Find row maximum
    float max_val = (row < M && col < N) ? sum : -INFINITY;
    if (threadIdx.x < 32) {
        max_val = warpReduceMax(max_val);
    }

    if (threadIdx.x == 0) {
        row_max[threadIdx.y] = max_val;
    }
    __syncthreads();

    // Broadcast max to all threads in row
    float row_max_val = row_max[threadIdx.y];

    // Step 2: Compute exp(x - max) and sum
    float exp_val = (row < M && col < N) ? expf(sum - row_max_val) : 0.0f;

    float sum_exp = exp_val;
    if (threadIdx.x < 32) {
        sum_exp = warpReduceSum(sum_exp);
    }

    if (threadIdx.x == 0) {
        row_sum[threadIdx.y] = sum_exp;
    }
    __syncthreads();

    // Step 3: Normalize
    if (row < M && col < N) {
        C[row * N + col] = exp_val / row_sum[threadIdx.y];
    }
}

// Simple row-wise softmax for verification
void cpuRowSoftmax(float* data, int M, int N) {
    for (int i = 0; i < M; ++i) {
        // Find max
        float max_val = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max_val = std::max(max_val, data[i * N + j]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            data[i * N + j] = std::exp(data[i * N + j] - max_val);
            sum += data[i * N + j];
        }

        // Normalize
        for (int j = 0; j < N; ++j) {
            data[i * N + j] /= sum;
        }
    }
}

int main() {
    const int M = 64, N = 64, K = 64;
    const int TILE_SIZE = 32;

    // Allocate matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    // Initialize with small values to avoid overflow
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (rand() % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    // Run fused GEMM + Softmax
    std::cout << "Running Fused GEMM + Softmax..." << std::endl;
    gemmSoftmax<TILE_SIZE><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, M, N, K
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference on CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    cpuRowSoftmax(h_C_ref.data(), M, N);

    // Verify softmax properties
    bool correct = true;
    for (int i = 0; i < M; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            row_sum += h_C[i * N + j];
            // Check that values are in [0, 1]
            if (h_C[i * N + j] < 0.0f || h_C[i * N + j] > 1.0f) {
                correct = false;
                break;
            }
        }
        // Check that row sums to 1
        if (std::abs(row_sum - 1.0f) > 1e-3) {
            correct = false;
            std::cout << "Row " << i << " sum: " << row_sum << std::endl;
            break;
        }
    }

    // Compare with reference
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = std::max(max_error, std::abs(h_C[i] - h_C_ref[i]));
    }

    std::cout << "GEMM + Softmax: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Max error vs reference: " << max_error << std::endl;
    std::cout << "First row sum: ";
    float first_row_sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        first_row_sum += h_C[j];
    }
    std::cout << first_row_sum << " (should be ~1.0)" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}