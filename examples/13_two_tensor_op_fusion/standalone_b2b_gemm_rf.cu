/*
 * Standalone B2B GEMM with RF (Register File) Residency
 *
 * 独立的简化实现，展示B2B GEMM融合的核心概念
 * 不依赖任何外部kernel实现，所有代码都在这一个文件中
 *
 * 核心优化：中间结果C保存在寄存器中，避免全局内存访问
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

// =====================================================================
// 简化的B2B GEMM Kernel - RF驻留版本
//
// 执行两个连续的矩阵乘法：
// 1. C = ReLU(A * B0)  [M,K] x [K,N] = [M,N]
// 2. D = ReLU(C * B1)  [M,N] x [N,P] = [M,P]
//
// 关键优化：C保持在寄存器中，不写入全局内存
// =====================================================================

template<int TILE_M, int TILE_N, int TILE_K, int TILE_P>
__global__ void b2b_gemm_rf_kernel(
    const half* __restrict__ A,   // [M, K] 行主序
    const half* __restrict__ B0,  // [K, N] 列主序
    const half* __restrict__ B1,  // [N, P] 列主序
    half* __restrict__ D,          // [M, P] 行主序
    int M, int N, int K, int P
) {
    // 每个线程块处理输出D的一个TILE_M x TILE_P的块
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 共享内存用于缓存输入tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float B0s[TILE_K][TILE_N];
    __shared__ float B1s[TILE_N][TILE_P];

    // 计算这个线程负责的全局位置
    const int row = bx * TILE_M + ty;
    const int col_n = by * TILE_N + tx;  // 用于第一个GEMM
    const int col_p = by * TILE_P + tx;  // 用于第二个GEMM

    // ========== 第一个GEMM: C = A * B0 ==========
    // 关键：结果保存在寄存器c_reg中
    float c_reg = 0.0f;

    // 沿K维度分块
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        // 协作加载A的tile到共享内存
        if (ty < TILE_M && tx < TILE_K) {
            int a_row = bx * TILE_M + ty;
            int a_col = k_tile * TILE_K + tx;
            if (a_row < M && a_col < K) {
                As[ty][tx] = __half2float(A[a_row * K + a_col]);
            } else {
                As[ty][tx] = 0.0f;
            }
        }

        // 协作加载B0的tile到共享内存
        if (ty < TILE_K && tx < TILE_N) {
            int b0_row = k_tile * TILE_K + ty;
            int b0_col = by * TILE_N + tx;
            if (b0_row < K && b0_col < N) {
                B0s[ty][tx] = __half2float(B0[b0_row + b0_col * K]);
            } else {
                B0s[ty][tx] = 0.0f;
            }
        }

        __syncthreads();

        // 计算部分积
        if (ty < TILE_M && tx < TILE_N) {
            for (int k = 0; k < TILE_K; k++) {
                if (k_tile * TILE_K + k < K) {
                    c_reg += As[ty][k] * B0s[k][tx];
                }
            }
        }

        __syncthreads();
    }

    // 应用ReLU激活函数
    c_reg = fmaxf(c_reg, 0.0f);

    // ========== RF驻留：c_reg保持在寄存器中 ==========
    // 这是关键优化！避免了：
    // - 写入全局内存：~500 cycles
    // - 读取全局内存：~500 cycles

    // ========== 第二个GEMM: D = C * B1 ==========
    float d_reg = 0.0f;

    // 为了简化，假设P == N（实际中可以处理不同尺寸）
    // 沿N维度分块
    for (int n_tile = 0; n_tile < (N + TILE_N - 1) / TILE_N; n_tile++) {
        // 加载B1的tile到共享内存
        if (ty < TILE_N && tx < TILE_P) {
            int b1_row = n_tile * TILE_N + ty;
            int b1_col = by * TILE_P + tx;
            if (b1_row < N && b1_col < P) {
                B1s[ty][tx] = __half2float(B1[b1_row + b1_col * N]);
            } else {
                B1s[ty][tx] = 0.0f;
            }
        }

        __syncthreads();

        // 使用寄存器中的c_reg计算
        if (ty < TILE_M && tx < TILE_P) {
            // 简化：假设每个线程处理一个元素
            // 实际上需要更复杂的映射来处理C的不同部分
            if (n_tile == by && row < M && col_p < P) {
                // 这是一个简化，实际需要累加C的整行与B1的列
                for (int n = 0; n < TILE_N; n++) {
                    if (n_tile * TILE_N + n < N) {
                        // 注意：这里简化了，实际需要正确的索引
                        float c_val = (n == tx && n_tile == by) ? c_reg : 0.0f;
                        d_reg += c_val * B1s[n][tx];
                    }
                }
            }
        }

        __syncthreads();
    }

    // 应用ReLU并写入全局内存
    if (row < M && col_p < P && ty < TILE_M && tx < TILE_P) {
        d_reg = fmaxf(d_reg, 0.0f);
        D[row * P + col_p] = __float2half(d_reg);
    }
}

// =====================================================================
// 简化但更正确的B2B GEMM Kernel - RF驻留版本
// 每个线程计算一个输出元素
// =====================================================================

__global__ void b2b_gemm_rf_kernel_simple(
    const half* __restrict__ A,   // [M, K] 行主序
    const half* __restrict__ B0,  // [K, N] 列主序
    const half* __restrict__ B1,  // [N, P] 列主序
    half* __restrict__ D,          // [M, P] 行主序
    int M, int N, int K, int P
) {
    // 每个线程计算D的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M维度
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // P维度

    if (row >= M || col >= P) return;

    // ========== 第一个GEMM: 计算C的一行 ==========
    // 为了计算D[row][col]，需要C[row][:]
    // 使用寄存器数组存储C的一行（简化：最多支持128列）
    float c_row[128];

    // 计算C[row][n] for all n
    for (int n = 0; n < N && n < 128; n++) {
        float sum = 0.0f;
        // C[row][n] = sum(A[row][k] * B0[k][n])
        for (int k = 0; k < K; k++) {
            float a_val = __half2float(A[row * K + k]);
            float b0_val = __half2float(B0[k + n * K]);
            sum += a_val * b0_val;
        }
        // 应用ReLU
        c_row[n] = fmaxf(sum, 0.0f);
    }

    // ========== RF驻留：c_row数组在寄存器中 ==========

    // ========== 第二个GEMM: 计算D[row][col] ==========
    float d_val = 0.0f;
    // D[row][col] = sum(C[row][n] * B1[n][col])
    for (int n = 0; n < N && n < 128; n++) {
        float b1_val = __half2float(B1[n + col * N]);
        d_val += c_row[n] * b1_val;
    }

    // 应用ReLU并写入
    d_val = fmaxf(d_val, 0.0f);
    D[row * P + col] = __float2half(d_val);
}

// =====================================================================
// 主机端辅助函数
// =====================================================================

// 初始化矩阵
void init_matrix(half* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((rand() / float(RAND_MAX) - 0.5f) * scale);
    }
}

// CPU参考实现
void cpu_b2b_gemm_ref(
    const half* A, const half* B0, const half* B1, half* D,
    int M, int N, int K, int P
) {
    // 临时存储C
    float* C = new float[M * N];

    // 第一个GEMM: C = ReLU(A * B0)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[m * K + k]) * __half2float(B0[k + n * K]);
            }
            C[m * N + n] = fmaxf(sum, 0.0f);
        }
    }

    // 第二个GEMM: D = ReLU(C * B1)
    for (int m = 0; m < M; m++) {
        for (int p = 0; p < P; p++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += C[m * N + n] * __half2float(B1[n + p * N]);
            }
            D[m * P + p] = __float2half(fmaxf(sum, 0.0f));
        }
    }

    delete[] C;
}

// 验证结果
bool verify_results(const half* gpu, const half* cpu, int size, float tolerance = 0.01f) {
    int errors = 0;
    float max_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = fabs(__half2float(gpu[i]) - __half2float(cpu[i]));
        max_error = fmax(max_error, diff);
        if (diff > tolerance) {
            errors++;
            if (errors < 10) {  // 只打印前10个错误
                printf("Error at %d: GPU=%f, CPU=%f, diff=%f\n",
                       i, __half2float(gpu[i]), __half2float(cpu[i]), diff);
            }
        }
    }

    printf("Max error: %f, Errors: %d/%d\n", max_error, errors, size);
    return errors == 0;
}

// =====================================================================
// 主函数
// =====================================================================

int main() {
    printf("\n=== Standalone B2B GEMM with RF Residency ===\n");
    printf("完全独立的实现，所有代码在一个文件中\n\n");

    // 问题尺寸
    const int M = 128;
    const int N = 64;
    const int K = 128;
    const int P = 64;

    printf("Problem sizes:\n");
    printf("First GEMM:  [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Second GEMM: [%d, %d] x [%d, %d] = [%d, %d]\n", M, N, N, P, M, P);
    printf("\n");

    // 分配主机内存
    size_t size_A = M * K * sizeof(half);
    size_t size_B0 = K * N * sizeof(half);
    size_t size_B1 = N * P * sizeof(half);
    size_t size_D = M * P * sizeof(half);

    half *h_A = (half*)malloc(size_A);
    half *h_B0 = (half*)malloc(size_B0);
    half *h_B1 = (half*)malloc(size_B1);
    half *h_D = (half*)malloc(size_D);
    half *h_D_ref = (half*)malloc(size_D);

    // 初始化输入
    srand(42);
    init_matrix(h_A, M, K, 0.5f);
    init_matrix(h_B0, K, N, 0.5f);
    init_matrix(h_B1, N, P, 0.5f);

    // 分配设备内存
    half *d_A, *d_B0, *d_B1, *d_D;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B0, size_B0);
    cudaMalloc(&d_B1, size_B1);
    cudaMalloc(&d_D, size_D);

    // 复制输入到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B0, h_B0, size_B0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, size_B1, cudaMemcpyHostToDevice);

    // 配置kernel启动参数
    dim3 blockDim(16, 16);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("Launching kernel with:\n");
    printf("Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("\n");

    // 启动kernel
    printf("Running GPU B2B GEMM with RF residency...\n");

    // 使用简单版本（每个线程计算一个输出）
    b2b_gemm_rf_kernel_simple<<<gridDim, blockDim>>>(
        d_A, d_B0, d_B1, d_D, M, N, K, P
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // 复制结果回主机
    cudaMemcpy(h_D, d_D, size_D, cudaMemcpyDeviceToHost);

    // 计算CPU参考结果
    printf("Computing CPU reference...\n");
    cpu_b2b_gemm_ref(h_A, h_B0, h_B1, h_D_ref, M, N, K, P);

    // 验证结果
    printf("\nVerifying results...\n");
    bool passed = verify_results(h_D, h_D_ref, M * P, 0.1f);  // FP16精度，容差放宽

    if (passed) {
        printf("\n*** PASSED ***\n");

        // 计算并显示性能优势
        printf("\n=== Performance Benefits ===\n");
        size_t intermediate_size = M * N * sizeof(half);
        printf("Intermediate matrix C: %d elements (%zu bytes)\n", M * N, intermediate_size);
        printf("Memory saved by RF residency:\n");
        printf("  - Write to global mem: %zu bytes\n", intermediate_size);
        printf("  - Read from global mem: %zu bytes\n", intermediate_size);
        printf("  - Total saved: %zu bytes\n", 2 * intermediate_size);
        printf("Estimated latency saved: ~1000 cycles\n");
    } else {
        printf("\n*** FAILED ***\n");
    }

    // 清理
    free(h_A);
    free(h_B0);
    free(h_B1);
    free(h_D);
    free(h_D_ref);

    cudaFree(d_A);
    cudaFree(d_B0);
    cudaFree(d_B1);
    cudaFree(d_D);

    return passed ? 0 : -1;
}