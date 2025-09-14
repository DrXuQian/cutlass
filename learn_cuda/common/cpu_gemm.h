#pragma once

#include <iostream>
#include <chrono>

namespace cpu_gemm {

// Basic CPU GEMM: C = alpha * A * B + beta * C
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix
template <typename T>
void gemm_cpu(int M, int N, int K,
              T alpha, const T* A, int lda,
              const T* B, int ldb,
              T beta, T* C, int ldc) {
    // First scale C by beta
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] *= beta;
        }
    }

    // Compute alpha * A * B and accumulate
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// CPU GEMM with transposition options
// transA: 'N' (no transpose) or 'T' (transpose)
// transB: 'N' (no transpose) or 'T' (transpose)
template <typename T>
void gemm_cpu_ex(char transA, char transB,
                 int M, int N, int K,
                 T alpha, const T* A, int lda,
                 const T* B, int ldb,
                 T beta, T* C, int ldc) {
    // First scale C by beta
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] *= beta;
        }
    }

    // Compute alpha * op(A) * op(B) and accumulate
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                T a_val, b_val;

                // Handle A indexing based on transpose
                if (transA == 'N' || transA == 'n') {
                    a_val = A[i * lda + k];
                } else {
                    a_val = A[k * lda + i];
                }

                // Handle B indexing based on transpose
                if (transB == 'N' || transB == 'n') {
                    b_val = B[k * ldb + j];
                } else {
                    b_val = B[j * ldb + k];
                }

                sum += a_val * b_val;
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// Optimized blocked CPU GEMM for better cache utilization
template <typename T>
void gemm_cpu_blocked(int M, int N, int K,
                      T alpha, const T* A, int lda,
                      const T* B, int ldb,
                      T beta, T* C, int ldc,
                      int block_size = 64) {
    // First scale C by beta
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] *= beta;
        }
    }

    // Blocked matrix multiplication
    for (int i = 0; i < M; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < K; k += block_size) {
                // Compute block boundaries
                int i_end = std::min(i + block_size, M);
                int j_end = std::min(j + block_size, N);
                int k_end = std::min(k + block_size, K);

                // Multiply blocks
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        T sum = 0;
                        for (int kk = k; kk < k_end; ++kk) {
                            sum += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                        C[ii * ldc + jj] += alpha * sum;
                    }
                }
            }
        }
    }
}

// Verify GPU GEMM result against CPU computation
template <typename T>
bool verify_gemm(int M, int N, int K,
                 T alpha, const T* A, int lda,
                 const T* B, int ldb,
                 T beta, const T* C_gpu, int ldc,
                 T tolerance = 1e-5, bool verbose = false) {
    // Allocate memory for CPU result
    T* C_cpu = new T[M * N];

    // Initialize C_cpu with zeros
    for (int i = 0; i < M * N; ++i) {
        C_cpu[i] = 0;
    }

    // Compute CPU GEMM
    gemm_cpu(M, N, K, alpha, A, lda, B, ldb, beta, C_cpu, ldc);

    // Compare results
    bool all_close = true;
    T max_diff = 0;
    int diff_count = 0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * ldc + j;
            T diff = std::abs(C_gpu[idx] - C_cpu[idx]);

            if (diff > tolerance) {
                all_close = false;
                diff_count++;
                max_diff = std::max(max_diff, diff);

                if (verbose && diff_count <= 10) {
                    std::cout << "Mismatch at [" << i << ", " << j << "]: "
                             << "GPU=" << C_gpu[idx] << " vs CPU=" << C_cpu[idx]
                             << " (diff: " << diff << ")" << std::endl;
                }
            }
        }
    }

    if (verbose) {
        if (all_close) {
            std::cout << "GEMM verification PASSED (tolerance: " << tolerance << ")" << std::endl;
        } else {
            std::cout << "GEMM verification FAILED: " << diff_count
                     << " elements out of " << (M * N) << " exceed tolerance" << std::endl;
            std::cout << "Maximum difference: " << max_diff << std::endl;
        }
    }

    delete[] C_cpu;
    return all_close;
}

// Benchmark CPU GEMM performance
template <typename T>
double benchmark_cpu_gemm(int M, int N, int K,
                         T alpha, const T* A, int lda,
                         const T* B, int ldb,
                         T beta, T* C, int ldc,
                         int iterations = 10) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        gemm_cpu(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double avg_time_ms = (diff.count() * 1000.0) / iterations;
    double gflops = (2.0 * M * N * K * 1e-9) / (avg_time_ms * 1e-3);

    std::cout << "CPU GEMM Performance:" << std::endl;
    std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;

    return avg_time_ms;
}

} // namespace cpu_gemm