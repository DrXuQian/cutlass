#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

namespace cuda_utils {

// Allocate device memory for a matrix
template <typename T>
cudaError_t allocateDeviceMatrix(T** d_matrix, int rows, int cols) {
    size_t size = sizeof(T) * rows * cols;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d_matrix), size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device matrix: "
                  << cudaGetErrorString(err) << std::endl;
    }
    return err;
}

// Allocate host memory for a matrix
template <typename T>
T* allocateHostMatrix(int rows, int cols) {
    return new T[rows * cols];
}

// Free device memory
template <typename T>
void freeDeviceMatrix(T* d_matrix) {
    if (d_matrix) {
        cudaFree(d_matrix);
    }
}

// Free host memory
template <typename T>
void freeHostMatrix(T* h_matrix) {
    if (h_matrix) {
        delete[] h_matrix;
    }
}

// Initialize matrix with random values
template <typename T>
void initializeRandomMatrix(T* matrix, int rows, int cols,
                           T min_val = -1.0, T max_val = 1.0,
                           unsigned int seed = 0) {
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<T> dist(min_val, max_val);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

// Initialize matrix with sequential values (for debugging)
template <typename T>
void initializeSequentialMatrix(T* matrix, int rows, int cols, T start_val = 0) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = start_val + static_cast<T>(i);
    }
}

// Initialize matrix with constant value
template <typename T>
void initializeConstantMatrix(T* matrix, int rows, int cols, T value) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = value;
    }
}

// Initialize identity matrix
template <typename T>
void initializeIdentityMatrix(T* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
}

// Copy matrix from host to device
template <typename T>
cudaError_t copyHostToDevice(T* d_matrix, const T* h_matrix, int rows, int cols) {
    size_t size = sizeof(T) * rows * cols;
    return cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
}

// Copy matrix from device to host
template <typename T>
cudaError_t copyDeviceToHost(T* h_matrix, const T* d_matrix, int rows, int cols) {
    size_t size = sizeof(T) * rows * cols;
    return cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
}

// Print matrix (for debugging)
template <typename T>
void printMatrix(const T* matrix, int rows, int cols, const char* name = "Matrix",
                int max_rows = 10, int max_cols = 10) {
    std::cout << name << " (" << rows << " x " << cols << "):" << std::endl;

    int display_rows = std::min(rows, max_rows);
    int display_cols = std::min(cols, max_cols);

    for (int i = 0; i < display_rows; ++i) {
        for (int j = 0; j < display_cols; ++j) {
            std::cout << std::setw(10) << std::setprecision(4)
                     << matrix[i * cols + j] << " ";
        }
        if (cols > max_cols) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    if (rows > max_rows) {
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

// Compare two matrices with tolerance
template <typename T>
bool compareMatrices(const T* matrix1, const T* matrix2, int rows, int cols,
                    T tolerance = 1e-5, bool verbose = false) {
    bool all_close = true;
    T max_diff = 0;
    int diff_count = 0;

    for (int i = 0; i < rows * cols; ++i) {
        T diff = std::abs(matrix1[i] - matrix2[i]);
        if (diff > tolerance) {
            all_close = false;
            diff_count++;
            max_diff = std::max(max_diff, diff);

            if (verbose && diff_count <= 10) {
                int row = i / cols;
                int col = i % cols;
                std::cout << "Mismatch at [" << row << ", " << col << "]: "
                         << matrix1[i] << " vs " << matrix2[i]
                         << " (diff: " << diff << ")" << std::endl;
            }
        }
    }

    if (verbose) {
        if (all_close) {
            std::cout << "Matrices match within tolerance " << tolerance << std::endl;
        } else {
            std::cout << "Matrices differ: " << diff_count << " elements out of "
                     << (rows * cols) << " exceed tolerance" << std::endl;
            std::cout << "Maximum difference: " << max_diff << std::endl;
        }
    }

    return all_close;
}

// Compute matrix norm (Frobenius norm)
template <typename T>
T computeMatrixNorm(const T* matrix, int rows, int cols) {
    T sum = 0;
    for (int i = 0; i < rows * cols; ++i) {
        sum += matrix[i] * matrix[i];
    }
    return std::sqrt(sum);
}

// Transpose matrix
template <typename T>
void transposeMatrix(T* out, const T* in, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

} // namespace cuda_utils