#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <sstream>

// Macro for checking CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Macro for checking the last CUDA error
#define CHECK_LAST_CUDA_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

namespace cuda_utils {

// Get device properties
inline void printDeviceInfo(int device = 0) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "=== CUDA Device Information ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Clock rate: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory clock rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak memory bandwidth: "
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
              << " GB/s" << std::endl;
    std::cout << "==============================\n" << std::endl;
}

// Calculate optimal block size for a kernel
inline dim3 calculateBlockSize(int total_threads, int max_threads_per_block = 256) {
    int threads = std::min(total_threads, max_threads_per_block);

    // Try to make it a multiple of warp size (32)
    if (threads > 32) {
        threads = (threads / 32) * 32;
    }

    return dim3(threads);
}

// Calculate grid size given total work and block size
inline dim3 calculateGridSize(int total_work, int block_size) {
    return dim3((total_work + block_size - 1) / block_size);
}

// Calculate 2D block dimensions
inline dim3 calculate2DBlockSize(int width, int height,
                                 int max_threads_per_block = 256) {
    // Common 2D block sizes
    if (width * height <= 256) {
        if (width <= 16 && height <= 16) {
            return dim3(16, 16);
        } else if (width <= 32 && height <= 8) {
            return dim3(32, 8);
        } else if (width <= 8 && height <= 32) {
            return dim3(8, 32);
        }
    }

    // Default to 16x16
    return dim3(16, 16);
}

// Calculate 2D grid size
inline dim3 calculate2DGridSize(int width, int height, dim3 block_size) {
    return dim3((width + block_size.x - 1) / block_size.x,
                (height + block_size.y - 1) / block_size.y);
}

// Check if a number is a power of 2
inline bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Round up to the nearest power of 2
inline int roundUpToPowerOfTwo(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Get available device memory
inline void getMemoryInfo(size_t& free_bytes, size_t& total_bytes) {
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
}

// Print memory usage
inline void printMemoryUsage() {
    size_t free_bytes, total_bytes;
    getMemoryInfo(free_bytes, total_bytes);

    double free_mb = free_bytes / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    double used_mb = total_mb - free_mb;

    std::cout << "GPU Memory: " << used_mb << " MB / " << total_mb << " MB used ("
              << (used_mb / total_mb * 100.0) << "%)" << std::endl;
}

// Synchronize and check for errors
inline void syncAndCheck() {
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();
}

// Format bytes to human readable string
inline std::string formatBytes(size_t bytes) {
    std::ostringstream oss;
    if (bytes < 1024) {
        oss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        oss << (bytes / 1024.0) << " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        oss << (bytes / (1024.0 * 1024.0)) << " MB";
    } else {
        oss << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
    }
    return oss.str();
}

// Calculate theoretical occupancy
inline float calculateOccupancy(int num_blocks, int threads_per_block, int device = 0) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int num_sms = prop.multiProcessorCount;

    int active_threads = num_blocks * threads_per_block;
    int max_threads = num_sms * max_threads_per_sm;

    return std::min(1.0f, (float)active_threads / max_threads);
}

} // namespace cuda_utils