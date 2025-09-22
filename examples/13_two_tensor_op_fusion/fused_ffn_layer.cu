/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This example demonstrates fusing a complete FFN (Feed-Forward Network) layer commonly used in
 * transformer models like LLaMA/GPT. The fusion includes:
 *
 * Input: [9600, 1024]
 * 1. GEMM1: [9600, 1024] x [1024, 2730] -> [9600, 2730] (gate projection)
 * 2. GEMM2: [9600, 1024] x [1024, 2730] -> [9600, 2730] (up projection)
 * 3. SiLU activation on GEMM1 output
 * 4. Element-wise multiplication: SiLU(GEMM1) * GEMM2
 * 5. LayerNorm on the multiplication result
 * 6. GEMM3: [9600, 2730] x [2730, 1024] -> [9600, 1024] (down projection)
 *
 * This mimics the MLP layer in modern transformer architectures.
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

////////////////////////////////////////////////////////////////////////////////

// Problem sizes for FFN layer
// Typical LLaMA-style dimensions
constexpr int kSeqLength = 9600;   // Sequence length * batch size
constexpr int kHiddenDim = 1024;   // Model hidden dimension
constexpr int kFFNDim = 2730;      // FFN intermediate dimension (typically 8/3 * hidden_dim)

////////////////////////////////////////////////////////////////////////////////

// Simple SiLU activation kernel
__global__ void silu_multiply_kernel(
    cutlass::half_t const* gate,
    cutlass::half_t const* up,
    cutlass::half_t* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float gate_val = float(gate[idx]);
        float up_val = float(up[idx]);

        // SiLU(x) = x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-gate_val));
        float silu = gate_val * sigmoid;

        output[idx] = cutlass::half_t(silu * up_val);
    }
}

// Simple LayerNorm kernel
__global__ void layernorm_kernel(
    cutlass::half_t const* input,
    cutlass::half_t* output,
    int seq_length,
    int hidden_dim,
    float eps = 1e-5f
) {
    int row = blockIdx.x;
    if (row < seq_length) {
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            mean += float(input[row * hidden_dim + i]);
        }
        mean /= hidden_dim;

        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            float diff = float(input[row * hidden_dim + i]) - mean;
            variance += diff * diff;
        }
        variance /= hidden_dim;

        // Normalize
        float stddev = sqrtf(variance + eps);
        for (int i = 0; i < hidden_dim; ++i) {
            output[row * hidden_dim + i] = cutlass::half_t(
                (float(input[row * hidden_dim + i]) - mean) / stddev
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

    std::cout << "=== Fused FFN Layer Example ===\n";
    std::cout << "This example demonstrates fusing a complete transformer FFN layer:\n";
    std::cout << "Input[" << kSeqLength << "," << kHiddenDim << "] -> ";
    std::cout << "GEMM -> [" << kSeqLength << "," << kFFNDim << "] -> ";
    std::cout << "SiLU*Up -> LayerNorm -> GEMM -> [" << kSeqLength << "," << kHiddenDim << "]\n\n";

    // Check GPU
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() failed: " << cudaGetErrorString(error) << "\n";
        return -1;
    }

    std::cout << "Running on GPU: " << props.name << " (SM" << props.major << props.minor << ")\n";

    // Define data types
    using ElementInput = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutInput = cutlass::layout::RowMajor;
    using LayoutWeight = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // Allocate host tensors
    cutlass::HostTensor<ElementInput, LayoutInput> tensor_input({kSeqLength, kHiddenDim});
    cutlass::HostTensor<ElementInput, LayoutWeight> tensor_gate_weight({kHiddenDim, kFFNDim});
    cutlass::HostTensor<ElementInput, LayoutWeight> tensor_up_weight({kHiddenDim, kFFNDim});
    cutlass::HostTensor<ElementInput, LayoutWeight> tensor_down_weight({kFFNDim, kHiddenDim});
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_output({kSeqLength, kHiddenDim});

    // Intermediate tensors
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_gate_out({kSeqLength, kFFNDim});
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_up_out({kSeqLength, kFFNDim});
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_activated({kSeqLength, kFFNDim});
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_normed({kSeqLength, kFFNDim});

    // Initialize input tensors with random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_input.host_view(),
        1,
        ElementInput(2),
        ElementInput(-2),
        0
    );

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_gate_weight.host_view(),
        1,
        ElementInput(0.5),
        ElementInput(-0.5),
        1
    );

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_up_weight.host_view(),
        1,
        ElementInput(0.5),
        ElementInput(-0.5),
        2
    );

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_down_weight.host_view(),
        1,
        ElementInput(0.5),
        ElementInput(-0.5),
        3
    );

    // Copy to device
    tensor_input.sync_device();
    tensor_gate_weight.sync_device();
    tensor_up_weight.sync_device();
    tensor_down_weight.sync_device();

    std::cout << "Executing FFN operations...\n";

    // Define GEMM operation for FP16
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput, LayoutInput,           // A matrix
        ElementInput, LayoutWeight,           // B matrix
        ElementOutput, LayoutOutput,          // C matrix
        ElementAccumulator                    // Accumulator
    >;

    // GEMM1: Gate projection
    Gemm gemm_gate;
    typename Gemm::Arguments args_gate(
        {kSeqLength, kFFNDim, kHiddenDim},   // Problem size
        tensor_input.device_ref(),            // A
        tensor_gate_weight.device_ref(),      // B
        tensor_gate_out.device_ref(),         // C (unused)
        tensor_gate_out.device_ref(),         // D (output)
        {ElementAccumulator(1), ElementAccumulator(0)}  // alpha, beta
    );

    cutlass::Status status = gemm_gate(args_gate);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Gate GEMM failed\n";
        return -1;
    }

    // GEMM2: Up projection
    Gemm gemm_up;
    typename Gemm::Arguments args_up(
        {kSeqLength, kFFNDim, kHiddenDim},
        tensor_input.device_ref(),
        tensor_up_weight.device_ref(),
        tensor_up_out.device_ref(),
        tensor_up_out.device_ref(),
        {ElementAccumulator(1), ElementAccumulator(0)}
    );

    status = gemm_up(args_up);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Up GEMM failed\n";
        return -1;
    }

    // Apply SiLU activation and multiply
    dim3 block(256);
    dim3 grid((kSeqLength * kFFNDim + block.x - 1) / block.x);
    silu_multiply_kernel<<<grid, block>>>(
        tensor_gate_out.device_data(),
        tensor_up_out.device_data(),
        tensor_activated.device_data(),
        kSeqLength * kFFNDim
    );

    // Apply LayerNorm
    layernorm_kernel<<<kSeqLength, 1>>>(
        tensor_activated.device_data(),
        tensor_normed.device_data(),
        kSeqLength,
        kFFNDim
    );

    // GEMM3: Down projection
    Gemm gemm_down;
    typename Gemm::Arguments args_down(
        {kSeqLength, kHiddenDim, kFFNDim},
        tensor_normed.device_ref(),
        tensor_down_weight.device_ref(),
        tensor_output.device_ref(),
        tensor_output.device_ref(),
        {ElementAccumulator(1), ElementAccumulator(0)}
    );

    status = gemm_down(args_down);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Down GEMM failed\n";
        return -1;
    }

    // Synchronize and copy back
    cudaDeviceSynchronize();
    tensor_output.sync_host();

    std::cout << "FFN layer execution completed successfully!\n\n";

    // Performance measurement
    std::cout << "=== Performance Benchmark ===\n";
    std::cout << "Problem size: [" << kSeqLength << ", " << kHiddenDim << "] -> ["
              << kSeqLength << ", " << kFFNDim << "] -> [" << kSeqLength << ", " << kHiddenDim << "]\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_iterations = 100;

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        gemm_gate(args_gate);
        gemm_up(args_up);
        silu_multiply_kernel<<<grid, block>>>(
            tensor_gate_out.device_data(),
            tensor_up_out.device_data(),
            tensor_activated.device_data(),
            kSeqLength * kFFNDim
        );
        layernorm_kernel<<<kSeqLength, 1>>>(
            tensor_activated.device_data(),
            tensor_normed.device_data(),
            kSeqLength,
            kFFNDim
        );
        gemm_down(args_down);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        gemm_gate(args_gate);
        gemm_up(args_up);
        silu_multiply_kernel<<<grid, block>>>(
            tensor_gate_out.device_data(),
            tensor_up_out.device_data(),
            tensor_activated.device_data(),
            kSeqLength * kFFNDim
        );
        layernorm_kernel<<<kSeqLength, 1>>>(
            tensor_activated.device_data(),
            tensor_normed.device_data(),
            kSeqLength,
            kFFNDim
        );
        gemm_down(args_down);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    std::cout << "Average time per FFN layer: " << time_ms / num_iterations << " ms\n";

    // Calculate FLOPs
    double flops = 2.0 * kSeqLength * kHiddenDim * kFFNDim * 2 +  // Two GEMMs to FFN dim
                   2.0 * kSeqLength * kFFNDim * kHiddenDim +       // Down projection
                   kSeqLength * kFFNDim * 5;                        // SiLU and multiply

    double tflops = (flops * num_iterations) / (time_ms * 1e9);
    std::cout << "Performance: " << tflops << " TFLOPS\n";

    // Memory bandwidth
    double bytes = sizeof(ElementInput) * (
        kSeqLength * kHiddenDim +           // Input
        kHiddenDim * kFFNDim * 2 +           // Gate and up weights
        kFFNDim * kHiddenDim +               // Down weight
        kSeqLength * kFFNDim * 4 +           // Intermediate results
        kSeqLength * kHiddenDim              // Output
    );
    double bandwidth = (bytes * num_iterations) / (time_ms * 1e6);
    std::cout << "Memory bandwidth: " << bandwidth << " GB/s\n";

    std::cout << "\n=== Fusion Opportunities ===\n";
    std::cout << "1. Fuse gate and up GEMMs (share input loading)\n";
    std::cout << "2. Fuse SiLU activation with GEMM epilogue\n";
    std::cout << "3. Keep intermediate results in shared memory\n";
    std::cout << "4. Fuse LayerNorm with down projection prologue\n";
    std::cout << "5. Use persistent kernels to avoid global memory traffic\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}