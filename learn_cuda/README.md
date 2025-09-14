# CUDA Learning Examples with CUTLASS

This repository contains CUDA programming examples and CUTLASS integration tutorials, focusing on GEMM operations and various optimizations.

## Repository Structure

```
learn_cuda/
├── common/                    # Common utility functions
│   ├── matrix_utils.h        # Matrix allocation, initialization, comparison
│   ├── cpu_gemm.h            # CPU GEMM reference implementation
│   ├── cuda_timer.h          # CUDA timing utilities
│   └── cuda_utils.h          # CUDA error checking and device utilities
│
├── 0_basic_gemm/             # Basic GEMM with CUTLASS
│   ├── basic_gemm.cu        # CUTLASS GEMM with verification
│   └── Makefile
│
└── 1_gemm_relu/              # GEMM + ReLU fusion examples
    ├── gemm_relu.cu          # Multiple GEMM+ReLU implementations
    ├── gemm_relu_cutlass.cu  # CUTLASS with fused ReLU epilogue
    ├── gemm_custom_epilogue.cu # Custom epilogue implementation
    ├── test_swizzle.cu       # Thread block swizzle testing
    ├── README.md             # Detailed epilogue design documentation
    └── Makefile
```

## Features

### Common Utilities (`common/`)
- **Matrix utilities**: Memory allocation, initialization (random, sequential, constant, identity)
- **CPU GEMM**: Reference implementation for verification
- **CUDA utilities**: Error checking, device info, memory management
- **Timing**: High-precision CUDA event-based timing

### Basic GEMM (`0_basic_gemm/`)
- CUTLASS GEMM implementation
- CPU verification
- Performance benchmarking
- Matrix layout handling (Row-major/Column-major)

### GEMM + ReLU Fusion (`1_gemm_relu/`)
- **Naive implementation**: Simple GEMM + ReLU kernel
- **Tiled implementation**: Shared memory optimization
- **cuBLAS + ReLU**: Separate kernels approach
- **CUTLASS fusion**: Built-in `LinearCombinationRelu` epilogue
- **Custom epilogue**: User-defined epilogue functor implementation

## Building and Running

### Prerequisites
- CUDA Toolkit (11.0 or later)
- CUTLASS library
- C++ compiler with C++17 support
- GPU with compute capability 7.0 or higher

### Build Examples

```bash
# Build basic GEMM
cd 0_basic_gemm
make

# Build GEMM + ReLU examples
cd 1_gemm_relu
make all

# Run examples
./basic_gemm
./gemm_relu
./gemm_relu_cutlass
./gemm_custom_epilogue
```

### Compilation Flags
```makefile
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr
INCLUDES = -I../../include -I../common
```

## Performance Results

On NVIDIA GeForce RTX 5070:

### Basic GEMM (512×512×512)
- CUTLASS: 1313 GFLOPS
- CPU Reference: 2.6 GFLOPS
- Speedup: ~505x

### GEMM + ReLU (1024×1024×1024)
- Naive kernel: 1959 GFLOPS
- Tiled kernel: 2654 GFLOPS
- cuBLAS + ReLU: 15,690 GFLOPS
- CUTLASS fused: 9,672 GFLOPS
- Speedup vs CPU: ~10,000x

## Key Concepts

### 1. Memory Layout
- Row-major vs Column-major storage
- Leading dimension handling
- Memory coalescing patterns

### 2. Kernel Fusion
- Reduces memory bandwidth requirements
- Eliminates intermediate memory writes
- Improves cache utilization

### 3. CUTLASS Epilogue
- Fuses post-processing with GEMM
- Custom epilogue functors for flexibility
- Vectorized operations with Fragments

### 4. Thread Block Swizzle
- Reorders thread block execution
- Improves memory access patterns
- Reduces cache conflicts

## Custom Epilogue Design

The custom epilogue implementation demonstrates:

1. **Fragment-based processing**: Vectorized operations on data chunks
2. **Parameterized computation**: Alpha/Beta scaling factors
3. **Conditional memory access**: Skip unnecessary reads when beta=0
4. **Activation functions**: Easy to extend with different activations

Example custom epilogue structure:
```cpp
template<typename ElementOutput, int Count>
class CustomLinearCombinationRelu {
    FragmentOutput operator()(
        FragmentAccumulator const &accumulator,
        FragmentC const &source) {
        // Linear combination + ReLU activation
        for (int i = 0; i < Count; ++i) {
            result = alpha * accumulator[i] + beta * source[i];
            output[i] = max(0, result);  // ReLU
        }
    }
};
```

## Learning Path

1. **Start with basic GEMM** (`0_basic_gemm/`)
   - Understand CUTLASS API
   - Learn about verification and benchmarking

2. **Explore kernel fusion** (`1_gemm_relu/`)
   - Compare different implementation approaches
   - Understand performance trade-offs

3. **Study custom epilogues** (`gemm_custom_epilogue.cu`)
   - Learn CUTLASS internals
   - Implement custom operations

4. **Experiment with optimizations**
   - Try different tile sizes
   - Test various swizzle strategies
   - Profile and analyze performance

## Extensions

Possible extensions to explore:
- Other activation functions (Sigmoid, Tanh, LeakyReLU)
- Bias addition
- Batch GEMM operations
- Mixed precision (FP16, INT8)
- Tensor Core utilization

## References

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

## License

This project is for educational purposes. Please refer to NVIDIA CUTLASS license for CUTLASS components.

## Author

Created as part of CUDA and CUTLASS learning journey.