# Simplified B2B GEMM Implementation Guide

This directory contains simplified implementations of Back-to-Back GEMM fusion to help understand the core concepts without CUTLASS complexity.

## Core Concept: B2B GEMM Fusion

Back-to-Back GEMM performs two consecutive matrix multiplications:
```
First GEMM:  C = A * B0   [M,K] * [K,N] = [M,N]
Second GEMM: D = C * B1   [M,N] * [N,P] = [M,P]
```

The key optimization is avoiding writing intermediate result C to global memory.

## Two Fusion Strategies

### 1. RF (Register File) Residency (`simple_b2b_gemm_rf.cu`)

**Key Idea**: Keep intermediate results in registers between GEMMs.

```cuda
__global__ void simple_b2b_gemm_rf_kernel(...) {
    // First GEMM: compute C element
    float c_val = 0.0f;  // Keep in register!
    for (int k = 0; k < K; ++k) {
        c_val += A[row][k] * B0[k][col];
    }

    // RF Residency: c_val stays in register
    // No write to global memory!

    // Second GEMM: use c_val from register
    float d_val = 0.0f;
    for (int n = 0; n < N; ++n) {
        d_val += c_val * B1[n][colP];  // Use register value
    }

    D[row][colP] = d_val;  // Only write final result
}
```

**Advantages**:
- Fastest access (register is fastest memory)
- No shared memory needed
- Minimal memory traffic

**Limitations**:
- Limited register count
- Works best for small tiles
- Complex for large problems

### 2. Shared Memory Residency (`simple_b2b_gemm_shmem.cu`)

**Key Idea**: Store intermediate results in shared memory between GEMMs.

```cuda
__global__ void simple_b2b_gemm_shmem_kernel(...) {
    __shared__ float tile_C[TILE_M][TILE_N];  // Intermediate storage

    // First GEMM: compute C tile
    float c_accumulator = 0.0f;
    for (int k_tile = 0; k_tile < K/TILE_K; ++k_tile) {
        // Load tiles to shared memory
        __syncthreads();
        // Compute partial products
        for (int k = 0; k < TILE_K; ++k) {
            c_accumulator += tile_A[ty][k] * tile_B[k][tx];
        }
    }

    // Store in shared memory (not global!)
    tile_C[ty][tx] = c_accumulator;
    __syncthreads();

    // Second GEMM: use C from shared memory
    float d_accumulator = 0.0f;
    for (int n = 0; n < TILE_N; ++n) {
        d_accumulator += tile_C[ty][n] * tile_B1[n][tx];
    }

    D[row][colP] = d_accumulator;  // Write final result
}
```

**Advantages**:
- Can handle larger tiles
- Shared across threads in block
- More flexible than RF

**Limitations**:
- Slower than registers
- Limited shared memory size (48KB-164KB)
- Requires synchronization

## Memory Hierarchy & Performance

```
Register File (RF)     < 1 cycle    ~256KB total/SM
    ↓
Shared Memory         ~30 cycles    48-164KB/SM
    ↓
L1 Cache              ~100 cycles   128KB/SM
    ↓
L2 Cache              ~200 cycles   6MB (global)
    ↓
Global Memory         ~500 cycles   8-24GB
```

## Compilation and Running

### Build
```bash
# RF version
nvcc -arch=sm_80 -I/home/qianxu/cutlass/include \
     -I/home/qianxu/cutlass/tools/util/include \
     simple_b2b_gemm_rf.cu -o simple_b2b_gemm_rf

# Shared memory version
nvcc -arch=sm_80 -I/home/qianxu/cutlass/include \
     -I/home/qianxu/cutlass/tools/util/include \
     simple_b2b_gemm_shmem.cu -o simple_b2b_gemm_shmem
```

### Run
```bash
./simple_b2b_gemm_rf
./simple_b2b_gemm_shmem
```

## Key Differences from Full CUTLASS

### Simplified Version
- Basic tiling strategy
- Simple thread-to-data mapping
- No Tensor Core usage
- Fixed tile sizes
- Limited problem size support

### Full CUTLASS B2B GEMM
- Complex multi-level tiling
- Tensor Core acceleration
- Software pipelining
- Warp-level primitives
- Epilogue fusion
- Arbitrary problem sizes
- Multiple precision support

## Performance Comparison

| Strategy | Memory Writes | Intermediate Storage | Best For |
|----------|--------------|---------------------|----------|
| Naive | 2x (C + D) | Global Memory | Baseline |
| RF Fusion | 1x (D only) | Registers | Small tiles |
| Shmem Fusion | 1x (D only) | Shared Memory | Medium tiles |
| CUTLASS | 1x (D only) | RF + Shmem + Tensor Cores | Production |

## When to Use Each Strategy

### RF Residency
- Small intermediate matrices
- Simple element-wise operations
- Maximum performance critical

### Shared Memory Residency
- Larger intermediate matrices
- Complex access patterns
- Need thread cooperation

## Learning Path

1. **Start here**: Understand basic concepts with these simplified versions
2. **Next**: Study CUTLASS threadblock and warp implementations
3. **Advanced**: Explore Tensor Core integration and epilogue fusion
4. **Expert**: Custom kernel fusion for specific workloads

## Files in This Directory

- `simple_b2b_gemm_rf.cu` - Simplified RF-resident implementation
- `simple_b2b_gemm_shmem.cu` - Simplified shared memory implementation
- `fused_two_gemms_*_rf.cu` - Full CUTLASS RF implementations
- `fused_two_gemms_*_shmem.cu` - Full CUTLASS shared memory implementations

## Debugging Tips

1. **Start small**: Use 16x16 or 32x32 matrices
2. **Print intermediate values**: Add printf in kernels
3. **Check memory access**: Use cuda-memcheck
4. **Profile**: Use nvprof or Nsight Compute
5. **Compare strategies**: Run both RF and Shmem versions

## Common Issues

### Issue: Results don't match
- Check matrix layouts (row vs column major)
- Verify tile boundary conditions
- Ensure proper synchronization

### Issue: Performance not improved
- Problem size too small (kernel launch overhead)
- Not memory bandwidth limited
- Need Tensor Core acceleration

## Next Steps

After understanding these simplified versions:

1. Study the full CUTLASS implementations in `device/b2b_gemm.h`
2. Learn about software pipelining in `threadblock/b2b_mma_multistage.h`
3. Explore Tensor Core usage in `arch/mma_sm80.h`
4. Understand epilogue fusion in `epilogue/threadblock/`

These simplified implementations demonstrate the core idea of B2B GEMM fusion. The full CUTLASS versions add many optimizations but follow the same fundamental principle: keep intermediate results in fast memory!