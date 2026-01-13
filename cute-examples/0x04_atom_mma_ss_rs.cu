#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// SS Mode Atom Layout (from CUTLASS)
template <int M, int N>
using SS_Layout = Layout<
    Shape <_128, Shape <Int<M>, Int<N>>>,
    Stride<  _0, Stride<    _1, Int<M>>>
>;

// RS Mode Atom Layout for A (from CUTLASS)
using RS_ALayout_64x16 = Layout<
    Shape <Shape <  _4, _8, _4>, Shape < _2, _2,  _2>>,
    Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>
>;

// C Accumulator Layout
template <int N>
using CLayout_64xN = Layout<
    Shape <Shape <  _4, _8, _4>, Shape < _2, _2, Int<N/8>>>,
    Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>
>;

int main() {
    printf("============================================================\n");
    printf("    0x04 CUTLASS Atom MMA: SS vs RS Mode\n");
    printf("============================================================\n\n");

    //==========================================================================
    // SS Mode: Both A and B from SMEM
    //==========================================================================

    printf("=== SS Mode (Shared-Shared) ===\n\n");

    printf("In SS Mode, both A and B operands come from shared memory.\n");
    printf("All threads share the same SMEM descriptor.\n\n");

    {
        using ALayout = SS_Layout<64, 16>;
        using BLayout = SS_Layout<16, 16>;
        using CLayout = CLayout_64xN<16>;

        printf("A Layout (64x16):\n");
        printf("  "); print(ALayout{}); printf("\n");
        printf("  Thr stride = 0 (all threads share same data)\n");
        printf("  Val shape = (64,16) = 1024 values\n\n");

        printf("B Layout (16x16):\n");
        printf("  "); print(BLayout{}); printf("\n");
        printf("  Thr stride = 0 (all threads share same data)\n");
        printf("  Val shape = (16,16) = 256 values\n\n");

        printf("C Layout (64x16):\n");
        printf("  "); print(CLayout{}); printf("\n");
        printf("  Thr shape = (4,8,4) = 128 threads\n");
        printf("  Val shape = (2,2,2) = 8 values/thread\n\n");
    }

    printf("PTX for SS Mode:\n");
    printf("  wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n");
    printf("      {d0..d7},      // 8 output registers\n");
    printf("      desc_a,        // SMEM descriptor for A\n");
    printf("      desc_b,        // SMEM descriptor for B\n");
    printf("      scale_d, ...;  // scale and transpose flags\n\n");

    //==========================================================================
    // RS Mode: A from Register, B from SMEM
    //==========================================================================

    printf("=== RS Mode (Register-Shared) ===\n\n");

    printf("In RS Mode, A operand comes from registers, B from SMEM.\n");
    printf("Each thread has its own A data in registers.\n\n");

    {
        using ALayout = RS_ALayout_64x16;
        using BLayout = SS_Layout<16, 16>;
        using CLayout = CLayout_64xN<16>;

        printf("A Layout (64x16, RS mode):\n");
        printf("  "); print(ALayout{}); printf("\n");
        printf("  Thr shape = (4,8,4) = 128 threads\n");
        printf("  Val shape = (2,2,2) = 8 values/thread\n");
        printf("  Thr stride != 0 (each thread has different data)\n\n");

        printf("Thread 0's A values:\n");
        for (int v = 0; v < 8; ++v) {
            int offset = ALayout{}(0, v);
            int m = offset % 64;
            int k = offset / 64;
            printf("  V%d: offset=%3d -> (m=%2d, k=%2d)\n", v, offset, m, k);
        }
        printf("\n");

        printf("B Layout (16x16, still SS):\n");
        printf("  "); print(BLayout{}); printf("\n");
        printf("  Thr stride = 0 (shared via descriptor)\n\n");

        printf("C Layout (same as SS mode):\n");
        printf("  "); print(CLayout{}); printf("\n\n");
    }

    printf("PTX for RS Mode:\n");
    printf("  wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n");
    printf("      {d0..d7},      // 8 output registers\n");
    printf("      {a0,a1,a2,a3}, // 4 input registers for A\n");
    printf("      desc_b,        // SMEM descriptor for B\n");
    printf("      scale_d, ...;  // scale and transpose flags\n\n");

    //==========================================================================
    // Comparison Table
    //==========================================================================

    printf("============================================================\n");
    printf("                  Mode Comparison\n");
    printf("============================================================\n\n");

    printf("+------------------+------------------------+------------------------+\n");
    printf("| Aspect           | SS Mode                | RS Mode                |\n");
    printf("+------------------+------------------------+------------------------+\n");
    printf("| A Source         | SMEM (descriptor)      | Register               |\n");
    printf("| B Source         | SMEM (descriptor)      | SMEM (descriptor)      |\n");
    printf("| A Thr Stride     | 0                      | != 0                   |\n");
    printf("| A per Thread     | 1024 (shared)          | 8 (private)            |\n");
    printf("| Use Case         | TMA direct load        | A needs preprocessing  |\n");
    printf("| A Flexibility    | Fixed layout           | Thread-local transform |\n");
    printf("+------------------+------------------------+------------------------+\n\n");

    printf("When to use each mode:\n");
    printf("  SS Mode:\n");
    printf("    - TMA loads A directly to SMEM\n");
    printf("    - No per-thread A transformation needed\n");
    printf("    - Simpler, lower register pressure\n\n");
    printf("  RS Mode:\n");
    printf("    - Need to transform A before MMA\n");
    printf("    - A data comes from previous computation\n");
    printf("    - More flexibility but uses more registers\n");

    //==========================================================================
    // Register Usage
    //==========================================================================

    printf("\n============================================================\n");
    printf("                  Register Usage\n");
    printf("============================================================\n\n");

    printf("For m64n16k16 MMA:\n\n");

    printf("SS Mode registers per thread:\n");
    printf("  A: 0 (uses descriptor)\n");
    printf("  B: 0 (uses descriptor)\n");
    printf("  C: 8 (accumulator)\n");
    printf("  Total: 8 registers\n\n");

    printf("RS Mode registers per thread:\n");
    printf("  A: 4 (a0,a1,a2,a3 for 8 f16 values)\n");
    printf("  B: 0 (uses descriptor)\n");
    printf("  C: 8 (accumulator)\n");
    printf("  Total: 12 registers\n\n");

    printf("Note: C accumulator layout is IDENTICAL in both modes.\n");
    printf("The difference is only in how A operand is provided.\n");

    return 0;
}
