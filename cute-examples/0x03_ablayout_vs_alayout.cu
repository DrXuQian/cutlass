#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// ALayout_64x16: RS mode TV Layout (128 threads x 8 values)
// Used for Register-Shared MMA where A operand comes from registers
using ALayout_64x16 = Layout<
    Shape <Shape <  _4, _8, _4>, Shape < _2, _2,  _2>>,
    Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>
>;

// ABLayout_64x16: SS mode TV Layout (128 threads x 1024 values)
// Used for Shared-Shared MMA where all threads see the same data
using ABLayout_64x16 = Layout<
    Shape <_128, Shape <_64, _16>>,
    Stride<  _0, Stride< _1, _64>>
>;

int main() {
    printf("============================================================\n");
    printf("    0x03 ABLayout vs ALayout: SS and RS Mode Comparison\n");
    printf("============================================================\n\n");

    //==========================================================================
    // ALayout_64x16 (RS Mode)
    //==========================================================================

    printf("=== ALayout_64x16 (RS Mode) ===\n\n");
    {
        auto layout = ALayout_64x16{};

        printf("Layout: "); print(layout); printf("\n\n");

        printf("Structure:\n");
        printf("  Thr shape:  "); print(shape<0>(layout));
        printf(" = %d threads\n", int(size<0>(layout)));
        printf("  Val shape:  "); print(shape<1>(layout));
        printf(" = %d values/thread\n", int(size<1>(layout)));
        printf("  Total: %d elements\n\n", int(size(layout)));

        printf("Key characteristics:\n");
        printf("  - Thr stride != 0 (threads access DIFFERENT data)\n");
        printf("  - Each thread has its own 8 register values\n");
        printf("  - Data lives in REGISTERS\n\n");

        printf("Thread 0's 8 values (offset -> (m,k)):\n");
        for (int v = 0; v < 8; ++v) {
            int offset = layout(0, v);
            int m = offset % 64;
            int k = offset / 64;
            printf("  V%d: offset=%3d -> (m=%2d, k=%2d)\n", v, offset, m, k);
        }

        printf("\nCompare Thread 0 vs Thread 1:\n");
        printf("Val | Thread 0       | Thread 1       | Same?\n");
        printf("----|----------------|----------------|------\n");
        for (int v = 0; v < 4; ++v) {
            int off0 = layout(0, v);
            int off1 = layout(1, v);
            printf(" %d  | offset=%3d     | offset=%3d     | %s\n",
                   v, off0, off1, off0 == off1 ? "YES" : "NO");
        }
        printf("\n-> Different threads access DIFFERENT offsets!\n");
    }

    //==========================================================================
    // ABLayout_64x16 (SS Mode)
    //==========================================================================

    printf("\n=== ABLayout_64x16 (SS Mode) ===\n\n");
    {
        auto layout = ABLayout_64x16{};

        printf("Layout: "); print(layout); printf("\n\n");

        printf("Structure:\n");
        printf("  Thr shape:  %d threads\n", int(size<0>(layout)));
        printf("  Val shape:  "); print(shape<1>(layout));
        printf(" = %d values/thread\n", int(size<1>(layout)));
        printf("  Total: %d elements\n\n", int(size(layout)));

        printf("Key characteristics:\n");
        printf("  - Thr stride = 0 (all threads see SAME data!)\n");
        printf("  - Each thread 'owns' entire 64x16 = 1024 values\n");
        printf("  - Data lives in SHARED MEMORY (accessed via descriptor)\n\n");

        printf("Thread 0's first 8 values (offset -> (m,k)):\n");
        for (int v = 0; v < 8; ++v) {
            int offset = layout(0, v);
            int m = offset % 64;
            int k = offset / 64;
            printf("  V%d: offset=%3d -> (m=%2d, k=%2d)\n", v, offset, m, k);
        }

        printf("\nCompare Thread 0 vs Thread 1:\n");
        printf("Val | Thread 0       | Thread 1       | Same?\n");
        printf("----|----------------|----------------|------\n");
        for (int v = 0; v < 4; ++v) {
            int off0 = layout(0, v);
            int off1 = layout(1, v);
            printf(" %d  | offset=%3d     | offset=%3d     | %s\n",
                   v, off0, off1, off0 == off1 ? "YES" : "NO");
        }
        printf("\n-> All threads see the SAME offsets (Thr stride = 0)!\n");
    }

    //==========================================================================
    // Side-by-side comparison
    //==========================================================================

    printf("\n============================================================\n");
    printf("                  Summary Comparison\n");
    printf("============================================================\n\n");

    printf("+------------------+---------------------+---------------------+\n");
    printf("| Property         | ALayout (RS Mode)   | ABLayout (SS Mode)  |\n");
    printf("+------------------+---------------------+---------------------+\n");
    printf("| Thr Stride       | != 0                | = 0                 |\n");
    printf("| Data Location    | Registers           | Shared Memory       |\n");
    printf("| Values/Thread    | 8                   | 1024 (64x16)        |\n");
    printf("| Thread Data      | Different per thread| Same for all threads|\n");
    printf("| MMA Operand      | A from registers    | A,B from SMEM desc  |\n");
    printf("+------------------+---------------------+---------------------+\n\n");

    printf("Usage in WGMMA:\n");
    printf("  RS Mode: wgmma.mma_async ... {a0,a1,a2,a3}, desc_b, ...\n");
    printf("           A comes from 4 registers, B from SMEM descriptor\n\n");
    printf("  SS Mode: wgmma.mma_async ... desc_a, desc_b, ...\n");
    printf("           Both A and B come from SMEM descriptors\n");

    return 0;
}
