#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// CLayout_64xN definition (from CUTLASS)
template <int N>
using CLayout_64xN = Layout<Shape <Shape <  _4, _8, _4>, Shape < _2, _2, Int<N/8>>>,
                            Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;

using CLayout_64x16 = CLayout_64xN<16>;

int main() {
    printf("============================================================\n");
    printf("    0x01 TV Layout (Thread-Value Layout) Examples\n");
    printf("============================================================\n\n");

    //==========================================================================
    // Example 1: Simple contiguous distribution
    //==========================================================================

    printf("=== Example 1: Contiguous Distribution ===\n");
    {
        // 32 elements distributed to 4 threads, 8 values per thread
        using TVLayout = Layout<Shape<_4, _8>, Stride<_8, _1>>;
        auto tv = TVLayout{};

        printf("TVLayout: "); print(tv); printf("\n");
        printf("  Shape: (Thr=%d, Val=%d)\n", int(size<0>(tv)), int(size<1>(tv)));
        printf("  Stride: (Thr_stride=%d, Val_stride=%d)\n",
               int(stride<0>(tv)), int(stride<1>(tv)));
        printf("\nFormula: offset = thread_id * 8 + value_id * 1\n\n");

        printf("TV Table:\n");
        printf("       Val: ");
        for (int v = 0; v < 8; ++v) printf("%3d ", v);
        printf("\n");
        printf("           +");
        for (int v = 0; v < 8; ++v) printf("---+");
        printf("\n");

        for (int t = 0; t < 4; ++t) {
            printf("Thr %d      |", t);
            for (int v = 0; v < 8; ++v) {
                printf("%3d|", int(tv(t, v)));
            }
            printf("\n");
            printf("           +");
            for (int v = 0; v < 8; ++v) printf("---+");
            printf("\n");
        }
    }

    //==========================================================================
    // Example 2: Interleaved distribution
    //==========================================================================

    printf("\n=== Example 2: Interleaved Distribution ===\n");
    {
        using TVLayout = Layout<Shape<_4, _8>, Stride<_1, _4>>;
        auto tv = TVLayout{};

        printf("TVLayout: "); print(tv); printf("\n");
        printf("  Stride: (Thr_stride=%d, Val_stride=%d)\n",
               int(stride<0>(tv)), int(stride<1>(tv)));
        printf("\nFormula: offset = thread_id * 1 + value_id * 4\n\n");

        printf("TV Table:\n");
        printf("       Val: ");
        for (int v = 0; v < 8; ++v) printf("%3d ", v);
        printf("\n");
        printf("           +");
        for (int v = 0; v < 8; ++v) printf("---+");
        printf("\n");

        for (int t = 0; t < 4; ++t) {
            printf("Thr %d      |", t);
            for (int v = 0; v < 8; ++v) {
                printf("%3d|", int(tv(t, v)));
            }
            printf("\n");
            printf("           +");
            for (int v = 0; v < 8; ++v) printf("---+");
            printf("\n");
        }
    }

    //==========================================================================
    // Example 3: Complex TV Layout (CLayout_64x16)
    //==========================================================================

    printf("\n=== Example 3: CLayout_64x16 (128 threads x 8 values) ===\n");
    {
        auto tv = CLayout_64x16{};

        printf("CLayout_64x16: "); print(tv); printf("\n");
        printf("  Thr shape: "); print(shape<0>(tv));
        printf(" = %d threads\n", int(size<0>(tv)));
        printf("  Val shape: "); print(shape<1>(tv));
        printf(" = %d values/thread\n", int(size<1>(tv)));
        printf("  Total: %d elements\n\n", int(size(tv)));

        printf("Thread 0's 8 values in 64x16 Atom:\n");
        printf("Val_idx -> offset -> (m, n)\n");
        for (int v = 0; v < int(size<1>(tv)); ++v) {
            int offset = tv(0, v);
            int m = offset % 64;
            int n = offset / 64;
            printf("  V%d -> offset %3d -> (m=%2d, n=%2d)\n", v, offset, m, n);
        }

        printf("\nComparing Thread 0-3:\n");
        printf("Thread | V0 (m,n) | V1 (m,n) | V2 (m,n) | V3 (m,n)\n");
        printf("-------|----------|----------|----------|----------\n");
        for (int t = 0; t < 4; ++t) {
            printf("   %d   |", t);
            for (int v = 0; v < 4; ++v) {
                int offset = tv(t, v);
                int m = offset % 64;
                int n = offset / 64;
                printf(" (%2d,%2d) |", m, n);
            }
            printf("\n");
        }
    }

    //==========================================================================
    // Example 4: SS mode (Thr stride = 0)
    //==========================================================================

    printf("\n=== Example 4: SS Mode (Thr stride = 0) ===\n");
    {
        using SSLayout = Layout<Shape<_4, _8>, Stride<_0, _1>>;
        auto tv = SSLayout{};

        printf("SS Layout: "); print(tv); printf("\n");
        printf("  Thr stride = 0 (all threads share same data!)\n\n");

        printf("TV Table (all rows identical):\n");
        printf("       Val: ");
        for (int v = 0; v < 8; ++v) printf("%3d ", v);
        printf("\n");
        printf("           +");
        for (int v = 0; v < 8; ++v) printf("---+");
        printf("\n");

        for (int t = 0; t < 4; ++t) {
            printf("Thr %d      |", t);
            for (int v = 0; v < 8; ++v) {
                printf("%3d|", int(tv(t, v)));
            }
            printf(" <- same!\n");
            printf("           +");
            for (int v = 0; v < 8; ++v) printf("---+");
            printf("\n");
        }
    }

    printf("\n============================================================\n");
    printf("Key Insights:\n");
    printf("  - TV Layout: (thread_id, value_id) -> offset\n");
    printf("  - Thr Slice: Base position for each thread\n");
    printf("  - Val Slice: Relative offset within thread's data\n");
    printf("  - SS mode: Thr stride = 0, all threads see same data\n");
    printf("============================================================\n");

    return 0;
}
