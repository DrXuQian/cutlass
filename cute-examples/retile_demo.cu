// Microbenchmark: MMA output retile for stmatrix
// MMA: SM80_16x8x16_F32F16F16F32_TN (Ampere)
// Copy: SM80 stmatrix-like 8x8x4 pattern

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/layout.hpp>

using namespace cute;

template <class Layout>
void print_layout_info(const char* name, Layout const& layout) {
    printf("%s:\n", name);
    printf("  Shape: "); print(shape(layout)); printf("\n");
    printf("  Stride: "); print(stride(layout)); printf("\n");
    printf("  Size: %d, Cosize: %d\n", (int)size(layout), (int)cosize(layout));
}

int main() {
    printf("=== CuTe retile_S Demo ===\n\n");

    // ========================================
    // 1. Define TiledMMA (Ampere SM80_16x8x16)
    // ========================================

    // MMA Atom: 16x8x16, F32 accumulator, F16 operands
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    // TiledMMA: 2x2 thread tiling = 4 warps = 128 threads
    // Tile size: 32x16 per iteration, with 2x4 iterations = 64x64 total
    using TiledMMA = TiledMMA<
        MMA_Atom,
        Layout<Shape<_2, _2, _1>>,  // 2x2 warps in M, N
        Tile<_32, _32, _16>         // Tile shape MxNxK
    >;

    TiledMMA tiled_mma;

    printf("1. TiledMMA Configuration:\n");
    printf("   Tile size: (%d, %d, %d)\n",
           (int)tile_size<0>(tiled_mma),
           (int)tile_size<1>(tiled_mma),
           (int)tile_size<2>(tiled_mma));
    printf("   Num threads: %d\n", (int)size(tiled_mma));

    // ========================================
    // 2. Get layoutC_TV from TiledMMA
    // ========================================

    auto layoutC_TV = tiled_mma.get_layoutC_TV();
    printf("\n2. TiledMMA layoutC_TV:\n");
    printf("   "); print(layoutC_TV); printf("\n");
    printf("   Shape: "); print(shape(layoutC_TV)); printf("\n");

    // layoutC_TV structure: (thr_idx, (FrgV, (RestM, RestN))) -> (M, N)
    // For SM80_16x8x16 with 2x2 tiling:
    // - FrgV = (2,2) = 4 values per thread per atom
    // - RestM, RestN = repetitions to cover tile

    // ========================================
    // 3. Simulate MMA output tensor (per thread)
    // ========================================

    // Each thread has its fragment of C
    // Shape: (FrgV, (RestM, RestN))
    // For this config: ((2,2), (1, 2)) = 8 values per thread

    auto thr_mma = tiled_mma.get_slice(0);  // Thread 0's view

    // Create a reference layout for C tile
    auto cC = make_identity_tensor(make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma)));
    auto tCcC = thr_mma.partition_C(cC);

    printf("\n3. Per-thread C fragment (tCcC) for thread 0:\n");
    printf("   Shape: "); print(shape(tCcC)); printf("\n");
    printf("   Layout: "); print(layout(tCcC)); printf("\n");

    // Print actual coordinates this thread is responsible for
    printf("   Coordinates (M,N) for each value:\n");
    for (int i = 0; i < size(tCcC); ++i) {
        auto coord = tCcC(i);
        printf("     val[%d] -> (%d, %d)\n", i, (int)get<0>(coord), (int)get<1>(coord));
    }

    // ========================================
    // 4. Create TiledCopy for R2S (register to shared)
    // ========================================

    // Copy Atom: 4 elements at a time (like stmatrix pattern)
    // Using UniversalCopy<uint128_t> = 16 bytes = 4 x FP32
    using R2SCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, float>;

    auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtom{}, tiled_mma);

    printf("\n4. TiledCopy (R2S) from make_tiled_copy_C:\n");

    // Get the TV layout used by TiledCopy
    using TiledCopyType = decltype(r2s_tiled_copy);
    using Tiler_MN = typename TiledCopyType::Tiler_MN;
    using TiledLayout_TV = typename TiledCopyType::TiledLayout_TV;
    using AtomNumVal = typename TiledCopyType::AtomNumVal;
    using TiledNumThr = typename TiledCopyType::TiledNumThr;
    using TiledNumVal = typename TiledCopyType::TiledNumVal;

    printf("   Tiler_MN: "); print(Tiler_MN{}); printf("\n");
    printf("   TiledLayout_TV: "); print(TiledLayout_TV{}); printf("\n");
    printf("   AtomNumVal: %d\n", (int)AtomNumVal{});
    printf("   TiledNumThr: %d\n", (int)TiledNumThr{});
    printf("   TiledNumVal: %d\n", (int)TiledNumVal{});

    // ========================================
    // 5. Demonstrate retile transformation
    // ========================================

    printf("\n5. Retile transformation analysis:\n");

    // The key variables in retile_S:
    // V = size<0>(tensor) = size of first mode of input
    auto V = size<0>(tCcC);
    printf("   V (input tensor mode-0 size): %d\n", (int)V);

    // frg_layout_mn computation
    auto inv_TV = right_inverse(TiledLayout_TV{});
    printf("   right_inverse(TiledLayout_TV): "); print(inv_TV); printf("\n");

    auto inv_TV_shaped = inv_TV.with_shape(shape(Tiler_MN{}));
    printf("   .with_shape(Tiler_MN): "); print(inv_TV_shaped); printf("\n");

    // upcast<TiledNumThr * V>
    constexpr int upcast_factor = TiledNumThr{} * decltype(V)::value;
    printf("   upcast factor (TiledNumThr * V): %d\n", upcast_factor);

    auto frg_layout_mn = upcast<upcast_factor>(inv_TV_shaped);
    printf("   frg_layout_mn after upcast: "); print(frg_layout_mn); printf("\n");
    printf("   frg_layout_mn maps (m,n) -> batch_idx\n");

    // right_inverse(frg_layout_mn)
    auto inv_frg_layout_mn = right_inverse(frg_layout_mn);
    printf("   right_inverse(frg_layout_mn): "); print(inv_frg_layout_mn); printf("\n");

    // logical_product
    auto lp = logical_product(make_layout(V), inv_frg_layout_mn);
    printf("   logical_product(make_layout(V), ...): "); print(lp); printf("\n");

    // zipped_divide by AtomNumVal
    auto frg_layout_v = zipped_divide(lp, make_layout(AtomNumVal{}));
    printf("   zipped_divide(..., AtomNumVal): "); print(frg_layout_v); printf("\n");
    printf("   This is (atom_vals, rest_vals) -> (v, m, n)\n");

    // ========================================
    // 6. Get thread's copy slice and retile
    // ========================================

    printf("\n6. Thread 0's retiled tensor:\n");

    auto thr_copy = r2s_tiled_copy.get_slice(0);

    // Create a register tensor with the same shape as tCcC
    // (In real code, this would be the actual accumulator)
    auto reg_tensor = make_tensor<float>(shape(tCcC));

    // Fill with sequential values for visualization
    for (int i = 0; i < size(reg_tensor); ++i) {
        reg_tensor(i) = float(i);
    }

    printf("   Original reg_tensor shape: "); print(shape(reg_tensor)); printf("\n");
    printf("   Original reg_tensor layout: "); print(layout(reg_tensor)); printf("\n");

    // Apply retile_S
    auto retiled = thr_copy.retile_S(reg_tensor);

    printf("   After retile_S shape: "); print(shape(retiled)); printf("\n");
    printf("   After retile_S layout: "); print(layout(retiled)); printf("\n");

    // ========================================
    // 7. Show copy iteration pattern
    // ========================================

    printf("\n7. Copy iteration pattern:\n");
    printf("   cute::copy iterates: for i in [0, size<1>(dst)):\n");
    printf("                           copy_atom.call(src(_, i), dst(_, i))\n\n");

    auto atom_vals = size<0>(retiled);
    auto rest_vals = size<1>(retiled);
    printf("   atom_vals (elements per copy_atom.call): %d\n", (int)atom_vals);
    printf("   rest_vals (number of iterations): %d\n", (int)rest_vals);

    // Print values using linear indexing
    printf("\n   All values in retiled order:\n   ");
    for (int i = 0; i < size(retiled); ++i) {
        if (i > 0 && i % (int)atom_vals == 0) {
            printf(" | ");  // separator between iterations
        }
        printf("%.0f ", retiled(i));
    }
    printf("\n   (separated by | every %d elements = one copy_atom.call)\n", (int)atom_vals);

    // ========================================
    // 8. Summary
    // ========================================

    printf("\n=== Summary ===\n");
    printf("MMA output: ((2,2), (RestM, RestN)) - nested structure from MMA atom\n");
    printf("Copy expects: (atom_vals, rest_vals) - flat structure for iteration\n");
    printf("retile_S transforms between these formats\n\n");

    printf("Key insight:\n");
    printf("  - upcast<TiledNumThr * V> groups coordinates by 'all threads' first V values'\n");
    printf("  - This creates batch indices for copy iterations\n");
    printf("  - zipped_divide by AtomNumVal regroups for copy atom granularity\n");

    return 0;
}
