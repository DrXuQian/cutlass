#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// Simplified MMA Atom traits for demonstration
// In real CUTLASS, these come from cute/atom/mma_traits_sm90_gmma.hpp

int main() {
    printf("============================================================\n");
    printf("    0x05 CUTLASS make_tiled_mma and MMA_Atom\n");
    printf("============================================================\n\n");

    //==========================================================================
    // MMA Atom Structure
    //==========================================================================

    printf("=== MMA Atom Structure ===\n\n");

    printf("An MMA Atom defines the smallest MMA unit:\n\n");

    printf("MMA_Atom<MMA_Traits<MMA_Op>>:\n");
    printf("  +-- MMA_Traits:\n");
    printf("  |   +-- Shape_MNK: The atom's (M, N, K) dimensions\n");
    printf("  |   +-- ThrID:     Thread layout within the atom\n");
    printf("  |   +-- ALayout:   Thread-Value layout for A operand\n");
    printf("  |   +-- BLayout:   Thread-Value layout for B operand\n");
    printf("  |   +-- CLayout:   Thread-Value layout for C accumulator\n");
    printf("  +-- fma():         The actual MMA operation\n\n");

    printf("Example: SM90 WGMMA m64n16k16 F32 += F16 * F16\n");
    printf("  Shape_MNK = (64, 16, 16)\n");
    printf("  ThrID     = 128 threads (1 warpgroup)\n");
    printf("  ALayout   = SS mode: (_128, (64,16)):(_0, (1,64))\n");
    printf("  BLayout   = SS mode: (_128, (16,16)):(_0, (1,16))\n");
    printf("  CLayout   = ((_4,_8,_4),(_2,_2,_2)):((_128,_1,_16),(_64,_8,_512))\n\n");

    //==========================================================================
    // make_tiled_mma
    //==========================================================================

    printf("=== make_tiled_mma ===\n\n");

    printf("make_tiled_mma tiles an MMA Atom to cover larger regions:\n\n");

    printf("auto tiled_mma = make_tiled_mma(\n");
    printf("    MMA_Atom{},           // Base atom\n");
    printf("    Layout<TileM, TileN, TileK>{},  // How to tile the atom\n");
    printf("    Permutation{}         // Optional: permute thread-value\n");
    printf(");\n\n");

    printf("Example: Tiling m64n16k16 atom to cover 128x64:\n\n");

    printf("  Base atom: 64 x 16\n");
    printf("  Tile layout: (2, 4, 1)  // 2 in M, 4 in N, 1 in K\n");
    printf("  Result: 64*2 x 16*4 = 128 x 64\n\n");

    printf("  Thread count: 128 (atom) * 2 (M-tile) * 4 (N-tile) = 1024?\n");
    printf("  NO! Threads are REUSED across tiles.\n");
    printf("  Each thread handles multiple tiles sequentially.\n\n");

    //==========================================================================
    // TiledMMA Components
    //==========================================================================

    printf("=== TiledMMA Components ===\n\n");

    printf("TiledMMA provides:\n\n");

    printf("1. get_slice(thread_idx):\n");
    printf("   Returns thread's view of the MMA operation\n\n");

    printf("2. partition_A(tensor_A):\n");
    printf("   Partitions A tensor for this thread\n");
    printf("   Returns: (MMA, MMA_M, MMA_K) shaped tensor\n\n");

    printf("3. partition_B(tensor_B):\n");
    printf("   Partitions B tensor for this thread\n");
    printf("   Returns: (MMA, MMA_N, MMA_K) shaped tensor\n\n");

    printf("4. partition_C(tensor_C):\n");
    printf("   Partitions C tensor for this thread\n");
    printf("   Returns: (MMA, MMA_M, MMA_N) shaped tensor\n\n");

    printf("5. make_fragment_A/B(partitioned_tensor):\n");
    printf("   Creates register fragment from partitioned view\n\n");

    printf("6. partition_fragment_C(tensor_C):\n");
    printf("   Creates accumulator fragment\n\n");

    //==========================================================================
    // Usage Pattern
    //==========================================================================

    printf("=== Usage Pattern ===\n\n");

    printf("```cpp\n");
    printf("// 1. Create TiledMMA\n");
    printf("using TiledMma = decltype(make_tiled_mma(\n");
    printf("    SM90_64x16x16_F32F16F16_SS{},\n");
    printf("    Layout<_2, _4, _1>{}\n");
    printf("));\n");
    printf("TiledMma tiled_mma;\n\n");

    printf("// 2. Get thread's slice\n");
    printf("auto thread_mma = tiled_mma.get_slice(threadIdx.x);\n\n");

    printf("// 3. Partition tensors\n");
    printf("Tensor tCsA = thread_mma.partition_A(sA);  // SMEM A\n");
    printf("Tensor tCsB = thread_mma.partition_B(sB);  // SMEM B\n\n");

    printf("// 4. Create fragments\n");
    printf("Tensor tCrA = thread_mma.make_fragment_A(tCsA);\n");
    printf("Tensor tCrB = thread_mma.make_fragment_B(tCsB);\n");
    printf("Tensor accum = partition_fragment_C(tiled_mma, tile_shape);\n\n");

    printf("// 5. Execute MMA\n");
    printf("cute::gemm(tiled_mma, tCrA, tCrB, accum);\n");
    printf("```\n\n");

    //==========================================================================
    // Layout Transformations
    //==========================================================================

    printf("=== Layout Transformations ===\n\n");

    printf("TiledMMA transforms layouts hierarchically:\n\n");

    printf("Original C tensor: (M, N)\n");
    printf("        |\n");
    printf("        v  partition_C\n");
    printf("Thread view: (MMA, MMA_M, MMA_N)\n");
    printf("             |     |      |\n");
    printf("             |     |      +-- N tiles for this thread\n");
    printf("             |     +--------- M tiles for this thread\n");
    printf("             +--------------- Values within MMA atom\n\n");

    printf("For example, with 128x64 C and 64x16 atom:\n");
    printf("  partition_C shape: (8, 2, 4)\n");
    printf("    8 = values per MMA (from CLayout)\n");
    printf("    2 = 128/64 M tiles\n");
    printf("    4 = 64/16 N tiles\n\n");

    //==========================================================================
    // Cooperative Tiling
    //==========================================================================

    printf("=== Cooperative Tiling ===\n\n");

    printf("For large tiles, multiple warpgroups cooperate:\n\n");

    printf("AtomLayoutMNK = Layout<Shape<_2, _1, _1>>\n");
    printf("  -> 2 warpgroups in M direction\n");
    printf("  -> Each covers 64 rows\n");
    printf("  -> Together cover 128 rows\n\n");

    printf("Thread assignment:\n");
    printf("  Threads 0-127:   Warpgroup 0, rows 0-63\n");
    printf("  Threads 128-255: Warpgroup 1, rows 64-127\n\n");

    printf("This is handled by tiled_product in thrfrg_C.\n");

    return 0;
}
