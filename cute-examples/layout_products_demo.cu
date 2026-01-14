// CuTe Layout Product and Divide Operations Demo
// Demonstrates: logical_divide, zipped_divide, logical_product, zipped_product,
//               blocked_product, raked_product

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// Helper to print 2D layout as a matrix
template <class Layout>
void print_layout_2d(const char* name, Layout const& layout) {
    printf("\n%s\n", name);
    printf("  Layout: "); print(layout); printf("\n");
    printf("  Shape: "); print(shape(layout)); printf("\n");
    printf("  Stride: "); print(stride(layout)); printf("\n");
}

// Helper to print nested layout
template <class Layout>
void print_nested_layout(const char* name, Layout const& layout) {
    printf("\n%s\n", name);
    printf("  Layout: "); print(layout); printf("\n");
    printf("  Shape: "); print(shape(layout)); printf("\n");
    printf("  Size: %d, Cosize: %d\n", (int)size(layout), (int)cosize(layout));

    // Print linear mapping
    printf("  Linear mapping: ");
    for (int i = 0; i < size(layout) && i < 32; ++i) {
        printf("%d ", (int)layout(i));
    }
    if (size(layout) > 32) printf("...");
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("CuTe Layout Product/Divide Operations\n");
    printf("========================================\n");

    // ========================================
    // Part 1: Basic Layout Setup
    // ========================================

    printf("\n=== Part 1: Basic Layouts ===\n");

    // A simple 8x8 row-major layout
    auto layout_8x8 = make_layout(make_shape(Int<8>{}, Int<8>{}));
    print_layout_2d("layout_8x8 (8,8):(8,1)", layout_8x8);

    // A tiler: 4x4 block
    auto tiler_4x4 = make_layout(make_shape(Int<4>{}, Int<4>{}));
    print_layout_2d("tiler_4x4 (4,4):(4,1)", tiler_4x4);

    // ========================================
    // Part 2: logical_divide
    // ========================================

    printf("\n=== Part 2: logical_divide ===\n");
    printf("Purpose: Divide a layout into tiles, creating ((Tile), (Rest)) structure\n");
    printf("Formula: composition(layout, make_layout(tiler, complement(tiler, shape(layout))))\n");

    auto ld_result = logical_divide(layout_8x8, tiler_4x4);
    print_nested_layout("logical_divide(8x8, 4x4)", ld_result);

    printf("\n  Interpretation:\n");
    printf("    Shape: ((4,4), (2,2)) = ((tile_m, tile_n), (rest_m, rest_n))\n");
    printf("    - First mode (4,4): coordinates within a 4x4 tile\n");
    printf("    - Second mode (2,2): which tile (2x2 = 4 tiles total)\n");

    // Access using linear indexing
    printf("\n  Linear access pattern (first 16 elements = tile 0):\n    ");
    for (int i = 0; i < 16; ++i) {
        printf("%2d ", (int)ld_result(i));
    }
    printf("\n  (next 16 elements = tile 1):\n    ");
    for (int i = 16; i < 32; ++i) {
        printf("%2d ", (int)ld_result(i));
    }
    printf("\n");

    // ========================================
    // Part 3: zipped_divide
    // ========================================

    printf("\n=== Part 3: zipped_divide ===\n");
    printf("Purpose: Same as logical_divide but zips tile modes together\n");
    printf("Formula: tile_unzip(logical_divide(layout, tiler), tiler)\n");

    auto zd_result = zipped_divide(layout_8x8, tiler_4x4);
    print_nested_layout("zipped_divide(8x8, 4x4)", zd_result);

    printf("\n  Interpretation:\n");
    printf("    Shape: ((4,4), (2,2)) - same structure but different stride organization\n");
    printf("    - Tile modes are 'zipped' for easier iteration\n");

    // Compare with logical_divide
    printf("\n  Comparison with logical_divide:\n");
    printf("    logical_divide layout: "); print(ld_result); printf("\n");
    printf("    zipped_divide  layout: "); print(zd_result); printf("\n");

    // ========================================
    // Part 4: logical_product
    // ========================================

    printf("\n=== Part 4: logical_product ===\n");
    printf("Purpose: Replicate a block layout over a tiler, creating (block, tiler) structure\n");
    printf("Formula: make_layout(block, composition(complement(block, size(block)*cosize(tiler)), tiler))\n");

    // Small block to replicate
    auto block_2x2 = make_layout(make_shape(Int<2>{}, Int<2>{}));
    auto tiler_3x3 = make_layout(make_shape(Int<3>{}, Int<3>{}));

    print_layout_2d("block_2x2", block_2x2);
    print_layout_2d("tiler_3x3", tiler_3x3);

    auto lp_result = logical_product(block_2x2, tiler_3x3);
    print_nested_layout("logical_product(2x2, 3x3)", lp_result);

    printf("\n  Interpretation:\n");
    printf("    Shape: ((2,2), (3,3)) = ((block_m, block_n), (tile_m, tile_n))\n");
    printf("    - First mode (2,2): the original 2x2 block\n");
    printf("    - Second mode (3,3): 3x3 = 9 copies of the block\n");
    printf("    - Total size: 2*2 * 3*3 = 36 elements\n");

    // Show block pattern using linear access
    printf("\n  Block replication pattern (linear access):\n");
    printf("    First 4 elements (block content): ");
    for (int i = 0; i < 4; ++i) printf("%d ", (int)lp_result(i));
    printf("\n    Elements 4-7 (next tiler pos): ");
    for (int i = 4; i < 8; ++i) printf("%d ", (int)lp_result(i));
    printf("\n    Elements 32-35 (last tiler pos): ");
    for (int i = 32; i < 36; ++i) printf("%d ", (int)lp_result(i));
    printf("\n");

    // ========================================
    // Part 5: zipped_product
    // ========================================

    printf("\n=== Part 5: zipped_product ===\n");
    printf("Purpose: Same as logical_product but zips modes together\n");

    auto zp_result = zipped_product(block_2x2, tiler_3x3);
    print_nested_layout("zipped_product(2x2, 3x3)", zp_result);

    printf("\n  Comparison:\n");
    printf("    logical_product: "); print(lp_result); printf("\n");
    printf("    zipped_product:  "); print(zp_result); printf("\n");

    // ========================================
    // Part 6: blocked_product
    // ========================================

    printf("\n=== Part 6: blocked_product ===\n");
    printf("Purpose: Replicate block over tiler, interleave per-mode (BLOCKED pattern)\n");
    printf("Result: Each mode contains (block_mode, tiler_mode) - blocks are contiguous\n");

    auto bp_result = blocked_product(block_2x2, tiler_3x3);
    print_nested_layout("blocked_product(2x2, 3x3)", bp_result);

    printf("\n  Interpretation:\n");
    printf("    Shape: ((2,3), (2,3)) = ((block_m,tile_m), (block_n,tile_n))\n");
    printf("    - Resulting shape: 6x6 matrix\n");
    printf("    - Blocks are placed contiguously, then repeated\n");

    // Visualize using linear access
    printf("\n  Linear view (first 36 elements):\n    ");
    for (int i = 0; i < 36; ++i) {
        if (i > 0 && i % 6 == 0) printf("\n    ");
        printf("%3d ", (int)bp_result(i));
    }
    printf("\n");

    // ========================================
    // Part 7: raked_product
    // ========================================

    printf("\n=== Part 7: raked_product ===\n");
    printf("Purpose: Replicate block over tiler, interleave per-mode (RAKED pattern)\n");
    printf("Result: Each mode contains (tiler_mode, block_mode) - blocks are interleaved\n");

    auto rp_result = raked_product(block_2x2, tiler_3x3);
    print_nested_layout("raked_product(2x2, 3x3)", rp_result);

    printf("\n  Interpretation:\n");
    printf("    Shape: ((3,2), (3,2)) = ((tile_m,block_m), (tile_n,block_n))\n");
    printf("    - Resulting shape: 6x6 matrix\n");
    printf("    - Blocks are interleaved (strided) across the tiler\n");

    // Visualize using linear access
    printf("\n  Linear view (first 36 elements):\n    ");
    for (int i = 0; i < 36; ++i) {
        if (i > 0 && i % 6 == 0) printf("\n    ");
        printf("%3d ", (int)rp_result(i));
    }
    printf("\n");

    // ========================================
    // Part 8: Comparison blocked vs raked
    // ========================================

    printf("\n=== Part 8: Blocked vs Raked Comparison ===\n");
    printf("\nblocked_product (blocks contiguous):\n");
    printf("  Element 0,1,2,3 from block 0 are adjacent\n");
    printf("  Layout: "); print(bp_result); printf("\n");

    printf("\nraked_product (blocks interleaved):\n");
    printf("  Element 0 from each tile comes first, then element 1, etc.\n");
    printf("  Layout: "); print(rp_result); printf("\n");

    printf("\nVisualization (2x2 block, 3x3 tiler):\n");
    printf("\n  BLOCKED (each 2x2 block is contiguous):\n");
    printf("    [B0 B0 | B1 B1 | B2 B2]\n");
    printf("    [B0 B0 | B1 B1 | B2 B2]\n");
    printf("    [B3 B3 | B4 B4 | B5 B5]\n");
    printf("    [B3 B3 | B4 B4 | B5 B5]\n");
    printf("    [B6 B6 | B7 B7 | B8 B8]\n");
    printf("    [B6 B6 | B7 B7 | B8 B8]\n");

    printf("\n  RAKED (blocks interleaved across tiler):\n");
    printf("    [B0 B3 B6 | B0 B3 B6]\n");
    printf("    [B1 B4 B7 | B1 B4 B7]\n");
    printf("    [B2 B5 B8 | B2 B5 B8]\n");
    printf("    [B0 B3 B6 | B0 B3 B6]\n");
    printf("    [B1 B4 B7 | B1 B4 B7]\n");
    printf("    [B2 B5 B8 | B2 B5 B8]\n");

    // ========================================
    // Part 9: Practical Example - Thread Tiling
    // ========================================

    printf("\n=== Part 9: Practical Example - Thread Tiling ===\n");

    // 128 elements, 32 threads, each thread handles 4 elements
    auto data_layout = make_layout(Int<128>{});  // 128 elements
    auto thread_layout = make_layout(Int<32>{});  // 32 threads

    printf("\nScenario: 128 elements, 32 threads\n");

    // Using blocked_product: each thread gets 4 contiguous elements
    auto blocked_assignment = blocked_product(make_layout(Int<4>{}), thread_layout);
    print_nested_layout("blocked_product(4, 32) - contiguous assignment", blocked_assignment);

    // Using raked_product: each thread gets 4 strided elements
    auto raked_assignment = raked_product(make_layout(Int<4>{}), thread_layout);
    print_nested_layout("raked_product(4, 32) - strided assignment", raked_assignment);

    // Use linear indexing to show thread assignments
    printf("\n  Thread 0 elements (blocked, linear idx 0-3): ");
    for (int i = 0; i < 4; ++i) printf("%d ", (int)blocked_assignment(i));
    printf("(contiguous)\n");

    printf("  Thread 0 elements (raked, linear idx 0-3):   ");
    for (int i = 0; i < 4; ++i) printf("%d ", (int)raked_assignment(i));
    printf("(strided by 32)\n");

    printf("\n  Thread 1 elements (blocked, linear idx 4-7): ");
    for (int i = 4; i < 8; ++i) printf("%d ", (int)blocked_assignment(i));
    printf("\n");

    printf("  Thread 1 elements (raked, linear idx 4-7):   ");
    for (int i = 4; i < 8; ++i) printf("%d ", (int)raked_assignment(i));
    printf("\n");

    // ========================================
    // Summary
    // ========================================

    printf("\n========================================\n");
    printf("Summary\n");
    printf("========================================\n");
    printf("\n");
    printf("DIVIDE operations (split existing layout):\n");
    printf("  logical_divide: ((Tile), (Rest)) - hierarchical\n");
    printf("  zipped_divide:  ((Tile), (Rest)) - zipped for iteration\n");
    printf("  tiled_divide:   ((Tile), Rest...) - unpacked rest\n");
    printf("  flat_divide:    (Tile..., Rest...) - fully flattened\n");
    printf("\n");
    printf("PRODUCT operations (replicate block over tiler):\n");
    printf("  logical_product: ((Block), (Tiler)) - hierarchical\n");
    printf("  zipped_product:  ((Block), (Tiler)) - zipped\n");
    printf("  blocked_product: ((blk,tile), ...) - blocks contiguous\n");
    printf("  raked_product:   ((tile,blk), ...) - blocks interleaved\n");
    printf("\n");
    printf("Key insight:\n");
    printf("  - DIVIDE: existing layout -> (tile, rest) structure\n");
    printf("  - PRODUCT: block layout -> replicated over tiler\n");
    printf("  - BLOCKED: preserves block locality (good for cache)\n");
    printf("  - RAKED: distributes blocks (good for load balancing)\n");

    return 0;
}
