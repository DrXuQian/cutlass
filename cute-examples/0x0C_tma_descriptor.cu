#include <cstdio>
#include <cstdint>

// TMA Descriptor structure (64 bytes, 512 bits)
// This demonstrates the descriptor fields without requiring SM90 hardware

struct TmaDescriptor {
    uint64_t data[8];  // 512 bits total
};

void print_tma_descriptor_fields() {
    printf("============================================================\n");
    printf("    0x0C CUTLASS SM90 TMA Descriptor Deep Dive\n");
    printf("============================================================\n\n");

    //==========================================================================
    // TMA Descriptor Overview
    //==========================================================================

    printf("=== TMA Descriptor Structure (512 bits) ===\n\n");

    printf("A TMA descriptor is a 64-byte structure that describes:\n");
    printf("  - Base address of the tensor in global memory\n");
    printf("  - Tensor dimensions and strides\n");
    printf("  - Tile/box dimensions for transfer\n");
    printf("  - Data type and swizzle pattern\n\n");

    printf("Layout (8 x 64-bit words):\n\n");

    printf("+--------+--------------------------------------------------+\n");
    printf("| Word   | Contents                                         |\n");
    printf("+--------+--------------------------------------------------+\n");
    printf("| 0      | Base address (48-bit) + flags                    |\n");
    printf("| 1      | Dimension sizes (d0, d1, ...)                    |\n");
    printf("| 2      | Dimension sizes continued                        |\n");
    printf("| 3      | Strides (s0, s1, ...)                            |\n");
    printf("| 4      | Strides continued                                |\n");
    printf("| 5      | Box/tile dimensions                              |\n");
    printf("| 6      | Box dimensions continued + element size          |\n");
    printf("| 7      | Swizzle pattern + interleave info                |\n");
    printf("+--------+--------------------------------------------------+\n\n");

    //==========================================================================
    // Key Fields
    //==========================================================================

    printf("=== Key Descriptor Fields ===\n\n");

    printf("1. Base Address (Word 0, bits 0-47):\n");
    printf("   - 48-bit global memory address\n");
    printf("   - Must be 16-byte aligned for TMA\n\n");

    printf("2. Tensor Dimensions (Words 1-2):\n");
    printf("   - Up to 5 dimensions supported\n");
    printf("   - Each dimension size in elements\n");
    printf("   - Example: (M=1024, K=512) for 2D tensor\n\n");

    printf("3. Tensor Strides (Words 3-4):\n");
    printf("   - Stride in bytes for each dimension\n");
    printf("   - Stride[0] = element size (implicit)\n");
    printf("   - Stride[i] = size of dimension i-1 * stride[i-1]\n\n");

    printf("4. Box Dimensions (Words 5-6):\n");
    printf("   - Size of tile to transfer\n");
    printf("   - Box must fit within tensor bounds\n");
    printf("   - Example: box=(64,16) for a 64x16 tile\n\n");

    printf("5. Element Size (Word 6):\n");
    printf("   - Size of each element in bytes\n");
    printf("   - 2 for f16/bf16, 4 for f32, etc.\n\n");

    printf("6. Swizzle Pattern (Word 7):\n");
    printf("   - Controls SMEM bank conflict avoidance\n");
    printf("   - Options: None, 32B, 64B, 128B\n\n");

    //==========================================================================
    // CUTLASS make_tma_copy
    //==========================================================================

    printf("=== CUTLASS TMA Descriptor Creation ===\n\n");

    printf("CUTLASS creates TMA descriptors via:\n\n");

    printf("```cpp\n");
    printf("auto tma_a = make_tma_copy(\n");
    printf("    SM90_TMA_LOAD{},           // or SM90_TMA_LOAD_MULTICAST\n");
    printf("    tensor_a,                  // Source tensor (gmem)\n");
    printf("    smem_layout(_,_,Int<0>{}), // SMEM layout for one tile\n");
    printf("    tile_shape,                // Tile dimensions\n");
    printf("    cluster_shape              // For multicast\n");
    printf(");\n");
    printf("```\n\n");

    printf("This internally calls cuTensorMapEncodeTiled() which:\n");
    printf("  1. Validates tensor alignment\n");
    printf("  2. Computes stride encoding\n");
    printf("  3. Sets up swizzle pattern\n");
    printf("  4. Returns 64-byte descriptor\n\n");

    //==========================================================================
    // Descriptor Usage
    //==========================================================================

    printf("=== Using TMA Descriptors ===\n\n");

    printf("1. Get descriptor for TMA operation:\n");
    printf("   auto* desc = tma_a.get_tma_descriptor();\n\n");

    printf("2. Prefetch descriptor to L2 (optional):\n");
    printf("   cute::prefetch_tma_descriptor(desc);\n");
    printf("   -> tensormap.prefetch [desc];\n\n");

    printf("3. Create tensor from descriptor:\n");
    printf("   Tensor gmem = tma_a.get_tma_tensor(shape);\n\n");

    printf("4. Get slice for current block:\n");
    printf("   auto block_tma = tma_a.get_slice(block_coord);\n\n");

    printf("5. Partition source and destination:\n");
    printf("   Tensor tAgA = block_tma.partition_S(gA);  // GMEM source\n");
    printf("   Tensor tAsA = block_tma.partition_D(sA);  // SMEM dest\n\n");

    printf("6. Execute TMA copy:\n");
    printf("   copy(tma_a.with(*barrier, mcast_mask), tAgA, tAsA);\n\n");

    //==========================================================================
    // PTX Instructions
    //==========================================================================

    printf("=== TMA PTX Instructions ===\n\n");

    printf("TMA Load (non-multicast):\n");
    printf("  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes\n");
    printf("      [smem], [tma_desc, {coords}], [barrier];\n\n");

    printf("TMA Load Multicast:\n");
    printf("  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster\n");
    printf("      [smem], [tma_desc, {coords}], [barrier], mcast_mask;\n\n");

    printf("TMA Store:\n");
    printf("  cp.async.bulk.tensor.2d.global.shared::cta\n");
    printf("      [tma_desc, {coords}], [smem];\n\n");

    printf("Prefetch Descriptor:\n");
    printf("  tensormap.prefetch [desc];\n\n");

    //==========================================================================
    // Alignment Requirements
    //==========================================================================

    printf("=== TMA Alignment Requirements ===\n\n");

    printf("+-------------------+------------------+\n");
    printf("| Item              | Alignment        |\n");
    printf("+-------------------+------------------+\n");
    printf("| Base address      | 16 bytes         |\n");
    printf("| Descriptor        | 64 bytes         |\n");
    printf("| SMEM destination  | 128 bytes        |\n");
    printf("| Transfer size     | 16 bytes minimum |\n");
    printf("+-------------------+------------------+\n\n");

    printf("If alignment is violated:\n");
    printf("  - can_implement() returns false\n");
    printf("  - Runtime error: 'misaligned address'\n");
}

int main() {
    print_tma_descriptor_fields();
    return 0;
}
