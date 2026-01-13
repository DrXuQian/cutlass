#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// CLayout_64xN 定义 (从 CUTLASS)
template <int N>
using CLayout_64xN = Layout<Shape <Shape <  _4, _8, _4>, Shape < _2, _2, Int<N/8>>>,
                            Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;

using CLayout_64x16 = CLayout_64xN<16>;

int main(int argc, char** argv) {
    
    printf("============================================================\n");
    printf("          thrfrg_C Step-by-Step Analysis\n");
    printf("============================================================\n\n");

    //==========================================================================
    // 参数设置
    //==========================================================================
    
    // AtomShape_MNK: 单个 Atom 的大小 (M=64, N=16, K=暂不关心)
    using AtomShapeM = Int<64>;
    using AtomShapeN = Int<16>;
    
    // AtomLayoutMNK: Cooperative tiling (M方向2个Atom, N方向1个, K方向1个)
    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;
    
    // AtomLayoutC_TV: 单个 Atom 内的 Thread-Value Layout
    using AtomLayoutC_TV = CLayout_64x16;
    
    printf("=== Configuration ===\n");
    printf("AtomShape_MNK: (M=%d, N=%d)\n", int(AtomShapeM{}), int(AtomShapeN{}));
    printf("AtomLayoutMNK (cooperative): "); print(AtomLayoutMNK{}); printf("\n");
    printf("AtomLayoutC_TV: "); print(AtomLayoutC_TV{}); printf("\n\n");

    //==========================================================================
    // Step 0: 原始 ctensor (128 x 128)
    //==========================================================================
    
    printf("=== Step 0: Original ctensor ===\n");
    auto ctensor = make_layout(make_shape(Int<128>{}, Int<128>{}));
    printf("ctensor shape: "); print(shape(ctensor)); printf("\n");
    printf("ctensor layout: "); print(ctensor); printf("\n");
    printf("Total elements: %d\n\n", int(size(ctensor)));

    //==========================================================================
    // Step 1: Permutation (identity, 不做任何变换)
    //==========================================================================
    
    printf("=== Step 1: Permutation (identity) ===\n");
    auto t_tensor = ctensor;
    printf("After permutation (no change): "); print(t_tensor); printf("\n\n");

    //==========================================================================
    // Step 2: zipped_divide by AtomShape
    //==========================================================================
    
    printf("=== Step 2: zipped_divide by AtomShape ===\n");
    auto c_tile = make_tile(make_layout(AtomShapeM{}),   // M: 64
                            make_layout(AtomShapeN{}));  // N: 16
    printf("c_tile (AtomShape): "); print(c_tile); printf("\n");
    
    auto c_tensor = zipped_divide(t_tensor, c_tile);
    printf("c_tensor layout: "); print(c_tensor); printf("\n");
    printf("  Shape: "); print(shape(c_tensor)); printf("\n");
    printf("  - First dim (AtomM, AtomN): "); print(shape<0>(c_tensor)); printf("\n");
    printf("  - Second dim (RestM, RestN): "); print(shape<1>(c_tensor)); printf("\n");
    printf("\nInterpretation:\n");
    printf("  - Each Atom covers: %d x %d = %d elements\n", 
           int(AtomShapeM{}), int(AtomShapeN{}), int(AtomShapeM{}) * int(AtomShapeN{}));
    printf("  - Number of Atoms: %d x %d = %d\n", 
           128 / int(AtomShapeM{}), 128 / int(AtomShapeN{}),
           (128 / int(AtomShapeM{})) * (128 / int(AtomShapeN{})));
    printf("\n");

    //==========================================================================
    // Step 3: compose with AtomLayoutC_TV
    //==========================================================================
    
    printf("=== Step 3: compose with AtomLayoutC_TV ===\n");
    printf("AtomLayoutC_TV (CLayout_64x16):\n");
    printf("  "); print(AtomLayoutC_TV{}); printf("\n");
    printf("  Shape: "); print(shape(AtomLayoutC_TV{})); printf("\n");
    
    printf("\nAtomLayoutC_TV breakdown:\n");
    printf("  Thr shape: "); print(shape<0>(AtomLayoutC_TV{})); 
    printf(" = %d threads\n", int(size<0>(AtomLayoutC_TV{})));
    printf("  Val shape: "); print(shape<1>(AtomLayoutC_TV{})); 
    printf(" = %d values/thread\n", int(size<1>(AtomLayoutC_TV{})));
    printf("  Total: %d threads x %d values = %d elements\n",
           int(size<0>(AtomLayoutC_TV{})), 
           int(size<1>(AtomLayoutC_TV{})),
           int(size(AtomLayoutC_TV{})));
    
    printf("\nAtomLayoutC_TV layout visualization (128 threads x 8 values):\n");
    print_layout(AtomLayoutC_TV{});
    printf("\n");
    
    auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _);
    printf("tv_tensor layout: "); print(tv_tensor); printf("\n");
    printf("  Shape: "); print(shape(tv_tensor)); printf("\n");
    printf("  - First dim (ThrV, FrgV): "); print(shape<0>(tv_tensor)); printf("\n");
    printf("  - Second dim (RestM, RestN): "); print(shape<1>(tv_tensor)); printf("\n");
    printf("\n");

    //==========================================================================
    // Step 4: zipped_divide for Thread Tiling
    //==========================================================================
    
    printf("=== Step 4: zipped_divide for Thread Tiling ===\n");
    
    auto AtomThrID = Layout<Int<128>>{};
    auto thr_layout_vmnk = tiled_product(AtomThrID, AtomLayoutMNK{});
    
    printf("AtomThrID: "); print(AtomThrID); printf("\n");
    printf("AtomLayoutMNK: "); print(AtomLayoutMNK{}); printf("\n");
    printf("thr_layout_vmnk = tiled_product(AtomThrID, AtomLayoutMNK):\n");
    printf("  "); print(thr_layout_vmnk); printf("\n");
    printf("  Shape: "); print(shape(thr_layout_vmnk)); printf("\n");
    printf("  Total threads: %d\n", int(size(thr_layout_vmnk)));
    
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<2>(thr_layout_vmnk))));
    printf("\nthr_tile: "); print(thr_tile); printf("\n");
    printf("  size<1>(thr_layout_vmnk) = ThrM = %d\n", int(size<1>(thr_layout_vmnk)));
    printf("  size<2>(thr_layout_vmnk) = ThrN = %d\n", int(size<2>(thr_layout_vmnk)));
    
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    printf("\nthr_tensor (final result): "); print(thr_tensor); printf("\n");
    printf("  Shape: "); print(shape(thr_tensor)); printf("\n");
    printf("  - First dim (ThrV, (ThrM, ThrN)): "); print(shape<0>(thr_tensor)); printf("\n");
    printf("  - Second dim (FrgV, (RestM', RestN')): "); print(shape<1>(thr_tensor)); printf("\n");
    printf("\n");

    //==========================================================================
    // 总结
    //==========================================================================
    
    printf("============================================================\n");
    printf("                      Summary\n");
    printf("============================================================\n");
    printf("Input:  ctensor = (128, 128) = %d elements\n", 128*128);
    printf("Output: thr_tensor shape = "); print(shape(thr_tensor)); printf("\n\n");
    
    int ThrV = int(size<0,0>(thr_tensor));
    int ThrM = int(size<0,1,0>(thr_tensor));
    int ThrN = int(size<0,1,1>(thr_tensor));
    int FrgV = int(size<1,0>(thr_tensor));
    int RestM = int(size<1,1,0>(thr_tensor));
    int RestN = int(size<1,1,1>(thr_tensor));
    
    printf("Thread coordinates:\n");
    printf("  ThrV  = %d (threads per Atom)\n", ThrV);
    printf("  ThrM  = %d (Atoms in M direction, cooperative)\n", ThrM);
    printf("  ThrN  = %d (Atoms in N direction, cooperative)\n", ThrN);
    printf("  Total threads = %d x %d x %d = %d\n", ThrV, ThrM, ThrN, ThrV*ThrM*ThrN);
    
    printf("\nPer-thread values:\n");
    printf("  FrgV   = %d (values per thread per Atom)\n", FrgV);
    printf("  RestM' = %d (remaining M tiles per thread)\n", RestM);
    printf("  RestN' = %d (remaining N tiles per thread)\n", RestN);
    printf("  Values per thread = %d x %d x %d = %d\n", FrgV, RestM, RestN, FrgV*RestM*RestN);
    
    printf("\nVerification:\n");
    printf("  Total elements = threads x values_per_thread\n");
    printf("                 = %d x %d = %d\n", 
           ThrV*ThrM*ThrN, FrgV*RestM*RestN, 
           ThrV*ThrM*ThrN * FrgV*RestM*RestN);
    printf("  Expected       = 128 x 128 = %d\n", 128*128);
    if (ThrV*ThrM*ThrN * FrgV*RestM*RestN == 128*128) {
        printf("  Match: YES ✓\n");
    } else {
        printf("  Match: NO ✗\n");
    }

    //==========================================================================
    // Thread 0 和 Thread 128 的数据示例
    //==========================================================================
    
    printf("\n============================================================\n");
    printf("               Thread Data Examples\n");
    printf("============================================================\n");
    
    printf("\nThread 0 (ThrV=0, ThrM=0, ThrN=0):\n");
    printf("  Belongs to: Atom 0 (Warpgroup 0)\n");
    printf("  Covers M rows: 0-%d\n", int(AtomShapeM{})-1);
    
    printf("\nThread 128 (ThrV=0, ThrM=1, ThrN=0):\n");
    printf("  Belongs to: Atom 1 (Warpgroup 1)\n");
    printf("  Covers M rows: %d-%d\n", int(AtomShapeM{}), 2*int(AtomShapeM{})-1);

    //==========================================================================
    // Thread 0 的 value 位置
    //==========================================================================
    
    printf("\n============================================================\n");
    printf("         Thread 0's Values in AtomLayoutC_TV\n");
    printf("============================================================\n");
    
    auto atom_tv = AtomLayoutC_TV{};
    printf("\nThread 0's %d values in 64x16 Atom:\n", int(size<1>(atom_tv)));
    printf("  Val_idx -> offset -> (m, n)\n");
    for (int v = 0; v < int(size<1>(atom_tv)); ++v) {
        int offset = atom_tv(0, v);
        int m = offset % 64;
        int n = offset / 64;
        printf("  V%d -> offset %3d -> (m=%2d, n=%2d)\n", v, offset, m, n);
    }

    //==========================================================================
    // 更多线程示例
    //==========================================================================
    
    printf("\n============================================================\n");
    printf("         Multiple Threads' Values Comparison\n");
    printf("============================================================\n");
    
    printf("\nComparing Thread 0, 1, 2, 3 in 64x16 Atom:\n");
    printf("Thread | V0 offset (m,n) | V1 offset (m,n) | V2 offset (m,n) | V3 offset (m,n)\n");
    printf("-------|-----------------|-----------------|-----------------|----------------\n");
    for (int t = 0; t < 4; ++t) {
        printf("   %d   |", t);
        for (int v = 0; v < 4; ++v) {
            int offset = atom_tv(t, v);
            int m = offset % 64;
            int n = offset / 64;
            printf("  %3d (%2d,%2d)  |", offset, m, n);
        }
        printf("\n");
    }

    return 0;
}
