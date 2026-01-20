# CuTe Layout Transformations for MMA and Copy (sgemm_sm80.cu)

This document provides a comprehensive guide to understanding CuTe's TiledMMA, MMA_Atom, TiledCopy, and Copy_Atom internals, using `examples/cute/tutorial/sgemm_sm80.cu` as a concrete example.

## Table of Contents

1. [Overview](#overview)
2. [How to Print Layouts](#how-to-print-layouts)
3. [Concrete Layout Values from sgemm_sm80](#concrete-layout-values-from-sgemm_sm80)
4. [MMA_Atom and TiledMMA](#mma_atom-and-tiledmma)
5. [Copy_Atom and TiledCopy](#copy_atom-and-tiledcopy)
6. [Data Flow in sgemm_sm80](#data-flow-in-sgemm_sm80)

---

## Overview

CuTe uses a **hierarchical tiling** approach:
- **Atom**: The smallest unit (e.g., one MMA instruction, one memory transaction)
- **Tiled**: Multiple atoms arranged across threads

Key abstractions:
| Abstraction | Purpose | Key Layout |
|-------------|---------|------------|
| `MMA_Atom` | Single MMA instruction | `LayoutA_TV`, `LayoutB_TV`, `LayoutC_TV` |
| `TiledMMA` | Multiple atoms across threads | `thr_layout_vmnk`, `get_layoutA_TV()` |
| `Copy_Atom` | Single copy instruction | `ValLayoutSrc`, `ValLayoutDst`, `ValLayoutRef` |
| `TiledCopy` | Multiple atoms across threads | `TiledLayout_TV`, `get_layoutS_TV()` |

The `_TV` suffix means **(Thread, Value)** layout: maps `(thread_idx, value_idx) -> coordinate`.

---

## How to Print Layouts

Compile with `CUTE_SGEMM_SM80_PRINT_LAYOUT=1`:
```bash
nvcc -DCUTE_SGEMM_SM80_PRINT_LAYOUT=1 -std=c++17 -arch=sm_80 \
  --expt-relaxed-constexpr -O2 \
  -I./include -I./tools/util/include \
  examples/cute/tutorial/sgemm_sm80.cu -o sgemm_sm80_debug
```

Run with small problem size:
```bash
./sgemm_sm80_debug 128 128 64 T N
```

---

## Concrete Layout Values from sgemm_sm80

Using `TN` layout with `SM80_16x8x16_F16F16F16F16_TN`, tile size `128x128x64`:

### CTA Tiler
```
cta_tiler: (_128, _128, _64)
```
- BLK_M = 128, BLK_N = 128, BLK_K = 64

### Shared Memory Layout (Swizzled)
```
swizzle_atom: Sw<3,3,3> o _0 o (_8,(_8,_8)):(_8,(_1,_64))
sA layout: Sw<3,3,3> o _0 o ((_8,_16),((_8,_8),_1),(_1,_3)):((_8,_512),((_1,_64),_0),(_0,_8192))
```

The swizzle pattern `Sw<3,3,3>` XORs bits [3:5] of the row with bits [3:5] of the column to avoid bank conflicts.

### G2S TiledCopy (copyA/copyB)

```
TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Tiler_MN: (_16, _64)
layout S_TV: ((_8,_16),(_8,_1)):((_128,_1),(_16,_0))
layout D_TV: ((_8,_16),(_8,_1)):((_128,_1),(_16,_0))
```

**Interpretation**:
- **TiledLayout_TV**: Shape `(128 threads, 8 values)` → coordinate in tile
  - `((_8,_16),_8)` = 128 threads, 8 values per thread
  - Stride `((_128,_1),_16)`: threads 0-7 handle columns 0,16,32,...; thread 8 handles column 1, etc.
- **Tiler_MN**: `(_16, _64)` = each copy iteration handles 16×64 elements
- 128 threads × 8 values = 1024 elements = 16×64 ✓

### MMA Atom (SM80_16x8x16_F16F16F16F16_TN)

```
Shape_MNK: (16, 8, 16)   # One MMA instruction computes 16×8 output with K=16
ThrID: 32 threads per MMA

LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

**LayoutA_TV Explained**:
- Shape: `(32 threads, 8 values)` → (M,K) coordinate
- `(_4,_8)` = 32 threads, `(_2,_2,_2)` = 8 values per thread
- For A matrix (16×16), each thread holds 8 FP16 values

### TiledMMA (2×2 atom arrangement)

```
thr_layout_vmnk: (_32,_2,_2,_1):(_1,_32,_64,_0)
tile_shape: (_32, _32, _16)
```

**thr_layout_vmnk**:
- Shape: `(ThrV=32, ThrM=2, ThrN=2, ThrK=1)` = 128 threads total
- `ThrV=32`: threads within one MMA atom
- `ThrM=2, ThrN=2`: 2×2 tiling of atoms
- Stride `(_1,_32,_64,_0)`: thread_idx = ThrV + 32*ThrM + 64*ThrN

```
layoutA_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
layoutB_TV: ((_4,_8,_2,_2),((_2,_2),(_2,_1))):((_64,_1,_0,_8),((_32,_256),(_16,_0)))
layoutC_TV: ((_4,_8,_2,_2),((_2,_2),(_1,_2))):((_64,_1,_16,_256),((_32,_8),(_0,_512)))
```

**layoutA_TV Interpretation**:
- Thread shape: `(_4,_8,_2,_2)` = 128 threads
  - `_4,_8` = 32 threads per atom
  - `_2,_2` = 2×2 atom tiling (but only ThrM=2 matters for A)
- Value shape: `((_2,_2,_2),(_1,_1))` = 8 values per thread
  - `(_2,_2,_2)` = 8 values from atom
  - `(_1,_1)` = 1×1 repetition (no extra tiling in M×K)

### S2R TiledCopy (ldmatrix)

```
s2r atom ValLayoutSrc: (_32,_8):(_8,_1)
s2r_copy_a TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
s2r_copy_a Tiler_MN: (_32, _16)
```

The S2R copy layout **matches** `mmaC.get_layoutA_TV()` because `make_tiled_copy_A` uses the MMA layout.

---

## MMA_Atom and TiledMMA

### MMA_Atom

Wraps hardware MMA traits defining:
- `Shape_MNK`: Per-instruction tile shape (e.g., 16×8×16)
- `ThrID`: Thread layout within one instruction
- `LayoutA_TV`, `LayoutB_TV`, `LayoutC_TV`: (thread, value) → coordinate

### TiledMMA Construction

```cpp
TiledMMA mma = make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},  // Atom
    Layout<Shape<_2,_2>>{},          // 2×2 atom arrangement
    Tile<_32,_32,_16>{});            // Per-warp tile size
```

Key state:
```cpp
thr_layout_vmnk_ = tiled_product(AtomThrID{}, thr_layout_mnk)
// Result: (ThrV, ThrM, ThrN, ThrK) → thread_idx
```

### thrfrg_C Transform

Transforms tensor from `(M,N,...)` to `((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))`:

```cpp
template <class CTensor>
auto thrfrg_C(CTensor&& ctensor) const {
    // Step 1: Apply permutation
    auto t_tile = make_tile(permutation_mnk<0>(), permutation_mnk<1>());
    auto t_tensor = logical_divide(ctensor, t_tile);        // (PermM,PermN)

    // Step 2: Tile for atom
    auto c_tile = make_tile(make_layout(AtomM), make_layout(AtomN));
    auto c_tensor = zipped_divide(t_tensor, c_tile);        // ((AtomM,AtomN),(RestM,RestN))

    // Step 3: Transform to (Thr,Val)
    auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _); // ((ThrV,FrgV),(RestM,RestN))

    // Step 4: Tile for threads
    auto thr_tile = make_tile(_, make_tile(Layout(ThrM), Layout(ThrN)));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);   // ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))

    return thr_tensor;
}
```

**Visualization for 32×32 C tile with 2×2 atoms (16×8 each)**:
```
Input: (M=32, N=32)
       ↓ logical_divide with permutation
Step 1: (32, 32)   # No permutation change
       ↓ zipped_divide with (16, 8)
Step 2: ((16,8), (2,4))  # 2 atoms in M, 4 atoms in N
       ↓ compose with LayoutC_TV (32 threads, 4 values)
Step 3: ((32,4), (2,4))  # (ThrV,FrgV), (RestM,RestN)
       ↓ zipped_divide with (_, (2,2))
Step 4: ((32,(2,2)), (4,(1,2)))  # Final layout
```

### thrfrg_A and thrfrg_B

Similar transforms for A(M,K) and B(N,K):
- `thrfrg_A`: `(M,K) → ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))`
- `thrfrg_B`: `(N,K) → ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))`

### get_layoutA/B/C_TV

Returns `(thread_idx, value_idx) → coordinate`:
```cpp
auto get_layoutC_TV() const {
    auto ref_C = make_layout(tile_size<0>(), tile_size<1>());
    auto thridx_2_thrid = /* maps thread_idx to (ThrV,ThrM,ThrN,ThrK) */;
    return thrfrg_C(ref_C).compose(thridx_2_thrid, _);
}
```

---

## Copy_Atom and TiledCopy

### Copy_Atom

Wraps copy instruction traits:
- `ValLayoutSrc/Dst/Ref`: Value layouts for source, destination, reference
- `NumValSrc/NumValDst`: Vector width (e.g., 8 for 128-bit load)

For `SM75_U32x4_LDSM_N` (ldmatrix.x4):
```
ValLayoutSrc: (_32,_8):(_8,_1)  # 32 threads, 8 values each
```

### TiledCopy Construction

```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16,_8>, Stride<_8,_1>>{},  // Thread layout: 16×8 threads, K-major
    Layout<Shape<_1,_8>>{});                  // Value layout: 1×8 values per thread
```

Key types:
- `TiledLayout_TV`: `(thread_idx, value_idx) → tile_coord`
- `Tiler_MN`: Shape of the tile this copy handles

### tile2thrfrg Transform

Core transform: `(M,N,...) → (Thr, (FrgV,FrgX), (RestM,RestN,...))`

```cpp
template <class Tensor, class Ref2TrgLayout>
auto tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg) {
    // Step 1: Split TiledLayout_TV by atom size
    auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, (AtomNumThr, AtomNumVal));
    // ((atom_tid,atom_val),(rest_tid,rest_val)) → coord

    // Step 2: Transform to target layout
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid,trg_val),(rest_tid,rest_val)) → coord

    // Step 3: Coalesce threads
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
    // ((trg_tid,rest_tid),(trg_val,rest_val)) → coord

    // Step 4: Apply to tensor
    auto tv_tensor = tensor.compose(thrval2mn, _);
    return tv_tensor(make_coord(_,_), _);
}
```

### retile Transform

Reinterprets a fragment tensor to match copy atom's expected layout:

```cpp
auto retile(Tensor&& tensor) {
    auto V = size<0>(tensor);  // Fragment size

    // Build layout mapping (m,n) → v_idx
    auto frg_layout_mn = upcast<NumThr * V>(right_inverse(TiledLayout_TV{}).with_shape(Tiler_MN));

    // Split into (atom_vals, rest_vals)
    auto frg_layout_v = zipped_divide(logical_product(...), AtomNumVal);

    // Apply to tensor
    auto t_tensor = zipped_divide(tensor, ...);
    return t_tensor.compose(frg_layout_v, _);
}
```

This is crucial for `retile_D` which aligns S2R copy with MMA fragments.

### make_tiled_copy_A/B

Creates a TiledCopy aligned with MMA's operand layout:

```cpp
auto make_tiled_copy_A(CopyAtom const& atom, TiledMMA const& mma) {
    return make_tiled_copy_impl(
        atom,
        mma.get_layoutA_TV(),          // Use MMA's A layout
        make_shape(tile_size_mnk<0>(mma), tile_size_mnk<2>(mma))  // (M,K) tile
    );
}
```

---

## Data Flow in sgemm_sm80

### 1. Global → Shared (G2S) with cp.async

```cpp
// Create TiledCopy
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16,_8>, Stride<_8,_1>>{},  // 128 threads, K-major
    Layout<Shape<_1,_8>>{});                  // 8 values per thread

// Partition
ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);  // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);  // (CPY,CPY_M,CPY_K,PIPE)

// Copy
copy(copyA, tAgA(_,_,_,k), tAsA(_,_,_,pipe));
```

### 2. Shared → Register (S2R) with ldmatrix

```cpp
// Create S2R copy aligned with MMA
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_A, mma);

// Partition smem and retile register fragment
ThrCopy s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
Tensor tXsA = s2r_thr_a.partition_S(sA);   // (CPY,MMA_M,MMA_K,PIPE)
Tensor tXrA = s2r_thr_a.retile_D(tCrA);    // (CPY,MMA_M,MMA_K) - retiled fragment

// Copy
copy(s2r_atom_A, tXsA(_,_,k_block), tXrA(_,_,k_block));
```

**Why retile_D?**
- `tCrA` is MMA fragment: `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))`
- ldmatrix expects: `(atom_vals, rest_vals, ...)`
- `retile_D` transforms the layout so copy lands on correct elements

### 3. MMA Compute

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCrA = thr_mma.partition_fragment_A(sA);  // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB);  // (MMA,MMA_N,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);     // (MMA,MMA_M,MMA_N)

gemm(mma, tCrA(_,_,k), tCrB(_,_,k), tCrC);
```

### 4. Register → Global (Epilogue)

```cpp
Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)
axpby(alpha, tCrC, beta, tCgC);         // Direct copy to global
```

---

## Summary

| Operation | Transform | Input Shape | Output Shape |
|-----------|-----------|-------------|--------------|
| `thrfrg_C` | MMA C partition | `(M,N)` | `((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))` |
| `thrfrg_A` | MMA A partition | `(M,K)` | `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))` |
| `thrfrg_B` | MMA B partition | `(N,K)` | `((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))` |
| `tidfrg_S` | Copy src partition | `(M,N)` | `(Thr,(FrgV,FrgX),(RestM,RestN))` |
| `tidfrg_D` | Copy dst partition | `(M,N)` | `(Thr,(FrgV,FrgX),(RestM,RestN))` |
| `retile` | Fragment → Copy | `(V,...)` | `((atom_vals,rest_vals),...)` |

The key insight is that **MMA and Copy must agree on how data is distributed across threads**. `make_tiled_copy_A/B` ensures this alignment by using `mma.get_layoutA/B_TV()`.
