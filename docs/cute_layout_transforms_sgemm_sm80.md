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

### Host-only dump (no GPU required at runtime)

`examples/cute/tutorial/sgemm_sm80.cu` also supports a **host-only** layout/algebra dump path that exits before any CUDA device queries or kernel launches:

```bash
nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -O2 \
  -I./include -I./tools/util/include \
  examples/cute/tutorial/sgemm_sm80.cu -o sgemm_sm80_dump

./sgemm_sm80_dump --dump-layouts --dump-m 128 --dump-n 128 --dump-k 64
./sgemm_sm80_dump --dump-latex --outdir ./dump_tex
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
s2r atom ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
s2r atom ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
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

## 深入理解 copy_atom.hpp：tidfrg_D / tile2thrfrg / partition_S

本节只看 3 个函数（都在 `include/cute/atom/copy_atom.hpp`）：
- `TiledCopy::tidfrg_D`（以及对称的 `tidfrg_S`）
- `TiledCopy::tile2thrfrg`
- `ThrCopy::partition_S`（以及对称的 `partition_D`）

目标：把**一个 tile 内的 (m,n) 坐标空间**，转换成**线程 + 寄存器片段（fragment）**的坐标空间，这样 `copy(copy_atom, src, dst)` 才能在每个线程上拿到正确的那一小段。

### 0. 需要先区分 3 套“坐标系”

以 A 矩阵从 gmem 复制到 smem 的 `copyA` 为例（`sgemm_sm80.cu` 里的 `gemm_tn()`）：

1) **原始张量坐标**：例如 `gA` 的逻辑坐标是 `(M,K,kTile)`。

2) **tile 坐标**：每次 copy 覆盖一个 `Tiler_MN = (TileM,TileN)` 的二维 tile。
   - 对 A 来说，这个 tile 是 `(TileM, TileK)`，在代码里仍叫 `Tiler_MN`（历史命名）。

3) **线程-片段坐标（thr-frg）**：把 tile 内元素分配给线程与寄存器片段：
   - `Thr`：逻辑线程 id（最终会用 `threadIdx.x` 选中其中一个）
   - `(FrgV, FrgX)`：这个线程在一次或多次 CopyAtom 指令里要搬的“片段”坐标
     - `FrgV`：CopyAtom 指令本身的向量长度（每条指令搬多少个 value）
     - `FrgX`：在一个 tile 内，一个线程要发出多少条 CopyAtom 指令（如果需要多条）

### 1) tidfrg_D：先切 tile，再交给 tile2thrfrg

`tidfrg_D` 的代码在 `include/cute/atom/copy_atom.hpp:239`：

```cpp
return tile2thrfrg(zipped_divide(dtensor, Tiler_MN{}),
                   right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}));
```

这行分成两半理解就够了：

**A. `zipped_divide(dtensor, Tiler_MN{})`：把 (M,N,...) 切成 (Tile, Rest)**

`zipped_divide` 会把前 `rank(Tiler_MN)` 个 mode 拆成：
- tile 内坐标 `((TileM,TileN), ...)`
- tile 之外的坐标 `((RestM,RestN), ...)`

因此把一个 layout 从
`(M,N,Rest...) -> idx`
变成
`((TileM,TileN), (RestM,RestN,Rest...)) -> idx`

**B. `right_inverse(AtomLayoutRef).compose(AtomLayoutDst)`：把“dst 的 thr/val 顺序”对齐到“ref 的 thr/val 顺序”**

这里非常容易被名字误导：这不是“ref->dst”，而是 **dst->ref** 的映射：

```
dst2ref = right_inverse(Ref(thr,val)->bit) ∘ Dst(thr,val)->bit
```

含义：CopyAtom 的“参考位序（RefLayout）”规定了 (thr,val) 如何线性化成 bit offset。
如果 `DstLayout` 的位序不同，就必须先把 dst 的 (thr,val) 重新编号成 ref 的编号，否则 copy_unpack 里做 `recast<>` 时会把向量打乱。

对 `cp.async` 来说（`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`），`SrcLayout/DstLayout/RefLayout` 本来就是一致的，所以这一步是恒等变换；但对更复杂的指令（例如 `ldmatrix`）这一步就很关键。

### 2) tile2thrfrg：把 tile 坐标变成 (Thr,(FrgV,FrgX))

`tile2thrfrg` 在 `include/cute/atom/copy_atom.hpp:254`，按 4 步看：

#### Step 1: 把 TiledLayout_TV 拆成 “atom 内” 与 “atom 间”

```cpp
auto atom_layout_TV =
  zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
// ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)  (在 tile 内)
```

这里的核心是假设（源码也写了）：`AtomNumThr` / `AtomNumVal` 在 `TiledLayout_TV` 里是**连续且低位优先**的。
于是 TiledCopy 的线程/值坐标可以拆成：
- `atom_tid/atom_val`：一条 CopyAtom 指令内部的线程/向量 lane
- `rest_tid/rest_val`：在一个 tile 内，要发几条指令、或有几个 atom group

#### Step 2: 用 src2ref 或 dst2ref 纠正 atom 内部顺序

```cpp
auto trg_layout_TV = atom_layout_TV.compose(trg2ref, _);
```

如果你传入的是 `dst2ref`（来自 `tidfrg_D`），那它就是把 `(dst_tid,dst_val)` 重排成 `(ref_tid,ref_val)` 再交给 `atom_layout_TV`。

#### Step 3: 重新组织维度：把 thread 维度合并，把 value 维度保留成 (FrgV,FrgX)

```cpp
auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
// (Thr, (FrgV,FrgX)) -> (m,n)
```

`zip(trg_layout_TV)` 的作用：把
`((tid,val),(rest_tid,rest_val))`
换成
`((tid,rest_tid),(val,rest_val))`，
即把“线程相关的 mode”放一起，把“值相关的 mode”放一起。

`coalesce(..., Shape<_1,Shape<_1,_1>>{})` 的 profile 很关键：
- 第 0 个 mode（线程那边）用 `_1`：尽量 coalesce 成**一个** `Thr`
- 第 1 个 mode（值那边）用 `Shape<_1,_1>`：保留成**两个** mode：`(FrgV, FrgX)`

#### Step 4: 把 (Thr,Frg) 映射回原 tensor 的 idx

```cpp
auto tv_tensor = tensor.compose(thrval2mn, _);
return tv_tensor(make_coord(_,_), _);
```

`tensor` 是 `zipped_divide(dtensor, Tiler_MN)` 的结果，形状是 `((TileM,TileN),(Rest...))`。
`compose(thrval2mn,_)` 把 tile 内坐标替换成 `(Thr,(FrgV,FrgX))`，从而得到：
`(Thr,(FrgV,FrgX),(Rest...)) -> idx`

最后一行 `tv_tensor(make_coord(_,_),_)` 只是把第一个 tuple mode 展开成两个 mode，便于后续 `partition_S/D` 用 `thr_idx` 直接索引。

### 3) partition_S：把 tidfrg_S 的第 0 维切成“当前线程”

`partition_S` 在 `include/cute/atom/copy_atom.hpp:365`：

```cpp
auto thr_tensor = make_tensor(stensor.data(), TiledCopy::tidfrg_S(stensor.layout()));
return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
```

要点只有两个：

1) `make_tensor(data, new_layout)` 不搬数据，只是**重解释**同一块内存的坐标系。
   - 这一步把张量 layout 改成 `tidfrg_S` 的输出：
     `(Thr,(FrgV,FrgX),(Rest...)) -> idx`

2) `thr_tensor(thr_idx_, ...)` 取第 0 维 `Thr = thr_idx_` 的 slice，
   得到“这个线程应该搬的那一片”：
   `((FrgV,FrgX),(Rest...)) -> idx`

### 4) 把这些映射回 sgemm_sm80 的直觉（copyA 的具体数字）

在 `sgemm_sm80.cu` 的 TN 路径里：

- `copyA` 使用 `cp.async`，因此
  - `AtomNumThr = 1`
  - `AtomNumVal = 8`（16B = 8×FP16）
- `make_tiled_copy` 里你给的
  - `thr_layout = Layout<Shape<_16,_8>,Stride<_8,_1>>`（16×8 = 128 threads）
  - `val_layout = Layout<Shape<_1,_8>>`（每线程 8 个 value）
- 因此一个 tile 覆盖的元素数：
  - `Tiler_MN = (16,64)`，因为 16 行 × (8 threads × 8 vals) = 16×64

把 K 维写成 `k = 8*k_group + k_inner`：
- `thr_idx = m*8 + k_group`
- `val_idx = k_inner`
所以每条 `cp.async` 正好搬一段连续的 `k_inner=0..7`。

对整个 `gA` 的 tile（128×64）来说，M 维还有 `RestM = 128/16 = 8` 个这样的 tile 行块，
所以每个线程会发 `8` 次 `cp.async`，总共搬 `8*8 = 64` 个 FP16。

你可以用 host-only dump 直接看到这一点：

```bash
nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -O2 \
  -I./include -I./tools/util/include \
  examples/cute/tutorial/sgemm_sm80.cu -o /tmp/sgemm_sm80_dump

/tmp/sgemm_sm80_dump --dump-layouts
```

输出里的 `tAgA`（thread0 的 `partition_S(gA)`）会长得像：
`((_8,_1),_8,_1,1):...`
其中：
- `(_8,_1)` 就是 `(FrgV=8, FrgX=1)`
- 第二个 `_8` 是 `RestM=8`（沿 M 方向的 tile 个数）

这就是 `tidfrg_* / tile2thrfrg / partition_*` 三件套最终想要实现的形状语义。

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

#### ldmatrix S2R copy：`copy(s2r_atom_A, ...)` 里“发生了什么”

在 `sgemm_sm80.cu` 的 S2R 路径里，`copy(s2r_atom_A, ...)` 最终会落到 `include/cute/arch/copy_sm75.hpp` 的 PTX：
`ldmatrix.sync.aligned.x4.m8n8.shared.b16`（通过 `SM75_U32x4_LDSM_N::copy`）。

这条指令的要点是：
- **源端（SRegisters）**：每个线程提供一个 16B 对齐的 smem 地址（在实现里是 `uint128_t const& smem_src`）。
- **目的端（DRegisters）**：每个线程得到 4 个 `u32`（总共 16B），等价于 8 个 `half`，但 **寄存器里 half 的排列不是“每线程连续 8 个 half”**，而是按 ldmatrix 的矩阵语义进行打包/重排。

CuTe 用 `Copy_Atom` 的三套 value layout 来建模这种“语义重排”（定义语义见 `include/cute/atom/copy_traits.hpp` 的注释）：
- `ValLayoutSrc`: `(thr, vid) -> offset`（从源端看每个线程的 value 线性编号）
- `ValLayoutDst`: `(thr, vid) -> offset`（从目的端看寄存器打包后的 value 线性编号）
- `ValLayoutRef`: 参考编号空间（RefLayout），`TiledCopy` 的 `Layout_TV` 语义以它为准，然后按需映射到 Src/Dst

对 `SM75_U32x4_LDSM_N`（见 `include/cute/atom/copy_traits_sm75.hpp`）而言，`RefLayout = DstLayout`。直觉上：**硬件“返回寄存器”的顺序是不可选的**，所以把它当成 canonical reference，源端去对齐它更自然。

用 host-only dump（`./sgemm_sm80_dump --dump-layouts`）可以直接看到 ldmatrix 的 3 个 value layout 以及 src2ref/dst2ref：

```text
ldmatrix ValLayoutSrc (thr,val)->offset: (_32,_8):(_8,_1)
ldmatrix ValLayoutDst (thr,val)->offset: (_32,(_2,_4)):(_2,(_1,_64))
ldmatrix ValLayoutRef (thr,val)->offset: (_32,(_2,_4)):(_2,(_1,_64))

ldmatrix src2ref (src_tv->ref_tv): ((_8,_4),(_2,_4)):((_4,_64),(_32,_1))
ldmatrix dst2ref (dst_tv->ref_tv): (_32,(_2,_4)):(_1,(_32,_64))

ldmatrix src2ref samples (src_tid,src_vid)->(ref_tid,(lane,reg)):
  tid  0 : ( 0,(0,0)) ( 0,(1,0)) ( 1,(0,0)) ( 1,(1,0)) ( 2,(0,0)) ( 2,(1,0)) ( 3,(0,0)) ( 3,(1,0))
  tid  8 : ( 0,(0,1)) ( 0,(1,1)) ( 1,(0,1)) ( 1,(1,1)) ( 2,(0,1)) ( 2,(1,1)) ( 3,(0,1)) ( 3,(1,1))
  tid 16 : ( 0,(0,2)) ( 0,(1,2)) ( 1,(0,2)) ( 1,(1,2)) ( 2,(0,2)) ( 2,(1,2)) ( 3,(0,2)) ( 3,(1,2))
  tid 24 : ( 0,(0,3)) ( 0,(1,3)) ( 1,(0,3)) ( 1,(1,3)) ( 2,(0,3)) ( 2,(1,3)) ( 3,(0,3)) ( 3,(1,3))
```

如何读 `ValLayoutDst/Ref: (_32,(_2,_4)):(_2,(_1,_64))`：
- `(_2,_4)` 把每线程 8 个 half 分解成：`lane=2`（一个 `u32` 里 2 个 half）× `reg=4`（x4 的 4 个 `u32`）。
- `stride ... (_1,_64)` 表示：同一个 `reg` 内两个 half 相邻（stride 1），不同 `reg` 之间相差 64 个 half（也就是一个 8×8 tile 的大小）。
- `thr` 维的 stride 是 2：因为每个线程在一个 tile 里贡献的是一个 `u32`（=2 个 half），所以 thread 间以 2 为步进交错。

把这些放回到 `sgemm_sm80.cu` 的两行里看，就能理解 `partition_S / retile_D` 在“对齐什么”：
- `tXsA = s2r_thr_a.partition_S(sA)`：内部会走 `tidfrg_S`/`tile2thrfrg`，并使用 `src2ref = right_inverse(Ref) ∘ Src` 把源端的 `(thr,vid)` 重排到 Ref(Dst) 的语义下；对 ldmatrix 来说这一步不是恒等变换。
- `tXrA = s2r_thr_a.retile_D(tCrA)`：把 MMA fragment 的寄存器视图重排成 CopyAtom 期望的 `(thr, (lane,reg), ...)` 语义。
- 最终 `copy(s2r_atom_A, ...)` 执行 ldmatrix，硬件返回的寄存器打包顺序正是 `Dst/Ref` 的语义，因此数据能直接落到 `tXrA` 正确的位置，后续 `gemm(mma, ...)` 才能按 `mma.get_layoutA_TV()` 的语义消费它。

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
