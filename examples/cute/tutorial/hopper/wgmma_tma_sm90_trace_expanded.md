# wgmma_tma_sm90 TMA Trace 展开笔记

本文把当前调试打印链路按“代码 + 中间变量 + 日志字段”摊开，便于你逐步对照分析。

- 示例入口文件：`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu`
- 关键头文件：`include/cute/atom/copy_traits_sm90_tma.hpp`
- 调试开关：`#define CUTE_DEBUG_TMA_GBASIS 1`

## 1. 调用链总览

主调用链（host -> device -> header）：

```text
main
  -> gemm
    -> gemm_nt / gemm_tn
      -> make_tma_atom
        -> make_tma_copy_atom
          -> construct_tma_gbasis
          -> make_tma_copy_desc
      -> launch gemm_device
        -> tma_partition
```

代码位置：

- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:643` (`main`)
- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:605` (`gemm`)
- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:360` (`gemm_nt`)
- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:485` (`gemm_tn`)
- `include/cute/atom/copy_traits_sm90_tma.hpp:1717` (`make_tma_atom`)
- `include/cute/atom/copy_traits_sm90_tma.hpp:1450` (`make_tma_copy_atom`)
- `include/cute/atom/copy_traits_sm90_tma.hpp:753` (`construct_tma_gbasis`)
- `include/cute/atom/copy_traits_sm90_tma.hpp:1077` (`make_tma_copy_desc`)
- `include/cute/atom/copy_traits_sm90_tma.hpp:1755` (`tma_partition`)

## 2. host 侧入口参数展开（wgmma_tma_sm90.cu）

### 2.1 `main -> gemm`

```cpp
// [main] runtime args
// transA / transB / m / n / k

gemm(transA, transB, m, n, k,
     alpha,
     d_A.data().get(), ldA,
     d_B.data().get(), ldB,
     beta,
     d_C.data().get(), ldC);
```

对应日志：

- `[TMA trace][host] main`
- `[TMA trace][host] gemm`
- `dispatch -> gemm_nt` 或 `dispatch -> gemm_tn`

### 2.2 `gemm_nt / gemm_tn` 到 `make_tma_atom`

`gemm_nt`（`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:360`）和 `gemm_tn`（`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:485`）都执行：

```cpp
Tensor mA = make_tensor(A, make_shape(M,K), dA);
Tensor mB = make_tensor(B, make_shape(N,K), dB);

Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
```

对应日志（host）：

- `cta_tiler`
- `sA(_,_,0)`
- `sB(_,_,0)`

## 3. `make_tma_atom` 展开

位置：`include/cute/atom/copy_traits_sm90_tma.hpp:1717`

```cpp
auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
using TmaType = conditional_t<is_same<void, TmaInternalType>::value,
                              typename GEngine::value_type,
                              TmaInternalType>;

auto atom = detail::make_tma_copy_atom<TmaType>(copy_op,
                                                 gtensor, slayout,
                                                 size(cluster_size), cta_v_tile);
```

展开说明：

```cpp
// step-1: 给 gtensor 每个 mode 生成单位 basis（E<0>, E<1>, ...）
auto id_layout = make_identity_layout(shape(gtensor));

// step-2: 用 cta_tiler 在这些 basis 上做 compose，得到 CTA value index -> gmem mode 映射
auto cta_v_tile = id_layout.compose(cta_tiler);
// 对应打印: cta_v_tile (src: make_identity_layout(shape(gtensor)).compose(cta_tiler))

// step-3: 选择 TMA 内部类型
using TmaType = conditional_t<is_same<void, TmaInternalType>::value,
                              typename GEngine::value_type,
                              TmaInternalType>;
// 对应打印:
// - TmaType.bits
// - TmaType.from_default
```

你关心的 `cta_v_tile` 就在这里生成并打印。

## 4. `make_tma_copy_atom` 展开

位置：`include/cute/atom/copy_traits_sm90_tma.hpp:1450`

```cpp
auto smem_swizzle = get_swizzle_portion(slayout);
auto smem_layout  = get_nonswizzle_portion(slayout);

auto tma_gbasis = detail::construct_tma_gbasis<TmaInternalType>(gtensor,
                                                                 smem_layout,
                                                                 cta_v_map);

auto [tma_desc, aux_params] = detail::make_tma_copy_desc<TmaInternalType>(gtensor,
                                                                           tma_gbasis,
                                                                           smem_swizzle,
                                                                           num_multicast);
```

展开说明：

```cpp
// 把 slayout 拆成两部分
auto smem_swizzle = get_swizzle_portion(slayout);   // swizzle 元信息
auto smem_layout  = get_nonswizzle_portion(slayout);// 纯逻辑 layout

// 用“无 swizzle 的 smem_layout + cta_v_map + gtensor”推导 TMA basis
auto tma_gbasis = construct_tma_gbasis(...);

// 用 tma_gbasis + swizzle + multicast 组装 descriptor
auto [tma_desc, aux_params] = make_tma_copy_desc(...);
```

## 5. `construct_tma_gbasis` 展开（核心）

位置：`include/cute/atom/copy_traits_sm90_tma.hpp:753`

按你给的风格展开如下。

```cpp
// [输入] gtensor / slayout / cta_v_map

// smem idx -> smem coord
auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
// 对应打印: inv_smem_layout

// smem idx -> gmem mode（先 composition 再 coalesce）
auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));
// 对应打印: sidx2gmode_full

// 找到可用于 TMA 的有效 rank（basis_value != 1 的边界）
auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e){ ... });
// 对应打印: smem_rank

// 截断成前 smem_rank 个 mode
auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);
// 对应打印: sidx2gmode

// 用 TmaInternalType 重解释 gtensor，并映射到 sidx2gmode
auto tile_gstride = recast<TmaInternalType>(gtensor.compose(sidx2gmode)).layout();
// 对应打印: tile_gstride

// 合并成 TMA 支持的 box 粒度（<=256）
auto tma_gstride = coalesce_256(tile_gstride);
// 对应打印: tma_gstride

// 基于原 gtensor shape 生成 identity basis
auto gbasis = make_identity_layout(shape(gtensor));
// 对应打印: gbasis

auto tile_gbasis_tmp = gbasis.compose(sidx2gmode);
// 对应打印: tile_gbasis_tmp

auto tile_gbasis = make_layout(shape(tile_gstride), stride(tile_gbasis_tmp));
// 对应打印: tile_gbasis

auto tma_gbasis_tile = tile_gbasis.compose(make_layout(wrap(shape(tma_gstride))));
// 对应打印: tma_gbasis_tile

Tensor gtensor_T = recast<TmaInternalType>(gtensor);
// 对应打印: gtensor_T

// 查找还没被 tma_gbasis_tile 覆盖的 basis
auto tile_gbasis_remaining_stride = filter_tuple(...);
// 对应打印: tile_gbasis_remaining_stride

auto tile_gbasis_remaining_shape = repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{});
// 对应打印: tile_gbasis_remaining_shape

auto tma_gbasis_full = make_layout(tuple_cat(...), tuple_cat(...));
// 对应打印: tma_gbasis_full

// 最终 rank 限制为 TMA 支持的 max-rank
auto tma_gbasis = group<cute::min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full);
// 对应打印: tma_gbasis
```

## 6. `make_tma_copy_desc` 展开（descriptor 生成）

位置：`include/cute/atom/copy_traits_sm90_tma.hpp:1077`

```cpp
constexpr int tma_dim = decltype(rank(tma_gbasis))::value;
Tensor gtensor_T = recast<TmaInternalType>(gtensor);
void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
auto  gmem_layout  = gtensor_T.layout();

cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);

for(uint64_t& stride : gmem_prob_stride) {
  stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
}

cute::array<uint32_t, 5> smem_box_shape  = {1,1,1,1,1};
cute::array<uint32_t, 5> smem_box_stride = {1,1,1,1,1};
for_each(make_seq<tma_dim>{}, [&](auto i) {
  smem_box_shape[i] *= size<i>(tma_gbasis);
});

for (uint32_t i = tma_dim-1, multicast = num_multicast; multicast > 1; --i) {
  ...
}

CUresult result = cuTensorMapEncodeTiled(...);

auto recast_ratio = trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                sizeof_bits<TmaInternalType>{});
auto gbasis = make_basis_like(shape(gtensor));
auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) { ... });
```

日志对应重点字段：

- `gmem_prob_shape`
- `gmem_prob_stride[elem]`
- `gmem_prob_stride[byte]`
- `smem_box_shape pre-mcast`
- `smem_box_shape post-mcast`
- `tma_format / smem_swizzle / cuTensorMapEncodeTiled result`
- `recast_ratio`
- `gmem_tma_basis_stride`

## 7. `gemm_device -> tma_partition` 展开

`gemm_device` 位置：`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:107`

`tma_partition` 位置：`include/cute/atom/copy_traits_sm90_tma.hpp:1755`

`gemm_device` 中调用：

```cpp
auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                  group_modes<0,2>(sA), group_modes<0,2>(gA));

auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                  group_modes<0,2>(sB), group_modes<0,2>(gB));
```

`tma_partition` 内部展开：

```cpp
Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));

auto glayout_V = append<GLayout::rank>(layout_V, _);
auto slayout_V = append<SLayout::rank>(layout_V, _);

Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{});
Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{});

auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
auto multicast_coord  = make_coord(make_coord(multicast_offset, Int<0>{}));
auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});

Tensor gresult = domain_offset(gcoord, gtensor_v);
Tensor sresult = domain_offset(scoord, stensor_v);
```

日志对应字段：

- `inv_smem_layout`
- `layout_v`
- `tma_layout_v`
- `layout_V`
- `glayout_V`
- `slayout_V`
- `gtensor_v`
- `stensor_v`
- `multicast_offset`
- `multicast_coord`
- `gcoord`
- `scoord`
- `gresult`
- `sresult`

## 8. 已执行并填充（实测）

实测环境和命令：

- GPU: `NVIDIA GeForce RTX 5070 (compute 12.0)`
- 命令: `/tmp/wgmma_tma_sm90_force 128 128 64 N T`
- 日志: `/tmp/wgmma_tma_sm90_force.log`
- 说明: 由于非 Hopper，运行到 WGMMA 阶段会触发 `CUTE_ARCH_MMA_SM90A_ENABLED` 断言；但在断言前，TMA 链路打印已完整输出并可用于分析。

按“代码 + 实测值”填充如下（取第一次 warmup 中 A/B 第一组）：

```cpp
// [make_tma_atom]
auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
// [print] cta_v_tile: (_128,_64):(_1@0,_1@1)

// [make_tma_copy_atom]
auto smem_swizzle = get_swizzle_portion(slayout);
// [print] smem_swizzle: Sw<3,4,3>
auto smem_layout  = get_nonswizzle_portion(slayout);
// [print] smem_layout: ((_64,_2),(_8,_8)):((_1,_512),(_64,_1024))

// [construct_tma_gbasis]
auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
// [print] inv_smem_layout: (_64,_8,_2,_8):(_1,_128,_64,_1024)

auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));
// [print] sidx2gmode_full: (_64,_8,_2,_8):(_1@0,_1@1,_64@0,_8@1)

auto smem_rank = find_if(stride(sidx2gmode_full), ...);
// [print] smem_rank: _2

auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);
// [print] sidx2gmode: (_64,_8):(_1@0,_1@1)

auto tile_gstride = recast<TmaInternalType>(gtensor.compose(sidx2gmode)).layout();
// [print] tile_gstride: (_64,_8):(_1,128)

auto tma_gstride  = coalesce_256(tile_gstride);
// [print] tma_gstride: (_64,_8):(_1,128)

auto tma_gbasis = group<cute::min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full);
// [print] tma_gbasis: (_64,_8):(_1@0,_1@1)

// [make_tma_copy_desc]
fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);
// [print] gmem_prob_shape: [128, 64, 1, 1, 1]
// [print] gmem_prob_stride[elem]: [1, 128, 0, 0, 0]

for (uint64_t& stride : gmem_prob_stride) {
  stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
}
// [print] gmem_prob_stride[byte]: [2, 256, 0, 0, 0]
// [print] smem_box_shape pre-mcast: [64, 8, 1, 1, 1]
// [print] smem_box_shape post-mcast: [64, 8, 1, 1, 1]
// [print] tma_format: 6
// [print] smem_swizzle(enum): 3
// [print] recast_ratio: _16/_16

auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) { ... });
// [print] gmem_tma_basis_stride: (_1@0,_1@1)

// [tma_partition]
Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
// [print] inv_smem_layout: (_64,_8,_2,_8):(_1,_128,_64,_1024)

Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));
// [print] layout_v: (((_64,_8,_2,_8),_1)):(((_1,_128,_64,_1024),_0))

Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
// [print] tma_layout_v: _512:_1

auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));
// [print] layout_V: (((_64,_8),(_2,_8)):((_1,_128),(_64,_1024)))

auto glayout_V = append<GLayout::rank>(layout_V, _);
// [print] glayout_V: (((_64,_8),(_2,_8)):((_1,_128),(_64,_1024)),_)

auto slayout_V = append<SLayout::rank>(layout_V, _);
// [print] slayout_V: (((_64,_8),(_2,_8)):((_1,_128),(_64,_1024)),_)

Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{});
// [print] gtensor_v: ArithTuple(0,_0) o (((_64,_8),(_2,_8)),1):(((_1@0,_1@1),(_64@0,_8@1)),_64@1)

Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{});
// [print] stensor_v: Sw<3,4,3>_smem_ptr[16b](0x7ef800000400) o ((_512,_16),(_1,_3)):((_1,_512),(_0,_8192))

auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
// [print] multicast_offset: _0
auto multicast_coord  = make_coord(make_coord(multicast_offset, Int<0>{}));
// [print] multicast_coord: ((_0,_0))
auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
// [print] gcoord: ((_0,_0),_0)
auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});
// [print] scoord: ((_0,_0),_0)

Tensor gresult = domain_offset(gcoord, gtensor_v);
// [print] gresult: ArithTuple(0,_0) o (((_64,_8),(_2,_8)),1):(((_1@0,_1@1),(_64@0,_8@1)),_64@1)

Tensor sresult = domain_offset(scoord, stensor_v);
// [print] sresult: Sw<3,4,3>_smem_ptr[16b](0x7ef800000400) o ((_512,_16),(_1,_3)):((_1,_512),(_0,_8192))

// [gemm_device]
// [print] tAgA: ArithTuple(0,_0) o (((_64,_8),(_2,_8)),1):(((_1@0,_1@1),(_64@0,_8@1)),_64@1)
// [print] tAsA: Sw<3,4,3>_smem_ptr[16b](0x7ef800000400) o ((_512,_16),(_1,_3)):((_1,_512),(_0,_8192))
// [print] tBgB: ArithTuple(0,_0) o (((_64,_8),(_2,_8)),1):(((_1@0,_1@1),(_64@0,_8@1)),_64@1)
// [print] tBsB: Sw<3,4,3>_smem_ptr[16b](0x7ef80000c400) o ((_512,_16),(_1,_3)):((_1,_512),(_0,_8192))
// [print] tma_transaction_bytes: 32768
// [print] K_PIPE_MAX: _3
// [print] k_tile_count init: 1
```

## 9. 复现建议

1. 若你在 Hopper（SM90a）上运行，可直接用原始 `wgmma_tma_sm90` 可执行并获得完整（不截断）日志。
2. 若在非 Hopper 上，只能稳定拿到本文这种“到 WGMMA 前”的 TMA 链路日志。
