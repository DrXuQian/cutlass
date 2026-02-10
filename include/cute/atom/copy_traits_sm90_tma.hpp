/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

#include <cute/atom/copy_traits_sm90_tma_swizzle.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/algorithm/prefetch.hpp>

#include <cute/numeric/integral_ratio.hpp>

#include <cutlass/cuda_host_adapter.hpp>

namespace cute
{

template <class GmemTmaBasisStrides_, class TmaGmemBasis_, class TmaSwizzle_>
struct AuxTmaParams {
  using GmemStrides  = GmemTmaBasisStrides_;    // Strides for Gmem mode -> Tma coord mode, may be dynamic
  GmemStrides g_stride_;
  using TmaGmemBasis = TmaGmemBasis_;           // Layout for Tma box shape -> Gmem mode(s), always static
  static_assert(is_static<TmaGmemBasis>::value);
  using TmaSwizzle   = TmaSwizzle_;             // Tma swizzle, always Swizzle<B,M,S>
  static_assert(is_static<TmaSwizzle>::value);
};

// Utility for unpacking TMA_LOAD arguments into a CopyOp
template <class CopyOp, class... Args>
struct TMA_LOAD_Unpack
{
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    static_assert(is_smem<TD>::value, "SM90_TMA_LOAD requires the destination be shared memory.");

    auto src_coord = src.data().coord_;
    void* dst_ptr = cute::raw_pointer_cast(dst.data());
#if 0
    auto [c0,c1,c2,c3,c4] = append<5>(src_coord, 0);
    printf("THR (%d,%d,%d) BLK (%d,%d,%d) TMACRD (%d,%d,%d,%d,%d) SMEMADDR (%p)\n",
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z,
          int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), dst_ptr);
#endif
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                 make_tuple(dst_ptr), seq<0>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_OP : SM90_TMA_LOAD {};

// The non-executable SM90_TMA_LOAD with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;

  // Construct with updated TMA descriptor only (no barrier change)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
  with(TmaDescriptor const* new_tma_desc) const {
    return {*new_tma_desc, aux_params_};
  }
};

// The executable SM90_TMA_LOAD with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  : TMA_LOAD_Unpack<SM90_TMA_LOAD_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint64_t   // cache hint
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint64_t cache)
    : opargs_(desc, mbar, cache) {}

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return get<0>(opargs_);
  }
};

// The prefetch for SM90_TMA_LOAD with tma_desc
template <class NumBitsPerTMA, class... Args>
struct Copy_Traits<SM90_TMA_LOAD::PREFETCH, NumBitsPerTMA, Args...>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD::PREFETCH arguments
  tuple<TmaDescriptor const*> const opargs_;

  // Construct with any other Traits' TMA Desc
  template <class OtherTraits>
  CUTE_HOST_DEVICE
  Copy_Traits(OtherTraits const& traits)
    : opargs_({traits.get_tma_descriptor()}) {}

  // Construct directly with a TMA descriptor pointer
  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc)
    : opargs_({desc}) {}

  // Build a new Prefetch traits with a different TMA descriptor pointer
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD::PREFETCH, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc) const {
    return {new_tma_desc};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    auto src_coord = src.data().coord_;
    return detail::explode_tuple(detail::CallCOPY<SM90_TMA_LOAD::PREFETCH>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_MULTICAST_OP : SM90_TMA_LOAD_MULTICAST {};

// The non-executable SM90_TMA_LOAD_MULTICAST with tma_desc and no tma_mbar
// Use .with(tma_mbar, multicast_mask) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD_MULTICAST with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_load_mbar,
    uint16_t const& multicast_mask,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    return {&tma_desc_, &tma_load_mbar, multicast_mask, static_cast<uint64_t>(cache_hint)};
  }

  // Construct an executable SM90_TMA_LOAD_MULTICAST_OP with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_load_mbar,
    uint16_t const& multicast_mask,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    return {new_tma_desc, &tma_load_mbar, multicast_mask, static_cast<uint64_t>(cache_hint)};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD_MULTICAST before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable SM90_TMA_LOAD_MULTICAST with tma_desc and tma_mbar and multicast_mask
template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>
  : TMA_LOAD_Unpack<SM90_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint16_t,  // multicast mask
  uint64_t   // cache hint
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint16_t mask, uint64_t hint)
    : opargs_(desc, mbar, mask, hint) {}

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return get<0>(opargs_);
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_STORE //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_STORE_PTR : SM90_TMA_STORE {};

// The executable SM90_TMA_STORE with tma_desc
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_STORE, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_STORE arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Construct new TMA_STORE with (unsafe) swapped out TMA descriptor ptr (for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_STORE_PTR, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc) const {
    return {new_tma_desc};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for SM90_TMA_STORE");
    //static_assert(is_gmem<TD>::value, "Expected gmem dst for SM90_TMA_STORE");  // TMA spoofed src tensor

    void const* const desc_ptr = &(traits.tma_desc_);
    void const* const src_ptr  = cute::raw_pointer_cast(src.data());
    auto dst_coord = dst.data().coord_;
#if 0
    auto [c0,c1,c2,c3,c4] = append<5>(dst_coord, 0);
    printf("THR (%d,%d,%d) BLK (%d,%d,%d) TMACRD (%d,%d,%d,%d,%d) SMEMADDR (%p)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), src_ptr);
#endif
    return detail::explode_tuple(detail::CallCOPY<SM90_TMA_STORE>{},
                                 make_tuple(desc_ptr, src_ptr), seq<0,1>{},
                                 dst_coord, tuple_seq<decltype(dst_coord)>{});
  }
};

// Same as SM90_TMA_STORE, but with an unsafe TMA Desc PTR instead
template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_STORE_PTR, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_STORE arguments
  TmaDescriptor const* tma_desc_;

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for SM90_TMA_STORE");
    //static_assert(is_gmem<TD>::value, "Expected gmem dst for SM90_TMA_STORE");  // TMA spoofed src tensor

    void const* const desc_ptr = traits.tma_desc_;
    void const* const src_ptr  = cute::raw_pointer_cast(src.data());
    auto dst_coord = dst.data().coord_;
#if 0
    auto [c0,c1,c2,c3,c4] = append<5>(dst_coord, 0);
    printf("THR (%d,%d,%d) BLK (%d,%d,%d) TMACRD (%d,%d,%d,%d,%d) SMEMADDR (%p)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), src_ptr);
#endif
    return detail::explode_tuple(detail::CallCOPY<SM90_TMA_STORE_PTR>{},
                                 make_tuple(desc_ptr, src_ptr), seq<0,1>{},
                                 dst_coord, tuple_seq<decltype(dst_coord)>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_REDUCE_ADD //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// The executable SM90_TMA_REDUCE_ADD with tma_desc
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_REDUCE_ADD, NumBitsPerTMA, AuxParams_>
{
  using ThrID   = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_REDUCE_ADD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void const* const src_ptr,
               Coord const& dst_coord, seq<Is...>) const
  {
#if 0
    auto [c0,c1,c2,c3,c4] = append<5>(dst_coord, 0);
    printf("THR (%d,%d,%d) BLK (%d,%d,%d) TMACRD (%d,%d,%d,%d,%d) SMEMADDR (%p)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), src_ptr);
#endif

    SM90_TMA_REDUCE_ADD::copy(&tma_desc_,
                         src_ptr, get<Is>(dst_coord)...);
  }

  // This is the copy_unpack dispatch for this Copy_Traits
  // Src needs to be a smem tensor
  // Dst needs to be a gmem tensor with TmaCoordIterator .data()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for SM90_TMA_REDUCE_ADD");
    //static_assert(is_gmem<TD>::value, "Expected gmem dst for SM90_TMA_REDUCE_ADD");  // TMA spoofed src tensor

    traits.copy_unpack_(cute::raw_pointer_cast(src.data()), dst.data().coord_, tuple_seq<decltype(dst.data().coord_)>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// BULK COPY //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <class NumBitsPerTMA, class... OpArgs>
struct Copy_Traits<SM90_BULK_COPY_G2S, NumBitsPerTMA, OpArgs...>
{
  static_assert(int32_t(NumBitsPerTMA::value / 8) % 16 == 0,
                "Bulk Copy requires copy vector size align to 16B.");

  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_BULK_COPY_G2S arguments
  // 0: uint64_t* bulk_load_memory_barrier
  cute::tuple<OpArgs...> bulk_load_mbar_;

  // Record the memory barrier for the instruction
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_BULK_COPY_G2S, NumBitsPerTMA, uint64_t*>
  with(uint64_t& bulk_mbar) const {
    return {&bulk_mbar};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_same<cute::tuple<OpArgs...>, cute::tuple<uint64_t*>>::value,
                  "Extra arguments not set. Set .with() before use.");
    static_assert(is_gmem<TS>::value, "Expected gmem src for SM90_BULK_COPY_G2S");
    static_assert(is_smem<TD>::value, "Expected smem dst for SM90_BULK_COPY_G2S");
    SM90_BULK_COPY_G2S::copy(raw_pointer_cast(src.data()), get<0>(traits.bulk_load_mbar_),
                             raw_pointer_cast(dst.data()), int32_t(NumBitsPerTMA::value / 8));
  }
};

template <class NumBitsPerTMA, class... Args>
struct Copy_Traits<SM90_BULK_COPY_G2S::PREFETCH, NumBitsPerTMA, Args...>
     : Copy_Traits<SM90_BULK_COPY_G2S, NumBitsPerTMA>
{
  template <class... CopyArgs>
  CUTE_HOST_DEVICE
  Copy_Traits(Copy_Traits<CopyArgs...> const& traits) {}

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for SM90_BULK_PREFETCH");
    SM90_BULK_COPY_G2S::PREFETCH::copy(raw_pointer_cast(src.data()), int32_t(NumBitsPerTMA::value / 8));
  }
};

template <class NumBitsPerTMA>
struct Copy_Traits<SM90_BULK_COPY_S2G, NumBitsPerTMA>
{
  static_assert(int32_t(NumBitsPerTMA::value / 8) % 16 == 0,
                "Bulk Copy requires copy vector size align to 16B.");

  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for SM90_BULK_COPY_S2G");
    static_assert(is_gmem<TD>::value, "Expected gmem dst for SM90_BULK_COPY_S2G");
    SM90_BULK_COPY_S2G::copy(raw_pointer_cast(src.data()), raw_pointer_cast(dst.data()), int32_t(NumBitsPerTMA::value / 8));
  }
};

//
// Placeholder for the bulk copy algorithm's default, auto-vectorizing behavior
//

template <class... OpArgs>
struct Copy_Traits<SM90_BULK_COPY_AUTO, OpArgs...>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,_1>, Stride<_0,_0>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,_1>, Stride<_0,_0>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_UBULK_COPY arguments
  // 0: uint64_t* bulk_load_memory_barrier [if this is a BULK_LOAD_G2S]
  cute::tuple<OpArgs...> opargs_;

  // Record the memory barrier for the instruction
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_BULK_COPY_AUTO, uint64_t*>
  with(uint64_t& bulk_mbar) const {
    return {&bulk_mbar};
  }
};

//
// MAKE_TMA_COPY and related
//

namespace detail {

// Custom version of coalesce that greedily combines modes only up to size-256
// Look at each element and the back of the stack (in order of priority)
// back(NewLayout)  get<I>(OldLayout)
//      s0:d0           _1:d1     =>  continue
//      _1:d0           s1:d1     =>  replace_back     s1:d1
//      s0:d0           s1:s0*d0  =>  replace_back  s0*s1:d0   if s0*s1 <= 256
//      s0:d0           s1:d1     =>  append           s1:d1
//
// @pre OldShape and OldStride are flat
template <int I, class OldShape, class OldStride, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_256_impl(OldShape const& old_shape, OldStride const& old_stride,
                  NewShape const& new_shape, NewStride const& new_stride)
{
  if constexpr (I == rank_v<OldShape>) {
    // Base case, we're done
#if CUTE_DEBUG_TMA_GBASIS
    print("    [coalesce_256_impl] I="); print(Int<I>{});
    print(" base-case reached, new_shape="); print(new_shape);
    print(" new_stride="); print(new_stride); print("\n");
#endif
    if constexpr (is_constant<1, NewShape>::value) {
      return Layout<_1,_0>{};
    } else {
      return Layout<NewShape,NewStride>{new_shape,new_stride};
    }
  } else if constexpr (is_constant<1, decltype(get<I>(old_shape))>::value) {
    // shape<I>(layout) == _1, skip it and continue
#if CUTE_DEBUG_TMA_GBASIS
    print("    [coalesce_256_impl] I="); print(Int<I>{});
    print(" skip old mode because shape==1, old_shape[I]="); print(get<I>(old_shape));
    print(" old_stride[I]="); print(get<I>(old_stride)); print("\n");
#endif
    return coalesce_256_impl<I+1>(old_shape, old_stride, new_shape, new_stride);
  } else if constexpr (is_constant<1, NewShape>::value) {
    // Replace our shape-1 with anything (Can only happen on input new_shape/new_stride)
#if CUTE_DEBUG_TMA_GBASIS
    print("    [coalesce_256_impl] I="); print(Int<I>{});
    print(" replace seed shape-1 with old mode, old_shape[I]="); print(get<I>(old_shape));
    print(" old_stride[I]="); print(get<I>(old_stride)); print("\n");
#endif
    return coalesce_256_impl<I+1>(old_shape, old_stride, get<I>(old_shape), get<I>(old_stride));
  } else if constexpr (is_constant<true, decltype(back(new_shape) * back(new_stride) == get<I>(old_stride) &&
                                                  get<I>(old_shape) * back(new_shape) <= Int<256>{})>::value) {
    // Merge modes because the shapes and strides match and the merge is 256 or less
#if CUTE_DEBUG_TMA_GBASIS
    print("    [coalesce_256_impl] I="); print(Int<I>{});
    print(" merge mode, back(shape/stride)=("); print(back(new_shape)); print(",");
    print(back(new_stride)); print(") old(shape/stride)=("); print(get<I>(old_shape));
    print(","); print(get<I>(old_stride)); print(")\n");
#endif
    return coalesce_256_impl<I+1>(old_shape, old_stride,
                                  replace_back(new_shape, get<I>(old_shape) * back(new_shape)),
                                  new_stride);
  } else {
    // Can't replace or merge, so append a new mode
#if CUTE_DEBUG_TMA_GBASIS
    print("    [coalesce_256_impl] I="); print(Int<I>{});
    print(" append new mode, old(shape/stride)=("); print(get<I>(old_shape));
    print(","); print(get<I>(old_stride)); print(")\n");
#endif
    return coalesce_256_impl<I+1>(old_shape, old_stride,
                                  append(new_shape,  get<I>(old_shape)),
                                  append(new_stride, get<I>(old_stride)));
  }

  CUTE_GCC_UNREACHABLE;
}

// Combine all the modes that are possible to combine
// Does not respect the profile of the layout, but does preserve total size
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_256(Layout<Shape,Stride> const& layout)
{
  auto flat_shape  = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());
#if CUTE_DEBUG_TMA_GBASIS
  print("  [coalesce_256] input layout : "); print(layout); print("\n");
  print("  [coalesce_256] flat_shape   : "); print(flat_shape); print("\n");
  print("  [coalesce_256] flat_stride  : "); print(flat_stride); print("\n");
#endif
  return coalesce_256_impl<1>(flat_shape, flat_stride, get<0>(flat_shape), get<0>(flat_stride));
}

// ============================================================================
// construct_tma_gbasis
// ============================================================================
// 功能：
//   从 (gtensor, slayout, cta_v_map) 推导出 tma_gbasis（TMA box -> GMEM mode 映射）。
//
// 输入：
//   - gtensor: 原始 GMEM Tensor
//   - slayout: CTA tile 对应的 SMEM layout（可含 swizzle）
//   - cta_v_map: CTA value index -> GMEM mode 的映射
//
// 输出：
//   - tma_gbasis: 供后续 make_tma_copy_desc 使用的 basis layout
//
// 主要阶段：
//   1) 反演/组合 SMEM 映射，得到 smem_idx -> gmem_mode 的向量化映射；
//   2) 截断不兼容模式，得到可用于 TMA 的 gmode 子空间；
//   3) 按 TmaInternalType 重解释并 coalesce；
//   4) 补齐遗漏 basis，并将 rank 限制到 TMA 支持的最大维度。
// ============================================================================
#ifndef CUTE_DEBUG_TMA_GBASIS
#define CUTE_DEBUG_TMA_GBASIS 0
#endif

template <class TmaInternalType,
          class GEngine, class GLayout,
          class SShape, class SStride,
          class VShape, class VStride>
CUTE_HOST_DEVICE constexpr
auto
construct_tma_gbasis(Tensor<GEngine,GLayout> const& gtensor,       // The original GMEM Tensor
                     Layout<SShape,SStride>  const& slayout,       // The layout of SMEM
                     Layout<VShape,VStride>  const& cta_v_map)     // smem_idx to hier gmode
{
  //
  // TMA parameter checking
  //

  // CUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
  //                      "TMA requires CTA_Tile and SLayout top-level shape equivalence.");
  CUTE_STATIC_ASSERT_V(size(slayout) == size(cta_v_map),
                       "TMA requires CTA_Tile and SLayout top-level size equivalence.");

#if CUTE_DEBUG_TMA_GBASIS
  print("\n[construct_tma_gbasis] begin\n");
  print("  gtensor  (src: input gtensor)  : "); print(gtensor); print("\n");
  print("  slayout  (src: input slayout)  : "); print(slayout); print("\n");
  print("  cta_v_map(src: input cta_v_map): "); print(cta_v_map); print("\n");
#endif

  //
  // TMA slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
#if CUTE_DEBUG_TMA_GBASIS
  print("  inv_smem_layout (src: right_inverse(get_nonswizzle_portion(slayout))) : ");
  print(inv_smem_layout); print("\n");
#endif

  // Compose with the V-Map to convert smem coord (CTA val idx) to gmem mode
  // smem idx -> gmem mode
  auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));
#if CUTE_DEBUG_TMA_GBASIS
  print("  sidx2gmode_full (src: coalesce(composition(cta_v_map, inv_smem_layout))) : ");
  print(sidx2gmode_full); print("\n");
#endif

  //
  // TMA gtensor truncation
  //

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  static_assert(smem_rank > 0, "Could not find a common tile-gmem vectorization. Does the Tile select out major GMEM modes?");
#if CUTE_DEBUG_TMA_GBASIS
  print("  smem_rank (src: find_if(stride(sidx2gmode_full), basis_value!=1)) : ");
  print(smem_rank); print("\n");
#endif

  // Keep only the static-1 basis modes into gmem
  auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);
#if CUTE_DEBUG_TMA_GBASIS
  print("  sidx2gmode (src: take<0,smem_rank>(sidx2gmode_full)) : ");
  print(sidx2gmode); print("\n");
#endif

  //
  // TMA gtensor manipulation
  //

  // The smem vector is the same units as gtensor, so compose first and then recast
  // tma_val_idx:gmem_strides
  auto tile_gstride = recast<TmaInternalType>(gtensor.compose(sidx2gmode)).layout();
#if CUTE_DEBUG_TMA_GBASIS
  print("  tile_gstride (src: recast<TmaInternalType>(gtensor.compose(sidx2gmode)).layout()) : ");
  print(tile_gstride); print("\n");
#endif
  // Coalesce modes up to size-256 (the maximum TMA box extent in units of TmaInternalType)
  // tma_box_shape:gmem_strides
  auto tma_gstride  = coalesce_256(tile_gstride);
#if CUTE_DEBUG_TMA_GBASIS
  print("  tma_gstride (src: coalesce_256(tile_gstride)) : ");
  print(tma_gstride); print("\n");
#endif

  // Perform the tiling, recast, and coalesce to the gmem vector again, but with indirections to the gtensor modes
  auto gbasis = make_identity_layout(shape(gtensor));
#if CUTE_DEBUG_TMA_GBASIS
  print("  gbasis (src: make_identity_layout(shape(gtensor))) : ");
  print(gbasis); print("\n");
#endif
  auto tile_gbasis_tmp = gbasis.compose(sidx2gmode);
#if CUTE_DEBUG_TMA_GBASIS
  print("  tile_gbasis_tmp (src: gbasis.compose(sidx2gmode)) : ");
  print(tile_gbasis_tmp); print("\n");
#endif

  // Instead of the recast (gbasis doesn't have type info), replace the shape with the already-recasted shape
  // tma_box_shape:gmem_mode
  auto tile_gbasis = make_layout(shape(tile_gstride), stride(tile_gbasis_tmp));
#if CUTE_DEBUG_TMA_GBASIS
  print("  tile_gbasis (src: make_layout(shape(tile_gstride), stride(tile_gbasis_tmp))) : ");
  print(tile_gbasis); print("\n");
#endif

  // "Coalesce" the tile basis into a compatible shape with the tma_gstride
  auto tma_gbasis_tile = tile_gbasis.compose(make_layout(wrap(shape(tma_gstride))));
#if CUTE_DEBUG_TMA_GBASIS
  print("  tma_gbasis_tile (src: tile_gbasis.compose(make_layout(wrap(shape(tma_gstride))))) : ");
  print(tma_gbasis_tile); print("\n");
#endif

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);
#if CUTE_DEBUG_TMA_GBASIS
  print("  gtensor_T (src: recast<TmaInternalType>(gtensor)) : ");
  print(gtensor_T); print("\n");
#endif

  // Find missing bases that don't appear in tile_gbasis
  auto tile_gbasis_remaining_stride = filter_tuple(flatten(shape (gtensor_T)), flatten(stride(gtensor_T)),
                                                   flatten(stride(gbasis)),
                                                   [&](auto s, auto d, auto e)
  {
    if constexpr (is_constant<1, decltype(s)>::value || is_constant<0, decltype(d)>::value) {
      return cute::tuple<>{};          // If size-1 or stride-0, then don't append
    } else {
      using E = decltype(e);
      auto has_e = any_of(flatten(stride(tma_gbasis_tile)), [] (auto tb) { return tb == E{}; });
      if constexpr (decltype(has_e)::value) {
        return cute::tuple<>{};        // If d was found, then don't append
      } else {
        return cute::tuple<E>(e);      // Else, this is missing so append
      }
    }
  });
#if CUTE_DEBUG_TMA_GBASIS
  print("  tile_gbasis_remaining_stride (src: filter_tuple(..., flatten(stride(tma_gbasis_tile)) membership check)) : ");
  print(tile_gbasis_remaining_stride); print("\n");
#endif

  // Append the remaining basis modes that contribute to the TMA with size-1
  auto tile_gbasis_remaining_shape = repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{});
#if CUTE_DEBUG_TMA_GBASIS
  print("  tile_gbasis_remaining_shape (src: repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{})) : ");
  print(tile_gbasis_remaining_shape); print("\n");
#endif
  auto tma_gbasis_full = make_layout(tuple_cat(wrap( shape(tma_gbasis_tile)), wrap(tile_gbasis_remaining_shape )),
                                     tuple_cat(wrap(stride(tma_gbasis_tile)), wrap(tile_gbasis_remaining_stride)));
#if CUTE_DEBUG_TMA_GBASIS
  print("  tma_gbasis_full (src: make_layout(tuple_cat(shape parts), tuple_cat(stride parts))) : ");
  print(tma_gbasis_full); print("\n");
#endif

  // Group the trailing modes to make this max rank-5 -- TMA rank limitation
  // tma_box_shape:gmem_mode
  auto tma_gbasis = group<cute::min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full);
#if CUTE_DEBUG_TMA_GBASIS
  print("  tma_gbasis (src: group<min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full)) : ");
  print(tma_gbasis); print("\n");
  print("[construct_tma_gbasis] end\n");
#endif

  return tma_gbasis;
}

// ============================================================================
// fill_tma_gmem_shape_stride
// ============================================================================
// 功能：根据 GMEM tensor 的 layout 和 TMA basis stride 映射，
//       填充 TMA descriptor 所需的 GMEM shape 和 stride 数组
//
// 参数：
//   gtensor           - GMEM Tensor，其 value_type 已经是 TmaInternalType
//   tma_gbasis_stride - TMA mode → GMEM mode(s) 的映射
//                       例如：(E<0>, E<1>) 表示 TMA dim0 ← GMEM dim0, TMA dim1 ← GMEM dim1
//                       也可以是 ((E<0>, E<2>), E<1>) 表示 TMA dim0 ← GMEM dim0 和 dim2 合并
//   gmem_prob_shape   - 输出：TMA 各维度的 size（最多 5 维）
//   gmem_prob_stride  - 输出：TMA 各维度的 stride（元素为单位，后续会转为字节）
//
// 工作原理：
//   遍历每个 TMA 维度 i，根据 tma_gbasis_stride[i] 确定它对应哪些 GMEM 维度，
//   然后提取或合并这些维度的 shape 和 stride。
//
// 示例：
//   gtensor: shape=(128, 64), stride=(64, 1)  (row-major 128x64 矩阵)
//   tma_gbasis_stride = (E<1>, E<0>)  // 转置：TMA 连续维度取 GMEM 的 dim1
//   输出：gmem_prob_shape = [64, 128], gmem_prob_stride = [1, 64]
// ============================================================================
template <class GEngine, class GLayout,
          class TmaGmemBasisStride,
          class ShapeT, size_t TmaRank>
CUTE_HOST_DEVICE constexpr
void
fill_tma_gmem_shape_stride(Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, in units of TmaInternalType
                           TmaGmemBasisStride        const& tma_gbasis_stride, // Map Tma mode idx -> Gmem mode(s)
                           cute::array<ShapeT,   TmaRank> & gmem_prob_shape,   // Tma Shapes, uint32_t or uin64_t
                           cute::array<uint64_t, TmaRank> & gmem_prob_stride)  // Tma Strides
{
  static_assert(is_tuple<TmaGmemBasisStride>::value);
  static_assert(is_same<uint32_t, ShapeT>::value || is_same<uint64_t, ShapeT>::value);

  using TmaInternalType = typename GEngine::value_type;
  constexpr int tma_rank = decltype(rank(tma_gbasis_stride))::value;
  static_assert(TmaRank >= tma_rank);

  auto gmem_shape  =  shape(gtensor);
  auto gmem_stride = stride(gtensor);

#if CUTE_DEBUG_TMA_GBASIS
  auto print_arr_local = [](char const* name, auto const& arr) {
    print("  "); print(name); print(" : [");
    for (int ii = 0; ii < int(TmaRank); ++ii) {
      print(arr[ii]);
      if (ii + 1 < int(TmaRank)) { print(", "); }
    }
    print("]\n");
  };
  print("\n[fill_tma_gmem_shape_stride] begin\n");
  print("  gtensor (src: input gtensor) : "); print(gtensor); print("\n");
  print("  tma_gbasis_stride (src: input mapping) : "); print(tma_gbasis_stride); print("\n");
  print_arr_local("gmem_prob_shape init", gmem_prob_shape);
  print_arr_local("gmem_prob_stride init", gmem_prob_stride);
#endif

  // 遍历每个 TMA 维度，使用 tma_gbasis_stride 中的间接索引来构建 TMA 的 shape/stride
  // Use the indirections in tma_gbasis_stride into gtensor to construct the tma gmem shapes/strides
  for_each(make_seq<tma_rank>{}, [&](auto i) {
    constexpr int tma_i_rank = decltype(rank<i>(tma_gbasis_stride))::value;

#if CUTE_DEBUG_TMA_GBASIS
    print("  [fill_tma_gmem_shape_stride] tma_dim i="); print(i);
    print(" rank<i>(mapping)="); print(Int<tma_i_rank>{}); print("\n");
#endif

    if constexpr (tma_i_rank == 1) {
      // ========== Case 1: 简单映射 ==========
      // 一个 TMA 维度对应一个 GMEM 维度
      // 例如：tma_gbasis_stride[i] = E<j>
      // Trivial contribution of this gmem mode to this tma mode
      auto ej = unwrap(get<i>(tma_gbasis_stride));
      gmem_prob_shape[i]  = basis_get(ej, gmem_shape);   // 直接取 GMEM 第 j 维的 size
      gmem_prob_stride[i] = basis_get(ej, gmem_stride);  // 直接取 GMEM 第 j 维的 stride
#if CUTE_DEBUG_TMA_GBASIS
      print("    case-1 ej : "); print(ej); print("\n");
      print("    case-1 shape/stride -> ("); print(gmem_prob_shape[i]); print(", ");
      print(gmem_prob_stride[i]); print(")\n");
#endif
    } else {
      // ========== Case 2: 复合映射（多个 GMEM 维度合并到一个 TMA 维度）==========
      // 例如：tma_gbasis_stride[i] = (E<j1>, E<j2>, ...)
      // 需要使用递推公式计算合并后的 shape 和 stride
      //
      // 递推公式原理：
      //   当多个维度合并时，计算它们共同覆盖的"逻辑范围"
      //   - 合并后的 stride = gcd(所有参与维度的 stride)
      //   - 合并后的 shape = 所有维度覆盖的总范围 / 合并后的 stride
      //
      // 例如：dim A: size=4, stride=1 → 覆盖 [0,1,2,3]
      //       dim B: size=3, stride=4 → 覆盖 [0,4,8]
      //       合并：stride=gcd(1,4)=1, shape = 3+(4-1)*(1/1)+(3-1)*(4/1)+1 = 12
      //             覆盖 [0..11]
      // Apply a recurrence to each gmem mode that contributes to this tma mode
#if CUTE_DEBUG_TMA_GBASIS
      print("    case-2 mapping tuple : "); print(get<i>(tma_gbasis_stride)); print("\n");
#endif
      for_each(get<i>(tma_gbasis_stride), [&](auto ej) {
        // Problem shape
        uint64_t shape_j  = basis_get(ej, gmem_shape);
        // Problem stride (in bytes)
        uint64_t stride_j = basis_get(ej, gmem_stride);
        uint64_t old_stride = gmem_prob_stride[i];
        auto old_shape = gmem_prob_shape[i];

        // 新 stride = gcd(原 stride, 当前维度的 stride)
        gmem_prob_stride[i] = gcd(gmem_prob_stride[i], stride_j);

        if (gmem_prob_stride[i] != 0) {
          // 递推公式：合并后的 shape
          // g_shape_new = (g_shape_old - 1) * (old_stride / new_stride)
          //             + (shape_j - 1) * (stride_j / new_stride)
          //             + 1
          // 这计算的是所有参与维度覆盖的最大偏移量 / 最小步长 + 1
          // Recurrence: g_shape = (s_i - 1) * (d_i / gcd_j d_j) + 1
          gmem_prob_shape[i] = (gmem_prob_shape[i]-1) * (old_stride / gmem_prob_stride[i])
                             +            (shape_j-1) * (stride_j   / gmem_prob_stride[i])
                             + 1;
        } else {
          // stride 为 0 的特殊情况（broadcast）
          gmem_prob_shape[i] = shape_j;
        }

#if CUTE_DEBUG_TMA_GBASIS
        print("      ej : "); print(ej);
        print(" shape_j="); print(shape_j);
        print(" stride_j="); print(stride_j);
        print(" old(shape,stride)=("); print(old_shape); print(","); print(old_stride); print(")");
        print(" new(shape,stride)=("); print(gmem_prob_shape[i]); print(","); print(gmem_prob_stride[i]); print(")\n");
#endif
      });
    }

#if CUTE_DEBUG_TMA_GBASIS
    print_arr_local("gmem_prob_shape running", gmem_prob_shape);
    print_arr_local("gmem_prob_stride running", gmem_prob_stride);
#endif
  });

#if CUTE_DEBUG_TMA_GBASIS
  print("[fill_tma_gmem_shape_stride] end\n");
#endif
}

// Overload for an existing Copy_Traits
template <class GEngine, class GLayout,
          class Op, class Bits, class Aux,
          class ShapeT, size_t TmaRank>
CUTE_HOST_DEVICE constexpr
void
fill_tma_gmem_shape_stride(Copy_Traits<Op,Bits,Aux>  const& tma_traits,
                           Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, value_type = TmaInternalType
                           cute::array<ShapeT,   TmaRank> & gmem_prob_shape,   // Tma Shapes, uint32_t or uin64_t
                           cute::array<uint64_t, TmaRank> & gmem_prob_stride)  // Tma Strides
{
  return fill_tma_gmem_shape_stride(gtensor, stride(typename Aux::TmaGmemBasis{}),
                                    gmem_prob_shape, gmem_prob_stride);
}

// ============================================================================
// make_tma_copy_desc
// ============================================================================
// 功能：创建 TMA descriptor，这是 TMA 操作的核心配置结构
//
// 主要工作：
//   1. 从 GMEM tensor 提取 shape/stride 信息，转换为 TMA 格式
//   2. 计算 SMEM box 尺寸（考虑 multicast）
//   3. 调用 CUDA Driver API cuTensorMapEncodeTiled 创建 descriptor
//   4. 构建 gmem_tma_basis_stride：GMEM 坐标 → TMA 坐标的映射
//
// 参数：
//   gtensor      - 原始 GMEM Tensor（完整矩阵）
//   tma_gbasis   - TMA box 的 shape 和 stride，描述 box 如何映射到 GMEM
//                  shape: box 各维度大小
//                  stride: 包含 E<i> basis，指示该 TMA 维度对应哪个 GMEM 维度
//   swizzle      - SMEM swizzle 模式 (Swizzle<B,M,S>)
//   num_multicast - multicast 的 CTA 数量（1 表示无 multicast）
//
// 返回值：
//   tuple<TmaDescriptor, AuxParams>
//   - TmaDescriptor: 传给 TMA PTX 指令的 128 字节描述符
//   - AuxParams: 包含 gmem_tma_basis_stride，用于运行时计算 TMA 坐标
//
// 数据流：
//   gtensor (GMEM layout) + tma_gbasis (box 映射)
//          ↓
//   gmem_prob_shape/stride (TMA 格式的 GMEM 信息)
//          ↓
//   smem_box_shape (考虑 multicast 缩小后的 box)
//          ↓
//   cuTensorMapEncodeTiled (CUDA Driver API)
//          ↓
//   TmaDescriptor + gmem_tma_basis_stride (坐标转换)
// ============================================================================
//
// Use a sidx2gmode to read through the GMEM tensor
//   and construct a TMA Descriptor for the resulting instruction
// At the same time, construct the Tma Tensor's Stride to generate
//   the TMA coordinates that the instruction consumes.
//
template <class TmaInternalType,
          class GEngine, class GLayout,
          class TShape, class TStride,
          int B, int M, int S>
CUTE_HOST_RTC
auto
make_tma_copy_desc(Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
                   Layout<TShape,TStride>  const& tma_gbasis,      // TMA mode -> GMEM mode mapping
                   Swizzle<B,M,S>          const& swizzle,         // Swizzle fn on smem_idx
                   uint32_t                       num_multicast)   // The number of CTAs in multicasting
{
#if CUTE_DEBUG_TMA_GBASIS
  // Lightweight array printer used by debug traces below.
  auto print_arr = [](char const* name, auto const& arr) {
    print("  "); print(name); print(" : [");
    for (int i = 0; i < 5; ++i) {
      print(arr[i]);
      if (i != 4) { print(", "); }
    }
    print("]\n");
  };
#endif

  // ========================================================================
  // 第一部分：提取 GMEM 信息
  // ========================================================================
  //
  // TMA desc creation
  //

  // TMA 维度数（最多 5 维）
  // rank(layout) 返回 Int<N> 类型
  // decltype 获取这个类型
  // ::value 提取编译期常量 N
  constexpr int tma_dim = decltype(rank(tma_gbasis))::value;
#if CUTE_DEBUG_TMA_GBASIS
  print("\n[make_tma_copy_desc] begin\n");
  print("  gtensor (src: input gtensor) : "); print(gtensor); print("\n");
  print("  tma_gbasis (src: input tma_gbasis) : "); print(tma_gbasis); print("\n");
  print("  swizzle (src: input swizzle) : "); print(swizzle); print("\n");
  print("  num_multicast (src: input num_multicast) : "); print(num_multicast); print("\n");
  print("  tma_dim (src: rank(tma_gbasis)) : "); print(tma_dim); print("\n");
#endif

  //
  // TMA gmem desc info
  //

  // 将 tensor 转换为 TmaInternalType 类型进行 shape/stride 检查
  // 例如：原始类型是 half2，TmaInternalType 是 half，会重新计算维度
  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);
#if CUTE_DEBUG_TMA_GBASIS
  print("  gtensor_T (src: recast<TmaInternalType>(gtensor)) : "); print(gtensor_T); print("\n");
#endif

  // 提取 GMEM 基地址和 layout
  void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();
#if CUTE_DEBUG_TMA_GBASIS
  print("  gmem_address (src: raw_pointer_cast(gtensor_T.data())) : ");
  print("%p\n", gmem_address);
  print("  gmem_layout (src: gtensor_T.layout()) : "); print(gmem_layout); print("\n");
#endif

  // TMA descriptor 需要的 GMEM shape 和 stride（最多 5 维）
  // 初始化为 {1,1,1,1,1} 和 {0,0,0,0,0}
  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

  // 根据 tma_gbasis 的 stride（包含 E<i> basis）填充 shape 和 stride
  fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);
#if CUTE_DEBUG_TMA_GBASIS
  print_arr("gmem_prob_shape (src: fill_tma_gmem_shape_stride(...))", gmem_prob_shape);
  print_arr("gmem_prob_stride[elem] (src: fill_tma_gmem_shape_stride(...))", gmem_prob_stride);
#endif

  // ========== GMEM 地址约束检查 ==========
  // TMA 要求地址 16 字节对齐
  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);  // Address must be 16B-aligned

  // ========== GMEM shape 约束检查 ==========
  // 每个维度：1 ≤ size ≤ 2^32
  assert(gmem_prob_shape[0] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));         // Size must be max 2^32

  // TMA 要求第 0 维（最内层维度）的 stride 必须是 1（连续存储）
  // 这确保了 SMEM 的主维度和 GMEM 的主维度匹配
  // TMA descriptor does not store the zeroth stride and assumes it is 1 (TmaInternalType element).
  assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");

  // 将元素 stride 转换为字节 stride（TMA descriptor 使用字节 stride）
  // convert strides to byte strides
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
  }
#if CUTE_DEBUG_TMA_GBASIS
  print_arr("gmem_prob_stride[byte] (src: stride *= sizeof_bits_v<TmaInternalType>/8)", gmem_prob_stride);
#endif

  // ========== GMEM stride 约束检查（字节单位）==========
  // 每个维度：stride < 2^40，且必须是 16 字节的倍数
  // Assert the byte strides. Tma Descriptor uses byte strides
  assert((gmem_prob_stride[1]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)

  // ========================================================================
  // 第二部分：计算 SMEM Box 信息
  // ========================================================================
  //
  // TMA smem desc info
  //

  // SMEM box 的 shape 和 stride
  cute::array<uint32_t, 5> smem_box_shape  = {1,1,1,1,1};
  cute::array<uint32_t, 5> smem_box_stride = {1,1,1,1,1};

  // Box shape = tma_gbasis 各维度的 size
  // 例如：tma_gbasis = Layout<Shape<_64, _32>, ...> → box = [64, 32, 1, 1, 1]
  // The smem box is simply given by the sizes of the modes in tma_gbasis
  for_each(make_seq<tma_dim>{}, [&](auto i) {
    smem_box_shape[i] *= size<i>(tma_gbasis);
  });
#if CUTE_DEBUG_TMA_GBASIS
  print_arr("smem_box_shape pre-mcast (src: for_each size<i>(tma_gbasis))", smem_box_shape);
#endif

  // 如果有 multicast，需要缩小 box 尺寸
  // multicast 时多个 CTA 共同读取数据，每个 CTA 只负责一部分
  // 从最后一个维度开始缩小，直到 multicast 因子消耗完
  //
  // 例如：box = [64, 32], multicast = 4
  //   第一轮：smem_box_shape[1] = 32/4 = 8, multicast = 1
  //   结果：box = [64, 8]（每个 CTA 读取 1/4）
  // Finally, truncate the tma box by the num_multicast
  for (uint32_t i = tma_dim-1, multicast = num_multicast; multicast > 1; --i) {
    assert(smem_box_shape[i] % multicast == 0 || multicast % smem_box_shape[i] == 0);
    uint32_t new_mult = ceil_div(multicast, smem_box_shape[i]);
    smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
    multicast = new_mult;
  }
#if CUTE_DEBUG_TMA_GBASIS
  print_arr("smem_box_shape post-mcast (src: truncation loop over num_multicast)", smem_box_shape);
  print_arr("smem_box_stride (src: initialized constant {1,1,1,1,1})", smem_box_stride);
#endif

  // ========== SMEM box shape 约束检查 ==========
  // 每个维度：1 ≤ size ≤ 256 (2^8)
  assert(smem_box_shape[0] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[0] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[1] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[2] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[3] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[4] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256

  // ========== SMEM box stride 约束检查 ==========
  // 每个维度：1 ≤ stride ≤ 8 (2^3)
  // 注意：这里的 stride 是元素 stride，不是字节 stride
  assert(smem_box_stride[0] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));               // Stride must be max 2^3 = 8

  // ========================================================================
  // 第三部分：调用 CUDA Driver API 创建 TMA Descriptor
  // ========================================================================
    //
    // Construct the descriptor
    //

    TmaDescriptor tma_desc{};

    //
    // TMA general info
    //

  #if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)

    // 数据类型映射：CuTe 类型 → CUDA 枚举
    // 例如：half → CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    CUtensorMapDataType     tma_format      = TMA::to_CUtensorMapDataType<TmaInternalType>();

    // 交织模式：通常为 NONE
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;

    // L2 缓存提升策略：128 字节
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;

    // 越界填充：NONE 表示不填充（越界会出错）
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // SMEM swizzle 配置
    // Swizzle<B,M,S> 映射到 CUtensorMapSwizzle 枚举：
    //   B=0 → NONE, B=1 → 32B, B=2 → 64B, B=3 → 128B
    // TMA smem swizzle type
    TMA::SmemSwizzleBits swizzle_bits = get_tma_swizzle_bits(swizzle);
    TMA::SmemSwizzleBase swizzle_base = get_tma_swizzle_base(swizzle);
    CUtensorMapSwizzle smem_swizzle = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);
#if CUTE_DEBUG_TMA_GBASIS
    print("  tma_format (src: to_CUtensorMapDataType<TmaInternalType>()) : "); print(int(tma_format)); print("\n");
    print("  tma_interleave (src: CU_TENSOR_MAP_INTERLEAVE_NONE) : "); print(int(tma_interleave)); print("\n");
    print("  tma_l2Promotion (src: CU_TENSOR_MAP_L2_PROMOTION_L2_128B) : "); print(int(tma_l2Promotion)); print("\n");
    print("  tma_oobFill (src: CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) : "); print(int(tma_oobFill)); print("\n");
    print("  swizzle_bits (src: get_tma_swizzle_bits(swizzle)) : "); print(int(swizzle_bits)); print("\n");
    print("  swizzle_base (src: get_tma_swizzle_base(swizzle)) : "); print(int(swizzle_base)); print("\n");
    print("  smem_swizzle (src: to_CUtensorMapSwizzle(swizzle_bits, swizzle_base)) : "); print(int(smem_swizzle)); print("\n");
#endif

    // 调用 CUDA Driver API 创建 TMA descriptor
    // 这个 API 会验证所有参数并填充 128 字节的 descriptor 结构
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tma_desc,
        tma_format,                     // 数据类型
        tma_dim,                        // 维度数（1-5）
        gmem_address,                   // GMEM 基地址（16B 对齐）
        gmem_prob_shape.data(),         // GMEM 各维度 size
        gmem_prob_stride.data() + 1,    // GMEM 各维度 byte stride（跳过 dim0，它隐含为 1 元素）
        smem_box_shape.data(),          // Box 各维度 size
        smem_box_stride.data(),         // Box 各维度 element stride
        tma_interleave,                 // 交织模式
        smem_swizzle,                   // Swizzle 模式
        tma_l2Promotion,                // L2 提升策略
        tma_oobFill);                   // 越界填充
#if CUTE_DEBUG_TMA_GBASIS
    print("  cuTensorMapEncodeTiled result (src: CUTLASS_CUDA_DRIVER_WRAPPER_CALL(...)) : ");
    print(int(result)); print("\n");
#endif

    // 错误处理：打印所有参数帮助调试
    if (result != CUDA_SUCCESS) {
      std::cerr << "TMA Desc Addr:   " << &tma_desc
                << "\nformat         " << tma_format
                << "\ndim            " << tma_dim
                << "\ngmem_address   " << gmem_address
                << "\nglobalDim      " << gmem_prob_shape
                << "\nglobalStrides  " << gmem_prob_stride
                << "\nboxDim         " << smem_box_shape
                << "\nelementStrides " << smem_box_stride
                << "\ninterleave     " << tma_interleave
                << "\nswizzle        " << smem_swizzle
                << "\nl2Promotion    " << tma_l2Promotion
                << "\noobFill        " << tma_oobFill << std::endl;
      std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;
      assert(false);
    }

  #endif // (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)

  // ========================================================================
  // 第四部分：构建 gmem_tma_basis_stride（GMEM 坐标 → TMA 坐标的映射）
  // ========================================================================
  //
  // 这是最复杂的部分。目的是构建一个映射，使得：
  //   给定 GMEM tensor 中的坐标 (i, j, k, ...)
  //   可以计算出对应的 TMA 坐标 (tma_coord_0, tma_coord_1, ...)
  //
  // 这个映射在运行时用于：
  //   tma_coord = gtensor_coord * gmem_tma_basis_stride
  //
  // 例如：
  //   gtensor: shape=(M, K), stride=(K, 1)  -- row-major
  //   tma_gbasis: stride=(E<1>, E<0>)       -- TMA dim0 ← K, TMA dim1 ← M
  //   gmem_tma_basis_stride = (E<1>, E<0>)  -- GMEM coord → TMA coord 的逆映射

  // recast_ratio: 原始类型和 TMA 内部类型的大小比
  // 例如：half2 / half = 2
  auto recast_ratio = cute::trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                        sizeof_bits<             TmaInternalType>{});
#if CUTE_DEBUG_TMA_GBASIS
  print("  recast_ratio (src: trait_ratio(sizeof_bits<GEngine::value_type>, sizeof_bits<TmaInternalType>)) : ");
  print(recast_ratio); print("\n");
#endif

  // 为 gtensor 的每个维度创建 basis (E<0>, E<1>, ...)
  auto gbasis = make_basis_like(shape(gtensor));
#if CUTE_DEBUG_TMA_GBASIS
  print("  gbasis (src: make_basis_like(shape(gtensor))) : "); print(gbasis); print("\n");
#endif

  // transform_leaf 遍历 gbasis 的每个叶子节点（每个 E<i>）
  // 对于每个 E<i>，计算它对应的 TMA 坐标贡献
  //
  // 本质上是构建 tma_gbasis 的"逆映射"：
  //   tma_gbasis.stride = (E<j0>, E<j1>, ...)  -- TMA dim → GMEM dim
  //   gmem_tma_basis_stride = (E<k0>, E<k1>, ...)  -- GMEM dim → TMA dim
  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem stride
  auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) {
    // ei 是当前处理的 GMEM 维度的 basis（如 E<0>、E<1>）

    // 获取该 GMEM 维度的 size 和 stride
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));

#if CUTE_DEBUG_TMA_GBASIS
    print("    [transform_leaf] ei="); print(ei);
    print(" si="); print(si);
    print(" di="); print(di); print("\n");
#endif

    // Case 1: 如果这个维度的 size=1 或 stride=0，它对 TMA 没有贡献
    // 返回 Int<0>{} 作为算术恒等元（加法单位元）
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
#if CUTE_DEBUG_TMA_GBASIS
      print("      -> case-1 size==1 or stride==0, return Int<0>\n");
#endif
      return Int<0>{};                  // If size-1 or stride-0, return arithmetic identity -- no contribution to the TMA
    } else {
      auto tma_gmem_basis_stride = stride(tma_gbasis);

      // 在 tma_gbasis.stride 中查找包含 E<i> 的 TMA 维度 j
      // 例如：tma_gbasis.stride = (E<1>, E<0>)
      //       如果 ei = E<0>，找到 j = 1（因为 E<0> 在 stride[1] 中）
      // Find j such that E<i> is in stride<j>(tma_gbasis)
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tma_gmem_basis_stride, [&](auto tma_stride_j) { return any_of(tma_stride_j, [&](auto dj) { return dj == EI{}; }); });

#if CUTE_DEBUG_TMA_GBASIS
      print("      candidate j (src: find_if over stride(tma_gbasis)) : "); print(j); print("\n");
#endif

      // Case 2: 如果找不到（这个 GMEM 维度不在任何 TMA 维度中）
      // 例如：batch 维度可能不参与 TMA
      if constexpr (decltype(j == rank(tma_gmem_basis_stride))::value) {
#if CUTE_DEBUG_TMA_GBASIS
        print("      -> case-2 not found in tma_gbasis, return Int<0>\n");
#endif
        return Int<0>{};               // If not-found, return arithmetic identity -- no contribution to the TMA
      } else
      // Case 3: 如果在 TMA dim0 中找到（连续维度）
      // 需要考虑 recast 比例（如 half2 → half 时，坐标需要乘 2）
      if constexpr (decltype(j == Int<0>{})::value) {
        auto scale = recast_ratio * basis_get(ei, stride(gtensor));
#if CUTE_DEBUG_TMA_GBASIS
        print("      -> case-3 j==0, scale="); print(scale);
        print(" return="); print(E<j>{} * scale); print("\n");
#endif
        return E<j>{} * scale;         // Return TMA Coord basis -- with a recast scale factor
      } else
      // Case 4: 如果 TMA 维度 j 只对应一个 GMEM 维度
      // scale 已知为 1
      if constexpr (decltype(rank<j>(tma_gmem_basis_stride) == Int<1>{})::value) {
#if CUTE_DEBUG_TMA_GBASIS
        print("      -> case-4 single-mode rank<j>==1, return="); print(E<j>{}); print("\n");
#endif
        return E<j>{};                 // Return TMA Coord basis -- known scale of Int<1>{}
      } else {
        // Case 5: TMA 维度 j 对应多个 GMEM 维度（合并情况）
        // 需要动态计算 scale
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<TmaInternalType> / cute::max(gmem_prob_stride[j], uint64_t{16})), 8);
#if CUTE_DEBUG_TMA_GBASIS
        print("      -> case-5 merged mode, dynamic scale="); print(scale);
        print(" return="); print(E<j>{} * scale); print("\n");
#endif
        return E<j>{} * scale;         // Return TMA Coord basis -- with a dynamic scale factor
      }
    }
  });

#if CUTE_DEBUG_TMA_GBASIS
  {
    auto const* tma_desc_words = reinterpret_cast<uint32_t const*>(&tma_desc);
    print("  tma_desc_words[0..7] (src: reinterpret_cast<uint32_t const*>(&tma_desc)) : [");
    for (int wi = 0; wi < 8; ++wi) {
      print(tma_desc_words[wi]);
      if (wi != 7) { print(", "); }
    }
    print("]\n");
  }
  print("  gmem_tma_basis_stride (src: transform_leaf(gbasis, ...)) : "); print(gmem_tma_basis_stride); print("\n");
  print("[make_tma_copy_desc] end\n");
#endif

  // ========================================================================
  // 第五部分：打包返回值
  // ========================================================================
  // 返回 tuple<TmaDescriptor, AuxParams>
  // - TmaDescriptor: 128 字节的 TMA 配置，传给 cp.async.bulk.tensor 指令
  // - AuxParams: 包含 gmem_tma_basis_stride，用于运行时计算 TMA 坐标偏移
  using AuxParams = AuxTmaParams<decltype(gmem_tma_basis_stride),
                                 decltype(tma_gbasis),
                                 decltype(swizzle)>;
  return cute::make_tuple(tma_desc, AuxParams{gmem_tma_basis_stride});
}

template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_tma_copy_atom(CopyOp,
                   Tensor<GEngine,GLayout> const& gtensor,       // Full GMEM Tensor
                   SLayout                 const& slayout,       // CTA Tile of SMEM, potentially swizzled
                   uint32_t                const& num_multicast, // The number of CTAs involved in multicasting
                   Layout<VShape,VStride>  const& cta_v_map)     // V: CTA val idx -> gmem mode
{
  //
  // TMA truncated layout
  //

  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

#if CUTE_DEBUG_TMA_GBASIS
  print("\n[make_tma_copy_atom] begin\n");
  print("  gtensor (src: input gtensor) : "); print(gtensor); print("\n");
  print("  slayout (src: input slayout) : "); print(slayout); print("\n");
  print("  cta_v_map (src: input cta_v_map) : "); print(cta_v_map); print("\n");
  print("  num_multicast (src: input num_multicast) : "); print(num_multicast); print("\n");
  print("  smem_swizzle (src: get_swizzle_portion(slayout)) : "); print(smem_swizzle); print("\n");
  print("  smem_layout (src: get_nonswizzle_portion(slayout)) : "); print(smem_layout); print("\n");
  print("  call chain: make_tma_copy_atom -> construct_tma_gbasis -> make_tma_copy_desc\n");
#endif

  auto tma_gbasis = detail::construct_tma_gbasis<TmaInternalType>(gtensor, smem_layout, cta_v_map);
#if CUTE_DEBUG_TMA_GBASIS
  print("  tma_gbasis (src: construct_tma_gbasis(...)) : "); print(tma_gbasis); print("\n");
#endif

  //
  // Construct the TMA Desc and the strides of the TMA Tensor
  //

  auto [tma_desc, aux_params] = detail::make_tma_copy_desc<TmaInternalType>(gtensor,
                                                                            tma_gbasis,
                                                                            smem_swizzle,
                                                                            num_multicast);
#if CUTE_DEBUG_TMA_GBASIS
  print("  tma_desc created (src: make_tma_copy_desc(...).first)\n");
#endif

  //
  // Construct the Copy_Traits
  //

  constexpr int num_bits_per_tma = size(tma_gbasis) * sizeof_bits_v<TmaInternalType>;
  using Traits = Copy_Traits<CopyOp, cute::C<num_bits_per_tma>, decltype(aux_params)>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;
#if CUTE_DEBUG_TMA_GBASIS
  print("  num_bits_per_tma (src: size(tma_gbasis) * sizeof_bits_v<TmaInternalType>) : ");
  print(num_bits_per_tma); print("\n");
#endif

  Traits tma_traits{tma_desc, aux_params};

#if 0
  print("num_bits_per_tma :  "); print(num_bits_per_tma); print("\n");
  print("g_stride_bases   :  "); print(tma_traits.aux_params_.g_stride_); print("\n");
#endif

  // Return the Copy_Atom
#if CUTE_DEBUG_TMA_GBASIS
  print("[make_tma_copy_atom] end\n");
#endif
  return Atom{tma_traits};
}

// The "logical TMA tid" is a map from the CTA rank to its logical id
// within the instruction.  It works like a mask or ordering on the
// CTAs.  For non-multicast TMA, all CTAs should map to 0.  For
// multicast TMA of size 4, CTAs will be mapped to {0,1,2,3}.
template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_tma_copy_tiled(CopyOp                  const& copy_op,
                    Tensor<GEngine,GLayout> const& gtensor,     // Full GMEM Tensor
                    SLayout                 const& slayout,     // CTA Tile of SMEM
                    Layout<TShape,TStride>  const& cta_t_map,   // T: CTA thr idx -> logical TMA tid
                    Layout<VShape,VStride>  const& cta_v_map)   // V: CTA val idx -> gmem mode
{
  Copy_Atom atom = make_tma_copy_atom<TmaInternalType>(copy_op, gtensor, slayout,
                                                       cosize(cta_t_map), cta_v_map);

  //
  // Construct the TiledCopy
  //

  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GEngine::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  // CTA T -> smem idx
  auto layout_t = make_layout(cosize(cta_t_map), safe_div(num_elems_per_tma, cosize(cta_t_map)));
  // CTA TID -> smem coord
  auto layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map));
  // Combine with the T mapping
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);

#if CUTE_DEBUG_TMA_GBASIS
  print("\n[make_tma_copy_tiled] summary\n");
  print("  cta_tiler (src: product_each(shape(cta_v_map))) : "); print(cta_tiler); print("\n");
  print("  num_elems_per_tma (src: RefLayout bits / sizeof_bits<value_type>) : "); print(num_elems_per_tma); print("\n");
  print("  inv_smem_layout (src: right_inverse(get_nonswizzle_portion(slayout))) : "); print(inv_smem_layout); print("\n");
  print("  layout_v (src: composition(inv_smem_layout, num_elems_per_tma)) : "); print(layout_v); print("\n");
  print("  layout_V (src: tile_to_shape(make_layout(layout_v), size(cta_v_map))) : "); print(layout_V); print("\n");
  print("  layout_t (src: make_layout(cosize(cta_t_map), safe_div(...))) : "); print(layout_t); print("\n");
  print("  layout_T (src: composition(inv_smem_layout, composition(layout_t, cta_t_map))) : "); print(layout_T); print("\n");
  print("  layout_TV (src: make_layout(layout_T, layout_V)) : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

} // end namespace detail

/** Make a CuTe CTA-collective TiledCopy for a TMA operation.
 *
 * @param CopyOp The target copy operation: SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST, SM90_TMA_STORE
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param cta_tile The CTA-local tile that each CTA will be tiling GMEM with.
 *                 This is often the blk_shape that is used to tile the GMEM for CTAs:
 *                   local_tile(gtensor, blk_shape, blk_coord) -> CTA-local tile of gtensor
 * @param cluster_size When using SM90_TMA_LOAD_MULTICAST, this can be a (static) power-of-2 <= 16
 *                   defining the multicast size (used to further partition the SMEM)
 *                 Else, static-1
 *
 * This code attempts to maximize the TMA box size. It does this by tracing
 * the SMEM "vector" -- the inverse of the smem layout -- to find the largest
 * contiguous array of smem that can be written to/from global memory given
 * the constraints that the TMA instruction imposes.
 *
 * This is accomplished by assigning "basis" strides to the GMEM to track which
 * modes of SMEM map to which modes of GMEM, then reorder the modes of GMEM according
 * to the SMEM vector, and then using those GMEM/SMEM modes to fill in the desc.
 *
 * Examples:
     using T = float;
     T* gptr = nullptr;

    {
    // Simple 2D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 256), GenRowMajor{}); // K-Major GMEM
    auto slayout   = make_layout(make_shape(_64{}, _32{}), GenRowMajor{});    // K-Major SMEM
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);
    }

    {
    // GMMA 2D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 256));                                 // MN-Major GMEM
    auto slayout   = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(_128{},_64{})); // MN-Major Swizzled+Tiled 128x64 SMEM
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);
    }

    {
    // 3D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 32, 512), make_stride(64, Int<1>{}, 65536)); // GMEM
    auto slayout   = make_layout(make_shape(_16{}, _8{}, _2{}), make_stride(_16{}, _1{}, _8{}));     // SMEM w/ same major-mode
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);
    }

    {
    // cuTENSOR 4D
    auto layout = make_shape(make_shape(32,40),make_shape(make_shape(8,8),656)); // GMEM
    auto cta_tile    = make_shape(_128{},make_shape(_32{},_2{}));                // GMEM Tiling:
                                                                                 //   Take 128-elem from m: m0 must divide 128,
                                                                                 //                         m-last may be predicated
                                                                                 //   Take 32-elem from k0, 2-elem from k1
    auto slayout = make_layout(cta_tile);                                        // Col-Major SMEM
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout, cta_tile, Int<1>{});
    }
 *
 * Check the TMA box size and desc:
    print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");
    print("TMA desc     : "); print(tma.tma_desc_); print("\n");
 *
 * Usage:
     Tensor mA = tma_a.get_tma_tensor(make_shape(M,N));        // (M,N) TMA coord tensor
     Tensor gA = local_tile(mA, cta_tile, cta_coord);          // (BLK_M,BLK_N) TMA coord tensor for this CTA
     Tensor sA = make_tensor(make_smem_ptr<T>(sptr), slayout); // (BLK_M,BLK_N) SMEM tensor

     auto cta_tma = tma.get_slice(cta_idx_in_cluster);         // Slice for multicast partitioning
     Tensor tAgA = cta_tma.partition_S(gA);                    // Partition for src
     Tensor tAsA = cta_tma.partition_D(sA);                    // Partition for dst

     copy(tma.with(barrier, mcast_mask), tAgA, tAsA);          // copy with supporting TMA params
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size)
{
  if constexpr (cute::is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL> ||
                cute::is_same_v<CopyOp, SM90_TMA_STORE_IM2COL>) {
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler,
                                cluster_size);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
    auto cta_t_tile = make_layout(cluster_size);
    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
#if CUTE_DEBUG_TMA_GBASIS
    print("\n[make_tma_copy] begin\n");
    print("  gtensor (src: input gtensor) : "); print(gtensor); print("\n");
    print("  slayout (src: input slayout) : "); print(slayout); print("\n");
    print("  cta_tiler (src: input cta_tiler) : "); print(cta_tiler); print("\n");
    print("  cluster_size (src: input cluster_size) : "); print(cluster_size); print("\n");
    print("  cta_v_tile (src: make_identity_layout(shape(gtensor)).compose(cta_tiler)) : "); print(cta_v_tile); print("\n");
    print("  cta_t_tile (src: make_layout(cluster_size)) : "); print(cta_t_tile); print("\n");
    print("  TmaType.bits (src: sizeof_bits<TmaType>::value) : "); print(sizeof_bits<TmaType>::value); print("\n");
#endif
    return detail::make_tma_copy_tiled<TmaType>(copy_op,
                                                gtensor, slayout,
                                                cta_t_tile, cta_v_tile);
  }
}

// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout)
{
  return make_tma_copy(copy_op, gtensor, slayout, product_each(shape(slayout)), Int<1>{});
}

// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              Cluster_Size            const& cluster_size)
{
  return make_tma_copy(copy_op, gtensor, slayout, product_each(shape(slayout)), cluster_size);
}

////////////////////////////////////
// Experimental Make TMA Atom and Partitioner
///////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size = Int<1>>
CUTE_HOST_RTC
auto
make_tma_atom(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size = {})
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
#if CUTE_DEBUG_TMA_GBASIS
  print("\n[make_tma_atom] begin\n");
  print("  gtensor (src: input gtensor) : "); print(gtensor); print("\n");
  print("  slayout (src: input slayout) : "); print(slayout); print("\n");
  print("  cta_tiler (src: input cta_tiler) : "); print(cta_tiler); print("\n");
  print("  cluster_size (src: input cluster_size) : "); print(cluster_size); print("\n");
  print("  cta_v_tile (src: make_identity_layout(shape(gtensor)).compose(cta_tiler)) : "); print(cta_v_tile); print("\n");
  print("  TmaType.bits (src: sizeof_bits<TmaType>::value) : "); print(sizeof_bits<TmaType>::value); print("\n");
  print("  TmaType.from_default (src: is_same<void, TmaInternalType>) : "); print(int(is_same<void, TmaInternalType>::value)); print("\n");
  print("  call chain: make_tma_atom -> make_tma_copy_atom -> construct_tma_gbasis -> make_tma_copy_desc\n");
#endif

  auto atom = detail::make_tma_copy_atom<TmaType>(copy_op,
                                                  gtensor, slayout,
                                                  size(cluster_size), cta_v_tile);
#if CUTE_DEBUG_TMA_GBASIS
  print("[make_tma_atom] end\n");
#endif
  return atom;
}

// The "VectorCopy Partitioner" for TMA
template <class... Args,
          class CtaCoord,
          class TShape, class TStride,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              CtaCoord                const& cta_coord,
              Layout<TShape,TStride>  const& cta_layout,  // T: CTA coord -> logical multicast id
              Tensor<SEngine,SLayout> const& stensor,     // SMEM Tensor (TMATile, Rest...)
              Tensor<GEngine,GLayout> const& gtensor)     // GMEM Tensor (TMATile, Rest...)
{
  CUTE_STATIC_ASSERT_V(size<0>(stensor) == size<0>(gtensor));

#if CUTE_DEBUG_TMA_GBASIS && defined(__CUDA_ARCH__)
  bool do_print = (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                   threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (do_print) {
    print("\n[tma_partition] begin\n");
    print("  copy_atom (src: input copy_atom) : "); print(copy_atom); print("\n");
    print("  cta_coord (src: input cta_coord) : "); print(cta_coord); print("\n");
    print("  cta_layout (src: input cta_layout) : "); print(cta_layout); print("\n");
    print("  stensor (src: input stensor) : "); print(stensor); print("\n");
    print("  gtensor (src: input gtensor) : "); print(gtensor); print("\n");
  }
#endif

  // Invert the smem to get the largest contiguous vector in the smem layout
  Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
  // Scale that up to cover all of the smem_coords
  Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

#if CUTE_DEBUG_TMA_GBASIS && defined(__CUDA_ARCH__)
  if (do_print) {
    print("  inv_smem_layout (src: right_inverse(get_nonswizzle_portion(layout<0>(stensor)))) : ");
    print(inv_smem_layout); print("\n");
    print("  layout_v (src: tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor))) : ");
    print(layout_v); print("\n");
  }
#endif

  // Factor out the single-instrucion portion
  Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
  auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));

#if CUTE_DEBUG_TMA_GBASIS && defined(__CUDA_ARCH__)
  if (do_print) {
    print("  tma_layout_v (src: make_layout(Int<Copy_Atom<Args...>::NumValSrc>{})) : ");
    print(tma_layout_v); print("\n");
    print("  layout_V (src: make_tile(logical_divide(layout_v, tma_layout_v))) : ");
    print(layout_V); print("\n");
  }
#endif

  // Append with _ until we cover all Rest... modes
  auto glayout_V = append<GLayout::rank>(layout_V, _);
  auto slayout_V = append<SLayout::rank>(layout_V, _);
  // Transform tile mode and coalesce
  Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)

#if CUTE_DEBUG_TMA_GBASIS && defined(__CUDA_ARCH__)
  if (do_print) {
    print("  glayout_V (src: append<GLayout::rank>(layout_V, _)) : "); print(glayout_V); print("\n");
    print("  slayout_V (src: append<SLayout::rank>(layout_V, _)) : "); print(slayout_V); print("\n");
    print("  gtensor_v (src: coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{})) : ");
    print(gtensor_v); print("\n");
    print("  stensor_v (src: coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{})) : ");
    print(stensor_v); print("\n");
  }
#endif

  // Offset inside the TMA-mode for the multicast
  auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
  auto multicast_coord  = make_coord(make_coord(multicast_offset, Int<0>{}));
  auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
  auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});

#if CUTE_DEBUG_TMA_GBASIS && defined(__CUDA_ARCH__)
  if (do_print) {
    print("  multicast_offset (src: cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout))) : ");
    print(multicast_offset); print("\n");
    print("  multicast_coord (src: make_coord(make_coord(multicast_offset, Int<0>{}))) : ");
    print(multicast_coord); print("\n");
    print("  gcoord (src: append<GLayout::rank>(multicast_coord, Int<0>{})) : "); print(gcoord); print("\n");
    print("  scoord (src: append<SLayout::rank>(multicast_coord, Int<0>{})) : "); print(scoord); print("\n");
  }
#endif

  Tensor gresult = domain_offset(gcoord, gtensor_v);
  Tensor sresult = domain_offset(scoord, stensor_v);

#if CUTE_DEBUG_TMA_GBASIS && defined(__CUDA_ARCH__)
  if (do_print) {
    print("  gresult (src: domain_offset(gcoord, gtensor_v)) : "); print(gresult); print("\n");
    print("  sresult (src: domain_offset(scoord, stensor_v)) : "); print(sresult); print("\n");
    print("[tma_partition] end\n");
  }
#endif

  return cute::make_tuple(gresult, sresult);
}

// Explicit defaults for cta_coord and cta_layout
template <class... Args,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              Tensor<SEngine,SLayout> const& stensor,     // SMEM Tensor (TMATile, Rest...)
              Tensor<GEngine,GLayout> const& gtensor)     // GMEM Tensor (TMATile, Rest...)
{
  return tma_partition(copy_atom, Int<0>{}, Layout<_1,_0>{}, stensor, gtensor);
}

// TMA Multicast Masks Calculation
template <class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t
create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                          CtaCoord  const& cta_coord_vmnk)
{
  auto [cta_layout, elected_cta] = slice_and_offset(cta_coord_vmnk, cta_layout_vmnk);

  uint16_t mcast_mask = 0;
  if constexpr (rank_v<decltype(cta_layout)> == 0) {
    // Trivial case with no additional ctas
    mcast_mask = uint16_t(1);
  } else
  if constexpr (rank_v<decltype(cta_layout)> == 1 and depth_v<decltype(cta_layout)> <= 1 and
                not is_static<decltype(cta_layout)>::value) {
    // Get the instruction code -- optimized for dynamic flat-rank-1 cta_layout
    mcast_mask = uint16_t(1);
    // Smear by stride<0> (may want to predicate on stride<0> mag?)
    mcast_mask |= mcast_mask << (1*stride<0>(cta_layout));
    mcast_mask |= mcast_mask << (2*stride<0>(cta_layout));
    mcast_mask |= mcast_mask << (4*stride<0>(cta_layout));
    mcast_mask |= mcast_mask << (8*stride<0>(cta_layout));
    // Select shape<0>
    mcast_mask &= (uint16_t(-1) >> (16 - shape<0>(cta_layout) * stride<0>(cta_layout)));
  } else {
    // Get the instruction code -- generic path
    for (int i = 0; i < size(cta_layout); ++i) {
      mcast_mask |= uint16_t(1) << cta_layout(i);
    }
  }
  // Shift by the instruction's elected block rank (dynamic)
  mcast_mask <<= elected_cta;
  return mcast_mask;
}

// Projections multicast mask
template <int Mode, int... Modes, class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t
create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                          CtaCoord  const& cta_coord_vmnk)
{
  return create_tma_multicast_mask<Modes...>(cta_layout_vmnk, replace<Mode>(cta_coord_vmnk, _));
}

////////////////////////////////////
// Make TMA copy A/B/C
///////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy_A_sm90(CopyOp                  const& copy_op,
                     Tensor<GEngine,GLayout> const& gtensor,
                     SLayout                 const& slayout,
                     CTA_Tiler               const& cta_tiler,
                     Cluster_Size            const& cluster_size)
{
  // Keep only MK modes from MNK
  auto cta_tiler_mk = remove<1>(cta_tiler);

  // mcast along N mode for this M load, if any
  auto cluster_size_n = size<1>(cluster_size);

  if constexpr (cute::is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL>) {
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler_mk,
                                cluster_size_n);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_mk);
    auto cta_t_tile = make_layout(cluster_size_n);

    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    auto tma_copy = detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_tile, cta_v_tile);
    return tma_copy;
  }
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy_B_sm90(CopyOp                  const& copy_op,
                     Tensor<GEngine,GLayout> const& gtensor,
                     SLayout                 const& slayout,
                     CTA_Tiler               const& cta_tiler,
                     Cluster_Size            const& cluster_size)
{
  // Keep only NK modes from MNK
  auto cta_tiler_nk = remove<0>(cta_tiler);

  // mcast along M mode for this N load, if any
  auto cluster_size_m = size<0>(cluster_size);

  if constexpr (cute::is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL>) {
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler_nk,
                                cluster_size_m);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_nk);
    auto cta_t_tile = make_layout(cluster_size_m);

    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    auto tma_copy = detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_tile, cta_v_tile);
    return tma_copy;
  }
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler>
CUTE_HOST_RTC
auto
make_tma_copy_C_sm90(CopyOp                  const& copy_op,
                     Tensor<GEngine,GLayout> const& gtensor,
                     SLayout                 const& slayout,
                     CTA_Tiler               const& cta_tiler)
{
  // Keep only MN modes from MNK
  auto cta_tiler_mn = remove<2>(cta_tiler);

  if constexpr (cute::is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL> ||
      cute::is_same_v<CopyOp, SM90_TMA_STORE_IM2COL>) {
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler_mn,
                                _1{});
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_mn);

    // No multicast, so only 1 CTA involved
    auto cta_t_map = Layout<_1,_0>{};

    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    auto tma_copy = detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
    return tma_copy;
  }
}
} // end namespace cute
