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
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef CUTE_SGEMM_SM80_PRINT_LAYOUT
#define CUTE_SGEMM_SM80_PRINT_LAYOUT 0
#endif

namespace {
constexpr bool kPrintLayouts = CUTE_SGEMM_SM80_PRINT_LAYOUT != 0;

template <class Layout>
void print_layout_debug(const char* name, Layout const& layout) {
  printf("%s: ", name);
  cute::print(layout);
  printf("\n");
  if constexpr (decltype(cute::rank(layout))::value == 2) {
    if constexpr (cute::is_integral<decltype(layout(0, 0))>::value) {
      cute::print_layout(layout);
    } else {
      printf("  (rank == 2 but codomain is not integral, print_layout skipped)\n");
    }
  } else {
    printf("  (rank != 2, print_layout skipped)\n");
  }
}

#if !defined(_WIN32)
class ScopedStdoutRedirect {
 public:
  explicit ScopedStdoutRedirect(std::string const& path) {
    fflush(stdout);
    saved_fd_ = ::dup(fileno(stdout));
    if (saved_fd_ < 0) {
      ok_ = false;
      return;
    }

    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      ok_ = false;
      return;
    }

    if (::dup2(fd, fileno(stdout)) < 0) {
      ok_ = false;
    }
    ::close(fd);
  }

  ScopedStdoutRedirect(ScopedStdoutRedirect const&) = delete;
  ScopedStdoutRedirect& operator=(ScopedStdoutRedirect const&) = delete;

  ~ScopedStdoutRedirect() {
    if (saved_fd_ < 0) {
      return;
    }
    fflush(stdout);
    (void)::dup2(saved_fd_, fileno(stdout));
    ::close(saved_fd_);
  }

  bool ok() const { return ok_; }

 private:
  int saved_fd_{-1};
  bool ok_{true};
};

template <class Fn>
void dump_latex_to_file(std::string const& path, Fn&& fn) {
  ScopedStdoutRedirect redirect(path);
  if (!redirect.ok()) {
    fprintf(stderr, "Failed to redirect stdout to %s\n", path.c_str());
    return;
  }
  fn();
  fflush(stdout);
}
#endif

struct DumpOptions {
  bool dump_layouts{false};
  bool dump_latex{false};
  std::string outdir{"."};
  int m{128};
  int n{128};
  int k{64};
};

bool parse_dump_options(int argc, char** argv, DumpOptions* opts) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump-layouts") == 0) {
      opts->dump_layouts = true;
      continue;
    }
    if (std::strcmp(argv[i], "--dump-latex") == 0) {
      opts->dump_latex = true;
      continue;
    }
    if (std::strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) {
      opts->outdir = argv[++i];
      continue;
    }
    if (std::strcmp(argv[i], "--dump-m") == 0 && i + 1 < argc) {
      opts->m = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--dump-n") == 0 && i + 1 < argc) {
      opts->n = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--dump-k") == 0 && i + 1 < argc) {
      opts->k = std::atoi(argv[++i]);
      continue;
    }
  }
  return opts->dump_layouts || opts->dump_latex;
}

void dump_sgemm_sm80_layouts(DumpOptions const& opts) {
  using namespace cute;

  using TA = half_t;
  using TB = half_t;
  using TC = half_t;

  printf("==== sgemm_sm80.cu layout/algebra dump (host-only) ====\n");
  printf("Reference problem M,N,K: (%d,%d,%d)\n", opts.m, opts.n, opts.k);
  printf("NOTE: 这里仅构造 layout/tensor view，不会触发任何 CUDA kernel。\n\n");

  // Problem shape (dynamic)
  auto prob_shape = make_shape(opts.m, opts.n, opts.k);  // (M, N, K)

  // Match gemm_tn() default in this file: transA='T', transB='N'
  int ldA = opts.k;
  int ldB = opts.k;
  int ldC = opts.m;
  auto dA = make_stride(ldA, Int<1>{});          // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});          // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);          // (dM, dN)

  printf("-- Global problem / strides --\n");
  printf("shape_MNK: "); print(prob_shape); printf("\n");
  printf("dA (MK):   "); print(dA);         printf("\n");
  printf("dB (NK):   "); print(dB);         printf("\n");
  printf("dC (MN):   "); print(dC);         printf("\n\n");

  // CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bP = Int<3>{};  // Pipeline stages
  auto cta_tiler = make_shape(bM, bN, bK);

  printf("-- CTA tiler (static) --\n");
  printf("cta_tiler: "); print(cta_tiler); printf("\n\n");

  // Shared memory swizzle layout (static)
  using SwizzleBase =
      Layout<Shape<_8, Shape<_8, _8>>,
             Stride<_8, Stride<_1, _64>>>;
  SwizzleBase swizzle_base{};
  auto swizzle_atom = composition(Swizzle<3,3,3>{}, swizzle_base);

  auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
  auto sC_layout = make_layout(make_shape(bM, bN));

  printf("-- Shared memory layouts (static) --\n");
  print_layout_debug("swizzle_base", swizzle_base);
  print_layout_debug("swizzle_atom", swizzle_atom);
  print_layout_debug("sA layout (M,K,PIPE)", sA_layout);
  print_layout_debug("sB layout (N,K,PIPE)", sB_layout);
  print_layout_debug("sC layout (M,N)", sC_layout);
  printf("\n");

  // Thread-level G2S copies (static)
  auto copyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16,_8>,Stride<_8,_1>>{},   // 16x8 threads, k-major
      Layout<Shape<_1,_8>>{});                 // 1x8 values
  auto copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16,_8>,Stride<_8,_1>>{},   // 16x8 threads, k-major
      Layout<Shape<_1,_8>>{});                 // 1x8 values

  // MMA (static)
  auto mmaC = make_tiled_mma(
      SM80_16x8x16_F16F16F16F16_TN{},
      Layout<Shape<_2,_2>>{},   // 2x2 atom arrangement
      Tile<_32,_32,_16>{});     // overall tile (for permutation/ldsm friendliness)

  // S2R copy atoms
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;
  auto s2r_copy_a = make_tiled_copy_A(s2r_atom_A, mmaC);
  auto s2r_copy_b = make_tiled_copy_B(s2r_atom_B, mmaC);

  printf("-- LDMATRIX CopyAtom (SM75_U32x4_LDSM_N) --\n");
  {
    using LdsAtom = decltype(s2r_atom_A);
    print_layout_debug("ldmatrix ValLayoutSrc (thr,val)->offset", typename LdsAtom::ValLayoutSrc{});
    print_layout_debug("ldmatrix ValLayoutDst (thr,val)->offset", typename LdsAtom::ValLayoutDst{});
    print_layout_debug("ldmatrix ValLayoutRef (thr,val)->offset", typename LdsAtom::ValLayoutRef{});
    auto src2ref = right_inverse(typename LdsAtom::ValLayoutRef{}).compose(typename LdsAtom::ValLayoutSrc{});
    auto dst2ref = right_inverse(typename LdsAtom::ValLayoutRef{}).compose(typename LdsAtom::ValLayoutDst{});
    printf("ldmatrix src2ref (src_tv->ref_tv): "); print(src2ref); printf("\n");
    printf("ldmatrix dst2ref (dst_tv->ref_tv): "); print(dst2ref); printf("\n");

	    printf("ldmatrix src2ref samples (src_tid,src_vid)->(ref_tid,(lane,reg)):\n");
	    for (int tid = 0; tid < 32; tid += 8) {
	      printf("  tid %2d : ", tid);
	      for (int vid = 0; vid < 8; ++vid) {
	        auto src_off = typename LdsAtom::ValLayoutSrc{}(tid, vid);
	        auto ref_coord = typename LdsAtom::ValLayoutRef{}.get_hier_coord(src_off);
	        auto rtid = get<0>(ref_coord);
	        auto rlane = get<1,0>(ref_coord);
	        auto rreg = get<1,1>(ref_coord);
	        printf("(%2d,(%d,%d))%s", int(rtid), int(rlane), int(rreg), (vid == 7 ? "" : " "));
	      }
	      printf("\n");
	    }
	    printf("\n");
	  }

  printf("-- High-level TiledCopy / TiledMMA objects --\n");
  printf("copyA: "); print(copyA); printf("\n");
  printf("copyB: "); print(copyB); printf("\n");
  printf("mmaC:  "); print(mmaC);  printf("\n\n");

  // Dump the same internal layouts that are used by copy_atom.hpp / mma_atom.hpp.
  printf("==== [copy_atom.hpp] make_tiled_copy() algebra (copyA) ====\n");
  {
    auto thr_layout = Layout<Shape<_16,_8>,Stride<_8,_1>>{};
    auto val_layout = Layout<Shape<_1,_8>>{};
    auto layout_mn  = raked_product(thr_layout, val_layout);
    auto layout_tv  = right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));
    auto tiler      = product_each(shape(layout_mn));
    print_layout_debug("thr_layout (m,n)->thr_idx", thr_layout);
    print_layout_debug("val_layout (m,n)->val_idx", val_layout);
    print_layout_debug("layout_mn  (m,n)->(thr,val)", layout_mn);
    print_layout_debug("layout_tv  (thr,val)->(m,n)", layout_tv);
    printf("tiler (TileM,TileN): "); print(tiler); printf("\n\n");
  }

  printf("==== [copy_atom.hpp] TiledCopy internals ====\n");
  {
    using CopyA = decltype(copyA);
    print_layout_debug("copyA TiledLayout_TV", typename CopyA::TiledLayout_TV{});
    printf("copyA Tiler_MN: "); print(typename CopyA::Tiler_MN{}); printf("\n");
    print_layout_debug("copyA layout S_TV", CopyA::get_layoutS_TV());
    print_layout_debug("copyA layout D_TV", CopyA::get_layoutD_TV());
    printf("\n");

    // Replicate tile2thrfrg() step-by-step (S side)
    printf("-- tile2thrfrg() step-by-step (S) --\n");
    using AtomLayoutRef = typename CopyA::AtomLayoutRef;
    using AtomLayoutSrc = typename CopyA::AtomLayoutSrc;
    auto src2ref = right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{});
    print_layout_debug("src2ref (src_tv)->(ref_tv)", src2ref);

    auto atom_layout_TV = zipped_divide(typename CopyA::TiledLayout_TV{}, make_shape(typename CopyA::AtomNumThr{}, typename CopyA::AtomNumVal{}));
    print_layout_debug("atom_layout_TV", atom_layout_TV);
    auto src_layout_TV = atom_layout_TV.compose(src2ref, _);
    print_layout_debug("src_layout_TV = atom_layout_TV.compose(src2ref,_)", src_layout_TV);
    auto thrval2mn = coalesce(zip(src_layout_TV), Shape<_1,Shape<_1,_1>>{});
    print_layout_debug("thrval2mn (coalesce(zip(src_layout_TV)))", thrval2mn);
    printf("\n");

    printf("-- tile2thrfrg() step-by-step (D) --\n");
    using AtomLayoutDst = typename CopyA::AtomLayoutDst;
    auto dst2ref = right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{});
    print_layout_debug("dst2ref (dst_tv)->(ref_tv)", dst2ref);
    auto dst_layout_TV = atom_layout_TV.compose(dst2ref, _);
    print_layout_debug("dst_layout_TV = atom_layout_TV.compose(dst2ref,_)", dst_layout_TV);
    auto thrval2mn_d = coalesce(zip(dst_layout_TV), Shape<_1,Shape<_1,_1>>{});
    print_layout_debug("thrval2mn (coalesce(zip(dst_layout_TV)))", thrval2mn_d);
    printf("\n");
  }

  printf("==== [mma_atom.hpp] TiledMMA internals ====\n");
  {
    using MmaAtom = typename decltype(mmaC)::Atom;
    print_layout_debug("mma atom LayoutA_TV", typename MmaAtom::LayoutA_TV{});
    print_layout_debug("mma atom LayoutB_TV", typename MmaAtom::LayoutB_TV{});
    print_layout_debug("mma atom LayoutC_TV", typename MmaAtom::LayoutC_TV{});
    printf("mmaC thr_layout_vmnk: "); print(mmaC.get_thr_layout_vmnk()); printf("\n");
    printf("mmaC permutation_mnk: (");
    print(mmaC.template permutation_mnk<0>()); printf(",");
    print(mmaC.template permutation_mnk<1>()); printf(",");
    print(mmaC.template permutation_mnk<2>()); printf(")\n");
    printf("mmaC tile_shape: "); print(tile_shape(mmaC)); printf("\n\n");

    // Reproduce thrfrg_C() algebra (see include/cute/atom/mma_atom.hpp)
    printf("-- thrfrg_C() step-by-step --\n");
    auto ref_C = make_layout(make_shape(mmaC.template tile_size_mnk<0>(), mmaC.template tile_size_mnk<1>()));
    print_layout_debug("ref_C (M,N)->idx", ref_C);
    auto t_tile_C = make_tile(mmaC.template permutation_mnk<0>(), mmaC.template permutation_mnk<1>());
    printf("t_tile_C: "); print(t_tile_C); printf("\n");
    auto t_tensor_C = logical_divide(ref_C, t_tile_C);
    print_layout_debug("t_tensor_C = logical_divide(ref_C, t_tile_C)", t_tensor_C);
    auto c_tile = make_tile(make_layout(size<0>(typename decltype(mmaC)::AtomShape_MNK{})),
                            make_layout(size<1>(typename decltype(mmaC)::AtomShape_MNK{})));
    printf("c_tile (AtomM,AtomN): "); print(c_tile); printf("\n");
    auto c_tensor = zipped_divide(t_tensor_C, c_tile);
    print_layout_debug("c_tensor = zipped_divide(t_tensor_C, c_tile)", c_tensor);
    auto tv_tensor = c_tensor.compose(typename decltype(mmaC)::AtomLayoutC_TV{}, _);
    print_layout_debug("tv_tensor = c_tensor.compose(AtomLayoutC_TV, _)", tv_tensor);
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(mmaC.get_thr_layout_vmnk())),
                                        make_layout(size<2>(mmaC.get_thr_layout_vmnk()))));
    printf("thr_tile (ThrM,ThrN): "); print(thr_tile); printf("\n");
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    print_layout_debug("thr_tensor = zipped_divide(tv_tensor, thr_tile)", thr_tensor);
    printf("\n");
  }

  printf("==== [sgemm_sm80.cu] end-to-end tensor views (thread0) ====\n");
  {
    TA const* A_ptr = nullptr;
    TB const* B_ptr = nullptr;
    TC* C_ptr = nullptr;

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr), select<0,2>(prob_shape), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr), select<1,2>(prob_shape), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr), select<0,1>(prob_shape), dC); // (M,N)

    auto cta_coord = make_coord(0, 0, _);  // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    Tensor sA = make_tensor(make_smem_ptr((TA*)nullptr), sA_layout);      // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr((TB*)nullptr), sB_layout);      // (BLK_N,BLK_K,PIPE)

    printf("mA: "); print(mA.layout()); printf("\n");
    printf("gA: "); print(gA.layout()); printf("\n");
    printf("sA: "); print(sA.layout()); printf("\n");
    printf("mB: "); print(mB.layout()); printf("\n");
    printf("gB: "); print(gB.layout()); printf("\n");
    printf("sB: "); print(sB.layout()); printf("\n");
    printf("mC: "); print(mC.layout()); printf("\n");
    printf("gC: "); print(gC.layout()); printf("\n\n");

    auto thr_copy_a = copyA.get_slice(0);
    auto thr_copy_b = copyB.get_slice(0);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    printf("tAgA: "); print(tAgA.layout()); printf("\n");
    printf("tAsA: "); print(tAsA.layout()); printf("\n");
    printf("tBgB: "); print(tBgB.layout()); printf("\n");
    printf("tBsB: "); print(tBsB.layout()); printf("\n\n");

    auto thr_mma = mmaC.get_slice(0);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    printf("tCgC: "); print(tCgC.layout()); printf("\n");
    printf("tCrA: "); print(tCrA.layout()); printf("\n");
    printf("tCrB: "); print(tCrB.layout()); printf("\n");
    printf("tCrC: "); print(tCrC.layout()); printf("\n\n");

    auto s2r_thr_copy_a = s2r_copy_a.get_slice(0);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(0);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    printf("tXsA: "); print(tXsA.layout()); printf("\n");
    printf("tXrA: "); print(tXrA.layout()); printf("\n");
    printf("tXsB: "); print(tXsB.layout()); printf("\n");
    printf("tXrB: "); print(tXrB.layout()); printf("\n\n");
  }

#if !defined(_WIN32)
  if (opts.dump_latex) {
    (void)std::filesystem::create_directories(opts.outdir);
    std::string base = opts.outdir;
    if (!base.empty() && base.back() != '/') {
      base += "/";
    }

    printf("==== LaTeX dumps (written to %s) ====\n", opts.outdir.c_str());

    // 2D layouts
    dump_latex_to_file(base + "swizzle_base.tex", [&] { print_latex(swizzle_base); });
    dump_latex_to_file(base + "swizzle_atom.tex", [&] { print_latex(swizzle_atom); });

    // Smem pipe0 slices
    {
      Tensor sA = make_tensor(make_smem_ptr((TA*)nullptr), sA_layout);
      Tensor sB = make_tensor(make_smem_ptr((TB*)nullptr), sB_layout);
      dump_latex_to_file(base + "sA_pipe0.tex", [&] { print_latex(sA(_,_,0).layout()); });
      dump_latex_to_file(base + "sB_pipe0.tex", [&] { print_latex(sB(_,_,0).layout()); });
    }
    dump_latex_to_file(base + "sC_layout.tex", [&] { print_latex(sC_layout); });

    // Copy and MMA figures
    dump_latex_to_file(base + "copyA.tex", [&] { print_latex(copyA); });
    dump_latex_to_file(base + "copyB.tex", [&] { print_latex(copyB); });
    dump_latex_to_file(base + "mmaC.tex",  [&] { print_latex(mmaC); });

    // TV visualizations (extra helpful)
    dump_latex_to_file(base + "mma_layoutA_TV.tex", [&] {
      auto tile_mk = make_shape(tile_size<0>(mmaC), tile_size<2>(mmaC));
      auto refA = make_identity_tensor(tile_mk);
      auto tensorA_TV = composition(refA, mmaC.get_layoutA_TV());
      print_latex_tv(tensorA_TV, tile_mk);
    });
    dump_latex_to_file(base + "mma_layoutB_TV.tex", [&] {
      auto tile_nk = make_shape(tile_size<1>(mmaC), tile_size<2>(mmaC));
      auto refB = make_identity_tensor(tile_nk);
      auto tensorB_TV = composition(refB, mmaC.get_layoutB_TV());
      print_latex_tv(tensorB_TV, tile_nk);
    });
    dump_latex_to_file(base + "mma_layoutC_TV.tex", [&] {
      auto tile_mn = make_shape(tile_size<0>(mmaC), tile_size<1>(mmaC));
      auto refC = make_identity_tensor(tile_mn);
      auto tensorC_TV = composition(refC, mmaC.get_layoutC_TV());
      print_latex_tv(tensorC_TV, tile_mn);
    });

    dump_latex_to_file(base + "copyA_layoutS_TV.tex", [&] {
      auto tiler_mn = typename decltype(copyA)::Tiler_MN{};
      auto tile_mn = product_each(shape(logical_divide(make_layout(Shape<_1,_1>{}), tiler_mn)));
      auto refS = make_identity_tensor(tile_mn);
      auto layoutS_TV = copyA.tidfrg_S(refS)(_,_,Int<0>{});
      print_latex_tv(layoutS_TV, tile_mn);
    });
    dump_latex_to_file(base + "copyA_layoutD_TV.tex", [&] {
      auto tiler_mn = typename decltype(copyA)::Tiler_MN{};
      auto tile_mn = product_each(shape(logical_divide(make_layout(Shape<_1,_1>{}), tiler_mn)));
      auto refD = make_identity_tensor(tile_mn);
      auto layoutD_TV = copyA.tidfrg_D(refD)(_,_,Int<0>{});
      print_latex_tv(layoutD_TV, tile_mn);
    });

    // S2R aligned copies
    dump_latex_to_file(base + "s2r_copy_a.tex", [&] { print_latex(s2r_copy_a); });
    dump_latex_to_file(base + "s2r_copy_b.tex", [&] { print_latex(s2r_copy_b); });
  }
#endif

  printf("==== dump done ====\n");
}
}  // namespace

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");

    print("tXsA : "); print(tXsA); print("\n");
    print("tXrA : "); print(tXrA); print("\n");
    print("tXsB : "); print(tXsB); print("\n");
    print("tXrB : "); print(tXrB); print("\n");
  }
#endif

#if 1

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
  //     Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

template <class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
{
  assert(false && "Not implemented");
}

// Setup params for a TN HGEMM
template <class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  using SwizzleBase =
      Layout<Shape<_8, Shape<_8, _8>>,
             Stride<_8, Stride<_1, _64>>>;
  auto swizzle_base = SwizzleBase{};
  auto swizzle_atom = composition(Swizzle<3,3,3>{}, swizzle_base);
  if constexpr (kPrintLayouts) {
    print_layout_debug("swizzle_base", swizzle_base);
    print_layout_debug("swizzle_atom", swizzle_atom);
  }
  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
  auto sC = make_layout(make_shape(bM, bN));

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 n-major

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

  if constexpr (kPrintLayouts) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      printf("\n==== sgemm_sm80 layout dump (gemm_tn) ====\n");
      printf("cta_tiler: "); print(cta_tiler); printf("\n");
      print_layout_debug("swizzle_atom", swizzle_atom);
      print_layout_debug("smem sA layout", sA);
      print_layout_debug("smem sB layout", sB);
      print_layout_debug("smem sC layout", sC);

      printf("\n-- G2S TiledCopy (copyA/copyB) --\n");
      print_layout_debug("copyA TiledLayout_TV", typename decltype(copyA)::TiledLayout_TV{});
      printf("copyA Tiler_MN: "); print(typename decltype(copyA)::Tiler_MN{}); printf("\n");
      print_layout_debug("copyA layout S_TV", decltype(copyA)::get_layoutS_TV());
      print_layout_debug("copyA layout D_TV", decltype(copyA)::get_layoutD_TV());
      print_layout_debug("copyB TiledLayout_TV", typename decltype(copyB)::TiledLayout_TV{});
      printf("copyB Tiler_MN: "); print(typename decltype(copyB)::Tiler_MN{}); printf("\n");
      print_layout_debug("copyB layout S_TV", decltype(copyB)::get_layoutS_TV());
      print_layout_debug("copyB layout D_TV", decltype(copyB)::get_layoutD_TV());

      using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
      print_layout_debug("g2s atom ValLayoutSrc", typename G2SCopyAtom::ValLayoutSrc{});
      print_layout_debug("g2s atom ValLayoutDst", typename G2SCopyAtom::ValLayoutDst{});
      print_layout_debug("g2s atom ValLayoutRef", typename G2SCopyAtom::ValLayoutRef{});

      printf("\n-- MMA Atom and TiledMMA --\n");
      using MmaAtom = typename decltype(mmaC)::Atom;
      print_layout_debug("mma atom LayoutA_TV", typename MmaAtom::LayoutA_TV{});
      print_layout_debug("mma atom LayoutB_TV", typename MmaAtom::LayoutB_TV{});
      print_layout_debug("mma atom LayoutC_TV", typename MmaAtom::LayoutC_TV{});
      printf("mmaC thr_layout_vmnk: "); print(mmaC.get_thr_layout_vmnk()); printf("\n");
      printf("mmaC tile_shape: "); print(tile_shape(mmaC)); printf("\n");
      print_layout_debug("mmaC layoutA_TV", mmaC.get_layoutA_TV());
      print_layout_debug("mmaC layoutB_TV", mmaC.get_layoutB_TV());
      print_layout_debug("mmaC layoutC_TV", mmaC.get_layoutC_TV());

      printf("\n-- S2R Copy Atom and TiledCopy --\n");
      using S2RCopyAtom = decltype(s2r_atom_A);
      print_layout_debug("s2r atom ValLayoutSrc", typename S2RCopyAtom::ValLayoutSrc{});
      print_layout_debug("s2r atom ValLayoutDst", typename S2RCopyAtom::ValLayoutDst{});
      print_layout_debug("s2r atom ValLayoutRef", typename S2RCopyAtom::ValLayoutRef{});
      auto s2r_copy_a = make_tiled_copy_A(s2r_atom_A, mmaC);
      auto s2r_copy_b = make_tiled_copy_B(s2r_atom_B, mmaC);
      print_layout_debug("s2r_copy_a TiledLayout_TV", typename decltype(s2r_copy_a)::TiledLayout_TV{});
      printf("s2r_copy_a Tiler_MN: "); print(typename decltype(s2r_copy_a)::Tiler_MN{}); printf("\n");
      print_layout_debug("s2r_copy_a layout S_TV", decltype(s2r_copy_a)::get_layoutS_TV());
      print_layout_debug("s2r_copy_a layout D_TV", decltype(s2r_copy_a)::get_layoutD_TV());
      print_layout_debug("s2r_copy_b TiledLayout_TV", typename decltype(s2r_copy_b)::TiledLayout_TV{});
      printf("s2r_copy_b Tiler_MN: "); print(typename decltype(s2r_copy_b)::Tiler_MN{}); printf("\n");
      print_layout_debug("s2r_copy_b layout S_TV", decltype(s2r_copy_b)::get_layoutS_TV());
      print_layout_debug("s2r_copy_b layout D_TV", decltype(s2r_copy_b)::get_layoutD_TV());
    }
  }

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  auto kernel_fptr = gemm_device<
    decltype(prob_shape), decltype(cta_tiler),
    cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
    cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
    cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
    decltype(alpha), decltype(beta)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, s2r_atom_A,
       B, dB, sB, copyB, s2r_atom_B,
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK, bP));             // (m,k,p) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK, bP));             // (n,k,p) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 n-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 n-major

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, Copy_Atom<AutoVectorizingCopy, TA>{},
       B, dB, sB, copyB, Copy_Atom<AutoVectorizingCopy, TB>{},
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA_atom                  = make_layout(make_shape (      bM,          bK),
                                              make_stride(Int<1>{}, bM+Int<1>{})); // (m,k) -> smem_idx; padded m-major
  [[maybe_unused]] auto sB_atom = make_layout(make_shape (      bN,          bK),
                                              make_stride(Int<1>{}, bN+Int<1>{})); // (n,k) -> smem_idx; padded n-major
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(sA_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, Copy_Atom<AutoVectorizingCopy, TA>{},
       B, dB, sB, copyB, Copy_Atom<AutoVectorizingCopy, TB>{},
       C, dC, sC, mmaC,
       alpha, beta);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}


int main(int argc, char** argv)
{
  DumpOptions dump_opts;
  if (parse_dump_options(argc, argv, &dump_opts)) {
    dump_sgemm_sm80_layouts(dump_opts);
    return 0;
  }

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 8) {
    std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  std::cout << "Using device 0: " << props.name
            << " (SM" << props.major * 10 + props.minor
            << ", " << props.multiProcessorCount
            << ")" << std::endl;

  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'T';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'N';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = cute::half_t;

  TI alpha = static_cast<TI>(1.0f);
  TI beta  = static_cast<TI>(0.0f);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  d_C = h_C;
  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}
