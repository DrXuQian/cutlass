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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FusionCallbacks - Dispatch Interface
//
// For detailed documentation with examples, see:
//   docs/epilogue_fusion_architecture.md
//
// This is the bridge between:
//   - FusionOperation (operations.hpp): Abstract definition of WHAT to compute
//   - EVT Implementation (sm90_callbacks_*.hpp): Concrete HOW to compute
//
// Dispatch Flow:
//   User specifies: FusionCallbacks<Sm90TmaWarpSpecialized, LinearCombination, ...>
//                                      ↓
//   Template specialization in sm90_callbacks_tma_warpspecialized.hpp matches
//                                      ↓
//   Maps to EVT: Sm90EVT<Sm90Compute<multiply_add>, Sm90ScalarBroadcast, Sm90SrcFetch, ...>
//
// The CollectiveEpilogue receives FusionCallbacks as template parameter and calls:
//   - fusion_callbacks.get_producer_load_callbacks() → pld_callbacks
//   - fusion_callbacks.get_consumer_store_callbacks() → cst_callbacks
//
// Example:
//   // Define fusion operation
//   using FusionOp = fusion::LinearCombination<half_t, float, half_t>;
//
//   // FusionCallbacks dispatches to EVT implementation
//   using Callbacks = FusionCallbacks<
//       Sm90TmaWarpSpecialized<4, 4, 2, false, false>,  // DispatchPolicy
//       FusionOp,                                        // Operation
//       Shape<_128, _128, _64>,                          // CtaTile_MNK
//       Shape<_64, _32>                                  // EpilogueTile_MN
//   >;
//   // → Resolves to Sm90EVT<Sm90Compute<multiply_add>, ...> internally
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Primary template - static_assert fires if no specialization matches
template <
  class DispatchPolicy,  // Collective's dispatch policy (e.g., Sm90TmaWarpSpecialized)
  class Operation,       // Fusion operation (e.g., LinearCombination, LinCombPerRowBias)
  class CtaTile_MNK,     // CTA tile shape
  class EpilogueTile_MN, // Epilogue subtile shape
  class... Args          // Additional implementation-dependent args
>
struct FusionCallbacks {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy, Operation>,
    "Could not find a FusionCallbacks specialization. "
    "Check that the Operation is supported for this DispatchPolicy.");
};

// Traits helper to extract metadata from FusionCallbacks or custom EVT types
template <class T>
struct FusionCallbacksTraits {
  using DispatchPolicy = void;
  using Callbacks = T;
  using Operation = FusionOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = void;
};

template <
  class DispatchPolicy_,
  class Operation_,
  class CtaTile_MNK_,
  class EpilogueTile_MN_,
  class... Args
>
struct FusionCallbacksTraits<
  FusionCallbacks<DispatchPolicy_, Operation_, CtaTile_MNK_, EpilogueTile_MN_, Args...>
> {
  using DispatchPolicy = DispatchPolicy_;
  using Callbacks = FusionCallbacks<DispatchPolicy_, Operation_, CtaTile_MNK_, EpilogueTile_MN_, Args...>;
  using Operation = Operation_;
  using CtaTile_MNK = CtaTile_MNK_;
  using EpilogueTile_MN = EpilogueTile_MN_;
  using ElementCompute = typename Operation::ElementCompute;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
