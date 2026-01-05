#pragma once

#include "cutlass/integer_subbyte.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cuda_bf16.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Biased integer subbyte types for quantization
// These represent values with a bias, e.g., vllm_uint4b8_t stores values in range [0,15]
// but represents values in range [-8, 7] (bias = 8)
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, int Bias, bool Signed = false>
struct biased_integer_subbyte : public integer_subbyte<Bits, Signed> {
  using Base = integer_subbyte<Bits, Signed>;
  using Storage = typename Base::Storage;
  using xint_t = typename Base::xint_t;
  using Base::bits_mask_;
  using Base::sign_mask_;
  using Base::storage;

  biased_integer_subbyte() = default;

  CUTLASS_HOST_DEVICE explicit biased_integer_subbyte(int value) : Base(value) {}
  CUTLASS_HOST_DEVICE explicit biased_integer_subbyte(unsigned value) : Base(value) {}
  CUTLASS_HOST_DEVICE explicit biased_integer_subbyte(double value) : Base(value) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Type aliases for common quantization schemes
///////////////////////////////////////////////////////////////////////////////////////////////////

// GPTQ/AWQ style: 4-bit unsigned with bias 8 (represents [-8, 7])
using uint4b8_t = biased_integer_subbyte<4, 8>;
// 8-bit unsigned with bias 128 (represents [-128, 127])
using uint8b128_t = biased_integer_subbyte<8, 128>;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, int Bias, bool Signed>
struct sizeof_bits<biased_integer_subbyte<Bits, Bias, Signed>> {
  static constexpr int value = Bits;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Type name utilities for debugging
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct nameof {
  static constexpr char const* value = "unknown";
};

template <typename T>
inline constexpr auto nameof_v = nameof<T>::value;

#define MACHETE_NAMEOF_TYPE(T) \
  template <> struct nameof<T> { static constexpr char const* value = #T; };

MACHETE_NAMEOF_TYPE(float_e4m3_t)
MACHETE_NAMEOF_TYPE(float_e5m2_t)
MACHETE_NAMEOF_TYPE(half_t)
MACHETE_NAMEOF_TYPE(nv_bfloat16)
MACHETE_NAMEOF_TYPE(bfloat16_t)
MACHETE_NAMEOF_TYPE(float)
MACHETE_NAMEOF_TYPE(int4b_t)
MACHETE_NAMEOF_TYPE(int8_t)
MACHETE_NAMEOF_TYPE(int32_t)
MACHETE_NAMEOF_TYPE(uint4b8_t)
MACHETE_NAMEOF_TYPE(uint4b_t)
MACHETE_NAMEOF_TYPE(uint4_t)
MACHETE_NAMEOF_TYPE(uint8_t)
MACHETE_NAMEOF_TYPE(uint8b128_t)

#undef MACHETE_NAMEOF_TYPE

}  // namespace cutlass
