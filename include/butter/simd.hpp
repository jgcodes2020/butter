#ifndef _BUTTER_SIMD_HPP_
#define _BUTTER_SIMD_HPP_

#include <nmmintrin.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "butter/support.hpp"

namespace butter {
  namespace details {
    template <size_t L, class ISeq = std::make_index_sequence<L>>
      requires(2 <= L && L <= 4)
    struct basic_fvec;

    template <size_t L, size_t... Is>
    struct basic_fvec<L, std::index_sequence<Is...>> {
      __m128 xmm;

      // Zero-initializes the underlying XMM register.
      basic_fvec() : xmm(_mm_setzero_ps()) {}

      // Sets the value of this register using 3 floats.
      explicit basic_fvec(sink_index<float, Is>... vs) :
        xmm([&]<size_t... Rest>(std::index_sequence<Rest...>) {
          return mm_setr_ps(vs..., (void(Rest), 0.0f)...);
        }(std::make_index_sequence<4 - L> {})) {}

      // Sets this vector to the contents of an existing XMM register. Zeros
      // unused elements.
      basic_fvec(__m128 vec) :
        xmm([&]() {
          const __m128 andps_mask = [&]<size_t... Rest>(
            std::index_sequence<Rest...>) {
            return _mm_castsi128_ps(
              _mm_set_epi32((void(Is), -1)..., (void(Rest), 0)...));
          }
          (std::make_index_sequence<4 - L> {});
          return _mm_and_ps(vec, andps_mask);
        }()) {}

      // Assigns another basic_fvec to this vector, padding with zeros or
      // truncating where necessary.
      template <size_t M, size_t... Js>
      basic_fvec(basic_fvec<M, std::index_sequence<Js...>> vec) :
        xmm([&]() {
          if constexpr (M > L) {
            const __m128 andps_mask = [&]<size_t... Rest>(
              std::index_sequence<Rest...>) {
              return _mm_castsi128_ps(
                _mm_set_epi32((void(Is), -1)..., (void(Rest), 0)...));
            }
            (std::make_index_sequence<4 - L> {});
            return _mm_and_ps(vec, andps_mask);
          }
          else {
            return vec;
          }
        }()) {}

      basic_fvec& operator=(__m128 x) {
        xmm = x;
        return *this;
      }
      operator __m128() { return xmm; }

      static basic_fvec broadcast(float x) { return _mm_set1_ps(x); }

      class const_reference {
        basic_fvec& ref;
        const uint8_t i;

      public:
        operator float() {
          // based on Clang assembly output with builtin vectors
          float dump[4];
          _mm_storeu_ps(dump, ref);
          return dump[i];
        }
      };
      class reference {
        basic_fvec& ref;
        const uint8_t i;

      public:
        operator float() {
          float dump[4];
          _mm_storeu_ps(dump, ref);
          return dump[i];
        }
        reference& operator=(float x) {
          // based on Clang assembly output with builtin vectors
          __m128 broadcast = _mm_set1_ps(x);
          __m128 blendvps_mask =
            _mm_cmpeq_ps(broadcast, _mm_setr_ps(0, 1, 2, 3));
          ref = _mm_blendv_ps(ref, broadcast, blendvps_mask);
          return *this;
        }
      };

      reference operator[](size_t x) { return reference {*this, x & 3}; }
      const_reference operator[](size_t x) const {
        return const_reference {*this, x & 3};
      }

      friend basic_fvec operator+(basic_fvec a, basic_fvec b) {
        return _mm_add_ps(a, b);
      }
      friend basic_fvec& operator+=(basic_fvec& a, basic_fvec b) {
        return a = _mm_add_ps(a, b);
      }
      friend basic_fvec operator-(basic_fvec a, basic_fvec b) {
        return _mm_sub_ps(a, b);
      }
      friend basic_fvec& operator-=(basic_fvec& a, basic_fvec b) {
        return a = _mm_sub_ps(a, b);
      }

      friend basic_fvec operator*(basic_fvec a, basic_fvec b) {
        return _mm_mul_ps(a, b);
      }
      friend basic_fvec operator*(basic_fvec a, float b) {
        return _mm_mul_ps(a, _mm_set1_ps(b));
      }
      friend basic_fvec operator*(float a, basic_fvec b) {
        return _mm_mul_ps(b, _mm_set1_ps(a));
      }

      friend basic_fvec operator/(basic_fvec a, basic_fvec b) {
        return _mm_div_ps(a, b);
      }
      friend basic_fvec operator/(basic_fvec a, float b) {
        return _mm_div_ps(a, _mm_set1_ps(b));
      }
    };
  }  // namespace details

  template <size_t L>
  using vec  = details::basic_fvec<L>;
  using vec2 = details::basic_fvec<2>;
  using vec3 = details::basic_fvec<3>;
  using vec4 = details::basic_fvec<4>;

  template <size_t L>
  inline float dot(vec<L> a, vec<L> b) {
    // DPPS adds left-to-right (viewed from memory ordering)
    constexpr uint8_t dpps_mask = (((1 << L) - 1) << 4) | 0x01;
    __m128 result = _mm_dp_ps(a, b, dpps_mask);
    return _mm_cvtss_f32(result);
  }
  inline vec3 cross(vec3 a, vec3 b) {
    // Uses this:
    // https://geometrian.com/programming/tutorials/cross-product/index.php
    const __m128 at = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 bt = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));

    a = _mm_mul_ps(a, bt);
    b = _mm_mul_ps(b, at);

    __m128 res = _mm_sub_ps(a, b);
    // Shuffle into the right order
    return _mm_shuffle_ps(res, res, _MM_SHUFFLE(3, 0, 2, 1));
  }
  template <size_t L>
  inline vec<L> normalize(vec<L> x) {
    // DPPS adds left-to-right (viewed from memory ordering)
    constexpr uint8_t dpps_mask = (((1 << L) - 1) << 4) | 0x0F;
    __m128 dist_sq = _mm_dp_ps(x, x, dpps_mask);
    return _mm_div_ps(x, _mm_sqrt_ps(dist_sq));
  }
}  // namespace butter
#endif