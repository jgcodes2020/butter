#ifndef _BUTTER_SIMD_HPP_
#define _BUTTER_SIMD_HPP_

#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "butter/support.hpp"

namespace butter {

  namespace details {
    template <class T, size_t L, class ISeq = std::make_index_sequence<L>>
    struct basic_vec;

    template <size_t L, size_t... Is>
      requires(L >= 2 && L <= 4)
    struct basic_vec<float, L, std::index_sequence<Is...>> {
    private:
      __m128 _m_xmm;

    public:
      // Zero-initializes the underlying XMM register.
      basic_vec() : _m_xmm(_mm_setzero_ps()) {}

      // Sets the value of this register using 3 floats.
      basic_vec(sink_index<float, Is>... vs) :
        _m_xmm([&]<size_t... Rest>(std::index_sequence<Rest...>) {
          return _mm_setr_ps(vs..., (void(Rest), 0.0f)...);
        }(std::make_index_sequence<4 - L> {})) {}

      // Sets this vector to the contents of an existing XMM register. Zeros
      // unused elements.
      basic_vec(__m128 vec) :
        _m_xmm([&]() {
          if constexpr (L < 4) {
            const __m128i andps_mask = _mm_setr_epi32(
              -uint32_t(L >= 1), -uint32_t(L >= 2), -uint32_t(L >= 3),
              -uint32_t(L >= 4));
            return _mm_and_ps(vec, _mm_castsi128_ps(andps_mask));
          }
          else {
            return vec;
          }
        }()) {}

      // Assigns another basic_fvec to this vector, padding with zeros or
      // truncating where necessary.
      template <size_t M, size_t... Js>
      basic_vec(basic_vec<float, M, std::index_sequence<Js...>> vec) :
        _m_xmm([&]() {
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

      basic_vec& operator=(__m128 x) {
        const __m128 andps_mask = [&]<size_t... Rest>(
          std::index_sequence<Rest...>) {
          return _mm_castsi128_ps(
            _mm_set_epi32((void(Is), -1)..., (void(Rest), 0)...));
        } (std::make_index_sequence<4 - L> {});
        _m_xmm = _mm_and_ps(x, andps_mask);
        return *this;
      }
      operator __m128() const { return _m_xmm; }

      static basic_vec broadcast(float x) {
        return _mm_setr_ps(
          ((L >= 1) ? x : 0.0f), ((L >= 2) ? x : 0.0f), ((L >= 3) ? x : 0.0f),
          ((L >= 4) ? x : 0.0f));
      }
      static basic_vec load(void* x) {
        if constexpr (L == 2) {
          return _mm_load_sd(std::bit_cast<double*>(x));
        }
        else if constexpr (L == 3) {
          __m128 a = _mm_load_sd(std::bit_cast<double*>(x));
          __m128 b = _mm_load_ss(std::bit_cast<float*>(x) + 2);
          return _mm_movelh_ps(a, b);
        }
        else if constexpr (L == 4) {
          return _mm_loadu_ps(std::bit_cast<float*>(x));
        }
      }

      class const_reference {
        basic_vec& ref;
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
        basic_vec& ref;
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
      template <size_t I>
      class fixed_const_reference {
        static_assert(I < 4, "Invalid index");
        basic_vec& ref;

        operator float() {
          if constexpr (I == 0) {
            return _mm_cvtss_f32(ref);
          }
          else if constexpr (I == 1) {
            return _mm_cvtss_f32(_mm_movehdup_ps(ref));
          }
          else if constexpr (I == 2) {
            return _mm_cvtss_f32(_mm_movehl_ps(ref, ref));
          }
          else if constexpr (I == 3) {
            return _mm_cvtss_f32(
              _mm_shuffle_ps(ref, ref, _MM_SHUFFLE(3, 3, 3, 3)));
          }
        }
      };
      template <size_t I>
      class fixed_reference {
        static_assert(I < 4, "Invalid index");
        basic_vec& ref;

        operator float() {
          if constexpr (I == 0) {
            return _mm_cvtss_f32(ref);
          }
          else if constexpr (I == 1) {
            return _mm_cvtss_f32(_mm_movehdup_ps(ref));
          }
          else if constexpr (I == 2) {
            return _mm_cvtss_f32(_mm_movehl_ps(ref, ref));
          }
          else if constexpr (I == 3) {
            return _mm_cvtss_f32(
              _mm_shuffle_ps(ref, ref, _MM_SHUFFLE(3, 3, 3, 3)));
          }
        }

        reference& operator=(float x) {
          uint8_t imm8 = I << 4;
          __m128 xmm   = _mm_set_ss(x);
          ref          = _mm_insert_ps(ref, xmm, imm8);
          return *this;
        }
      };

      reference operator[](size_t x) { return reference {*this, x & 3}; }
      const_reference operator[](size_t x) const {
        return const_reference {*this, x & 3};
      }
      template <size_t I>
      fixed_reference<I> operator[](std::integral_constant<size_t, I>) {
        return fixed_reference<I>(*this);
      }
      template <size_t I>
      fixed_const_reference<I> operator[](
        std::integral_constant<size_t, I>) const {
        return fixed_const_reference<I>(*this);
      }

      friend basic_vec operator+(basic_vec a, basic_vec b) {
        return _mm_add_ps(a, b);
      }
      friend basic_vec& operator+=(basic_vec& a, basic_vec b) {
        return a = _mm_add_ps(a, b);
      }
      friend basic_vec operator-(basic_vec a, basic_vec b) {
        return _mm_sub_ps(a, b);
      }
      friend basic_vec& operator-=(basic_vec& a, basic_vec b) {
        return a = _mm_sub_ps(a, b);
      }

      friend basic_vec operator*(basic_vec a, basic_vec b) {
        return _mm_mul_ps(a, b);
      }
      friend basic_vec operator*(basic_vec a, float b) {
        return _mm_mul_ps(a, _mm_set1_ps(b));
      }
      friend basic_vec operator*(float a, basic_vec b) {
        return _mm_mul_ps(b, _mm_set1_ps(a));
      }
      friend basic_vec& operator*=(basic_vec& a, float b) {
        return a = _mm_mul_ps(a, _mm_set1_ps(b));
      }

      friend basic_vec operator/(basic_vec a, basic_vec b) {
        return _mm_div_ps(a, b);
      }
      friend basic_vec operator/(basic_vec a, float b) {
        return _mm_div_ps(a, _mm_set1_ps(b));
      }
      friend basic_vec& operator/=(basic_vec& a, float b) {
        return a = _mm_div_ps(a, _mm_set1_ps(b));
      }
    };

    template <std::integral T, size_t L, size_t... Is>
      requires(sizeof(T) == sizeof(int16_t))
    struct basic_vec<T, L, std::index_sequence<Is...>> {
    private:
      __m128i _m_xmm;

    public:
      // Zero-initializes the underlying XMM register.
      basic_vec() : _m_xmm(_mm_setzero_si128()) {}

      basic_vec(details::sink_index<T, Is>... vs) :
        _m_xmm([&]<size_t... Rest>(std::index_sequence<Rest...>) {
          return _mm_setr_epi16(vs..., (void(Rest), 0)...);
        }(std::make_index_sequence<(sizeof(__m128i) / sizeof(T)) - L> {})) {}

      basic_vec(__m128i vec) :
        _m_xmm([&]<size_t... RIs>(std::index_sequence<RIs...>) {
          const __m128i pand_mask = _mm_setr_epi16(-uint16_t(RIs < L)...);
          return _mm_and_si128(vec, pand_mask);
        }(std::make_index_sequence<sizeof(__m128i) / sizeof(T)> {})) {}

      template <size_t M, size_t... Js>
      basic_vec(basic_vec<T, M, std::index_sequence<Js...>> vec) :
        _m_xmm([&]() {
          if constexpr (M > L) {
            const __m128i pand_mask = [&]<size_t... RIs>(
              std::index_sequence<RIs...>) {
              return _mm_setr_epi16(-uint16_t(RIs < L)...);
            }
            (std::make_index_sequence<sizeof(__m128i) / sizeof(T)> {});
            return _mm_and_si128(vec, pand_mask);
          }
          else {
            return static_cast<__m128i>(vec);
          }
        }) {}
      
      struct const_reference {
        basic_vec& vec;
        uint8_t idx;
        
        operator T() {
          
        }
      };
      struct reference {
        basic_vec& vec;
        uint8_t idx;
        
        operator T() {
          
        }
      };
      template <size_t I>
      struct fixed_const_reference {
        basic_vec& vec;
        
        operator T() {
          return _mm_extract_epi16(vec, I);
        }
      };
      template <size_t I>
      struct fixed_reference {
        basic_vec& vec;
        
        operator T() {
          return _mm_extract_epi16(vec, I);
        }
      };
    };
  }  // namespace details

  consteval uint8_t shuffle_mask(
    uint8_t a = 0, uint8_t b = 1, uint8_t c = 2, uint8_t d = 3) {
    if (a >= 4 || b >= 4 || c >= 4 || d >= 4)
      throw std::out_of_range("All parameters must be less than 4");
    return (a) | (b << 2) | (c << 4) | (d << 6);
  }

  template <size_t L>
  using fvec  = details::basic_vec<float, L>;
  using fvec2 = details::basic_vec<float, 2>;
  using fvec3 = details::basic_vec<float, 3>;
  using fvec4 = details::basic_vec<float, 4>;

  template <size_t L>
  inline float dot(fvec<L> a, fvec<L> b) {
    // DPPS adds left-to-right (viewed from memory ordering)
    constexpr uint8_t dpps_mask = (((1 << L) - 1) << 4) | 0x01;
    __m128 result               = _mm_dp_ps(a, b, dpps_mask);
    return _mm_cvtss_f32(result);
  }
  inline fvec3 cross(fvec3 a, fvec3 b) {
    // Uses this:
    // https://geometrian.com/programming/tutorials/cross-product/index.php
    const __m128 at = _mm_shuffle_ps(a, a, shuffle_mask(1, 2, 0, 3));
    const __m128 bt = _mm_shuffle_ps(b, b, shuffle_mask(1, 2, 0, 3));

    a = _mm_mul_ps(a, bt);
    b = _mm_mul_ps(b, at);

    __m128 res = _mm_sub_ps(a, b);
    // Shuffle into the right order
    return _mm_shuffle_ps(res, res, shuffle_mask(1, 2, 0, 3));
  }
  template <size_t L>
  inline fvec<L> normalize(fvec<L> x) {
    // DPPS adds left-to-right (viewed from memory ordering)
    constexpr uint8_t dpps_mask = (((1 << L) - 1) << 4) | 0x0F;
    __m128 dist_sq              = _mm_dp_ps(x, x, dpps_mask);
    return _mm_div_ps(x, _mm_sqrt_ps(dist_sq));
  }
  template <uint8_t Mask, size_t R, size_t L>
  inline fvec<R> swizzle(fvec<L> x) {
    constexpr uint8_t low_mask = uint8_t(-1) >> ((4 - R) * 2);
    constexpr uint8_t pshufb_c =
      (Mask & low_mask) | (shuffle_mask() & ~low_mask);
    return _mm_shuffle_ps(x, x, pshufb_c);
  }
  template <size_t L>
  inline float distance_sq(fvec<L> a, fvec<L> b) {
    fvec<L> diff = a - b;
    return dot(diff, diff);
  }
  template <size_t L>
  inline float distance(fvec<L> a, fvec<L> b) {
    fvec<L> diff = a - b;
    return std::sqrt(dot(diff, diff));
  }

  // Only doing 4x4 float matrix because SM64 only uses those
  class mat4 {
  private:
    fvec4 m_cols[4];

  public:
    // Initializes the matrix with 16 floats in column-major order.
    mat4(
      float a, float b, float c, float d, float e, float f, float g, float h,
      float i, float j, float k, float l, float m, float n, float o, float p) :
      m_cols {{a, b, c, d}, {e, f, g, h}, {i, j, k, l}, {m, n, o, p}} {}

    // Initializes with a list of column vectors.
    mat4(fvec4 c0, fvec4 c1, fvec4 c2, fvec4 c3) : m_cols {c0, c1, c2, c3} {}

    fvec4& operator[](size_t x) { return m_cols[x & 3]; }
    const fvec4& operator[](size_t x) const { return m_cols[x & 3]; }

    // Returns the identity matrix.
    static mat4 identity() {
      // clang-format off
      return mat4 {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      };
      // clang-format on
    }

    mat4 transpose() {
      mat4 res = *this;
      _MM_TRANSPOSE4_PS(
        res.m_cols[0], res.m_cols[1], res.m_cols[2], res.m_cols[3]);
      return res;
    }
    mat4& transpose_self() {
      _MM_TRANSPOSE4_PS(m_cols[0], m_cols[1], m_cols[2], m_cols[3]);
      return *this;
    }

    friend mat4 operator+(const mat4& x, const mat4& y) {
      return mat4 {x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]};
    }
    friend mat4& operator+=(mat4& x, const mat4& y) {
      x[0] += y[0];
      x[1] += y[1];
      x[2] += y[2];
      x[3] += y[3];
      return x;
    }

    friend mat4 operator-(const mat4& x, const mat4& y) {
      return mat4 {x[0] - y[0], x[1] - y[1], x[2] - y[2], x[3] - y[3]};
    }
    friend mat4& operator-=(mat4& x, const mat4& y) {
      x[0] -= y[0];
      x[1] -= y[1];
      x[2] -= y[2];
      x[3] -= y[3];
      return x;
    }

    friend mat4 operator*(const mat4& x, float y) {
      return mat4 {x[0] * y, x[1] * y, x[2] * y, x[3] * y};
    }
    friend mat4 operator*(float x, const mat4& y) {
      return mat4 {y[0] * x, y[1] * x, y[2] * x, y[3] * x};
    }
    friend mat4& operator*=(mat4& x, float y) {
      x[0] *= y;
      x[1] *= y;
      x[2] *= y;
      x[3] *= y;
      return x;
    }

    friend mat4 operator/(const mat4& x, float y) {
      return mat4 {x[0] / y, x[1] / y, x[2] / y, x[3] / y};
    }
    friend mat4& operator/=(mat4& x, float y) {
      x[0] /= y;
      x[1] /= y;
      x[2] /= y;
      x[3] /= y;
      return x;
    }
    // Multiplies a vector by a matrix on the rules of linear algebra.
    friend fvec4 operator*(const mat4& x, fvec4 y) {
      __m128 tmp0 = _mm_shuffle_ps(y, y, shuffle_mask(0, 0, 0, 0));
      __m128 tmp1 = _mm_shuffle_ps(y, y, shuffle_mask(1, 1, 1, 1));
      __m128 tmp2 = _mm_shuffle_ps(y, y, shuffle_mask(2, 2, 2, 2));
      __m128 tmp3 = _mm_shuffle_ps(y, y, shuffle_mask(3, 3, 3, 3));
      if constexpr (config::use_exact_fp) {
        tmp0 = _mm_mul_ps(tmp0, x.m_cols[0]);
        tmp1 = _mm_mul_ps(tmp1, x.m_cols[1]);
        tmp2 = _mm_mul_ps(tmp2, x.m_cols[2]);
        tmp3 = _mm_mul_ps(tmp3, x.m_cols[3]);

        __m128 res = _mm_add_ps(tmp0, tmp1);
        res        = _mm_add_ps(res, tmp2);
        res        = _mm_add_ps(res, tmp3);
        return res;
      }
      else {
        __m128 res = _mm_mul_ps(tmp0, x.m_cols[0]);
        res        = _mm_fmadd_ps(tmp1, x.m_cols[1], res);
        res        = _mm_fmadd_ps(tmp2, x.m_cols[2], res);
        res        = _mm_fmadd_ps(tmp3, x.m_cols[3], res);
        return res;
      }
    }
    // Multiplies two matrices by the rules of linear algebra.
    friend mat4 operator*(const mat4& x, const mat4& y) {
      return mat4 {x * y[0], x * y[1], x * y[2], x * y[3]};
    }
  };
}  // namespace butter
#endif