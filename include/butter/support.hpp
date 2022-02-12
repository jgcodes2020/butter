#ifndef _BUTTER_SUPPORT_HPP_
#define _BUTTER_SUPPORT_HPP_

#include <algorithm>
#include <limits>
#if defined(_MSC_VER)
  #define BUTTER_FORCE_INLINE __forceinline
#elif defined(__GNUC__)
  #define BUTTER_FORCE_INLINE [[gnu::always_inline]]
#else
  #define BUTTER_FORCE_INLINE
#endif

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace butter::details {
  template <class T, size_t I>
  using sink_index = T;

  template <char... Cs>
  consteval bool check_index_size() {
    struct local {
      static consteval auto stringify() {
        size_t n = std::numeric_limits<size_t>::max();
        std::array<char, std::numeric_limits<size_t>::digits10> digits;
        for (size_t i = digits.size(); i > 0; --i) {
          digits[i - 1] = n % 10;
          n %= 10;
        }
        return digits;
      }
    };
    
    if (sizeof...(Cs) > std::numeric_limits<size_t>::digits10) {
      return false;
    }
    else if (sizeof...(Cs) == std::numeric_limits<size_t>::digits10) {
      std::array<char, std::numeric_limits<size_t>::digits10> arr {Cs...};
      std::array<char, std::numeric_limits<size_t>::digits10> cmp = local::stringify();
      return std::lexicographical_compare(arr.begin(), arr.end(), cmp.begin(), cmp.end());
    }
    else {
      return true;
    }
  }

  template <char... Cs>
  consteval size_t parse_index_constant() {
    static_assert(((Cs >= '0' && Cs <= '9') && ...), "Invalid digits");
    static_assert(check_index_size<Cs...>(), "Number is too large");
    
    size_t num = 0;
    ((num = num * 10 + (Cs - '0')), ...);
    return num;
  }
}  // namespace butter::details
namespace butter::inline literals {
  template<char... Cs>
  consteval auto operator""_idxc() {
    return std::integral_constant<size_t, details::parse_index_constant<Cs...>()> {};
  }
}
namespace butter::config {
#ifdef BUTTER_CFG_EXACT_FP
  static constexpr bool use_exact_fp = true;
#else
  static constexpr bool use_exact_fp = false;
#endif
}  // namespace butter::config

#endif