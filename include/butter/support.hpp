#ifndef _BUTTER_SUPPORT_HPP_
#define _BUTTER_SUPPORT_HPP_

#include <type_traits>
#include <utility>
#if defined(_MSC_VER)
  #define BUTTER_FORCE_INLINE __forceinline
#elif defined(__GNUC__)
  #define BUTTER_FORCE_INLINE [[gnu::always_inline]]
#else
  #define BUTTER_FORCE_INLINE
#endif



#include <cstddef>

namespace butter::details {
  template <class T, size_t I>
  using sink_index = T;
  
  
}
namespace butter::config {
  #ifdef BUTTER_CFG_EXACT_FP
  static constexpr bool use_exact_fp = true;
  #else
  static constexpr bool use_exact_fp = false;
  #endif
}

#endif