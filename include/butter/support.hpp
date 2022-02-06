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

#endif