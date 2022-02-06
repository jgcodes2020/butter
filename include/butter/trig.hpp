#ifndef _BUTTER_TRIG_HPP_
#define _BUTTER_TRIG_HPP_
#include <array>
#include <cstdint>
#include <numbers>

#include <butter/support.hpp>

namespace butter {
  namespace tables {
    extern const std::array<float, 1024 + 4096> sin_table;
    extern const std::array<uint16_t, 1025> atan_table;
  }
  
  BUTTER_FORCE_INLINE inline float sins(uint16_t x) {
    return tables::sin_table[x >> 4];
  }
  
  BUTTER_FORCE_INLINE inline float coss(uint16_t x) {
    return tables::sin_table[(x >> 4) + 1024];
  }
  
  uint16_t atan2s(float y, float x);
  
  inline float atan2f(float y, float x) {
    // std::numbers::pi is a double.
    // pi as a float is std::numbers::pi_v<float>
    return float(atan2s(y, x)) * std::numbers::pi;
  }
}  // namespace butter
#endif