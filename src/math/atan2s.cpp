#include <butter/trig.hpp>

namespace {
  /// Performs lookup based on y/x, assuming 0° < atan2(y, x) < 45°
  uint16_t atan2_lookup(float y, float x) {
    using namespace butter;
    return (x == 0) ? 0 : tables::atan_table[uint32_t(y / x * 1024 + 0.5f)];
  }
}  // namespace

namespace butter {
  uint16_t atan2s(float y, float x) {
    if (x >= 0) {
      if (y >= 0) {
        if (y >= x) {
          // 45°-90°
          return atan2_lookup(x, y);
        }
        else {
          // 0°-45°
          return 16384 - atan2_lookup(y, x);
        }
      }
      else {
        y = -y;
        if (y < x) {
          // 315°-360°
          return 16384 + atan2_lookup(y, x);
        }
        else {
          // 270°-315°
          return 32768 - atan2_lookup(x, y);
        }
      }
    }
    else {
      x = -x;
      if (y < 0) {
        y = -y;
        if (y >= x) {
          // 225°-270°
          return 32768 + atan2_lookup(x, y);
        }
        else {
          // 180°-225°
          return 49152 - atan2_lookup(y, x);
        }
      }
      else {
        if (y < x) {
          // 135°-180°
          return 49152 + atan2_lookup(y, x);
        }
        else {
          // 90°-135°
          return -atan2_lookup(x, y);
        }
      }
    }
  }
}  // namespace butter