#ifndef _BUTTER_SURFACE_HPP_
#define _BUTTER_SURFACE_HPP_
#include <cstdint>
namespace butter {
  struct surface {
    uint16_t type;
    uint16_t force;
    uint8_t flags;
    uint8_t room;
    uint16_t lower_y;
    uint16_t upper_y;
    
  };
}
#endif