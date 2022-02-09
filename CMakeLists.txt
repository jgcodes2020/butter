set(CMAKE_EXPORT_COMPILE_COMMANDS true)
cmake_minimum_required(VERSION 3.18)

set(BUTTER_EXACT_FP true CACHE BOOL "Ensure SM64-accurate FP calculations.")

project(butter
  LANGUAGES CXX
)

add_library(butter
  src/rng_table.cpp
  src/math/atan2s.cpp
  src/math/trig_tables.cpp
)
target_include_directories(butter PUBLIC include)
target_compile_features(butter PUBLIC cxx_std_20)
if (BUTTER_EXACT_FP)
  target_compile_definitions(butter PUBLIC "BUTTER_CFG_EXACT_FP")
endif()