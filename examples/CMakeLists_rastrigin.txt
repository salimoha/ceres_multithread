cmake_minimum_required(VERSION 2.8)

project(rastrigin)

find_package(Ceres REQUIRED COMPONENTS C++11 OpenMP)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
include_directories(${CERES_INCLUDE_DIRS})

# rastrigin
add_executable(rastrigin rastrigin.cc)
if (NOT MSVC)
  target_link_libraries(rastrigin m)
endif (NOT MSVC)

target_link_libraries(rastrigin ${CERES_LIBRARIES})
