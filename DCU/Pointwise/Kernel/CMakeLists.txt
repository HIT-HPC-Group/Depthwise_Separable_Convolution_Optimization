cmake_minimum_required (VERSION 3.1)
project (PointwiseConvolution)

# HIP_PATH
SET(HIP_PATH "/public/software/compiler/dtk-23.04/hip")
SET(MIOPEN_PATH "/public/software/compiler/dtk-23.04/miopen")
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  "${HIP_PATH}/cmake" ${MIOPEN_PATH}/lib/cmake/miopen)
add_definitions(-std=c++11)

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)
find_package(miopen REQUIRED)
include_directories(${MIOpen_INCLUDE_DIRS})

add_definitions(-DAMD_PLATFORM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
if(USE_GFLAGS)
  add_definitions(-DUSE_GFLAGS)
endif()

SET(CMAKE_CXX_COMPILER "/opt/dtk-23.04.1/bin/hipcc")
FILE(GLOB HIPDNNSRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/*.cpp")
INCLUDE_DIRECTORIES(${MIOPEN_INCLUDE_DIR})
LINK_DIRECTORIES(${MIOPEN_LIBRARY_DIR})
include_directories(${HIP_PATH}/include ${HIP_PATH} )

add_executable(kernel DCU_Pointwise_Kernel.cpp)
target_link_libraries(kernel MIOpen)