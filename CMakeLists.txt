cmake_minimum_required(VERSION 3.20.0)
project(kapy LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

option(KAPY_BUILD_PYTHON_MODULE "Build Kapy Python bindings" OFF)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

set(KAPY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(KAPY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Python module
if (KAPY_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  set(PYTHON_SRC_FILES ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/ir.cc)

  include_directories(${PYBIND11_INCLUDE_DIR})
  include_directories(${PYTHON_SRC_PATH})
  if (PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
  else()
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    include_directories(${Python3_INCLUDE_DIRS})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()

  add_library(kapy SHARED ${PYTHON_SRC_FILES})
  set(KAPY_LIBRARIES ${dialect_libs} ${conversion_libs})
  target_link_libraries(kapy ${KAPY_LIBRARIES} z)
endif()
