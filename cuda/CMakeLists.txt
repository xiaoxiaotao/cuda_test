
project(cuda_lib)
cmake_minimum_required(VERSION 3.5)
  
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON) 
find_package( CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};--disable-warnings;
        -O2
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
        -Wno-deprecated-gpu-targets
       
        )
# file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

cuda_add_executable(cuda_lib   ${SOURCE_FILES} )
