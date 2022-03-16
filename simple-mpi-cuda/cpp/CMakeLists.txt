cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project (simple-mpi-cuda LANGUAGES CXX CUDA)

enable_language (CXX)
enable_language (CUDA)

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES 60 70)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

find_package (MPI REQUIRED)

add_executable (simple-mpi-cuda simple-cuda.cu simple-mpi.cpp)
target_link_libraries (simple-mpi-cuda MPI::MPI_CXX)

install (TARGETS simple-mpi-cuda DESTINATION bin)
