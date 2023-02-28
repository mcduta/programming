#
# --- project details
project (paff)


#
# --- minimal cmake version
#       * CMake 3.12 and above use the ``find_package(<PackageName>)`` environment variable.
#       * See ``cmake --help-policy CMP0074`` for an explanation.
cmake_minimum_required (VERSION 3.12)


#
# --- default build type
set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message (STATUS "Setting build type to '${default_build_type}'.")
    set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif ()


#
# ===== CPU executables
#

#
# --- C support
enable_language (C)
set (CMAKE_C_STANDARD 99)
set (CMAKE_C_STANDARD_REQUIRED TRUE)


#
# --- NUMA support (NB: NO find_package support)
find_path (NUMA_INCLUDE_DIR NAME numa.h
    HINTS /usr/local/include /usr/include /opt/local/include
)
find_library (NUMA_LIBRARY NAME numa
    HINTS /usr/local/lib64 /usr/local/lib /usr/lib64 /usr/lib /opt/local/lib64 /opt/local/lib
)
if (NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
    set (NUMA_FOUND TRUE)
    message (STATUS "Found numa library: inc=${NUMA_INCLUDE_DIR}, lib=${NUMA_LIBRARY}")
else ()
    set (NUMA_FOUND FALSE)
    message (FATAL_ERROR "Could not find NUMA lib.")
endif ()


#
# --- OpenMP support (mandatory but easy to satisfy)
find_package (OpenMP REQUIRED)


#
# --- single process, multi-threaded
add_executable (paff paff.c)
target_compile_options (paff PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries  (paff PRIVATE ${OpenMP_C_FLAGS} ${NUMA_LIBRARY})
install (TARGETS paff DESTINATION bin)


#
# --- MPI support
find_package (MPI)


#
# --- multi-process, multi-threaded
if (MPI_FOUND)
    add_executable (paff_mpi paff.c)
    target_compile_options (paff_mpi PRIVATE ${OpenMP_C_FLAGS} -D_MPI)
    target_link_libraries  (paff_mpi PRIVATE ${OpenMP_C_FLAGS} ${NUMA_LIBRARY} MPI::MPI_C)
    install (TARGETS paff_mpi DESTINATION bin)
else ()
    message (WARNING "Could not build MPI version.")
endif ()




#
# ===== CPU & GPU executables
#

#
# --- CUDA support
enable_language (CUDA)
if (CMAKE_CUDA_COMPILER)
    set (CUDA_FOUND TRUE)
    message (STATUS "Found CUDA ver. ${CMAKE_CUDA_COMPILER_VERSION}")
    if (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
        set (CMAKE_CUDA_ARCHITECTURES 60 70)
    endif ()
        message (STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
else ()
    set (CUDA_FOUND FALSE)
    message (WARNING "Could not find CUDA.")
endif ()

if (CUDA_FOUND)
    add_library (paff_utils SHARED paff_utils.cu paff_utils.h)
endif ()


#
# --- CUDA extension
if (CUDA_FOUND)
    #
    # --- single process, multi-threaded
    add_executable (paff_cuda paff.c)
    target_compile_options (paff_cuda PRIVATE ${OpenMP_C_FLAGS} -D_CUDA)
    target_link_libraries  (paff_cuda PRIVATE ${OpenMP_C_FLAGS} ${NUMA_LIBRARY} paff_utils)
    install (TARGETS paff_cuda DESTINATION bin)

    #
    # --- multi-process, multi-threaded
    if (MPI_FOUND)
        add_executable (paff_mpi_cuda paff.c)
        target_compile_options (paff_mpi_cuda PRIVATE ${OpenMP_C_FLAGS} -D_CUDA -D_MPI)
        target_link_libraries  (paff_mpi_cuda PRIVATE ${OpenMP_C_FLAGS} ${NUMA_LIBRARY} MPI::MPI_C paff_utils)
        install (TARGETS paff_mpi_cuda DESTINATION bin)
    else ()
       message (WARNING "Could not build CUDA MPI version.")
    endif ()
endif ()



# CUDA_FOUND
# set_target_properties( particles
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# add_executable(particle_test test.cu)
 
# set_property(TARGET particle_test 
#              PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(particle_test PRIVATE particles)


# add_library(greeting
#   SHARED
#     greeting.cpp
#     greeting.hpp
#   )


# #
# # --- MPI support
# find_package (MPI)








# enable_language (CXX)
# set (CMAKE_CXX_STANDARD 17)

# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     message (STATUS "Setting AVX vectorisation flags.")
#     if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#         set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
#     elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
#         set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCORE-AVX2 -axCORE-AVX512")
#     endif ()
# endif ()



# #
# # --- "make install"

# if (OPENGL_FOUND AND GLUT_FOUND)
#     install (TARGETS rdGL DESTINATION bin)
# endif ()
