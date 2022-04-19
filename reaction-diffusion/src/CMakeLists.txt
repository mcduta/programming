#
# --- project details
project (ReactionDiffusion)


#
# --- minimal cmake version
#       * CMake 3.12 and above use the ``find_package(<PackageName>)`` environmen variable.
#       * See ``cmake --help-policy CMP0074`` for an explanation.
cmake_minimum_required (VERSION 3.12)


# 
# --- option: float (default) or double?
option (USE_DOUBLE_PRECISION "Build with 8 byte reals." OFF)
if (USE_DOUBLE_PRECISION)
    add_compile_definitions (_DOUBLE_PRECISION)
endif ()


#
# --- default build type
set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message (STATUS "Setting build type to '${default_build_type}'.")
    set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif ()


#
# --- C++ support (with AVX)
enable_language (CXX)
set (CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message (STATUS "Setting AVX vectorisation flags.")
    if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCORE-AVX2 -axCORE-AVX512")
    endif ()
endif ()


#
# --- OpenGL support
find_package (OpenGL)
find_package (GLUT)
if (OPENGL_FOUND AND GLUT_FOUND)
    include_directories (${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
endif ()


#
# --- OpenMP support
find_package (OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else (OPENMP_FOUND)
    message ("WARNING: no OpenMP support.")
endif ()


#
# --- HDF5 support
find_package (HDF5 COMPONENTS CXX HL)


#
# --- "make"
add_executable (rd rd.cpp solution.cpp timer.cpp)

if (HDF5_FOUND)
    add_compile_definitions (_HDF5)
    include_directories (${HDF5_INCLUDE_DIRS})
    link_directories( ${HDF5_LIBRARY_DIRS} )
    target_link_libraries (rd ${HDF5_CXX_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif ()


if (OPENGL_FOUND AND GLUT_FOUND)
    add_executable (rdGL rdGL.cpp solution.cpp renderer.cpp timer.cpp)
    target_link_libraries (rdGL ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
    if (HDF5_FOUND)
        target_link_libraries (rdGL ${HDF5_CXX_LIBRARIES} ${HDF5_HL_LIBRARIES})
    endif ()
endif ()


#
# --- "make install"
install (TARGETS rd DESTINATION bin)
if (OPENGL_FOUND AND GLUT_FOUND)
    install (TARGETS rdGL DESTINATION bin)
endif ()
