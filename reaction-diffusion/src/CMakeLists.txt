cmake_minimum_required (VERSION 2.8.11)

#
# --- project details

project (ReactionDiffusion)


#
# --- C++ support (with AVX)
enable_language (CXX)
set (CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -xCORE-AVX2")
    endif ()
endif ()


#
# --- OpenGL support
find_package (OpenGL)
find_package (GLUT)
if (OPENGL_FOUND AND GLUT_FOUND)
    include_directories (${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})
endif ()

#
# --- OpenMP support
find_package (OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else (OPENMP_FOUND)
    message("WARNING: no OpenMP support.")
endif ()


#
# --- "make"
add_executable (rd rd.cpp solution.cpp)
if (OPENGL_FOUND AND GLUT_FOUND)
    add_executable (rdGL rdGL.cpp solution.cpp renderer.cpp)
    target_link_libraries (rdGL ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
endif ()



#
# --- "make install"
install (TARGETS rd DESTINATION bin)
if (OPENGL_FOUND AND GLUT_FOUND)
    install (TARGETS rdGL DESTINATION bin)
endif ()
