//
// --- header guards to prevent multiple inclusions of the same header file
//     (equivalent to the non-standard # pragma once preprocessor directive)
//
# ifndef _TEST_UTILS_H_
# define _TEST_UTILS_H_

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <limits>

// default real is single precision
# ifndef REAL
# define REAL float
# endif

// sizes
# define NUM_MEGA (1<<20)
# define NUM_GIGA (1<<30)


//
// --- useful bits
//
template <class T> void randVec (T *x, const size_t n);
template <class T> T maxError (T *x, const size_t n);
template <class T> void printVec (T *x, const size_t n);
void printHelp (char *exeName);
size_t parseArgv (int argc, char *argv[]);
void printSize (const size_t totalBytes);


//
// --- include the template implementation source file to instantiate
//
# include "test-utils.cpp"

# endif
