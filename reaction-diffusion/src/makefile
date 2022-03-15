#======================================================================#
#                                                                      #
#      Makefile -- reaction-diffusion finite difference solution       #
#                                                                      #
#======================================================================#

# debug?
ifndef DEBUG
  DEBUG = 0
endif

# compiler
CXX = g++

INCS = 

# optimisation
ifeq ($(DEBUG), 0)
  CFLAGS = -O3 -mavx2 -fopenmp
else
  CFLAGS = -O0 -g
endif

# libraries
LIBS = -lGL -lGLU -lglut -fopenmp

# objects
OBJS = rd.o \
       solution.o \
       integrator.o \
       renderer.o

# target
EXE = ../bin/rd

# transformation rules
.SUFFIXES:
.SUFFIXES: .o .h .cpp

.cpp.o:
		$(CXX) $(INCS) $(CFLAGS) -c $<

# build
default:	all

all:		rd

rd:		$(OBJS)
		$(CXX) -o $(EXE) $(OBJS) $(LIBS)

clean:
		/bin/rm -f $(OBJS) *.hpp~ *.cpp~ makefile~ $(EXE)


# end
