# compiler details
# include make.gcc
# include make.intel
include make.llvm



# transformation rules
.SUFFIXES:
.SUFFIXES: .o .cpp

.c.o:
	$(CXX) $(CFLAGS) -c $<

# build
default:	all

all:		loops

loops:		loops.cpp
		$(CXX) $(CFLAGS) -o loops loops.cpp
