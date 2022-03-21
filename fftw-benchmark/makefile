#======================================================================#
#                                                                      #
#      Makefile -- explore code optimisation.                          #
#               -- Intel compilers on Intel cpus                       #
#                                                                      #
#======================================================================#


#
# ----- compiler
#
CC ?= gcc


#
# ----- must specify FFTW or MKL
#
FFTW_LIBRARY ?= fftw



#
# ----- default options: double precision, no threading
#
ifndef OMP
  OMP = false
endif

ifndef DBL
  DBL = true
endif

ifndef DBG
  DBL = false
endif


#
# ----- headers and libraries
#
ifeq ($(FFTW_LIBRARY), fftw)
  ifndef FFTWDIR
    $(error FFTWDIR is not defined)
  endif
  FFTW3_INCS = -I$(FFTWDIR)/include
  ifeq ($(DBL), true)
    FFTW3_LIBS += -L$(FFTWDIR)/lib -lfftw3 -lfftw3_omp -Wl,-rpath=$(FFTWDIR)/lib
  else
    FFTW3_LIBS += -L$(FFTWDIR)/lib -lfftw3f -lfftw3f_omp -Wl,-rpath=$(FFTWDIR)/lib
  endif
endif

ifeq ($(FFTW_LIBRARY), mkl)
  ifndef MKLROOT
    $(error MKLROOT is not defined)
  endif
  FFTW3_INCS = -I$(MKLROOT)/include/fftw
  ifeq ($(OMP), true)
    # FFTW3_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl
    FFTW3_LIBS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
  else
    # FFTW3_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl
    FFTW3_LIBS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
  endif
endif


#
# ----- basic flags
#
ifeq ($(OMP), true)
  CFLAGS = -DFFTW3_OMP
else
  CFLAGS = 
endif

ifeq (${DBL}, true)
  CFLAGS += -DFFTW3_DOUBLE
else
  CFLAGS += -DFFTW3_FLOAT
endif

#
# ----- optimisation flags
#
ifeq (${DBG}, true)
  COPTS = -g
else
  COPTS = -O3 -mavx
endif

ifeq (${OMP}, true)
  COPTS += -fopenmp
endif


#
# ----- FFTW include and library paths
#

# extra include and library paths
EXTRA_INCS = -DFFTW3_FILE_OUTPUT
EXTRA_LIBS = -lm

# include and library paths
INCS = ${FFTW3_INCS} ${EXTRA_INCS}
LIBS = ${FFTW3_LIBS} ${EXTRA_LIBS}


# transformation rules
.SUFFIXES:
.SUFFIXES: .o .c

.c.o:
	$(CC) ${INCS} ${COPTS} ${CFLAGS} -c $<


# build
default:	all

help:
		@echo
		@echo " usage:    make all"
		@echo "           make OMP=true all"
		@echo "           make OMP=true DBL=false all"
		@echo " defaults: OMP=false, DBL=true"
		@echo

all:		fftw3 fftw3s

fftw3:  	fftw3.o fftw3_aux.o
		${CC} ${COPTS} -o fftw3 fftw3.o fftw3_aux.o   ${LIBS}

fftw3s: 	fftw3s.o fftw3_aux.o
		${CC} ${COPTS} -o fftw3s fftw3s.o fftw3_aux.o ${LIBS}

.PHONY:		clean
clean:
		@/bin/rm -f *.o \
			*.c~ *.h~ makefile~ \
			fftw3 fftw3s \
			fftw3.out



# end
