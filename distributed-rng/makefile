
# begin

#
# makefile --- adapted from ${MKLROOT}/examples/vslc/makefile
#

# compiler
CC    = mpicc
COPTS = -w -vec-report0 -DMKL_VML_MIXED

# mkl library
MKL_PATH ?= ${MKLROOT}/lib/intel64
MKL_INCL  = -I${MKL_PATH}/include -I$(MKL_PATH)/examples/vslc/source
MKL_LIBS  = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_vml_avx2 -liomp5 -lpthread -lm -ldl

# transformation rules
.SUFFIXES: .c .o
.c.o:
	${CC} ${COPTS} ${MKL_INCL} -c $<

# linking
# vsl_uniform: vsl_uniform.o
# 	${CC} ${COPTS} ${MKL_INCL} -o vsl_uniform vsl_uniform.o ${MKL_LIBS}

vsl_mc: vsl_mc.o
	${CC} ${COPTS} ${MKL_INCL} -o vsl_mc vsl_mc.o ${MKL_LIBS}

# extra
clean:
	rm *.o *.c~ makefile~


# end
