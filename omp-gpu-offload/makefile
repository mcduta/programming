# compiler and flags
#  * add -D REAL=double fro double precision
CC      ?= nvc
CCFLAGS ?= -fast -Minfo=mp -mp=gpu,multicore

# source, executable and arguments
SRC = $(wildcard *.c)
EXE = $(patsubst %.c,%.x, $(SRC))
ARG = 20000 20000 10

# rule
%.x:	%.c
	$(CC) $(CCFLAGS) $< -o $@

# targets
.PHONY: all build run clean
all:	run
build:	$(EXE)
run:	build
	@for x in $(EXE); do \
		echo " running $$x..."; \
		$(RUN) ./$$x $(ARG); \
	done
clean:
	-@$(RM) $(wildcard $(EXE))
