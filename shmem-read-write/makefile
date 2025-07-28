CC     := gcc
CFLAGS :=

.SUFFIXES:
.SUFFIXES:	.c .h .o

.c.o:
		$(CC) $(CFLAGS) -c $<

default:	all

all:		shm_read shm_write

shm_read:	shm_read.o shm_read_write.h
		$(CC) $(CFLAGS) -o shm_read shm_read.o

shm_write:	shm_write.o shm_read_write.h
		$(CC) $(CFLAGS) -o shm_write shm_write.o

clean:
		/bin/rm -f *.o shm_read shm_write
