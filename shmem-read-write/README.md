# Program to test reading and writing shared memory segments

## Background
Written as a basic speed test for shared memory reading and writing.

## Build
Prerequisites are a C compiler (``gcc`` only was tested). ``make`` creates two executable:
  * ``shm_write`` -- write to shared memory;
  * ``shm_read``  -- read from shared memory.

## Usage
``shm_write`` attaches itself to a single shared memory segment (1GB in size) and repeatedly writes to it. The contents of the shared segment is copied (``memcpy``) from a process heap-allocated variable, whose values changes with each iteration. The program captures SIGINT (CTRL+C) so it detaches itself from the segment and destroys it cleanly. Write bandwidth is reported at each iteration. ``shm_read`` is very similar, repeatedly reads the contents of the shared segment written to by ``shm_write`` and reports the read bandwidth.

For simplicity, the segments utilise a fixed key, avoiding the use of ``ftok``. The parameters of the reads and writes (size of the segment, number of iterations) are defined in ``shm_read_write.h``.
