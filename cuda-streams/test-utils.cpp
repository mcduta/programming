/*

  CUDA kernel and utils shared by the test-*.cu examples

 */

//
// --- random initialisation of CPU memory
//     * nothing sophisticated, just fills x with some numbers
//
template <class T> void randVec (T *x, const size_t n)
{
  for (size_t i = 0; i < n; i++) {
    x[i] = (T) rand() / (T) ((unsigned) RAND_MAX + 1);
  }
}


//
// --- find max(x[i] - 1.0f) over all i (all x[i] expected to be 1.0)
//
template <class T> T maxError (T *x, const size_t n)
{
  T errorMax = 0.0;
  for (size_t i = 0; i < n; i++) {
    T error = fabs (x[i] - (T) 1.0);
    if (error > errorMax) errorMax = error;
  }
  return errorMax;
}


//
// --- inspect vector
//
template <class T> void printVec (T *x, const size_t n)
{
  const size_t nn=5;
  size_t n1, n2;
  if (n < 2*nn) {
    n1 = n;
    n2 = n;
  } else {
    n1 = nn;
    n2 = n - nn;
  }
  for (size_t i = 0; i < n1; i++) {
    printf (" [%d] = %f\n", i, x[i]);
  }
  if (n >= 2*nn) printf (" ...\n");
  for (size_t i = n2; i < n; i++) {
    printf (" [%d] = %f\n", i, x[i]);
  }
}


//
// --- command line parser
//
void printHelp (char *exeName) {
  fprintf (stderr, " usage: %s [-m|-g] N\n", exeName);
  fprintf (stderr, "        N is the number of MB (-m) or GB (-g) of GPU memory used\n", exeName);
}

size_t parseArgv (int argc, char *argv[])
{
  int opt;
  size_t totalBytes=NUM_GIGA; // 1GB default

  while ((opt = getopt(argc, argv, "m:g:h")) != -1) {
    switch(opt) {
    case 'm': totalBytes = ((size_t) atoi(optarg)) * (1<<20); break;
    case 'g': totalBytes = ((size_t) atoi(optarg)) * (1<<30); break;
    case 'h': printHelp (argv[0]); exit(0);
    default:
      fprintf (stderr, " *** warning: unknown option, using default\n");
    }
  }

  return totalBytes;
}


//
// --- report size
//
void printSize (const size_t totalBytes)
{
  printf(" used total GPU memory =");
  if (totalBytes < NUM_GIGA) {
    printf(" %6.4f MB\n", ((double) totalBytes)/((double) NUM_MEGA));
  } else {
    printf(" %6.4f GB\n", ((double) totalBytes)/((double) NUM_GIGA));
  }
  printf(" size of REAL = %6d bytes\n", sizeof(REAL));
}
