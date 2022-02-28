void mult (int n, double *x, double *y, double *z) {
  int i;
  for (i = 0; i < n; i++) {
    z[i] = x[i] + 2.0*y[i] + x[i]*y[i];
  }
}

void mult2 (const int n, double *restrict x, double *restrict y, double *restrict z) {
  int i;
  for (i = 0; i < n; i++) {
    z[i] = x[i] + 2.0*y[i] + x[i]*y[i];
  }
}
