/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    rendered.cpp -- renderer class

 */


// ----- headers
# include "solution.hpp"
# include "renderer.hpp"


// ----- local defs
# define RENDERER_DATA_MIN_INIT -1.e+10
# define RENDERER_DATA_MAX_INIT +1.e+10


// ----- constructor (default)
renderer :: renderer () {
  data    = NULL;
  dataMin = RENDERER_DATA_MIN_INIT;
  dataMax = RENDERER_DATA_MAX_INIT;
}


// ----- destructor
renderer :: ~renderer () {
  if (data) delete [] data;
}


// ----- constructor
void renderer :: initRenderer (class solution *sol) {
  data    = new unsigned char [(sol->M) * (sol->N) * 4];
  dataMin = RENDERER_DATA_MIN_INIT;
  dataMax = RENDERER_DATA_MAX_INIT;
  maxw    = sol->N / ( (float) MAX(sol->M, sol->N) );
  maxh    = sol->M / ( (float) MAX(sol->M, sol->N) );
}

// ----- get data limits
void renderer :: solutionExtrema (class solution *sol) {
  std::size_t M = sol->M,
              N = sol->N,
              k;


  REAL smax = -1.e+10;   // extrema initialised
  REAL smin = +1.e+10;   // to unlikely extreme values
  REAL *s = sol -> u;    // calculate extrema just for u

# pragma omp parallel default(none) \
                      shared(s,M,N) \
                      private(k) \
                      reduction(max:smax) \
                      reduction(min:smin)
  {
    # pragma omp for
    for (k=0; k<M*N; k++) {
      smax = MAX (smax , s[k] );
      smin = MIN (smin , s[k] );
    }
  }

  dataMin = smin;
  dataMax = smax;

}


// ----- transform solution to RGB data
void renderer :: solutionToImage (class solution *sol) {
  std::size_t M, N;             // image size
  std::size_t m, n, i, j, k;    // indices
  REAL *s;                      // array of values
  REAL sk;                      // value mapped to RGB

  // solution/image size
  M = sol->M;
  N = sol->N;
  s = sol->u;

  // find minimum and maximum value
  if ((dataMin==RENDERER_DATA_MIN_INIT) || (dataMax==RENDERER_DATA_MAX_INIT)) {
    std::cout << " *** error: solution rendering not calibrated" << std::endl;
    return;
  }

  // map data to RGB
  # pragma omp parallel default(none) \
                        shared(s,M,N, dataMin,dataMax, data,mapRGB) \
                        private(sk,m,n,k,i,j)
  {
    # pragma omp for
    for (n=0; n<N; n++) {
      for (m=0; m<M; m++) {
        k  = COL_MAJOR_INDEX_2D(M,N,m,n);
        sk = s[k];

        i = (int) ceil((sk - dataMin)/(dataMax - dataMin) * MAP_RGB_SIZE);
        i = MIN ( MAP_RGB_SIZE - 1, MAX ( 0, i ) );

        // NB: (M,N) -> (N,M) !!!
        j = ROW_MAJOR_INDEX_3D(N,M,4, n,m,0);

        data[j+0] = mapRGB[i][0];
        data[j+1] = mapRGB[i][1];
        data[j+2] = mapRGB[i][2];

        data[j+3] = MAP_RGB_COL_MAX;
      }
    }
  } // pragma omp parallel

  // load to memory
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, M,N,
                0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}


// ----- GL initialisation routine
void renderer :: initImage (class solution *sol) {
  glClearColor (0,0,0,0);
  glShadeModel (GL_FLAT);
  glEnable     (GL_DEPTH_TEST);

  glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

  glGenTextures (1, &texture);
  glBindTexture (GL_TEXTURE_2D, texture);

  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D    (GL_TEXTURE_2D, 0, GL_RGBA, sol->M,sol->N,
                   0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}


// ----- display routine
void renderer :: renderImage () {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
  glBindTexture(GL_TEXTURE_2D, texture);

  /*
    QUAD vertex are:
       (0,1) (1,1)       (0,maxh) (maxw,maxh)
       (0,0) (1,0)       (0,   0) (maxw,   0)
   */

  glBegin(GL_QUADS);
  glTexCoord2f( 0.0,  0.0); glVertex2f(-maxw, -maxh);
  glTexCoord2f( 0.0, maxh); glVertex2f(-maxw,  maxh);
  glTexCoord2f(maxw, maxh); glVertex2f( maxw,  maxh);
  glTexCoord2f(maxw,  0.0); glVertex2f( maxw, -maxh);
  glEnd();
  glFlush();
  glutSwapBuffers();
}


/* end */
