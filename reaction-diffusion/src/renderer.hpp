/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    rendered.h -- renderer class headers

 */


// ----- includes
# include <GL/gl.h>
# include <GL/glut.h>

# include "mapRGB.hpp"


// ----- class
class renderer {
  unsigned char *data;          // data in RGB format
  REAL dataMin;                 // minimum solution value
  REAL dataMax;                 // maximum solution value
  GLuint texture;               // GL texture

public:
  /* contructor */
  renderer ();
  /* destructor */
  ~renderer ();
  /* init */
  void initRenderer (class solution *sol);
  /* methods */
  void solutionExtrema (class solution *sol);
  void solutionToImage (class solution *sol);
  void initImage (class solution *sol);
  void renderImage ();
};


# ifndef GL_RENDER_ITERATION
  # define GL_RENDER_ITERATION
# endif


/* end */
