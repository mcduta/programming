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
  public:
    unsigned char *data;   // data in RGB format
    GLuint texture;        // GL texture
  private:
    REAL dataMin;          // minimum solution value
    REAL dataMax;          // maximum solution value
    GLfloat maxh=1.0,      // maximum and minumum
            maxw=1.0;      // texture horz and vert span

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
  void initImage       (class solution *sol);
  void renderImage ();
};


# ifndef GL_RENDER_ITERATION
  # define GL_RENDER_ITERATION
# endif


/* end */
