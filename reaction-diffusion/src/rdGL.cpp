/*

  rd -- finite difference solution of a reaction-diffusion equation
     -- OpenGL version

 */


//====================================================================//
//                                                                    //
//                             includes                               //
//                                                                    //
//====================================================================//

// general
# include "rd.hpp"
# include "solution.hpp"
# include "renderer.hpp"
# include "timer.hpp"


//====================================================================//
//                                                                    //
//                     rendering function headers                     //
//                                                                    //
//====================================================================//

// GL Window dimensions
# define GL_WINDOW_WIDTH  1024
# define GL_WINDOW_HEIGHT  740

// GL functions
void rdInit ();
void GLinitImage (int narg, char **varg);
void GLdisplayImage (void);
void GLreshapeImage (int width, int heigth);
void GLkeyboardImage (unsigned char key, int x, int y);
void GLrenderImage ();
void rdFinalize ();
void rdConfig (std::string configFileName, int *M, int *N);



//====================================================================//
//                                                                    //
//                         global variables                           //
//                                                                    //
//====================================================================//

// ----- global pointer class member
class solution *sol;

// ----- global renderer class member
class renderer *img;



//====================================================================//
//                                                                    //
//                            M  A  I  N                              //
//                                                                    //
//====================================================================//


int main (int narg, char **varg) {

  // configuration file
  std::string config_file = "config.in";
  if (narg > 1) config_file = varg[1];

  // new solution and renderer
  sol = new solution;
  img = new renderer;

  // initialise solution
  try {
    sol -> init (config_file);
  } catch (const std::string errMsg) {
    std::cerr << errMsg << std::endl;
    return EXIT_FAILURE;
  }

  // init GL
  GLinitImage (narg,varg);

  // display
  glutDisplayFunc  (GLdisplayImage);
  glutReshapeFunc  (GLreshapeImage);
  glutKeyboardFunc (GLkeyboardImage);
  glutIdleFunc     (GLrenderImage);
  glutMainLoop     ();

  // cleanup
  delete sol;
  delete img;

  return EXIT_SUCCESS;

}


//====================================================================//
//                                                                    //
//                        rendering function                          //
//                                                                    //
//                                                                    //
//       NB: These functions are not separated from main because      //
//           they are using global variables.  If kept in a separate  //
//           file, these global variables would lead to multiple      //
//           variable definitions.                                    //
//                                                                    //
//====================================================================//


//
// ----- initialisation routine
//

void GLinitImage (int narg, char **varg) {

  // initialise GLUT
  glutInit(&narg, varg);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2  - GL_WINDOW_WIDTH/2,
                          glutGet(GLUT_SCREEN_HEIGHT)/2 - GL_WINDOW_HEIGHT/2);
  glutInitWindowSize(GL_WINDOW_WIDTH, GL_WINDOW_HEIGHT);
  glutCreateWindow("2D Reaction-Diffussion Equation");

  // data initialisation
  img -> initRenderer (sol);

  // init image
  img -> solutionExtrema (sol);
  img -> solutionToImage (sol);
  img -> initImage (sol);
}


//
// ----- display routine
//

void GLdisplayImage (void) {
  img -> renderImage ( );
}


//
// ----- reshape routine
//

void GLreshapeImage (int width, int height) {
  if (width == 0 || height == 0) return;
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  gluPerspective (60.0, (GLfloat) width/(GLfloat) height, 1.0, 20.0);
  glMatrixMode (GL_MODELVIEW);
  glViewport (0,0, (GLsizei) width,(GLsizei) height);
  glLoadIdentity ();
  glTranslatef (0.0, 0.0, -3.6);
}


//
// ----- keyboard routine
//

void GLkeyboardImage (unsigned char key, int x, int y) {
  switch (key) {
  case 'h': // print help
    std::cout << std::endl;
    std::cout << " press q to terminate"           << std::endl;
    std::cout << "       h for help"               << std::endl;
    std::cout << "       s to save data"           << std::endl;
    std::cout << "       k to recalibrate colours" << std::endl;
    std::cout << std::endl;
    break;
  case 's': // dump current solution to disk
    std::cout << " dumping solution to file..." << std::endl;
    sol -> dump ("sol.dat");
    break;
  case 'k': // calibrate solution rendering
    std::cout << " calibrating rendering..." << std::endl;
    img -> solutionExtrema (sol);
    break;
  case 'q': // quit
    exit(0);
    break;
  default:
    break;
  }
}


//
// ----- idle routine
//

void GLrenderImage ( ) {

  auto timer = get_time_start ();

  // go through all iterations at once
  // NB: this is called by the GL idle function, so repeated ad infinitum!
  sol -> iterate (sol->niter);
  img -> solutionToImage (sol);
  img -> renderImage ( );

  auto wtime = get_time_elapsed (timer);
  std::cout << " *** finished " << sol->niter << " iterations in " << wtime << std::endl;

}


/* end */
