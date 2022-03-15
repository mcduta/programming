/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    solution.h -- solution class headers

 */

# include "rd.hpp"


///template <typename REAL>

class solution {

  private:
    std::size_t M,                // number of points in x direction
                N;                // number of points in y direction
    REAL *u, *v;                  // solution storage
    REAL Du, Dv,                  // Gray -- Scott equation
         alpha, beta;             // parameters

  public:
    std::size_t niter;            // number of time iterates

  public:
    /* contructor */
    solution ();
    /* destructor */
    ~solution ();
    /* data init */
    void init (const std::string);
    /* solution iteration */
    void iterate (std::size_t);
    /* solution dump */
    void dump (const std::string);

  // friends
  friend class renderer;
};


/* end */
