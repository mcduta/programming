use ndarray::{Array2, Axis};


//                          //
// ===== solver class ===== //
//                          //

//
// === data
//
pub struct Solver {
    nx: usize,          // grid size in x direction
    ny: usize,          // grid size in y direction

    du:    f32,         // diffusion coefficient for chemical species u
    dv:    f32,         // diffusion coefficient for chemical species v
    alpha: f32,         // feed rate
    beta:  f32,         // kill rate

    u:  Array2<f32>,    // grid storage for chemical species u
    v:  Array2<f32>,    // grid storage for chemical species v
    un: Array2<f32>,    // grid swap copy of u
    vn: Array2<f32>,    // grid swap copy of v
}

//
// === implementation
//
impl Solver {

    //
    // --- create object
    //
    pub fn new (nx: usize, ny: usize, du: f32, dv: f32, alpha: f32, beta: f32) -> Self {

        // grid storage
        let u = Array2::<f32>::zeros((nx, ny));
        let v = Array2::<f32>::zeros((nx, ny));
    
        // grid swap copies
        let un = Array2::<f32>::zeros((nx, ny));
        let vn = Array2::<f32>::zeros((nx, ny));

        // return Solver variable
        Solver {
            nx,
            ny,
            du,
            dv,
            alpha,
            beta,
            u,
            v,
            un,
            vn,
        }
    }


    //
    // --- initialise spots
    //
    pub fn init (&mut self) {
        // random numbers crate
        use rand;
        use rand_distr::{Normal, Distribution};

        // grid dimensions
        let nx = self.nx;
        let ny = self.ny;
    
        // check
        assert!(nx >= 3 && ny >= 3);

        // initialise all u on 1.0 (v is already 0 from default)
        self.u.par_mapv_inplace(|_| 1.0);

        // instantiate random number generator
        let mut rng = rand::rng();

        // instantiate normal distribution
        let normal = Normal::new(1.00, 0.25).unwrap();

        // number of initial spots
        let ns: usize = rand::random_range(1..=10);

        // initialise u and v at random spots
        for _ in 1..ns {
            // generate spot coordinates and spot size
            let xs:usize = rand::random_range(0..nx);
            let ys:usize = rand::random_range(0..ny);
            let ss:usize = rand::random_range(1..std::cmp::max(1, std::cmp::min(nx, ny)/10));
            // radius of spot squared
            let ss2 = 1 + ss*ss/4;
            // index limits
            let is_min = std::cmp::max(   1, xs-ss);
            let is_max = std::cmp::min(nx-1, xs+ss);
            let js_min = std::cmp::max(   1, ys-ss);
            let js_max = std::cmp::min(ny-1, ys+ss);
            // initialise chemical species at the spots
            for i in is_min..is_max {
                for j in js_min..js_max {
                    // index distance from spot center
                    let imxs: i64 = (i as i64) - (xs as i64);
                    let jmys: i64 = (j as i64) - (ys as i64);
                    // if within spot
                    if imxs*imxs + jmys*jmys < (ss2 as i64) {
                        self.u[(i, j)] = 0.50 * normal.sample(&mut rng);
                        self.v[(i, j)] = 0.25 * normal.sample(&mut rng);
                    }
                }
            }
        }

    }


    //
    // --- update grid point field values from the reaction-diffusion equation dynamics
    //
    fn update (&mut self) {
        // parallelise loops using rayon
        use ndarray::parallel::prelude::*;

        // grid dimensions
        let nx = self.nx;
        let ny = self.ny;
    
        // check
        assert!(nx >= 3 && ny >= 3);

        // capture immutable references for thread-safe closure usage
        let u     = &self.u;
        let v     = &self.v;
        let du    = self.du;
        let dv    = self.dv;
        let alpha = self.alpha;
        let beta  = self.beta;

        // parallel updates of interior region
        let mut un_view = self.un.slice_mut (ndarray::s![1..nx - 1, 1..ny - 1]);
        let mut vn_view = self.vn.slice_mut (ndarray::s![1..nx - 1, 1..ny - 1]);

        // loop over all grid nodes in interior region
        un_view
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(vn_view.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(ii, (mut un_row, mut vn_row))| {
                let i = ii + 1; // add an offset of 1 to account for the boundary

                for j in 1..ny - 1 {
                    // grid point field values
                    let u_c = u[(i, j)];
                    let v_c = v[(i, j)];
                    // laplacian values (5-point stencil)
                    let lu = u[(i - 1, j)] + u[(i + 1, j)] + u[(i, j - 1)] + u[(i, j + 1)] - 4.0 * u_c;
                    let lv = v[(i - 1, j)] + v[(i + 1, j)] + v[(i, j - 1)] + v[(i, j + 1)] - 4.0 * v_c;
                    // common term calculated once
                    let uvv = u_c * v_c * v_c;
                    // update grid point field values using a finite difference approxiation of the Scott-Gray equations
                    un_row[j - 1] = u_c + du * lu - uvv + alpha * (1.0 - u_c);
                    vn_row[j - 1] = v_c + dv * lv + uvv - (alpha + beta) * v_c;
                }
            });

        // periodic boundary conditions (sequential)
        // ... top/bottom rows
        for j in 0..ny {
            self.un[(0, j)] = self.un[(nx - 2, j)];
            self.vn[(0, j)] = self.vn[(nx - 2, j)];

            self.un[(nx - 1, j)] = self.un[(1, j)];
            self.vn[(nx - 1, j)] = self.vn[(1, j)];
        }

        // ... left/right columns
        for i in 0..nx {
            self.un[(i, 0)] = self.un[(i, ny - 2)];
            self.vn[(i, 0)] = self.vn[(i, ny - 2)];

            self.un[(i, ny - 1)] = self.un[(i, 1)];
            self.vn[(i, ny - 1)] = self.vn[(i, 1)];
        }

        // swap grid storage
        std::mem::swap(&mut self.u, &mut self.un);
        std::mem::swap(&mut self.v, &mut self.vn);
    }


    //
    // --- iterator wrapper function
    //
    pub fn iterate (&mut self, nt: usize) {
        for _ in 1..nt {
            self.update();
        }
    }


    //
    // --- iterator wrapper function that plots the grid values for chemical species u
    //
    pub fn ploterate (&mut self, nt: usize) {
        use minifb::{Key, Window, WindowOptions};
        use crate::rd_out;

        // plot framebuffer
        let mut buffer = vec![0u32; self.nx * self.ny];

        // open window
        let mut window = Window::new(
            "Reaction-Diffusion Iteration Viewer (ESC to quit)",
            self.ny,
            self.nx,
            WindowOptions::default(),
        ).unwrap();

        let mut it: usize = 0; 
        while it<nt { //&& window.is_open() && !window.is_key_down(Key::Escape) {
            self.update();
            rd_out::output::grid_to_rgb_buffer(&self.u, &mut buffer);
            it += 1;
            window
                .update_with_buffer(&buffer, self.ny, self.nx)
                .unwrap();
        }

        while window.is_open() && !window.is_key_down(Key::Escape) { }
    }

}
