use ndarray;
use ndarray::parallel::prelude::*;
use std::{mem, time};
use clap::Parser;

//
// --- input (command line args parsing)
//
#[derive(Parser, Debug)]
#[command(name = "jacobi")]
#[command(about = "Demo of 2D Jacobi update.", long_about = None)]
//    long_about = "Runs a 2D Jacobi update on a grid of specified dimensions
//                  for a specified numbe of iterations. Performance (GFlops)
//                  is reported.")
//    ]
struct CmdLineArgs {
    /// grid size in X direction
    #[arg(long)]
    nx: usize,

    /// grid size in Y direction
    #[arg(long)]
    ny: usize,

    /// number of iterations
    #[arg(long, default_value_t = 1000)]
    nt: usize,
}


//
// --- Jacobi update
//
pub fn jac_iter (old: &ndarray::Array2<f32>, new: &mut ndarray::Array2<f32>) {
    // minimal check
    assert_eq!(old.dim(), new.dim());
    // dimensions
    let (nx, ny) = old.dim();
    // at least one interior point needed
    assert!(nx >= 3 && ny >= 3);

    // copy boundaries unchanged
    new.row_mut(0).assign(&old.row(0));
    new.row_mut(nx - 1).assign(&old.row(nx - 1));
    new.column_mut(0).assign(&old.column(0));
    new.column_mut(ny - 1).assign(&old.column(ny - 1));

    // Jacobi update on interior grid points
    ndarray::Zip::from(new.slice_mut(ndarray::s![1..nx - 1, 1..ny - 1]))
                 .and(old.slice(ndarray::s![0..nx - 2, 1..ny - 1])) // up
                 .and(old.slice(ndarray::s![2..nx,     1..ny - 1])) // down
                 .and(old.slice(ndarray::s![1..nx - 1, 0..ny - 2])) // left
                 .and(old.slice(ndarray::s![1..nx - 1, 2..ny    ])) // right
                 .for_each(|new_ij, &up, &down, &left, &right| {
            *new_ij = 0.25 * (up + down + left + right);
        });
}


//
// --- Jacobi update (multithreaded version)
//
pub fn jac_iter_par (old: &ndarray::Array2<f32>, new: &mut ndarray::Array2<f32>) {
    // minimal check
    assert_eq!(old.dim(), new.dim());
    // dimensions
    let (nx, ny) = old.dim();
    // at least one interior point needed
    assert!(nx >= 3 && ny >= 3);

    // copy boundaries unchanged (sequential)
    new.row_mut(0).assign(&old.row(0));
    new.row_mut(nx - 1).assign(&old.row(nx - 1));
    new.column_mut(0).assign(&old.column(0));
    new.column_mut(ny - 1).assign(&old.column(ny - 1));

    // Jacobi update on interior grid points (parallel)
    new.slice_mut(ndarray::s![1..nx - 1, 1..ny - 1])
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(ii, mut row)| {
            // ii is 0-based slice-local indexing, so shift by +1 for global i indexing
            let i = ii + 1;

            for j in 1..ny - 1 {
                row[j - 1] = 0.25
                    * (old[(i - 1, j)] + old[(i + 1, j)] +
                       old[(i, j - 1)] + old[(i, j + 1)]);
            }
        });
    }


//
// --- main
//
fn main() {

    // --- parse input
    let args = CmdLineArgs::parse();
    let nx = args.nx;
    let ny = args.ny;
    let nt = args.nt;

    // --- sizes
    let mut u  = ndarray::Array2::<f32>::zeros((nx, ny));
    let mut u2 = ndarray::Array2::<f32>::zeros((nx, ny));

    // --- memory usage
    println!(" ... {nx}X{ny} matrix ({} GB)", ((mem::size_of::<f32>() * u.len()) as f64) / (1024.0 * 1024.0 * 1024.0));

    // Example boundary condition: top row = 1.0
    let mut ts = time::Instant::now();
    u.row_mut(0).fill(1.0);
    u2.row_mut(0).fill(1.0);
    let mut te = ts.elapsed();
    println!(" ... initialisation: {} sec", te.as_secs_f64());

    // Run a few Jacobi iterations
    ts = time::Instant::now();
    for _ in 0..nt {
        jac_iter_par(&u, &mut u2);
        std::mem::swap(&mut u, &mut u2);
    }
    te = ts.elapsed();
    println!(" ... iterations: {} sec @ {} Gflops", te.as_secs_f64(), 4.0*((nt*(nx-2)*(ny-2)) as f64)*1.0e-9/te.as_secs_f64());

}
