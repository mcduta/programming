use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "rd")]
#[command(about = "Demo of 2D reaction-diffusion equation.",
 long_about = "Time dependent solution of the 2D Scott-Gray equation with periodic boundary conditions, solved on a grid of specified dimensions for a specified numbee of iterations.")
 ]
//
// --- input (command line args parsing)
//
pub struct Input {

    /// grid size in X direction
    #[arg(long)]
    pub nx: usize,

    /// grid size in Y direction
    #[arg(long)]
    pub ny: usize,

    /// number of iterations
    #[arg(long, default_value_t = 1000)]
    pub nt: usize,

    /// equation parameters file name
    pub filename: String,

}