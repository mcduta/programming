//
// rd -- implements a solution for the reaction-diffusion type Gray-Scott equation
//

//
// --- public crates
//
use std;
use serde;
use clap::Parser;


//
// --- local crate
//
mod rd_inp;
mod rd_sol;
mod rd_out;


//
// --- Gray-Scott equation parameters
//
#[derive(Debug, serde::Deserialize)]
struct Config {
    du:    f32,   // diffusion coefficient for chemical species u
    dv:    f32,   // diffusion coefficient for chemical species v
    alpha: f32,   // feed rate
    beta:  f32,   // kill rate
}


//
// --- main
//
fn main() -> Result<(), Box<dyn std::error::Error>> {

    // --- parse command line input
    let args = rd_inp::Input::parse();

    // --- read config (equation parameters) from file specified from command line
    let json_text = std::fs::read_to_string(args.filename)?;
    let config: Config = serde_json::from_str(&json_text)?;

    // --- create new solver from command line arguments (grid size) and config file (equation parameters)
    let mut solver = rd_sol::Solver::new (args.nx, args.ny, config.du, config.dv, config.alpha, config.beta);

    // --- initial conditions
    solver.init();

    // --- solve solution (time-step iterate) and plot at every iteration
    //       * "solver.ploterate(args.nt);" plots solution at every iteration
    //       * "solver.iterate(args.nt);" only iterates the solution
    let ts = std::time::Instant::now();
    solver.ploterate(args.nt);
    let te = ts.elapsed();
    println!(" ... iterations: {} sec", te.as_secs_f64());

    Ok(())
}
