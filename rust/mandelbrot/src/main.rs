mod mb_inp;   pub use crate::mb_inp::input;
mod mb_tools; pub use crate::mb_tools::tools;
mod mb_out;   pub use crate::mb_out::output;

use std::env;
use std::time::Instant;


fn main() {

    //
    // ... process input 
    let args: Vec<String> = env::args().collect();

    if args.len() != 6 {
        eprintln! ("Usage: {} FILE PIXELS UPPERLEFT LOWERRIGHT POLICY", args[0]);
        eprintln! ("Example: {} mandel.png 1000x750 -1.20,0.35 -1,0.20 bands", args[0]);
        eprintln! ("         policy = single | bands | queue");
        std::process::exit (1);
    }

    let bounds = input::parse_pair::<usize> (&args[2], 'x')
        .expect ("error parsing image dimensions");
    let upper_left = input::parse_complex (&args[3])
        .expect ("error parsing upper left corner point");
    let lower_right = input::parse_complex (&args[4])
        .expect ("error parsing lower right corner point");
    let policy = &args[5];

    //
    // ... generate image
    let mut pixels = vec![0; bounds.0 * bounds.1];
    let time_start = Instant::now();
    tools::render_image (&mut pixels, bounds, upper_left, lower_right, policy);
    let time_elapsed = time_start.elapsed();
    println!("Time elapsed: {:?}", time_elapsed);


    //
    // ... save image
    output::write_image (&args[1], &pixels, bounds)
       .expect("error writing PNG file");
}
