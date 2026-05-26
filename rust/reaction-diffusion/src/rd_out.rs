
pub mod output {

    //
    // --- function that limits its input value to the interval between 0.0 and 1.0
    //
    fn clip (x: f32) -> f32 {
        if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
    }

    //
    // --- simple "jet-like" colormap (blue -> cyan -> green -> yellow -> red)
    //
    fn value_to_rgb (x: f32) -> (u8, u8, u8) {
        let x = clip (x);

        let r = clip (1.5 - (4.0 * (x - 0.75)).abs());
        let g = clip (1.5 - (4.0 * (x - 0.50)).abs());
        let b = clip (1.5 - (4.0 * (x - 0.25)).abs());

        ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    //
    // --- map grid values to colormap buffer
    //
    pub fn grid_to_rgb_buffer (grid: &ndarray::Array2<f32>, buffer: &mut Vec<u32>) {
        use rayon::prelude::*;

        // grid dims
        let (nx, ny) = grid.dim();
        buffer.resize(nx * ny, 0);

        // convert to RGB (multithreaded code)
        buffer
        .par_chunks_mut(ny)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..ny {
                let (r, g, b) = value_to_rgb (grid[(i, j)]);
                row[j] = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            }
        });

        /*
        // convert to RGB (serial code)
        for i in 0..nx {
            for j in 0..ny {
                let (r, g, b) = value_to_rgb (grid[(i, j)]);
                buffer[i * ny + j] = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            }
        }
        */
    }

}