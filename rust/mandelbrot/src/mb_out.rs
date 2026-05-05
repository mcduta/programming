pub mod output {

    use image::{ColorType, ImageEncoder};
    use image::codecs::png::PngEncoder;
    use std::fs::File;
    use std::io::BufWriter;

    /// Write the buffer `pixels`, whose dimensions are given by `bounds`, to the
    /// file named `filename`.
    pub fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<(), image::ImageError>
    {
        let output = File::create(filename)?;
        let writer = BufWriter::new(output);

        let encoder = PngEncoder::new(writer);
	
        encoder.write_image(pixels, bounds.0 as u32, bounds.1 as u32, ColorType::L8.into())?;

        Ok(())
    }

}