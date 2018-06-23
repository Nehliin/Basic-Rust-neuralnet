extern crate image;

use std::fs::File;
use std::io::prelude::*;
use std::io::Result;
use std::io::{self, BufReader};

const number_of_images:usize = 60000;
const number_of_rows:usize = 28;
const number_of_columns:usize = 28;

struct Image(Vec<u8>);

impl Image {
    fn create(&self) {
        let mut image_buffer = image::GrayImage::new(number_of_columns as u32, number_of_rows as u32);
        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let index:usize = (y as usize)*number_of_rows + x as usize;
            *pixel = image::Luma([self.0[index]]);
        }
        File::create("img.png").unwrap();
        image_buffer.save("img.png");
    }

}

fn load() -> Result<()> {
    let mut image_file = File::open("train-images-idx3-ubyte")?;
    let mut buf = BufReader::with_capacity(number_of_images*number_of_columns*number_of_rows, image_file);
    if let Ok(bytes) = buf.fill_buf() {
        println!("Start");
        let mut images:Vec<Image> = Vec::with_capacity(number_of_images);
        let mut pixels:Vec<u8> = Vec::with_capacity(number_of_columns*number_of_rows);
        for (i, pixel) in bytes[16..].iter().enumerate() { // first 16 bytes are metadata
            if i % (number_of_rows*number_of_columns) == 0 {
                images.push(Image(pixels.clone()));
                pixels.clear();
            }
            pixels.push(*pixel);
        }
        println!("Done");
        images[6].create();
    }

    Ok(())
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn loadTest(){
        load();
    }

}