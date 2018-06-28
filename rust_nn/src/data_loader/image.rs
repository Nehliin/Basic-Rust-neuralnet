extern crate image;

pub const NUMBER_OF_IMAGES:usize = 60000;
pub const NUMBER_OF_ROWS:usize = 28;
pub const NUMBER_OF_COLUMNS:usize = 28;

use std::fs::File;

pub struct Image(pub Vec<u8>);

impl Image {

    pub fn new(pixels: Vec<u8>) -> Image {
        Image(pixels)
    }

    pub fn create(&self) {
        let mut image_buffer = image::GrayImage::new(NUMBER_OF_COLUMNS as u32, NUMBER_OF_ROWS as u32);
        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let index:usize = (y as usize)* NUMBER_OF_ROWS + x as usize;
            *pixel = image::Luma([self.0[index]]);
        }
        File::create("img.png").unwrap();
        image_buffer.save("img.png");
    }

}