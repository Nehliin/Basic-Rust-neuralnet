pub mod image;

use std::fs::File;
use std::io::prelude::*;
use std::io::Result;
use std::io::{BufReader};
use self::image::*;



// use lifetimes instead this implementation is retarded
pub fn load_training_data() -> Result<Vec<(Image, Vec<f32>)>>{
    let label_list = load_training_labels()?;
    let image_list = old_load_training_images()?;
    let mut result = Vec::with_capacity(NUMBER_OF_IMAGES);
    for (image, label) in image_list.iter().zip(label_list) {
        result.push((Image::new(image.get_pixels().clone()), label));
    }
    Ok(result)
}

/*
The labels are actually a int between 0-9 but I return a probability vector that's
easy to use when calculating the cost to the cost function in the back-propagation.
*/
fn load_training_labels() -> Result<Vec<Vec<f32>>> {
    let label_file = File::open("train-labels-idx1-ubyte")?;
    let mut buf = BufReader::with_capacity(NUMBER_OF_IMAGES + 8, label_file); // 8 bytes of metadata
    let mut labes;

    let bytes = buf.fill_buf()?;
    labes = Vec::with_capacity(NUMBER_OF_IMAGES);
    let mut prob_vec = vec![0.0 as f32;10];
    for label in bytes[8..].iter() {
        //println!("{}",*label)
        prob_vec[*label as usize] = 1.0;
        labes.push(prob_vec.clone());
        prob_vec[*label as usize] = 0.0;
    }
    //labes.push(prob_vec.clone());
    Ok(labes)
}

/*
This loads all traningdata images as Image structs from submodule
*/
fn old_load_training_images() -> Result<Vec<Image>> {
    let image_file = File::open("train-images-idx3-ubyte")?;
    let mut buf = BufReader::with_capacity(NUMBER_OF_IMAGES * NUMBER_OF_COLUMNS * NUMBER_OF_ROWS + 16, image_file);
    let mut images;
    let bytes = buf.fill_buf()?;
    images = Vec::with_capacity(NUMBER_OF_IMAGES);
    let mut pixels = Vec::with_capacity(NUMBER_OF_COLUMNS * NUMBER_OF_ROWS);
    for (i, pixel) in bytes[16..].iter().enumerate() { // first 16 bytes are metadata
        if i % (NUMBER_OF_ROWS * NUMBER_OF_COLUMNS) == 0 && i != 0 {
            images.push(Image::new(pixels.clone()));
            pixels.clear();
        }
        pixels.push(*pixel);
    }
    Ok(images)
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn load_test(){
        let data = load_training_data().unwrap();
        let (ref img, ref vec) = data[2];
        img.create();
        println!("{:?}", vec);
        assert_eq!(1.0, vec[4]); //assert it's a 4 (0 is index 0)
        let mut modified = vec.clone();
        modified.remove(4);
        assert_eq!(vec![0.0 as f32; 9], modified);
    }

}