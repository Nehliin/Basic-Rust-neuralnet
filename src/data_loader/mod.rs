pub mod image;

use std::fs::File;
use std::io::prelude::*;
use std::io::Result;
use std::io::{BufReader};
use self::image::*;
use ndarray::*;
use std::f64::consts::E;


pub fn load_traning_data_new() -> Result<Vec<(Array2<f64>, Array2<f64>)>>{
    let number_of_images = 60000;
    let label_list = new_load_labels("train-labels-idx1-ubyte", number_of_images)?;
    let mut image_list = new_load_images("train-images-idx3-ubyte", number_of_images)?;
    let mut result = Vec::with_capacity(number_of_images);
    //let mut i = 0;
    for (image, label) in image_list.iter_mut().zip(label_list) {
        result.push((image.clone(), label.clone()));
      /*  if i == 43034 {
            let im = Image::new(image.iter_mut().map(|e| *e as u8).collect());
            im.create();
            println!("{}", label);
        }
        i += 1;*/
    }
    Ok(result)
}

pub fn load_test_data_new() -> Result<Vec<(Array2<f64>, Array2<f64>)>> {
    let number_of_images = 10000;
    let label_list = new_load_labels("t10k-labels.idx1-ubyte", number_of_images)?;
    let image_list = new_load_images("t10k-images.idx3-ubyte", number_of_images)?;
    let mut result = Vec::with_capacity(number_of_images);
    for (image, label) in image_list.iter().zip(label_list) {
        result.push((image.clone(), label));
    }
    Ok(result)
}




fn new_load_images(path: &str, number_of_images: usize) -> Result<Vec<Array2<f64>>> {
    let image_file = File::open(path)?;
    let mut buf = BufReader::with_capacity(number_of_images * NUMBER_OF_COLUMNS * NUMBER_OF_ROWS + 16, image_file);
    let bytes = buf.fill_buf()?;
    let mut images = Vec::with_capacity(number_of_images);
    let mut pixels: Array2<f64> = Array2::zeros((NUMBER_OF_COLUMNS * NUMBER_OF_ROWS, 1));
    for (i, pixel) in bytes[16..].iter().enumerate() { // first 16 bytes are metadata
        if i % (NUMBER_OF_ROWS * NUMBER_OF_COLUMNS) == 0 && i != 0 {
            images.push(pixels.clone());
            pixels = Array2::zeros((NUMBER_OF_COLUMNS * NUMBER_OF_ROWS, 1));
        }
        pixels[[i % (NUMBER_OF_ROWS * NUMBER_OF_COLUMNS),0]] = sigmoid(f64::from(*pixel));
    }
    Ok(images)
}

fn new_load_labels(path: &str, number_of_labels: usize) -> Result<Vec<Array2<f64>>> {
    let label_file = File::open(path)?;
    let mut buf = BufReader::with_capacity(number_of_labels + 8, label_file); // 8 bytes of metadata

    let bytes = buf.fill_buf()?;
    let mut labes = Vec::with_capacity(number_of_labels);
    let mut prob_vec = Array::zeros((10, 1));
   // println!("{}", prob_vec);
    for label in bytes[8..].iter() {
        prob_vec[[*label as usize,0]] = 1.0 as f64;
        labes.push(prob_vec.clone());
        prob_vec[[*label as usize, 0]] = 0.0 as f64
    }
    //labes.push(prob_vec.clone());
    Ok(labes)
}
fn sigmoid(x : f64) -> f64{
    1.0/(1.0 + E.powf(-x))
}

#[cfg(test)]
mod tests{
    use super::*;

   /* #[test]
    fn load_test(){
        let data = load_training_data().unwrap();
        let (ref img, ref vec) = data[2];
        img.create();
        println!("{:?}", vec);
        assert_eq!(1.0, vec[4]); //assert it's a 4 (0 is index 0)
        let mut modified = vec.clone();
        modified.remove(4);
        assert_eq!(vec![0.0 as f64; 9], modified);
    }*/

}