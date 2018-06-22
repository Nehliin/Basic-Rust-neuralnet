use std::fs::File;
use std::io::prelude::*;
use std::io::Result;
use std::io::{self, BufReader};

struct Image(Vec<Vec<u8>>);

impl Image {
    fn draw(&self) {
        for row in &self.0 {
            println!("{:?}",row);
        }
    }
}

fn load() -> Result<()> {
    let mut image_file = File::open("train-images-idx3-ubyte")?;
    const number_of_images:usize = 60000;
    const number_of_rows:usize = 28;
    const number_of_columns:usize = 28;
    let mut buf = BufReader::with_capacity(number_of_images*number_of_columns*number_of_rows, image_file);
  //  let mut buf =   [0;number_of_images*number_of_columns*number_of_rows];
  //  image_file.read(buf.as_mut())?;
    if let Ok(bytes) = buf.fill_buf() {
        print!("hej");
        let mut images:Vec<Image> = Vec::with_capacity(number_of_images);
        let mut rows = Vec::with_capacity(number_of_rows);
        let mut row = Vec::with_capacity(number_of_columns);
        print!("start");
        for (i, pixel) in bytes[16..].iter().enumerate() { // first 16 bytes are meta-data
            if i % number_of_rows*number_of_columns == 0 {
                println!("första {}", i % number_of_rows*number_of_columns);
                images.push(Image(rows.clone()));
                rows.clear();
                println!("images len, {}", images.len());
                if images.len() == 3 {
                    break;
                }
            }
            row.push(*pixel);
            if i % number_of_rows == 0 {
                println!("andra {}", i % number_of_rows);
                rows.push(row.clone());
                row.clear();
            }
        }
        images[2].draw();
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