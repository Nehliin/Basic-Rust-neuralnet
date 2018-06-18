extern crate rand;
use std;
use neural_net::math::rand::distributions::{Normal, Distribution};


pub fn normally_distributed() -> f32{
    let normal_distribution = Normal::new(0.0, 1.0);
    return normal_distribution.sample(&mut rand::thread_rng()) as f32;
}

pub fn generate_matrix(number_of_columns: &usize, number_of_rows: &usize) -> Vec<Vec<f32>> {
    let columns = vec![normally_distributed(); *number_of_rows];
    return vec![columns; *number_of_columns];
}

pub fn sigmoid(x : f32) -> f32{
    return 1.0/(1.0 + std::f32::consts::E.powf(-x));
}

pub fn matrix_multiply(matrix: &Vec<Vec<f32>>, vec: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
    if matrix.len() != vec.len() { //row
        return Err("Missmatch of row lenght and vec to be multiplied")
    } else {
        let mut result = Vec::with_capacity(matrix.len());
        let mut index:usize = 0;
        let mut calc:f32 = 0.0;
        for i in matrix {
            for matrixValue in i {
                calc += matrixValue*vec[index];
                index += 1;
            }
            result.push(calc);
            index = 0;
            calc = 0.0;
        }
        return Ok(result);
    }
}


