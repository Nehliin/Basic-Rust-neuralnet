extern crate rand;
use std;
use neural_net::math::rand::distributions::{Normal, Distribution};

#[derive(Debug)]
pub struct Matrix(pub Vec<Vec<f32>>);

pub fn normally_distributed() -> f32{
    let normal_distribution = Normal::new(0.0, 1.0);
    return normal_distribution.sample(&mut rand::thread_rng()) as f32;
}

pub fn generate_matrix(number_of_columns: &usize, number_of_rows: &usize) -> Matrix {
    let mut matrix = Vec::with_capacity(*number_of_columns);
    for _ in 0..*number_of_columns {
        matrix.push(vec![0.0; *number_of_rows].into_iter().map(|_| normally_distributed()).collect()); // generate column
    }

    return Matrix(matrix);
}

pub fn sigmoid(x : f32) -> f32{
    return 1.0/(1.0 + std::f32::consts::E.powf(-x));
}

pub fn matrix_multiply(matrix: &Matrix, vec: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
    if matrix.0.len() != vec.len() { //row
        return Err("Missmatch of row lenght and vec to be multiplied")
    } else {
        let mut result = Vec::with_capacity(matrix.0.len());
        let mut index:usize = 0;
        let mut calc:f32 = 0.0;
        for i in &matrix.0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_multiplication() {
        let matrix: Matrix = Matrix(vec![
            vec![1.0, 1.5, 1.2],
            vec![2.0, -3.0, 4.5],
            vec![1.1, 1.2, -1.4]
        ]);
        let wrong_size: Vec<f32> = vec![1.2, 2.3, 3.4, 4.0];
        assert_eq!(true, matrix_multiply(&matrix, &wrong_size).is_err());

        let right_size: Vec<f32> = vec![1.2, 2.3, 3.4];
        let answer: Vec<f32> = vec![8.73, 10.800001, -0.6800003];

        let calculated = matrix_multiply(&matrix, &right_size);

        let calculated = match calculated {
            Ok(result) => result,
            _ => vec![],
        };
        assert_eq!(answer, calculated);
    }

    #[test]
    fn matrix_generation() {
        let matrix = generate_matrix(&2,&3);
        assert_eq!(2, matrix.0.len());

        let mut prev_col =  vec![2.0;3]; // can never be greater than 1
        for col in &matrix.0 {
            assert_eq!(3, col.len());
            assert_ne!(prev_col, *col); // check that columns are unique
            let mut prev = 2.0 as f32; // a value that can never be taken
            for x in col {
                assert_ne!(prev, *x);
                println!("{}",*x);
                assert!( -5.0 <= *x && *x <= 5.0);
                prev = *x;
            }
            prev_col = col.clone();
        }
    }

    #[test]
    fn sigmoid_test() {
        assert_eq!(1.0, sigmoid(345.234));
        assert_eq!(0.09673856, sigmoid(-2.234));
        assert_eq!(0.99469614, sigmoid(5.234));
        assert_eq!(0.77451790 ,sigmoid(1.234))
    }
}

