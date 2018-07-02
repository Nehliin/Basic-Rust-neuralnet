extern crate rand;

pub mod matrix;
use neural_net::math::rand::distributions::{Normal, Distribution};
use neural_net::math::matrix::Matrix;
use std::f32::consts::E;

pub fn nabla_bias(activation: &Vec<f32>, z: &Vec<f32>, expected_output: &Vec<f32>) -> Vec<f32>{
    let cost: Vec<f32> = activation.iter()
        .zip(expected_output)
        .map(|(actual, expected)| *actual-*expected).collect();
    vec_sigmoidprime(z).iter().zip(cost).map(| (sigmoid_prime, cost) | 2.0*sigmoid_prime*cost).collect()
}

pub fn nabla_weight(delta_vec: &Vec<f32>, activation_vec: &Vec<f32>) -> Matrix {
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(delta_vec.len());
    let mut row: Vec<f32> = Vec::with_capacity(activation_vec.len());
    for d in delta_vec {
        for a in activation_vec {
            row.push((*d)*(*a));
        }
        rows.push(row.clone());
        row.clear();
    }
    Matrix(rows)
}

pub fn vec_adder(v1 : &Vec<f32>, v2: &Vec<f32>) -> Vec<f32>{
    v1.iter().zip(v2)
        .map(|(u, v)| *u + v).collect()
}

pub fn vec_sub(v1 : &Vec<f32>, v2: &Vec<f32>) -> Vec<f32>{
    v1.iter().zip(v2)
        .map(|(u, v)| *u - v).collect()
}

pub fn vec_get_max(vec: &Vec<f32>) -> usize {
    let mut max = vec[0];
    let mut index = 0;
    for (i, v) in vec.iter().enumerate() {
        if *v > max {
            max = *v;
            index = i;
        }
    }
    index
}

pub fn normally_distributed() -> f32{
    let normal_distribution = Normal::new(0.0, 1.0);
    return normal_distribution.sample(&mut rand::thread_rng()) as f32;
}

//herutian product


pub fn vec_sigmoid(v : &Vec<f32>) -> Vec<f32>{
    v.iter().map(| e |   sigmoid(*e)).collect()
}

pub fn vec_sigmoidprime(z: &Vec<f32>) -> Vec<f32> {
    z.iter().map(| e | sigmoid_prime(*e)).collect()
}

pub fn sigmoid(x : f32) -> f32{
    1.0/(1.0 + E.powf(-x))
}

fn sigmoid_prime(x : f32) -> f32 {
    sigmoid(x)*(1.0-sigmoid(x))
}



#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn sigmoid_test() {
        assert_eq!(1.0, sigmoid(345.234));
        assert_eq!(0.09673856, sigmoid(-2.234));
        assert_eq!(0.99469614, sigmoid(5.234));
        assert_eq!(0.77451790 ,sigmoid(1.234));
        let mut test: Vec<f32> =vec![-2.234,5.234,1.234];
        vec_sigmoid(&mut test);
        assert_eq!(vec![0.09673856, 0.99469614, 0.77451790], test);
    }


    #[test]
    fn vec_max() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![2.0, -2.0, 2.0];
        let v3 = vec![1.0, 3.0, -3.0];
        assert_eq!(2, vec_get_max(&v1));
        assert_eq!(0, vec_get_max(&v2));
        assert_eq!(1, vec_get_max(&v3));
    }

    #[test]
    fn nabla_weights() {
        let delta:Vec<f32> = vec![4.0,5.0];
        let activation:Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = nabla_weight(&delta, &activation);
        assert_eq!(vec![4.0, 8.0, 12.0], result.get_rows()[0]);
        assert_eq!(vec![5.0, 10.0, 15.0], result.get_rows()[1]);
    }

    #[test]
    fn sigmoid_prime_test(){
        let expected: Vec<f32> = vec![0.08281957, 0.1586849, 0.03125247];
        let to_be_calculated: Vec<f32> = vec![2.3, 1.4, -3.4];
        assert_eq!(expected, vec_sigmoidprime(&to_be_calculated))
    }
}

