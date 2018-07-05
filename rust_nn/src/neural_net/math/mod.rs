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
        let delta:Vec<f32> = vec![ 0.00418704095,
         0.00557962314,
         0.102673455,
         6.52345275e-02,
        2.65370253e-05,
        1.15101899e-01,
        4.95149885e-04,
        1.65037705e-03,
        5.46378319e-02,
        -5.80916191e-05];
        let activation:Vec<f32> = vec![0.83689182, 0.71944723, 0.62745113, 0.00615586, 0.79145588, 0.99963068,
        0.02803344, 0.9684408,  0.64020823, 0.19694865, 0.98593323, 0.89733453,
        0.98709725, 0.03776075, 0.40200459, 0.9515954 ];

        [[ 3.50410033e-02  3.01235501e-02  2.62716357e-02  2.57748319e-04
        3.31385817e-02  4.18549459e-02  1.17377174e-03  4.05490126e-02
        2.68057808e-02  8.24632073e-03  4.12814280e-02  3.75717643e-02
        4.13301661e-02  1.58105796e-03  1.68320968e-02  3.98436889e-02]
        [ 4.66954098e-02  4.01424442e-02  3.50094084e-02  3.43473710e-04
        4.41602554e-02  5.57756248e-02  1.56416048e-03  5.40353467e-02
        3.57212066e-02  1.09889926e-02  5.50113586e-02  5.00678852e-02
        5.50763066e-02  2.10690741e-03  2.24303412e-02  5.30954370e-02]
        [ 8.59265745e-02  7.38681325e-02  6.44225749e-02  6.32043266e-04
        8.12615093e-02  1.02635535e-01  2.87829046e-03  9.94331620e-02
        6.57323908e-02  2.02213986e-02  1.01229171e-01  9.21324364e-02
        1.01348685e-01  3.87702639e-03  4.12752001e-02  9.77035867e-02]
        [ 5.45942426e-02  4.69328002e-02  4.09314779e-02  4.01574526e-04
        5.16302503e-02  6.52104351e-02  1.82874842e-03  6.31757777e-02
        4.17636815e-02  1.28478523e-02  6.43168883e-02  5.85371943e-02
        6.43928228e-02  2.46330452e-03  2.62245796e-02  6.20768761e-02]
        [ 2.22086194e-05  1.90919893e-05  1.66506864e-05  1.63358175e-07
        2.10028847e-05  2.65272246e-05  7.43924190e-07  2.56995378e-05
        1.69892220e-05  5.22643137e-06  2.61637350e-05  2.38125892e-05
        2.61946247e-05  1.00205791e-06  1.06680060e-05  2.52525111e-05]
        [ 9.63278379e-02  8.28097424e-02  7.22208163e-02  7.08551012e-04
        9.10980746e-02  1.15059389e-01  3.22670254e-03  1.11469374e-01
        7.36891831e-02  2.26691639e-02  1.13482787e-01  1.03284909e-01
        1.13616768e-01  4.34633374e-03  4.62714918e-02  1.09530437e-01]
        [ 4.14386889e-04  3.56234213e-04  3.10682354e-04  3.04807267e-06
        3.91889287e-04  4.94967016e-04  1.38807562e-05  4.79523348e-04
        3.16999032e-04  9.75191027e-05  4.88184724e-04  4.44315090e-04
        4.88761090e-04  1.86972298e-05  1.99052527e-04  4.71182351e-04]
        [ 1.38118705e-03  1.18735919e-03  1.03553094e-03  1.01594877e-05
        1.30620062e-03  1.64976753e-03  4.62657511e-05  1.59829246e-03
        1.05658497e-03  3.25039536e-04  1.62716157e-03  1.48094031e-03
        1.62908264e-03  6.23194709e-05  6.63459149e-04  1.57049120e-03]
        [ 4.57259547e-02  3.93090368e-02  3.42825693e-02  3.36342766e-04
        4.32434333e-02  5.46176531e-02  1.53168656e-03  5.29135054e-02
        3.49795897e-02  1.07608474e-02  5.38692540e-02  4.90284133e-02
        5.39328537e-02  2.06316537e-03  2.19646593e-02  5.19931093e-02]
        [-4.86164009e-05 -4.17938545e-05 -3.64496519e-05 -3.57603792e-07
            -4.59769534e-05 -5.80701647e-05 -1.62850810e-06 -5.62582938e-05
            -3.71907327e-05 -1.14410661e-05 -5.72744575e-05 -5.21276158e-05
            -5.73420775e-05 -2.19358296e-06 -2.33530976e-05 -5.52797173e-05]]

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

