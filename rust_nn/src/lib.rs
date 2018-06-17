pub mod neural_net;

#[cfg(test)]
mod tests {
    use neural_net::layer::math;
    #[test]
    fn matrix_multiplication() {
        let matrix: Vec<Vec<f64>> = vec![
            vec![1.0, 1.5, 1.2],
            vec![2.0, -3.0, 4.5],
            vec![1.1, 1.2, -1.4]
        ];
        let wrong_size: Vec<f64> = vec![1.2, 2.3, 3.4, 4.0];
        assert_eq!(true, math::matrix_multiply(&matrix, &wrong_size).is_err());

        let right_size: Vec<f64> = vec![1.2, 2.3, 3.4];
        let answer: Vec<f64> = vec![8.73, 10.799999999999999, -0.6799999999999997];

        let calculated = math::matrix_multiply(&matrix, &right_size);

        let calculated = match calculated {
            Ok(result) => result,
            _ => vec![],
        };
        assert_eq!(answer, calculated);
    }

    #[test]
    fn sigmoid() {
        assert_eq!(1.0, math::sigmoid(345.234));
        assert_eq!(0.09673855591916831, math::sigmoid(-2.234));
        assert_eq!(0.9946961270943794, math::sigmoid(5.234));
        assert_eq!(0.7745179009360356 ,math::sigmoid(1.234))
    }

    #[test]
    fn propagate_forward() {

    }

    #[test]
    fn layer_generation() {

    }

}