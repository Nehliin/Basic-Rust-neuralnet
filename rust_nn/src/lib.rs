pub mod neural_net;

#[cfg(test)]
mod tests {
    use neural_net::layer::math;
    #[test]
    fn multiply() {
        let matrix: Vec<Vec<f64>> = vec![
            vec![1.0, 1.5, 1.2],
            vec![2.0, -3.0, 4.5],
            vec![1.1, 1.2, -1.4]
        ];
        let wrong_size: Vec<f64> = vec![1.2, 2.3, 3.4, 4.0];
        assert_eq!(true, math::matrix_multiply(&matrix, &wrong_size).is_err());

        let right_size: Vec<f64> = vec![1.2, 2.3, 3.4];
        let answer: Vec<f64> = vec![8.73, 10.800001, -0.6800003];

        let calculated = math::matrix_multiply(&matrix, &right_size);

        let calculated = match calculated {
            Ok(result) => result,
            _ => vec![],
        };
        assert_eq!(answer, calculated);
    }


}