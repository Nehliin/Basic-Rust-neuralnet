pub mod neural_net;

#[cfg(test)]
mod tests {
    use neural_net::math;
    #[test]
    fn matrix_multiplication() {
        let matrix: Vec<Vec<f32>> = vec![
            vec![1.0, 1.5, 1.2],
            vec![2.0, -3.0, 4.5],
            vec![1.1, 1.2, -1.4]
        ];
        let wrong_size: Vec<f32> = vec![1.2, 2.3, 3.4, 4.0];
        assert_eq!(true, math::matrix_multiply(&matrix, &wrong_size).is_err());

        let right_size: Vec<f32> = vec![1.2, 2.3, 3.4];
        let answer: Vec<f32> = vec![8.73, 10.800001, -0.6800003];

        let calculated = math::matrix_multiply(&matrix, &right_size);

        let calculated = match calculated {
            Ok(result) => result,
            _ => vec![],
        };
        assert_eq!(answer, calculated);
    }

    #[test]
    fn matrix_generation() {
        let matrix = math::generate_matrix(&2,&3);
        assert_eq!(2, matrix.len());

        let mut prev_col =  vec![2.0;3]; // can never be greater than 1
        for col in matrix {
            assert_eq!(3, col.len());
            assert_ne!(prev_col, col); // check that columns are unique
            let mut prev = 2.0; // a value that can never be taken
            for x in col {
                assert_ne!(prev, x);
                assert!( 0.0 <= x && x <= 1.0);
                prev = x;
            }
            prev_col = col.copy();
        }
    }

    #[test]
    fn sigmoid() {
        assert_eq!(1.0, math::sigmoid(345.234));
        assert_eq!(0.09673856, math::sigmoid(-2.234));
        assert_eq!(0.99469614, math::sigmoid(5.234));
        assert_eq!(0.77451790 ,math::sigmoid(1.234))
    }

    #[test]
    fn propagate_forward() {

    }


}