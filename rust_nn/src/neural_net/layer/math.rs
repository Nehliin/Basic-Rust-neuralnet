
use std;
pub fn sigmoid(x : f64) -> f64{
    return 1.0/(1.0 + std::f64::consts::E.powf(-x));
}

pub fn matrix_multiply(matrix: &Vec<Vec<f64>>, vec: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
    if matrix.len() != vec.len() { //row
        return Err("Missmatch of row lenght and vec to be multiplied")
    } else {
        let mut result = Vec::with_capacity(matrix.len());
        let mut index:usize = 0;
        let mut calc:f64 = 0.0;
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


