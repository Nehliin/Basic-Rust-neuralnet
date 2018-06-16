
#[derive(Debug)]
pub struct Layer {
    weight_matrix : Vec<Vec<f64>>,
    neurons: Vec<f64>,
    bias : Vec<f64>
}

impl Layer {

    //propagets the layer using feedforward and returns the next layers neurons
    pub fn propagate_forward(&self) -> Vec<f64> {
        let missing_bias = math::matrix_multiply(&self.weight_matrix, &self.neurons)?;
        let mut result:Vec<f64> = Vec::with_capacity(missing_bias.len());
        let mut index:usize = 0;
        for x in &missing_bias {
            result.push(math::sigmoid(x + &self.bias[index]));
            index += 1;
        }
        return result;
    }

}

pub mod math {
    use std;
    pub fn sigmoid(x : f64) -> f64{
        return 1/(1 + std::f64::consts::E.powf(-x));
    }

    pub fn matrix_multiply(matrix: &Vec<Vec<f64>>, vec: &Vec<f64>) -> Result<Vec<f64>, String> {
        if matrix.len() != vec.len() { //row
            return Err(String::from("Missmatch of row lenght and vec to be multiplied"))
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


}
