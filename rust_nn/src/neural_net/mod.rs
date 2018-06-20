

pub mod math;

use neural_net::math::Matrix;

#[derive(Debug)]
pub struct NeuralNet {
    weight_matrixes: Vec<Matrix>,
    biases: Vec<Vec<f32>>,
    structure: Vec<usize>
}

impl NeuralNet {

    pub fn new(structure: Vec<usize>) -> NeuralNet{
        let mut biases: Vec<Vec<f32>> = Vec::with_capacity(structure.len()-1); // first layer need no bias
        let mut weight_matrixes: Vec<Matrix> = Vec::with_capacity(structure.len()-1); // last layer have no matrix to feed forward

        for i in 1..structure.len() {
            let mut bias_layer: Vec<f32> = vec![0.0; structure[i]]; // allocate each bias vector for each layer
            biases.push(bias_layer.into_iter().map(|_| math::normally_distributed()).collect()); // fill with normally distributed values

            if let Some(nr_neurons) = structure.get(i) {
                weight_matrixes.push(math::generate_matrix(&structure[i-1], nr_neurons));
            }

        }
        return NeuralNet{
            weight_matrixes,
            biases,
            structure
        }

    }

    //propagets the layer using feedforward and returns the next layers neurons
    pub fn propagate_forward(&self, input: Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if input.len() != self.structure[0] { // if input vector isn't same size as neural_net
            return Err("wrong input vector lenght");
        } else {

            let mut calculated_layer: Vec<f32> = input;
            let mut index = 0;
            for m in &self.weight_matrixes {
                calculated_layer = math::matrix_multiply(m, &calculated_layer)?;
                for i in 0..self.biases[index].len() {
                    calculated_layer[i] += &self.biases[index][i]; // use zip with instead?
                }
                index += 1;
            }
            return Ok(calculated_layer);
        }
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_new() {
        let structure: Vec<usize> = vec![2,3, 5];

        let nn = NeuralNet::new(structure);

        assert_eq!(3, nn.structure.len());
        assert_eq!(2, nn.biases.len());
        assert_eq!(2, nn.weight_matrixes.len());

        assert_ne!(nn.biases[0], nn.biases[1]);

        assert_eq!(2, nn.weight_matrixes[0].getColumns().len()); //number of columns in connecting matrix between layer 0 and 1
        let row_count = 3;
        let mut prev_col = vec![0.0, 0.0, 0.0];
        for col in nn.weight_matrixes[0].getColumns() {
            assert_eq!(row_count, col.len());
            assert_ne!(prev_col, *col);
            prev_col = col.clone();
        }


        assert_eq!(3, nn.weight_matrixes[1].getColumns().len()); //number of columns in connecting matrix between layer 0 and 1
        let row_count = 5;
        let mut prev_col = vec![0.0;5];
        for col in nn.weight_matrixes[1].getColumns() {
            assert_eq!(row_count, col.len());
            assert_ne!(prev_col, *col);
            prev_col = col.clone();
        }
    }


    #[test]
    fn propagate_forward(){

    }
}
