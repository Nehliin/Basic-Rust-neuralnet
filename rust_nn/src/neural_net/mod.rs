

pub mod math;

use neural_net::math::Matrix;

#[derive(Debug)]
pub struct NeuralNet {
    weight_matrixes: Vec<Matrix>,
    biases: Vec<Vec<f32>>,
    number_of_layers: usize
}

impl NeuralNet {

    pub fn new(structure: Vec<usize>) -> NeuralNet{
        let mut biases: Vec<Vec<f32>> = Vec::with_capacity(structure.len()-1); // first layer need no bias
        let mut weight_matrixes: Vec<Matrix> = Vec::with_capacity(structure.len()-1); // last layer have no matrix to feed forward
        let number_of_layers = structure.len();

        for i in 1..number_of_layers {
            let mut bias_layer: Vec<f32> = vec![0.0; structure[i]]; // allocate each bias vector for each layer
            biases.push(bias_layer.into_iter().map(|_| math::normally_distributed()).collect()); // fill with normally distributed values

            if let Some(nr_neurons) = structure.get(i) {
                weight_matrixes.push(math::generate_matrix(&structure[i-1], nr_neurons));
            }

        }
        return NeuralNet{
            weight_matrixes,
            biases,
            number_of_layers
        }

    }

    //propagets the layer using feedforward and returns the next layers neurons
    pub fn propagate_forward(&self) -> Result<Vec<f64>, &'static str> {
      //  let missing_bias = math::matrix_multiply(&self.weight_matrix, &self.neurons)?;
       // let mut result:Vec<f64> = Vec::with_capacity(self.next_layer_size);
        //for index in [0..(self.next_layer_size)] {
        //    result.push(math::sigmoid(missing_bias[index] + &self.bias[index]));
        //}
        return Err("penis");
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_new() {
        let structure: Vec<usize> = vec![2,3, 5];

        let nn = NeuralNet::new(structure);

        assert_eq!(3, nn.number_of_layers);
        assert_eq!(2, nn.biases.len());
        assert_eq!(2, nn.weight_matrixes.len());

        assert_ne!(nn.biases[0], nn.biases[1]);

        assert_eq!(2, nn.weight_matrixes[0].0.len()); //number of columns in connecting matrix between layer 0 and 1
        let row_count = 3;
        let mut prev_col = vec![0.0, 0.0, 0.0];
        for col in &nn.weight_matrixes[0].0 {
            assert_eq!(row_count, col.len());
            assert_ne!(prev_col, *col);
            prev_col = col.clone();
        }


        assert_eq!(3, nn.weight_matrixes[1].0.len()); //number of columns in connecting matrix between layer 0 and 1
        let row_count = 5;
        let mut prev_col = vec![0.0;5];
        for col in &nn.weight_matrixes[1].0 {
            assert_eq!(row_count, col.len());
            assert_ne!(prev_col, *col);
            prev_col = col.clone();
        }
    }
}
