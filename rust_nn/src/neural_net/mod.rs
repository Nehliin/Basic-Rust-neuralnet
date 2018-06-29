mod math;

use neural_net::math::Matrix;
use data_loader::*;

#[derive(Debug)]
pub struct NeuralNet {
    weight_matrixes: Vec<Matrix>,
    biases: Vec<Vec<f32>>,
    structure: Vec<usize>
}

impl NeuralNet {

    pub fn generate_new(structure: Vec<usize>) -> NeuralNet{
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

    pub fn new(structure: Vec<usize>, weight_matrixes: Vec<Matrix>, biases: Vec<Vec<f32>>) -> NeuralNet{
        return NeuralNet{
            weight_matrixes,
            biases,
            structure
        }
    }

    //propagets the layer using feedforward and returns the next layers neurons
    pub fn propagate_forward<'a>(&self, input: Vec<f32>) -> Result<Vec<f32>, &'a str> {
        if input.len() != self.structure[0] { // if input vector isn't same size as neural_net
            return Err("wrong input vector lenght");
        }
        let mut calculated_layer: Vec<f32> = input;
        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            calculated_layer = math::matrix_multiply(m, &calculated_layer)?.iter()
                .zip(bias)
                .map(| (v, b) | math::sigmoid(v + b)).collect();
        }
        return Ok(calculated_layer);
    }

    fn backpropagation<'a>(&self, (input, expected_output):(image::Image, Vec<f32>)) -> Result<(), &'a str> {
        let number_of_layers = self.structure.len();
        let mut neurons: Vec<Vec<f32>> = Vec::with_capacity(number_of_layers); //neurons for each layer
        let mut z_vectors: Vec<Vec<f32>> = Vec::with_capacity(number_of_layers); // the z value vector for each layer

        let mut activation = input.get_pixels().iter().map(|a| *a as f32).collect();
        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            let mut z:Vec<f32> = math::matrix_multiply(m, &activation)?.iter()
                .zip(bias)
                .map(| (v, b) | v + b).collect(); //multiplies weights and adds bias
            z_vectors.push(z.clone());
            math::vec_sigmoid(&mut z);
            activation = z;
            neurons.push(activation.clone());
        }

        // räkna ut dC/db
        let mut nabla_biases = Vec::with_capacity(number_of_layers);
        let mut nabla_weights =  Vec::with_capacity(number_of_layers);
        let delta = math::nabla_bias(activation, expected_output);
        nabla_biases.push(delta);
        // dC/dw är  transponat av neuroner * dC/db
        nabla_weights.push(delta.iter().zip(math::transpose(neurons[neurons.len()-2])).map(| (d, n)| d*n).collect());
        // to be continnued

        Ok(())

    }

    pub fn gradient_decent(&self, learning_rate:f32, ) {
        /*let training_data = load_training_data().unwrap();
        // use sum image
        nabla_weights = Matrix::new(number_columns, number_rows);
        nabla_bias =
        for traning_pair in training_data {
            (delta_weights, delta_bias) = backpropagate(traning_pair);
            nabla_weights = nabla_weights.iter().zip(delta_weights)
                .map(|weight  |);

        }*/

    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_generate_new() {
        let structure: Vec<usize> = vec![2,3, 5];

        let nn = NeuralNet::generate_new(structure);

        assert_eq!(3, nn.structure.len());
        assert_eq!(2, nn.biases.len());
        assert_eq!(2, nn.weight_matrixes.len());

        assert_ne!(nn.biases[0], nn.biases[1]);

        assert_eq!(2, nn.weight_matrixes[0].getRows().len()); //number of columns in connecting matrix between layer 0 and 1
        let row_count = 3;
        let mut prev_col = vec![0.0, 0.0, 0.0];
        for col in nn.weight_matrixes[0].getRows() {
            assert_eq!(row_count, col.len());
            assert_ne!(prev_col, *col);
            prev_col = col.clone();
        }


        assert_eq!(3, nn.weight_matrixes[1].getRows().len()); //number of columns in connecting matrix between layer 0 and 1
        let row_count = 5;
        let mut prev_col = vec![0.0;5];
        for col in nn.weight_matrixes[1].getRows() {
            assert_eq!(row_count, col.len());
            assert_ne!(prev_col, *col);
            prev_col = col.clone();
        }
    }


    #[test]
    fn propagate_forward(){
        let structure = vec![2, 3, 5];
        let mut weight_matrixes: Vec<Matrix> = Vec::with_capacity(2);
        weight_matrixes.push(math::Matrix(vec![
            vec![3.0, 1.5],
            vec![2.0, -3.0],
            vec![-1.1, 1.2]
        ]));
        weight_matrixes.push(math::Matrix(vec![
            vec![1.0, 1.5, 1.2],
            vec![2.0, -3.0, 4.5],
            vec![1.1, 1.2, -1.4],
            vec![-0.1, -4.0, -5.5],
            vec![2.3, -3.0, 2.3],
        ]));
        let mut biases: Vec<Vec<f32>> = Vec::with_capacity(2);
        biases.push(vec![-3.0, 2.0, 1.0]);
        biases.push(vec![-2.0, -3.0, 5.0, 5.0, 3.0]);

        let nn = NeuralNet::new(structure, weight_matrixes, biases);

        let answer = vec![0.55528086, 0.9332355, 0.99274254, 0.36103073, 0.99901676];
        let mut failed = false;
        match nn.propagate_forward(vec![1.0, 2.0]) {
            Ok(calc) => assert_eq!(answer, calc),
            _ => failed = true
        }
        assert_eq!(false, failed);

        failed = false;
        match nn.propagate_forward(vec![1.0, 2.0, 3.4]) {
            Err(_) => failed = true,
            _ => ()
        }
        assert_eq!(true, failed);

    }

    #[test]
    fn propagate_backwards() {
        // test with test data
    }


}
