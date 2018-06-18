

pub mod math;


#[derive(Debug)]
struct NeuralNet {
    weight_matrixes: Vec<Vec<Vec<f32>>>, // list of matrixes
    biases: Vec<Vec<f32>>,
    number_of_layers: usize
}

impl NeuralNet {

    pub fn new(structure: Vec<usize>) -> NeuralNet{
        let mut biases: Vec<Vec<f32>> = Vec::with_capacity(structure.len()-1); // first layer need no bias
        let mut weight_matrixes: Vec<Vec<Vec<f32>>> = Vec::with_capacity(structure.len()-1); // last layer have no matrix to feed forward
        let number_of_layers = structure.len();

        for i in 1..number_of_layers {
            let mut bias_layer: Vec<f32> = vec![0.0; structure[i]]; // allocate each bias vector for each layer
            biases.push(bias_layer.into_iter().map(|_| math::normally_distributed()).collect()); // fill with normally distributed values

            if let Some(nr_neurons) = structure.get(i+1) {
                weight_matrixes.push(math::generate_matrix(&structure[i], nr_neurons));
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