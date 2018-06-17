extern crate rand;

pub mod math;

use neural_net::layer::rand::distributions::{Normal, Distribution};

#[derive(Debug)]
pub struct Layer {
    weight_matrix : Vec<Vec<f64>>,
    neurons: Vec<f64>,
    bias : Vec<f64>,
    next_layer_size : usize,
    input_layer : bool
}

impl Layer {


    // creates a new layer with given size and randomizes weights and biases with a corresponding
    // random weight matrix used to propagate to the connecting layer
    pub fn new(layer_size : usize, next_layer_size: usize, input_layer: bool) -> Layer {

        let mut neurons : Vec<f64> = Vec::with_capacity(layer_size);
        let mut bias : Vec<f64>;
        if !input_layer {
            bias = Vec::with_capacity(layer_size);
        }

        let normal_distribution = Normal::new(0.0, 1.0);

        for x in [0..layer_size] {
            neurons.push( normal_distribution.sample(&mut rand::thread_rng()));
            print!("neuron generated: {}", x);
            if !input_layer {
                bias.push(normal_distribution.sample(&mut rand::thread_rng()));
                print!("bias generated: {}", x);
            }
        }
        
        return Layer{
            weight_matrix,
            neurons,
            bias,
            next_layer_size,
            input_layer
        };

    }

    //propagets the layer using feedforward and returns the next layers neurons
    pub fn propagate_forward(&self) -> Result<Vec<f64>, String> {
        let missing_bias = math::matrix_multiply(&self.weight_matrix, &self.neurons)?;
        let mut result:Vec<f64> = Vec::with_capacity(self.next_layer_size);
        for index in [0..(self.next_layer_size)] {
            result.push(math::sigmoid(missing_bias[index] + &self.bias[index]));
        }
        return Ok(result);
    }

}


