mod math;

use neural_net::math::matrix::Matrix;
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
                weight_matrixes.push(math::matrix::generate_matrix(&structure[i-1], nr_neurons));
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
    pub fn propagate_forward<'a>(&self, input: image::Image) -> Result<Vec<f32>, &'a str> {

        let mut calculated_layer: Vec<f32> = input.get_pixels().iter().map(|a| *a as f32).collect();
        if calculated_layer.len() != self.structure[0] { // if input vector isn't same size as neural_net
            return Err("wrong input vector lenght");
        }

        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            calculated_layer = math::matrix::matrix_multiply(m, &calculated_layer)?.iter()
                .zip(bias)
                .map(| (v, b) | math::sigmoid(v + b)).collect();
        }
        return Ok(calculated_layer);
    }

    pub fn backpropagation<'a>(&self, (input, expected_output):(image::Image, Vec<f32>)) -> Result<(Vec<Matrix>, Vec<Vec<f32>>), &'a str> {

        let number_of_layers = self.structure.len();
        let mut neurons: Vec<Vec<f32>> = Vec::with_capacity(number_of_layers); //neurons for each layer
        let mut z_vectors: Vec<Vec<f32>> = Vec::with_capacity(number_of_layers); // the z value vector for each layer

        // feeds forward
        let mut activation: Vec<f32> = input.get_pixels().iter().map(|a| *a as f32).collect();
        neurons.push(activation.clone());

        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            let mut z:Vec<f32> = math::matrix::matrix_multiply(m, &activation)?.iter()
                .zip(bias)
                .map(| (v, b) | v + b).collect(); //multiplies weights and adds bias
            z_vectors.push(z.clone());
            activation = math::vec_sigmoid(&z);
         //   println!("ACTIVATION: {:?}", activation);
            neurons.push(activation.clone());
        }
        //println!("Z vectors {:?}", z_vectors);
        // calculate partial derivatives
       // println!("activation {:?}", &activation);
        let mut nabla_biases = Vec::with_capacity(number_of_layers-1);
        let mut nabla_weights: Vec<Matrix> =  Vec::with_capacity(number_of_layers-1);
        let mut delta = math::nabla_bias(&activation, &z_vectors[z_vectors.len()-1], &expected_output);
        nabla_biases.push(delta.clone()); // måste bli reversed
        // dC/dw är  transponat av neuroner * dC/db
        //println!("DELTA {:?}", delta.len());
        //println!("1 {:?}", math::nabla_weight(&delta, &neurons[neurons.len()-2]));
        nabla_weights.push(math::nabla_weight(&delta, &neurons[neurons.len()-2])); // måste reverseras
        //println!("Matrixs dim: rows {}, row len {}", nabla_weights[0].get_rows().len(), nabla_weights[0].get_rows()[0].len());
        println!("NEURONS: {:?}", &neurons);
        for l in 2..self.structure.len() {
            let mut z = z_vectors[(z_vectors.len()-l)].clone();
            println!("NEW Z {:?}", z);
            let mut sp = math::vec_sigmoidprime(&z);
            println!("prime {:?}", sp);
            if let Ok(result) = math::matrix::matrix_multiply(&self.weight_matrixes[self.weight_matrixes.len()-l+1].transpose(), &delta) {
                delta = result.iter().zip(sp).map(| (r, s) | *r*s).collect();
            } else {
                println!("PANIK");
            }
            println!("delta: {:?}", delta);
            println!("activation {:?}", &neurons[neurons.len()-l-1]);
            nabla_biases.push(delta.clone());
            println!("{}", math::nabla_weight(&delta, &neurons[neurons.len()-l-1]));
            nabla_weights.push(math::nabla_weight(&delta, &neurons[neurons.len()-l-1]));
        }

        nabla_weights.reverse();
        nabla_biases.reverse();
        //println!("{:?}", nabla_weights);
        Ok((nabla_weights, nabla_biases))
    }

    pub fn gradient_decent<'a>(&'a mut self, learning_rate:f32 ) -> Result<(), &'a str> {
        let training_data = load_training_data().unwrap();
        // use sum image
        let mut i:f32 = 0.0;
        let mut nabla_weights:Vec<Matrix> = Vec::with_capacity(self.weight_matrixes.len());
        let mut nabla_biases:Vec<Vec<f32>> = Vec::with_capacity(self.biases.len());
        println!("Started");
        for   traning_pair in  training_data { // avrage backprop diffs

            if i > 3.0 {
                break;
            }

            if i % 2000.0 == 0.0 {
                println!("traning: {}% complete", (i  / 60000.0)*100.0)
            }

            let mut weight_flag = false;
            let mut bias_flag = false;
            let (delta_weights, delta_biases) = self.backpropagation(traning_pair)?;
           // println!("deltaw len {}, deltab len {}", delta_weights.len(), delta_biases.len());
            for (i, (dw, db)) in delta_weights.iter().zip(delta_biases).enumerate() {

                if let Some(m1) = nabla_weights.get_mut(i) {
                    //println!("used weight");
                    //println!("before: m1 {}", *m1);
                    //println!("before: dw {}", dw);
                    *m1 = Matrix(m1.get_rows().clone()) + Matrix(dw.get_rows().clone());
                    //println!("after: m1 {}", *m1);
                    break;
                } else {
                    weight_flag = true;
                }

                if weight_flag {
                    println!("pushed weight");
                    nabla_weights.push(Matrix(dw.get_rows().clone()));
                    weight_flag = false;
                }


                if let Some(b1) = nabla_biases.get_mut(i) {
                    //println!("used bias");
                    *b1 = math::vec_adder(b1, &db);
                } else {
                    bias_flag = true;
                }

                if bias_flag {
                    //println!("pushed bias");
                    nabla_biases.push(db.clone());
                    bias_flag = false
                }
            }

            i += 1.0;
           // nabla_weights = nabla_weights + delta_weights; // add + operator to matrix
           // nabla_biases = nabla_biases + delta_bias; // use biases as matrix
        }

        self.weight_matrixes = self.weight_matrixes.iter().zip(nabla_weights)
            .map(|(m1, m2)| Matrix(m1.get_rows().clone()) - m2 ).collect();

        self.biases = self.biases.iter().zip(nabla_biases)
            .map(|(v1, v2)| math::vec_sub(v1, &v2)).collect();
        Ok(())
        // apply to network weights and biases
        // do 1 epoch or more
    }

    pub fn eval<'a>(self) -> Result<(), &'a str> {
        let test_data = load_test_data().unwrap();
        let mut correct = 0.0;
        for (img, expected) in test_data {
            let answ = self.propagate_forward(img)?;
            let actual = math::vec_get_max(&answ);
            if actual == math::vec_get_max(&expected) {
               // println!("expected: {:?}, actual {:?}", expected, kuk);
                correct += 1.0;
            }
        }
        println!("{}% classified correctly", (correct / 10000.0) * 100.0);
        Ok(())
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
        assert_eq!(3, nn.weight_matrixes[0].get_rows().len()); //number of rows in connecting matrix between layer 0 and 1
        assert_eq!(2, nn.weight_matrixes[0].get_rows()[0].len()); //number of columns in connecting matrix between layer 0 and 1
        let mut prev_row = vec![0.0, 0.0, 0.0];
        for row in nn.weight_matrixes[0].get_rows() {
            assert_eq!(2, row.len());
            assert_ne!(prev_row, *row);
            prev_row = row.clone();
        }

        assert_eq!(5, nn.weight_matrixes[1].get_rows().len()); //number of rows in connecting matrix between layer 1 and 2
        assert_eq!(3, nn.weight_matrixes[1].get_rows()[0].len()); //number of columns in connecting matrix between layer 1 and 2
        let mut prev_row = vec![0.0;3];
        for row in nn.weight_matrixes[1].get_rows() {
            assert_eq!(3, row.len());
            assert_ne!(prev_row, *row);
            prev_row = row.clone();
        }
    }


   /* #[test]
    fn propagate_forward(){
        let structure = vec![2, 3, 5];
        let mut weight_matrixes: Vec<Matrix> = Vec::with_capacity(2);
        weight_matrixes.push(Matrix(vec![
            vec![3.0, 1.5],
            vec![2.0, -3.0],
            vec![-1.1, 1.2]
        ]));
        weight_matrixes.push(Matrix(vec![
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

    }*/

    #[test]
    fn propagate_backwards() {
        let mut nn = NeuralNet::generate_new(vec![2,3,5]);
        println!("number of weight matrixes {}, number of rows {}, number of cols {}", nn.weight_matrixes.len(),
                 nn.weight_matrixes[0].get_rows().len(), nn.weight_matrixes[0].get_rows()[0].len());
        //for tuple in load_training_data().unwrap() {
            if let Err(some) =  nn.gradient_decent( 2.4) {
                println!("{}",some)
            }
          //  break;
        //}

    }


}
