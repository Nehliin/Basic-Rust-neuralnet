mod math;

use neural_net::math::matrix::Matrix;
use data_loader::*;

#[derive(Debug)]
pub struct NeuralNet {
    weight_matrixes: Vec<Matrix>,
    biases: Vec<Vec<f64>>,
    structure: Vec<usize>
}

impl NeuralNet {

    pub fn generate_new(structure: Vec<usize>) -> NeuralNet{
        let mut biases: Vec<Vec<f64>> = Vec::with_capacity(structure.len()-1); // first layer need no bias
        let mut weight_matrixes: Vec<Matrix> = Vec::with_capacity(structure.len()-1); // last layer have no matrix to feed forward

        for i in 1..structure.len() {
            let mut bias_layer: Vec<f64> = vec![0.0; structure[i]]; // allocate each bias vector for each layer
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

    pub fn new(structure: Vec<usize>, weight_matrixes: Vec<Matrix>, biases: Vec<Vec<f64>>) -> NeuralNet{
        return NeuralNet{
            weight_matrixes,
            biases,
            structure
        }
    }

    //propagets the layer using feedforward and returns the next layers neurons
    pub fn propagate_forward<'a>(&self, input: image::Image) -> Result<Vec<f64>, &'a str> {

        let mut calculated_layer: Vec<f64> = input.get_pixels().iter().map(|a| *a as f64).collect();
        if calculated_layer.len() != self.structure[0] { // if input vector isn't same size as neural_net
            return Err("wrong input vector lenght");
        }

        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            let mut z:Vec<f64> = math::matrix::matrix_multiply(m, &calculated_layer)?.iter()
                .zip(bias)
                .map(| (v, b) | v + b).collect(); //multiplies weights and adds bias
            calculated_layer = math::vec_sigmoid(&z);
        }
        return Ok(calculated_layer);
    }

    pub fn backpropagation<'a>(&self, (input, expected_output):(Vec<f64>, Vec<f64>)) -> Result<(Vec<Matrix>, Vec<Vec<f64>>), &'a str> {

        let number_of_layers = self.structure.len();
        let mut nabla_biases = Vec::with_capacity(number_of_layers-1);
        for len in self.structure[1..] {
            nabla_biases.push(vec![0.0;len])
        }

        let mut nabla_weights: Vec<Matrix> =  Vec::with_capacity(number_of_layers-1);
        let mut neurons: Vec<Vec<f64>> = Vec::with_capacity(number_of_layers); //neurons for each layer
        let mut z_vectors: Vec<Vec<f64>> = Vec::with_capacity(number_of_layers); // the z value vector for each layer

        // feeds forward
        let mut activation: Vec<f64> = input;
        neurons.push(activation.clone());
        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            if let Ok(mul) = math::matrix::matrix_multiply(m, &activation){
                let mut z:Vec<f64> = math::vec_adder(&mul, bias);
                z_vectors.push(z.clone());
                activation = math::vec_sigmoid(&z);
                neurons.push(activation.clone());
            } else {
                println!("PANIK")
            }
        }

        //println!("Activation {:?}", &activation);
        let mut nabla_biases = Vec::with_capacity(number_of_layers-1);
        let mut nabla_weights: Vec<Matrix> =  Vec::with_capacity(number_of_layers-1);
        let mut delta = math::nabla_bias(&neurons[neurons.len()-1], &math::vec_sigmoidprime(&z_vectors[z_vectors.len()-1]), &expected_output);

        nabla_biases.push(delta.clone()); // måste bli reversed
        // dC/dw är  transponat av neuroner * dC/db
       // println!("DELTA {:?}", delta);
        nabla_weights.push(math::nabla_weight(&delta, &neurons[neurons.len()-2])); // måste reverseras

        for l in 2..self.structure.len() {
            let mut z = z_vectors[(z_vectors.len()-l)].clone();
            let mut sp = math::vec_sigmoidprime(&z);
            //println!("not transpose : {:?}", &self.weight_matrixes[self.weight_matrixes.len()-l+1]);
            //println!("transpose : {:?}", &self.weight_matrixes[self.weight_matrixes.len()-l+1].transpose());
            //println!("delta: {:?}", delta);
            if let Ok(result) = math::matrix::matrix_multiply(&self.weight_matrixes[self.weight_matrixes.len()-l+1].transpose(), &delta) {
                delta = result.iter().zip(sp).map(|(r, s)| *r * s).collect();
            } else {
                println!("PANIK");
            }
            //println!("delta: {:?}", delta);
           // println!("activation {:?}", &neurons[neurons.len()-l-1]);
            nabla_biases.push(delta.clone());
           // println!("{}", math::nabla_weight(&delta, &neurons[neurons.len()-l-1]));
            nabla_weights.push(math::nabla_weight(&delta, &neurons[neurons.len()-l-1]));
        }

        nabla_weights.reverse();
        nabla_biases.reverse();
        //println!("{:?}", nabla_weights);
        Ok((nabla_weights, nabla_biases))
    }

    pub fn gradient_decent<'a>(&'a mut self, learning_rate:f64 ) -> Result<(), &'a str> {
        let training_data = load_training_data().unwrap();
        // use sum image
        let mut itera:f64 = 0.0;
      //  let mut nabla_weights:Vec<Matrix> = Vec::with_capacity(self.weight_matrixes.len());
       // let mut nabla_biases:Vec<Vec<f64>> = Vec::with_capacity(self.biases.len());
        println!("Started");
        for   (img, expected) in  training_data { // avrage backprop diffs

            //if i > 3.0 {
            //   break;
            //}

            if itera % 2000.0 == 0.0 {
                println!("traning: {}% complete", (itera  / 60000.0)*100.0)
            }
            let input:Vec<f64> = math::vec_sigmoid(&img.get_pixels().iter().map(|a| *a as f64).collect());
            //println!("input {:?}", input);
            //let mut weight_flag = false;
            //let mut bias_flag = false;
            let (delta_weights, delta_biases) = self.backpropagation((input, expected))?;
            // println!("deltaw len {}, deltab len {}", delta_weights.len(), delta_biases.len());
            for (i, (dw, db)) in delta_weights.iter().zip(delta_biases).enumerate() {

                let temp = Matrix(self.weight_matrixes[i].get_rows().clone()) - Matrix(dw.get_rows().clone());
                self.weight_matrixes[i] = math::matrix_mul(&temp, learning_rate);



                self.biases[i] = math::vec_sub(&self.biases[i], &db, learning_rate);
                /*if let Some(m1) = nabla_weights.get_mut(i) {
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
                   // println!("pushed weight");
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
                }*/
            }

            itera += 1.0;
            // nabla_weights = nabla_weights + delta_weights; // add + operator to matrix
            // nabla_biases = nabla_biases + delta_bias; // use biases as matrix
        }
        println!("traning: 100% complete");

        /*self.weight_matrixes = self.weight_matrixes.iter().zip(nabla_weights)
            .map(|(m1, m2)| Matrix(m1.get_rows().clone()) - m2 ).collect();

        self.biases = self.biases.iter().zip(nabla_biases)
            .map(|(v1, v2)| math::vec_sub(v1, &v2)).collect();*/
        Ok(())
        // apply to network weights and biases
        // do 1 epoch or more
    }

    pub fn eval<'a>(self) -> Result<(), &'a str> {
        let test_data = load_training_data().unwrap();
        let mut correct = 0.0;
        let data_len  = test_data.len() as f64;
        for (img, expected) in test_data {
            let mut activation:Vec<f64> = math::vec_sigmoid(&img.get_pixels().iter().map(|a| *a as f64).collect());
            for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
                if let Ok(mul) = math::matrix::matrix_multiply(m, &activation){
                    let mut z:Vec<f64> = math::vec_adder(&mul, bias);
                    activation = math::vec_sigmoid(&z);
                } else {
                    println!("PANIK")
                }
            }

            let actual = math::vec_get_max(&activation);
            if actual == math::vec_get_max(&expected) {
               // println!("expected: {:?}, actual {:?}", expected, kuk);
                correct += 1.0;
            }
        }
        println!("{}% classified correctly", (correct / data_len) * 100.0);
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


    #[test]
    fn propagate_forward() {
        let mut weights:Vec<Matrix> = Vec::new();
        weights.push(Matrix(vec![
            vec![-0.75017493,  0.63441085,  1.20964448, -1.11281677],
            vec![-1.28239165,  0.19359482, -0.30047025, -0.643115  ],
            vec![ 0.76829006,  0.13071221,  1.65978875, -0.30169557]
        ]));
        weights.push(Matrix(vec![
            vec![-0.93035666, -0.22974035, -0.8669485 ],
            vec![ 2.03457862,  0.01182334,  1.69269502]
        ]));
        let mut biases:Vec<Vec<f64>> = Vec::new();

        biases.push(vec![-0.52276612, -0.70665436, -0.75755805]);
        biases.push(vec![-0.5064255, -0.33532004]);

        let input: Vec<f64> = vec![
            -0.31136393,
            0.62555392,
            1.65537081,
            0.23020555];

        let mut expected: Vec<Vec<f64>> = Vec::new();
        expected.push(vec![
            -0.31136393,
             0.62555392,
             1.65537081,
             0.23020555]);
        expected.push(vec![
            0.86458472,
            0.30328612,
            0.85359818]);
        expected.push(vec![
            0.1071202,
            0.9464556]);

        let mut actual: Vec<Vec<f64>> = Vec::new();
        let mut activation: Vec<f64> = input;
        let mut z_vectors: Vec<Vec<f64>> = Vec::with_capacity(3); // the z value vector for each layer
        actual.push(activation.clone());
        for (m, bias) in weights.iter().zip(biases) {
            if let Ok(mul) = math::matrix::matrix_multiply(m, &activation){
                let mut z:Vec<f64> = math::vec_adder(&mul, &bias);
                z_vectors.push(z.clone());
                activation = math::vec_sigmoid(&z);

                actual.push(activation.clone());
            } else {
                println!("PANIK")
            }
        }
        assert_eq!(expected, actual);
    }

    #[test]
    fn propagate_backwards() {
        let mut weights:Vec<Matrix> = Vec::new();
        weights.push(Matrix(
            vec![
                vec![0.76387296, -0.87731381, -0.59489399, -0.59874781],
                vec![-2.1901988 , -0.0743342 ,  0.82774157, -1.15352468],
                vec![ 0.16367149,  1.3900156 ,  1.87726586,  0.90083509]
        ]));
        weights.push(Matrix(
            vec![
                vec![0.77082276,  1.59769716,  1.9896242],
                vec![0.88744456, -1.10635366,  0.53014915],
            ]));
        let mut biases:Vec<Vec<f64>> = Vec::new();
        biases.push(
            vec![
             0.0354712 ,
             0.63689698,
            -2.28648105]);
        biases.push(vec![
            1.71617244,
            0.66434508]);

        let nn = NeuralNet::new(vec![4,3,2], weights, biases);
        let input: Vec<f64> = vec![
             0.46421592,
            -0.91378042,
            -2.11136753,
             1.69515476];

        let expected_output = vec![0.0, 1.0];


        let mut expected_nw = Vec::new();
        expected_nw.push(Matrix(
            vec![
                vec![0.00185371, -0.00364891, -0.00843111,  0.00676909],
                vec![ 0.00121743, -0.00239644, -0.00553719,  0.00444565],
                vec![ 0.00015478, -0.00030467, -0.00070397,  0.0005652]
            ]));
        expected_nw.push(Matrix(vec![
            vec![5.76303139e-02,  1.26516745e-03,  1.91696116e-04],
            vec![-2.67015360e-02, -5.86183068e-04, -8.88175060e-05],
        ]));

        let mut expected_bw = Vec::new();
        expected_bw.push(vec![0.003993200867330706, 0.002622560220092743, 0.0003334189144546065]); // 0.0039932 , 0.00262256, 0.00033342
        expected_bw.push(vec![0.07138299359351796, -0.033073489216505554]); // 0.07138299, -0.03307349
        if let Ok((nabla_w, nabla_b)) = nn.backpropagation((input, expected_output)) {
<<<<<<< HEAD
            //for (m1, m2) in expected_nw.iter().zip(nabla_w) {
             //   assert_eq!(m1.get_rows(), m2.get_rows())
            //}
=======
            for (m1, m2) in expected_nw.iter().zip(nabla_w) {
                println!("Expected: {}", m1);
                println!("actual: {}", m2);
                //assert_eq!(m1.get_rows(), m2.get_rows())
            }
>>>>>>> 0e48a2e4474f065cf4ccb27357d923b63fb6f765
            assert_eq!(expected_bw, nabla_b);
        }
    }


}
