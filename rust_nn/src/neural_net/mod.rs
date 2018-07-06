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
        let mut neurons: Vec<Vec<f64>> = Vec::with_capacity(number_of_layers); //neurons for each layer
        let mut z_vectors: Vec<Vec<f64>> = Vec::with_capacity(number_of_layers); // the z value vector for each layer

        // feeds forward
        let mut activation: Vec<f64> = input;
        neurons.push(activation.clone());

        for (m, bias) in self.weight_matrixes.iter().zip(&self.biases) {
            let mut z:Vec<f64> = math::matrix::matrix_multiply(m, &activation)?.iter()
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
        let mut delta = math::nabla_bias(&activation, &math::vec_sigmoidprime(&z_vectors[z_vectors.len()-1]), &expected_output);
        nabla_biases.push(delta.clone()); // måste bli reversed
        // dC/dw är  transponat av neuroner * dC/db
        //println!("DELTA {:?}", delta.len());
        //println!("1 {:?}", math::nabla_weight(&delta, &neurons[neurons.len()-2]));
        nabla_weights.push(math::nabla_weight(&delta, &neurons[neurons.len()-2])); // måste reverseras
        //println!("Matrixs dim: rows {}, row len {}", nabla_weights[0].get_rows().len(), nabla_weights[0].get_rows()[0].len());
        //println!("NEURONS: {:?}", &neurons);
        for l in 2..self.structure.len() {
            let mut z = z_vectors[(z_vectors.len()-l)].clone();
          //  println!("NEW Z {:?}", z);
            let mut sp = math::vec_sigmoidprime(&z);
           // println!("prime {:?}", sp);
            if let Ok(result) = math::matrix::matrix_multiply(&self.weight_matrixes[self.weight_matrixes.len()-l+1].transpose(), &delta) {
                delta = result.iter().zip(sp).map(| (r, s) | *r*s).collect();
            } else {
                println!("PANIK");
            }
           // println!("delta: {:?}", delta);
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
            let input:Vec<f64> = img.get_pixels().iter().map(|a| *a as f64).collect();
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
        let mut biases: Vec<Vec<f64>> = Vec::with_capacity(2);
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
        let mut weights:Vec<Matrix> = Vec::new();
        weights.push(Matrix(vec![vec![-0.41349304, -0.68632649, 0.91983257, 0.46879769,  -1.10143961, 0.41353717
                               , -0.63210644],
                          vec![-0.11399703, 1.83181328, 1.4881615,  0.19030198, -0.37203935 -1.06950722
                               ,  0.06853642],
                          vec![-1.16237823, 0.83112924, 0.23195781, -1.93040103, 0.54744602, 0.40719859
                               , -0.88738743],
                          vec![-1.66524648, 0.18031383, 0.1630823,  -0.40666155, -0.24571867, 0.14421307
                               , -1.96200136],
                          vec![ 0.51116553, 0.17181706, -1.50245733, -1.09545316, -1.02617832, 0.67989883
                                ,  1.53252645]]));
        weights.push(Matrix(vec![
                                 vec![-0.99803862,1.22824448,0.0866717,-0.01480796,0.36305291],
                                 vec![-1.23275625,1.65827426,0.01053812,0.30560507,0.21211367],
                                 vec![-1.80560939,1.76615943,1.55560105,0.51834508,0.30002316]]));
        let mut biases:Vec<Vec<f64>> = Vec::new();
        biases.push(vec![
             0.21373571,
             0.86229568,
            -1.36403487,
            -1.42388866,
            -1.63735394]);
        biases.push(vec![
            2.28946578,
            -0.04004787,
            -0.43054496]);

        let nn = NeuralNet::new(vec![7,5,3], weights, biases);
        let input: Vec<f64> = vec![
                0.33949838,
                0.2927223 ,
                0.5927223 ,
                0.8927223 ,
                0.99223   ,
                0.00927223,
                0.423123  ];
        let expected_output = vec![0.0, 1.0, 0.0];


        let mut expected_nw = Vec::new();
        expected_nw.push(Matrix(vec![vec![5.21990930e-03,4.50071030e-03,9.11331784e-03,1.37259254e-02
                                          , 1.52558919e-02,1.42563860e-04,6.50566781e-03],
                                     vec![6.83559881e-03,5.89379014e-03,1.19341125e-02,1.79744348e-02
                                          , 1.99779634e-02,1.86690859e-04,8.51933100e-03],
                                     vec![4.05214248e-05,3.49383837e-05,7.07454101e-05,1.06552436e-04
                                          , 1.18429353e-04,1.10670328e-06,5.05025881e-05],
                                     vec![-3.23507415e-04,2.78934570e-04,5.64804048e-04,8.50673526e-04
                                          ,-9.45494240e-04,8.83549183e-06,4.03193170e-04],
                                     vec![2.87544505e-04,2.47926629e-04,5.02017243e-04,7.56107856e-04
                                          , 8.40387765e-04,7.85328870e-06,3.58371942e-04]]));
        expected_nw.push(Matrix(vec![vec![ 0.0158866,  0.03780694, 0.0021609,  0.00155223, 0.00108441],
        vec![ -0.03526334, -0.08391973, -0.00479654, -0.00344547, -0.00240707],
        vec![ 0.00171823, 0.00408906, 0.00023372, 0.00016788, 0.00011729]]));

        let mut expected_bw = Vec::new();
        expected_bw.push(vec![
             0.01537536,
             0.02013441,
             0.00011936,
            -0.0009529 ,
             0.00084697]);
        expected_bw.push(vec![
             0.04261743,
            -0.09459753,
             0.00460934]);
        if let Ok((nabla_w, nabla_b)) = nn.backpropagation((input, expected_output)) {
            for (m1, m2) in expected_nw.iter().zip(nabla_w) {
                assert_eq!(m1.get_rows(), m2.get_rows())
            }
            assert_eq!(expected_bw, nabla_b);
        }
    }


}
