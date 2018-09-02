extern crate rand;

use ndarray::*;
use neural_net::rand::distributions::{Normal, Distribution};
use std::f64::consts::E;
use self::rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct NeuralNet {
    pub weight_matrixes: Vec<Array2<f64>>,
    pub biases: Vec<Array2<f64>>,
    pub structure: Vec<usize>
}

impl NeuralNet {

    pub fn new(structure: Vec<usize>) -> NeuralNet {
        let mut weight_matrixes = Vec::with_capacity(structure.len());
        let mut biases = Vec::with_capacity(structure.len()-1);
        let normal_distribution = Normal::new(0.0, 1.0);

        for (x, y) in structure.clone()[..structure.len()-1].iter().zip(structure[1..].iter()) {
            let mut matrix = Array2::zeros((*y, *x));
            for e in matrix.iter_mut() {
                *e = normal_distribution.sample(&mut rand::thread_rng()) as f64;
            }
            weight_matrixes.push(matrix);

            let mut bias = Array2::zeros((*y,1));
            for e in bias.iter_mut() {
                *e = normal_distribution.sample(&mut rand::thread_rng()) as f64;
            }
            biases.push(bias);
        }

        NeuralNet {
            weight_matrixes,
            biases,
            structure
        }
    }

    pub fn feedforward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut result = input.clone();
        for (w, b) in self.weight_matrixes.iter().zip(&self.biases) {
            result = w.dot(&result) + b;
            result = vec_sigmoid(&result);
        }
        result
    }

    fn backprop(&self, input: &Array2<f64>, expected_output: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>){
        let mut nabla_b = Vec::with_capacity(self.biases.len());
        let mut nabla_w = Vec::with_capacity(self.weight_matrixes.len());

        for bias in &self.biases {
            nabla_b.push(Array2::zeros(bias.raw_dim()));
        }

        for weight in &self.weight_matrixes {
            nabla_w.push(Array2::zeros(weight.raw_dim()));
        }

        let mut activation = input.clone();
        let mut activations = Vec::with_capacity(self.structure.len());

        activations.push(activation.clone());

        let mut zs = Vec::with_capacity(self.structure.len());

        for (w,b) in self.weight_matrixes.iter().zip(&self.biases) {
            let z = w.dot(&activation) + b.clone();
            zs.push(z.clone());
            activation = vec_sigmoid(&z);
            activations.push(activation.clone());
        }

        let mut delta = (activations[activations.len()-1].clone() - expected_output) * vec_sigmoid_prime(&zs[zs.len()-1]);
        let mut nb_index = nabla_b.len()-1;
        let mut nw_index = nabla_w.len()-1;

        nabla_b[nb_index] = delta.clone();
        nabla_w[nw_index] = delta.dot(&activations[activations.len()-2].clone().reversed_axes());

        let mut z;
        let mut sp;

        for l in 2..self.structure.len() {
            nb_index = nabla_b.len()-l;
            nw_index = nabla_w.len()-l;
            z = zs[zs.len()-l].clone();
            sp = vec_sigmoid_prime(&z);
            delta = self.weight_matrixes[self.weight_matrixes.len()-l+1].clone().reversed_axes().dot(&delta) * sp;
            nabla_b[nb_index] = delta.clone();
            nabla_w[nw_index] = delta.dot(&activations[activations.len()-l-1].clone().reversed_axes());
        }
        (nabla_b, nabla_w)
    }


    fn update_mini_batch(&mut self, mini_batch:&[(Array2<f64>, Array2<f64>)], eta: f64) {
        let mut nabla_b = Vec::with_capacity(self.biases.len());
        let mut nabla_w = Vec::with_capacity(self.weight_matrixes.len());

        for w in &self.weight_matrixes {
            nabla_w.push(Array::zeros(w.raw_dim()));
        }
        for b in &self.biases {
            nabla_b.push(Array::zeros(b.raw_dim()));
        }
        for (x,y) in mini_batch {

            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
            for (w, dw) in nabla_w.iter_mut().zip(&delta_nabla_w) {
                *w = &w.view() + dw;
            }
            for (b, db) in nabla_b.iter_mut().zip(&delta_nabla_b) {
                *b = &b.view() + db;
            }
        }
        let minibatch_len = mini_batch.len() as f64;
        for (w, dw) in self.weight_matrixes.iter_mut().zip(&nabla_w) {
            //let te:&Array2<f64> = &(dw * 1.9);
            *w = &w.view() - &(dw * (eta/minibatch_len));
        }
        for (b, db) in self.biases.iter_mut().zip(&nabla_b) {
            *b = &b.view() - &(db * (eta/minibatch_len));
        }


    }
    /*pub fn gradient_decent(&mut self,training_data:Vec<(Array2<f64>, Array2<f64>)> ) {
        let mut itera:f64 = 0.0;
        println!("Started");
        for   (img, expected) in  training_data { // avrage backprop diffs
            if itera % 2000.0 == 0.0 {
                println!("traning: {}% complete", (itera  / 60000.0)*100.0)
            }
            let (delta_nabla_b, delta_nabla_w) = self.backprop(img, expected);
            for (w, dw) in self.weight_matrixes.iter_mut().zip(&delta_nabla_w) {
                *w = w.clone() - (dw.clone());
            }
            for (b, db) in self.biases.iter_mut().zip(&delta_nabla_b) {
                *b = b.clone() - (db.clone());
            }
            itera += 1.0;
        }
        println!("traning: 100% complete");
    }*/
    /* if test_data: n_test = len(test_data)
     n = len(training_data)
     for j in xrange(epochs):
     random.shuffle(training_data)
     mini_batches = [
     training_data[k:k+mini_batch_size]
     for k in xrange(0, n, mini_batch_size)]
     for mini_batch in mini_batches:
     self.update_mini_batch(mini_batch, eta)
     if test_data:
     print "Epoch {0}: {1} / {2}".format(
     j, self.evaluate(test_data), n_test)
     else:
     print "Epoch {0} complete".format(j)*/
    pub fn sdg(&mut self, training_data:&mut [(Array2<f64>, Array2<f64>)], epochs:usize, eta:f64, batch_size:usize, test_data:&[(Array2<f64>, Array2<f64>)]) {
        for i in 0..epochs {
            thread_rng().shuffle(training_data);
            let mut mini_batches = Vec::with_capacity(batch_size);
            for batch in training_data.chunks(batch_size) {
                mini_batches.push(batch.to_vec());
            }
            for batch in mini_batches {
                self.update_mini_batch(&batch, eta);
            }
            println!("Epoch: {}", i);
            self.eval(test_data);
        }


    }

    pub fn eval(&self, test_data: &[(Array2<f64>, Array2<f64>)]) {
        let mut correct = 0.0;
        let data_len = test_data.len() as f64;
        for (img, answer) in test_data {
            let calculated = self.feedforward(&img);
            if vec_max(&calculated) == vec_max(&answer) {
                correct += 1.0;
            }
        }
        println!("{}% classified correctly", (correct / data_len) * 100.0);
    }

}

fn vec_max(v :&Array2<f64>) -> usize {
    let mut max_index: usize = 0;
    let mut max = v[[0,0]];
    for (i, element) in v.iter().enumerate() {
        if *element > max {
            max = *element;
            max_index = i;
        }
    }
    max_index
}

fn vec_sigmoid(v : &Array2<f64>) -> Array2<f64>{
    let mut result = v.clone();
    for e in result.iter_mut() {
        *e = sigmoid(*e);
    }
    result
}

fn vec_sigmoid_prime(v : &Array2<f64>) -> Array2<f64> {
    let mut result = v.clone();
    for e in result.iter_mut() {
        *e = sigmoid_prime(*e);
    }
    result
}


fn sigmoid(x : f64) -> f64{
    1.0/(1.0 + E.powf(-x))
}

fn sigmoid_prime(x :f64) -> f64{
    sigmoid(x)*(1.0-sigmoid(x))
}

#[cfg(test)]
mod tests {

    use super::*;


    #[test]
    fn feedforward(){
        let structure = vec![3,2,1];
        let mut nn = NeuralNet::new(structure);

        let mut weight_matrixes = Vec::with_capacity(3);
        let mut biases = Vec::with_capacity(2);
        weight_matrixes.push(arr2(&[[2., 2., 2.],
            [2., 2., 2.]]));
        weight_matrixes.push(arr2(&[[2., 2.]]));
        biases.push(arr2(&[[1.], [1.]]));
        biases.push(arr2(&[[1.]]));

        nn.weight_matrixes = weight_matrixes;
        nn.biases = biases;
        let result = nn.feedforward(&arr2(&[[1.], [1.], [1.]]));
        assert_eq!(arr2(&[[29.]]), result); // fel

    }


    #[test]
    fn vec_max_test(){
        let a = arr2(&[[1.0], [2.2], [0.3], [2.4]]);
        assert_eq!(3, vec_max(&a));
    }


}


