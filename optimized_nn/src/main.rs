#[macro_use(s)]
extern crate ndarray;


mod neural_net;
mod data_loader;
use neural_net::NeuralNet;



fn main() {
    println!("Hello, world!");
    let mut nn = NeuralNet::new(vec![784,30,10]);
    let mut data = data_loader::load_traning_data_new().unwrap();
    let test_data = data_loader::load_test_data_new().unwrap();
    nn.sdg(&mut data, 300, 1.3, 90, &test_data);

}





