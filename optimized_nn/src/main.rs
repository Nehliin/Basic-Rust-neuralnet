#[macro_use(s)]
extern crate ndarray;

mod neural_net;
mod data_loader;
use neural_net::NeuralNet;



fn main() {
    println!("Hello, world!");
    let mut nn = NeuralNet::new(vec![784,16,16,10]);
    let mut data = data_loader::load_traning_data_new().unwrap();
    let test_data = data_loader::load_test_data_new().unwrap();
    for x in 0..2 {
        nn.sdg(&mut data, 3.0, 4000);
        nn.eval(&test_data);
    }
    //nn.gradient_decent(data_loader::load_traning_data_new().unwrap());

}
