pub mod neural_net;
pub mod data_loader;
use neural_net::NeuralNet;
fn main() {
    println!("Hello, world!");
    let mut nn = NeuralNet::generate_new(vec![784,30,10]);
    nn.gradient_decent(3.0);
    nn.eval();

}
