pub mod neural_net;
pub mod data_loader;
use neural_net::NeuralNet;
fn main() {
    println!("Hello, world!");
    let mut nn = NeuralNet::generate_new(vec![14,8,8,10]);
    nn.gradient_decent(1.0);
    nn.eval();

}
