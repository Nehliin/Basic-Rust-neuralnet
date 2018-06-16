pub mod layer;

#[derive(Debug)]
struct NeuralNet {
    //layer
    input_layer: layer::Layer,
    output_layer: layer::Layer,
    hidden_layers: Vec<layer::Layer>
}