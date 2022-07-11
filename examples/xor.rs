use nalgebra::DVector;
use neural_network_from_scratch::{ActivationFunction, NeuralNetwork};


fn main() {
    // Create network structure
    let mut neural_network = NeuralNetwork::new(2);
    neural_network.add_layer(2, ActivationFunction::Sigmoid);
    neural_network.add_layer(1, ActivationFunction::Sigmoid);
    neural_network.randomize();
    // Create training data
    let inputs = vec![
        DVector::from_column_slice(&[0.0, 0.0]),
        DVector::from_column_slice(&[1.0, 0.0]),
        DVector::from_column_slice(&[0.0, 1.0]),
        DVector::from_column_slice(&[1.0, 1.0])
    ];
    let outputs = vec![
        DVector::from_column_slice(&[0.0]),
        DVector::from_column_slice(&[1.0]),
        DVector::from_column_slice(&[1.0]),
        DVector::from_column_slice(&[0.0])
    ];
    // Print output before trained
    println!("{:?}", neural_network.feed(vec![0.0, 0.0]));
    println!("{:?}", neural_network.feed(vec![1.0, 0.0]));
    println!("{:?}", neural_network.feed(vec![0.0, 1.0]));
    println!("{:?}", neural_network.feed(vec![1.0, 1.0]));
    // Train network
    neural_network.fit(inputs, outputs, 100000, 4, 10000, |_, loss| loss * 0.5);
    // Print output after trained
    println!("{:?}", neural_network.feed(vec![0.0, 0.0]));
    println!("{:?}", neural_network.feed(vec![1.0, 0.0]));
    println!("{:?}", neural_network.feed(vec![0.0, 1.0]));
    println!("{:?}", neural_network.feed(vec![1.0, 1.0]));
}