use std::cmp::Ordering;
use std::io::Read;
use bytes::{Buf, Bytes};
use flate2::read::GzDecoder;
use nalgebra::{DVector};
use sussy_net::{NeuralNetwork, ActivationFunction};

fn get_training_file_bytes(path : &str) -> (usize, Bytes) {
    let file = std::fs::File::open(path).expect("Could not open mnist file: Did you download the MNIST data set gzip files ?");
    let mut gz_decoder = GzDecoder::new(file);
    let mut file_data = Vec::new();
    gz_decoder.read_to_end(&mut file_data).unwrap();
    let mut file_bytes = bytes::Bytes::from(file_data);
    let _magic_number = file_bytes.get_u32();
    let count = file_bytes.get_u32() as usize;
    (count, file_bytes)
}

fn get_input_image_vectors() -> Vec<DVector<f64>> {
    let (image_count, mut file_bytes) = get_training_file_bytes("train-images-idx3-ubyte.gz");
    let row_count = file_bytes.get_u32() as usize;
    let column_count = file_bytes.get_u32() as usize;

    let mut inputs = Vec::with_capacity(image_count);
    for _ in 0..image_count {
        let mut image_vector = DVector::zeros(row_count * column_count);
        for j in 0..row_count*column_count {
            image_vector[j] = file_bytes.get_u8() as f64 / 255.0;
        }
        inputs.push(image_vector);
    }
    inputs
}

fn get_output_lable_vectors() -> Vec<DVector<f64>> {
    let (label_count, mut file_bytes) = get_training_file_bytes("train-labels-idx1-ubyte.gz");
    let mut outputs = Vec::with_capacity(label_count);
    for _ in 0..label_count {
        let mut output_vector = DVector::zeros(10);
        output_vector[file_bytes.get_u8() as usize] = 1.0;
        outputs.push(output_vector);
    }
    outputs
}

fn test_accuarcy(neural_network : &NeuralNetwork, test_inputs : &Vec<DVector<f64>>, test_outputs : &Vec<DVector<f64>>) {
    let mut correct = 0;
    for (input, output) in test_inputs.iter().zip(test_outputs) {
        let correct_number = output.iter().enumerate().max_by(|(_, a), (_, b)| {
            if a > b { Ordering::Greater } else { Ordering::Less }
        }).unwrap().0;

        let prediction = neural_network.feed(input.iter().map(|v| *v).collect());

        let predicted_number = prediction.iter().enumerate().max_by(|(_, a), (_, b)| {
            if a > b { Ordering::Greater } else { Ordering::Less }
        }).unwrap().0;
        if correct_number == predicted_number {correct += 1}
    }
    println!("[Test] Accuracy: {:.3}%", correct as f32 / test_inputs.len() as f32 * 100.0);
}

fn main() {
    // Load training data
    let mut inputs = get_input_image_vectors();
    let mut outputs = get_output_lable_vectors();
    // Define network structure
    let mut neural_network = NeuralNetwork::new(784);
    neural_network.add_layer(128, ActivationFunction::Tanh);
    neural_network.add_layer(64, ActivationFunction::Tanh);
    neural_network.add_layer(64, ActivationFunction::Tanh);
    neural_network.add_layer(10, ActivationFunction::Sigmoid);
    neural_network.randomize();
    // Split dataset into test and train
    let validation_split = 0.1;
    let test_size = (inputs.len() as f32 * validation_split as f32).floor() as usize;
    let test_inputs : Vec<DVector<f64>> = inputs.drain(0..test_size).collect();
    let test_outputs : Vec<DVector<f64>> = outputs.drain(0..test_size).collect();
    // Train network for 10 epochs, but test accuracy before and after train
    test_accuarcy(&neural_network, &test_inputs, &test_outputs);
    neural_network.fit(inputs.clone(), outputs.clone(), 10, 2, 1, |_, loss| loss * 0.0005);
    test_accuarcy(&neural_network, &inputs, &outputs);
}
