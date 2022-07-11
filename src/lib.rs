use std::iter;
use nalgebra::{DMatrix, DVector};
use std::ops::{AddAssign};
use rand::Rng;

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Identity,
    Relu,
    LeakyRelu(f64),
    Sigmoid,
    Tanh
}

impl ActivationFunction {

    fn sigmoid(x : f64) -> f64 {
        1.0 / (1.0 + std::f64::consts::E.powf(-x))
    }

    pub fn execute(&self, input : f64) -> f64 {
        match self {
            ActivationFunction::Identity => input,
            ActivationFunction::Relu => input.max(0.0),
            ActivationFunction::LeakyRelu(v) => input.max(input*v),
            ActivationFunction::Sigmoid => Self::sigmoid(input),
            ActivationFunction::Tanh => input.tanh()
        }
    }

    pub fn execute_derivative(&self, input : f64) -> f64 {
        match self {
            ActivationFunction::Identity => 1.0,
            ActivationFunction::Relu => if input <= 0.0 {0.0} else {1.0},
            ActivationFunction::LeakyRelu(v) => if input <= 0.0 {*v} else {1.0},
            ActivationFunction::Sigmoid => Self::sigmoid(input) * (1.0 - Self::sigmoid(input)),
            ActivationFunction::Tanh => 1.0 - input.tanh().powi(2)
        }
    }

}

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights_matrix : DMatrix<f64>,
    pub bias_vector : DVector<f64>,
    pub activation_function : ActivationFunction
}

impl Layer {

    pub fn new(neurons : u16, activation_function : ActivationFunction, prev_neurons : u16) -> Self {
        Self {
            weights_matrix: DMatrix::zeros(neurons as usize, prev_neurons as usize),
            bias_vector: DVector::zeros(neurons as usize),
            activation_function
        }
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        self.weights_matrix = self.weights_matrix.map(|_| rng.gen_range(-1.0..1.0));
        self.bias_vector = self.bias_vector.map(|_| rng.gen_range(-1.0..1.0));
    }

    pub fn feed(&self, input_vector : &DVector<f64>, use_activation_function : bool) -> DVector<f64> {
        let mut output : DVector<f64> = &self.weights_matrix * input_vector;
        output.add_assign(&self.bias_vector);
        if use_activation_function {
            output.map(|v| self.activation_function.execute(v))
        } else {output}
    }

}

#[derive(Clone)]
pub struct NeuralNetwork {
    input_size : u16,
    pub layers : Vec<Layer>
}

impl NeuralNetwork {

    /// Create a new neural network based on the input size
    pub fn new(input_size : u16) -> Self {
        Self {
            input_size,
            layers: vec![]
        }
    }

    /// Randomizes the weights and biases in the network
    pub fn randomize(&mut self) {
        for layer in &mut self.layers {
            layer.randomize();
        }
    }

    /// Adds a fully connected layer with n neurons and a activation function to the end of the network
    pub fn add_layer(&mut self, neurons : u16, activation_function : ActivationFunction) {
        let prev_layer_neurons = match self.layers.last() {
            Some(l) => l.bias_vector.len() as u16,
            None => self.input_size
        };
        self.layers.push(Layer::new(neurons, activation_function, prev_layer_neurons));
    }

    pub fn get_layer_mut(&mut self, index : usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    fn _feed(&self, input_vector : DVector<f64>) -> DVector<f64> {
        let mut last_output = input_vector;
        for layer in &self.layers {
            last_output = layer.feed(&last_output, true);
        }
        last_output
    }

    /// Feed input to the neural network and get the resulting output (The prediction)
    pub fn feed(&self, inputs : Vec<f64>) -> Vec<f64> {
        let input_vector : DVector<f64> = DVector::from_vec(inputs);
        self._feed(input_vector).data.as_vec().clone()
    }

    /// Computes the error for each node in the network based on the last errors of the network (propagates the error backwards)
    fn get_layer_errors(&self, errors : DVector<f64>) -> Vec<DVector<f64>> {
        // Calculate errors for each node in every layer
        let mut layer_errors : Vec<DVector<f64>> = iter::repeat(DVector::zeros(0)).take(self.layers.len()).collect();
        layer_errors[self.layers.len()-1] = errors;
        for i in (1..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let weights_sum = 1.0; //layer.weights_matrix.sum();
            layer_errors[i-1] = (layer.weights_matrix.transpose() / weights_sum) * &layer_errors[i];
        }
        layer_errors
    }

    fn gradient_descent_one(&self, input : &DVector<f64>, desired_output : &DVector<f64>) -> (Vec<(DMatrix<f64>, DVector<f64>)>, f64) {
        // Feed forward, but remember the outputs for every layer (with and without the derivative applied)
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        let mut layer_outputs_no_activation = Vec::with_capacity(self.layers.len());
        let mut last_output = input;
        for (i, layer) in self.layers.iter().enumerate() {
            let output_no_activate = layer.feed(last_output, false);
            layer_outputs_no_activation.push(output_no_activate);
            // Apply activation function for next layer
            layer_outputs.push(layer_outputs_no_activation[i].map(|v| layer.activation_function.execute(v)));
            last_output = &layer_outputs[i];
        }
        // Calculate error value
        let absolute_squared_error = (desired_output - last_output).map(|v| v*v).sum();
        // Get the layer errors with gradient descent
        let layer_errors = self.get_layer_errors(desired_output - last_output);
        // Calculate how to change the parameters based on the errors for each layer and the outputs of each layer
        let mut layer_changes = Vec::with_capacity(self.layers.len());
        for (i, (layer, errors)) in self.layers.iter().zip(layer_errors).enumerate() {
            // Get the previous layers output (or if at the start the input)
            let previous_output = if i == 0 {input} else {&layer_outputs[i-1]};
            // Get the derivative of the current layer outputs activation function
            let output = &layer_outputs_no_activation[i];
            let output_derivative = output.map(|v| layer.activation_function.execute_derivative(v));
            // Get the errors of each node in the layer, taking into account the effect of the activation function
            // (If for instance the activation function has a non linear relationship with the input, the error should change according to the same non linear relationship, in order to be able to change the underlining parameters of the network in the correct direction)
            let errors_with_activation = errors.component_mul(&output_derivative);
            // Calculate the required change of every weight in the layer matrix, by matrix multiplying the error vector of every node with the previous layers output vector
            // The resulting output matrix has the same size as the weights matrix since the weights matrix has the shape: (Current Neurons, Previous Neurons)
            // Since were matrix multiplying a Vector of size "Current Neurons" with a vector of size "Previous Neurons" the resulting matrix has that same shape here
            let weights_change = &errors_with_activation * previous_output.transpose();
            // The bias is just a vector and it's change does not depend on the previous layers input (because its a constant).
            // The bias change does depend on the activation functions change though (because it's fed through it)
            // That's why the change is just the error that has been corrected with the activation functions derivative
            let bias_change = errors_with_activation;
            layer_changes.push((weights_change, bias_change));
        }
        (layer_changes, absolute_squared_error)
    }

    fn correct_parameters(&mut self, layer_changes : Vec<(DMatrix<f64>, DVector<f64>)>, learning_rate : f64) {
        for (layer, (weights_change, bias_change)) in self.layers.iter_mut().zip(layer_changes) {
            layer.weights_matrix += weights_change * learning_rate;
            layer.bias_vector += bias_change * learning_rate;
        }
    }

    fn get_batch_layer_changes(&self, batch_inputs : &[DVector<f64>], batch_outputs: &[DVector<f64>]) -> (Vec<(DMatrix<f64>, DVector<f64>)>, f64) {
        let mut batch_loss = 0.0;
        let mut batch_layer_changes = Vec::with_capacity(self.layers.len());
        for (i, input_vector) in batch_inputs.iter().enumerate() {
            let (layer_changes, ase) = self.gradient_descent_one(input_vector, &batch_outputs[i]);
            batch_loss += ase;
            if batch_layer_changes.len() == 0 {batch_layer_changes = layer_changes; continue}
            // Add layer changes for the current input output pair to total batch changes
            for (l, (weights_layer_change, bias_layer_change)) in layer_changes.iter().enumerate() {
                batch_layer_changes[l].0 = &batch_layer_changes[l].0 + weights_layer_change;
                batch_layer_changes[l].1 = &batch_layer_changes[l].1 + bias_layer_change;
            }
        }
        (batch_layer_changes, batch_loss)
    }

    pub fn fit(
        &mut self,
        inputs: Vec<DVector<f64>>,
        outputs: Vec<DVector<f64>>,
        epochs : usize,
        batch_size : usize,
        print_every : usize,
        learning_rate_func : fn(usize, f64) -> f64)
    {
        assert_eq!(inputs.len(), outputs.len(), "Number of training inputs and outputs must match");
        // Split training data into batches (Important to handle large training data sets)
        let input_batches : Vec<_> = inputs.chunks(batch_size).collect();
        let output_batches : Vec<_> = outputs.chunks(batch_size).collect();
        // Train for a set number of epochs
        for epoch in 0..epochs {
            let mut loss = 0.0;
            for (batch_inputs, batch_outputs) in input_batches.iter().zip(&output_batches) {
                // Calculate layer changes over all the batch training examples
                let (batch_layer_changes, batch_loss) = self.get_batch_layer_changes(batch_inputs, batch_outputs);
                loss = batch_loss;
                // Modify network parameters with the calculated changes of the current batch
                self.correct_parameters(batch_layer_changes, learning_rate_func(epoch, loss));
            }
            if epoch % print_every == 0 {
                println!("[Fit] Epoch: {}/{} loss: {}", epoch+1, epochs, loss);
            }
        }
    }

}