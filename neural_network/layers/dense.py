import numpy as np

from neural_network import activations, costs
from .layer import Layer

from neural_network import constants


class Dense(Layer):
    def __init__(self, num_input: int, num_output: int, activation: activations.Activation = constants.DEFAULT_ACTIVATION) -> None:
        self.num_input = num_input
        self.num_output = num_output
        self.activation = activation

        self.initialize_weights_biases()

    def initialize_weights_biases(self):
        self.weights = np.zeros((self.num_input, self.num_output))
        self.biases = np.zeros(self.num_output)

        self.gradient_weights = np.zeros((self.num_input, self.num_output))
        self.gradient_biases = np.zeros(self.num_output)

        self.initialize_random_weights()

    def __repr__(self):
        variables = [
            ("num_input", self.num_input),
            ("num_output", self.num_output),
            ("activation", self.activation.__name__),
        ]
        vars_string = ", ".join([f"{var[0]}: {var[1]}" for var in variables])
        return f"Layer({vars_string})"

    def calculate_outputs(self, inputs):
        self.inputs = inputs
        self.weighted_inputs = np.dot(inputs, self.weights) + self.biases
        # check if any of the inputs are nan
        if np.isnan(self.weighted_inputs).any():
            print(inputs, self.weighted_inputs)
        self.activations = self.activation.func(self.weighted_inputs)
        return self.activations

    def initialize_random_weights(self):
        self.weights = np.random.normal(0, 1, size=self.weights.shape) / np.sqrt(self.num_input)

    # Gradient Descent
    def apply_gradients(self, learn_rate: float):
        self.biases -= self.gradient_biases * learn_rate
        self.weights -= self.gradient_weights * learn_rate

    def reset_gradients(self):
        self.gradient_weights = np.zeros((self.num_input, self.num_output))
        self.gradient_biases = np.zeros(self.num_output)

    def update_gradients(self, node_values, max_norm=1.0):
        gradient_weights = np.outer(self.inputs, node_values)  # derivative cost with respect to weights
        gradient_biases = node_values  # derivative cost with respect to biases
            
        # Calculate the norm of the gradients
        norm = np.linalg.norm(gradient_weights) + np.linalg.norm(gradient_biases)

        # Scale the gradients if their norm exceeds the threshold
        if norm > max_norm:
            scale = max_norm / norm
            gradient_weights *= scale
            gradient_biases *= scale

        self.gradient_weights += gradient_weights
        self.gradient_biases += gradient_biases

    # Backpropagation
    def calculate_output_layer_node_values(self, expected_outputs, cost: costs.Cost):
        cost_derivative = cost.derivative(self.activations, expected_outputs)
        activation_derivative = self.activation.derivative(self.weighted_inputs)
        node_values = activation_derivative * cost_derivative
        return node_values

    def calculate_node_values(self, prev_layer, prev_node_values):
        node_values = np.dot(prev_layer.weights, prev_node_values)
        node_values *= self.activation.derivative(self.weighted_inputs)
        return node_values
