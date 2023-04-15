import numpy as np
import random
import math

from neural_network import activations, costs

class Layer:
    def __init__(self, num_input: int, num_output: int, activation: activations.Activation = activations.ReLU) -> None:
        self.num_input = num_input
        self.num_output = num_output
        self.activation = activation

        self.initialize()

    def initialize(self):
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
        self.activations = np.empty(self.num_output)
        self.weighted_inputs = np.empty(self.num_output)

        for out in range(self.num_output):
            weighted_input = self.biases[out]
            for inp in range(self.num_input):
                weighted_input += inputs[inp] * self.weights[inp][out]
            self.weighted_inputs[out] = weighted_input
            self.activations[out] = self.activation.func(weighted_input)

        return self.activations
    
    def initialize_random_weights(self):
        print("random normal", np.random.normal(0, 1, size=self.weights.shape))
        self.weights = np.random.normal(0, 1, size=self.weights.shape) / np.sqrt(self.num_input)

    # Gradient Descent
    def apply_gradients(self, learn_rate: float):
        for out in range(self.num_output):
            self.biases[out] -= self.gradient_biases[out] * learn_rate
            for inp in range(self.num_input):
                self.weights[inp][out] -= self.gradient_weights[inp][out] * learn_rate
    
    def reset_gradients(self):
        self.gradient_weights = np.zeros((self.num_input, self.num_output))
        self.gradient_biases = np.zeros(self.num_output)

    def update_gradients(self, node_values):
        for out in range(self.num_output):
            for inp in range(self.num_input):
                # derivative cost with respect to weight
                self.gradient_weights[inp][out] += self.inputs[inp] * node_values[out]
            # derivative cost with respect to bias
            self.gradient_biases += 1 * node_values[out]

    # Backpropagation
    def caculate_output_layer_node_values(self, expected_outputs, cost: costs.Cost):
        node_values = np.empty(self.num_output)
        for i in range(self.num_output):
            cost_derivative = cost.derivative(self.activations[i], expected_outputs[i])
            activation_derivative = self.activation.derivative(self.weighted_inputs[i])
            node_values[i] = activation_derivative * cost_derivative
        return node_values
    
    def caculate_hidden_layer_node_values(self, prev_layer, prev_node_values):
        node_values = np.empty(self.num_output)

        for i in range(len(node_values)):
            node_value = 0
            for j in range(len(prev_node_values)):
                weighted_input_derivative = prev_layer.weights[i][j]
                node_value += weighted_input_derivative * prev_node_values[j]
            node_value *= self.activation.derivative(self.weighted_inputs[i])
            node_values[i] = node_value

        return node_values
