import numpy as np
import random
import math

from neural_network import activations

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
        activations = np.empty(self.num_output)

        for out in range(self.num_output):
            weighted_input = self.biases[out]
            for inp in range(self.num_input):
                weighted_input += inputs[inp] * self.weights[inp][out]
            activations[out] = self.activation.func(weighted_input)

        return activations
    
    def initialize_random_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.normal(0, 1) / np.sqrt(len(self.weights))

    # Gradient Descent
    def apply_gradients(self, learn_rate: float):
        for out in range(self.num_output):
            self.biases[out] -= self.gradient_biases[out] * learn_rate
            for inp in range(self.num_input):
                self.weights[inp][out] -= self.gradient_weights[inp][out] * learn_rate
    
    def reset_gradients(self):
        self.gradient_weights = np.zeros((self.num_input, self.num_output))
        self.gradient_biases = np.zeros(self.num_output)
