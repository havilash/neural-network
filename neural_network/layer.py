import numpy as np

from neural_network import activation

class Layer:
    def __init__(self, num_input: int, num_output: int, activation: activation.Activation = activation.ReLU) -> None:
        self.num_nodes_in = num_input
        self.num_nodes_out = num_output
        self.activation = activation

        self.weights = np.zeros((num_input, num_output))
        self.biases = np.zeros(num_output)

    def calculate_outputs(self, inputs):
        activations = np.empty(self.num_nodes_out)

        for node_out in range(self.num_nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in][node_out]
            activations[node_out] = self.activation.func(weighted_input)

        return activations

    def __repr__(self):
        return f"Layer(num_nodes_in={self.num_nodes_in}, num_nodes_out={self.num_nodes_out})"
