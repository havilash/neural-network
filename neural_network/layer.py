import numpy as np


class Layer:
    def __init__(self, num_nodes_in: int, num_nodes_out: int) -> None:
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out

        self.weights = np.zeros((num_nodes_in, num_nodes_out))
        self.biases = np.zeros((num_nodes_out))

    def calculate_outputs(self, inputs):
        weighted_inputs = np.empty(self.num_nodes_out)

        for node_out in range(self.num_nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in][node_out]
            weighted_inputs[node_out] = weighted_input

        return weighted_inputs

    def __repr__(self):
        return f"Layer(num_nodes_in={self.num_nodes_in}, num_nodes_out={self.num_nodes_out})"
