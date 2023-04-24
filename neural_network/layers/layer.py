import numpy as np

from neural_network import activations, costs
from neural_network import constants


class Layer:
    def __init__(self) -> None:
        self.asdf = [1, 2, 3, 4, 5]

    def __repr__(self):
        name = str(self.__class__.__name__)
        dict = repr(self.__dict__)[1:-1]
        return f"{name}({dict})"

    def calculate_outputs(self, inputs):
        return inputs

    # Gradient Descent
    def apply_gradients(self, learn_rate: float):
        pass

    def reset_gradients(self):
        pass

    def update_gradients(self, node_values):
        pass

    # Backpropagation
    def calculate_output_layer_node_values(self, expected_outputs, cost: costs.Cost):
        pass
    
    def calculate_node_values(self, prev_layer, prev_node_values):
        return prev_node_values
