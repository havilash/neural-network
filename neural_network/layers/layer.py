import numpy as np

from neural_network import activations, costs
from neural_network import constants


class Layer:
    def __init__(self) -> None:
        pass

    def __repr__(self):
        name = self.__class__.__name__
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f"{name}({attrs})"

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
