import numpy as np
import math
import random

from neural_network.layer import Layer
from neural_network import costs


class NeuralNetwork:
    def __init__(self, layers) -> None:
        self.layers: list[Layer] = np.array(layers)

    def __repr__(self):
        return f"NeuralNetwork(\n layers={self.layers}\n)"

    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    def cost(self, inputs, expected_outputs, node_cost: costs.Cost = costs.MeanSquaredError):
        outputs = self.calculate_outputs(inputs) 
        cost = 0
        for output, expected_output in zip(outputs, expected_outputs):
            cost += node_cost.func(output, expected_output)
        
        return cost

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()


    def apply_gradients(self, learn_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def learn(self, inputs, expected_outputs, learn_rate: float, node_cost: costs.Cost = costs.MeanSquaredError):
        H = 0.000001
        original_cost = self.cost(inputs, expected_outputs, node_cost)
        for layer in self.layers:
            for inp in range(layer.num_input):                
                for out in range(layer.num_output):
                    layer.weights[inp][out] += H
                    delta_cost = self.cost(inputs, expected_outputs, node_cost) - original_cost
                    layer.weights[inp][out] -= H
                    layer.gradient_weights[inp][out] = delta_cost / H

            for out in range(layer.num_output):
                layer.weights[inp][out] += H
                delta_cost = self.cost(inputs, expected_outputs, node_cost) - original_cost
                layer.weights[inp][out] -= H
                layer.gradient_weights[inp][out] = delta_cost / H

        self.apply_gradients(learn_rate)