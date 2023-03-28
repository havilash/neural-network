import numpy as np
import math

from neural_network.layer import Layer


class NeuralNetwork:
    def __init__(self, layers) -> None:
        # self.layers = np.empty(len(layer_sizes) - 1, dtype=Layer)
        # for i in range(len(self.layers)):
        #     self.layers[i] = Layer(layer_sizes[i], layer_sizes[i + 1])
        self.layers = np.array(layers)

    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    @staticmethod
    def nodeCost(output, expectedOutput):
        error = output - expectedOutput
        return math.pow(error, 2)

    def cost(self, outputs, expected_outputs):  #? TEST
        if len(outputs) != len(expected_output): raise ValueError(f"The length of outputs ({outputs}) does not match the length of expectedOutputs ({expected_outputs}).")
        cost = 0
        for output, expected_output in zip(outputs, expected_outputs):
            print(output, expected_output)
            cost += self.nodeCost(output, expected_output)
        
        return cost