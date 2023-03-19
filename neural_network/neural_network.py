import numpy as np
from .layer import Layer


class NeuralNetwork:
    def __init__(self, layer_sizes) -> None:
        self.layers = np.empty(len(layer_sizes) - 1, dtype=Layer)
        for i in range(len(self.layers)):
            self.layers[i] = Layer(layer_sizes[i], layer_sizes[i + 1])

    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs):
        outputs = self.calculate_outputs(inputs)
        return np.max(outputs)
