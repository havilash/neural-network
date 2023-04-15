import numpy as np
import threading

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
    
    # Gradient Descent
    def cost(self, inputs, expected_outputs, node_cost: costs.Cost = costs.MeanSquaredError):
        outputs = self.calculate_outputs(inputs) 
        cost = np.sum(node_cost.func(outputs, expected_outputs))
        return cost

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

    def apply_gradients(self, learn_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def update_gradients(self, inputs, expected_outputs, cost: costs.Cost):
        self.calculate_outputs(inputs)

        output_layer = self.layers[-1]
        node_values = output_layer.caculate_output_layer_node_values(expected_outputs, cost)
        output_layer.update_gradients(node_values)

        for i in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[i]
            node_values = hidden_layer.caculate_hidden_layer_node_values(self.layers[i+1], node_values)
            hidden_layer.update_gradients(node_values)

    def learn(self, training_batch, learn_rate: float, cost: costs.Cost = costs.MeanSquaredError):
        """
        Update the gradients and apply them to the network based on a batch of training data.

        :param training_batch: A list of tuples containing input and target output data. Each tuple represents a single training example with the first element being the input data (features) and the second element being the target output data (label). For example: [(x1, y1), (x2, y2), (x3, y3)] where x1, x2, x3 are the input data and y1, y2, y3 are the target output data.
        :param learn_rate: The learning rate to use when applying the gradients.
        :param cost: The cost function to use when calculating the error between the network's output and the target output.
        """
        
        threads = []
        for data in training_batch:
            thread = threading.Thread(target=self.update_gradients, args=(data[0], data[1], cost))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        self.apply_gradients(learn_rate / len(training_batch))
        self.reset_gradients()