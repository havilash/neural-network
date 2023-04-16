import numpy as np
import math
import threading
import pickle

from neural_network.layer import Layer
from neural_network import costs
from neural_network.data import create_batches


class NeuralNetwork:
    def __init__(self, layers) -> None:
        self.layers: list[Layer] = np.array(layers)

    def __repr__(self):
        return f"NeuralNetwork(\n layers={self.layers}\n)"

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    # Gradient Descent
    def cost(self, inputs, expected_outputs, node_cost: costs.Cost = costs.MeanSquaredError):
        outputs = self.predict(inputs) 
        cost = np.sum(node_cost.func(outputs, expected_outputs))
        return cost

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

    def apply_gradients(self, learn_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def update_gradients(self, inputs, expected_outputs, cost: costs.Cost):
        self.predict(inputs)

        output_layer = self.layers[-1]
        node_values = output_layer.caculate_output_layer_node_values(expected_outputs, cost)
        output_layer.update_gradients(node_values)

        for i in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[i]
            node_values = hidden_layer.caculate_hidden_layer_node_values(self.layers[i+1], node_values)
            hidden_layer.update_gradients(node_values)

    def learn(self, train_batch, learn_rate: float, cost: costs.Cost = costs.MeanSquaredError):
        threads = []
        for data in train_batch:
            thread = threading.Thread(target=self.update_gradients, args=(data[0], data[1], cost))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        self.apply_gradients(learn_rate / len(train_batch))
        self.reset_gradients()

    def train(self, train_data, test_data, learn_rate: float, cost: costs.Cost = costs.MeanSquaredError, batch_size: int = 32, epochs: int = 5, save: bool = False, file_name: str = "neural_network.pkl"):
        """
        Update the gradients and apply them to the network based on a list of batches of training data.

        :param train_data: A list of batches of training data. Each batch is a list of tuples containing input and target output data. Each tuple represents a single training example with the first element being the input data (features) and the second element being the target output data (label). For example: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]] where x1, x2, x3, x4 are the input data and y1, y2, y3, y4 are the target output data.
        :param test_data: A list of tuples containing input and target output data for testing the accuracy of the model.
        :param learn_rate: The learning rate to use when applying the gradients.
        :param cost: The cost function to use when calculating the error between the network's output and the target output.
        :param batch_size: The size of each batch of training data.
        :param epochs: The number of times to iterate over the entire training dataset.
        :param save: Whether to save the model to a file after each epoch.
        :param file_name: The name of the file to save the model to if `save` is `True`.
        """
        
        if save: 
            file = open(file_name, 'wb')
        num_batches = math.ceil(len(train_data) / batch_size)
        bar_step = (num_batches / 50)
        print()
        print("Training ...")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            print("- Progress: [", end="", flush=True)
            epoch_cost = 0
            for i, batch in enumerate(create_batches(train_data, batch_size)):
                if i % bar_step < 1:
                    print("\u2588" * int(max(1 / bar_step, 1)), end="", flush=True)
                self.learn(batch, learn_rate)
            accuracy, epoch_cost = self.validate(test_data, cost)
            print(f"] Cost: {epoch_cost:.4f} | Accuracy: {accuracy:.4f}")
            if save: self.save(file)
        if save: file.close()

    def save(self, file):
        pickle.dump(self, file)

    @staticmethod
    def load(file):
        obj = pickle.load(file)
        if isinstance(obj, NeuralNetwork):
            return obj
        else:
            raise TypeError(f"Loaded object is not an instance of NeuralNetwork. Got {type(obj)} instead.")

    def validate(self, data, cost: costs.Cost = costs.MeanSquaredError):
        """
        Calculate the accuracy and average cost of the model on the data.

        :param data: A list of tuples containing input and target output data. Each tuple represents a single example with the first element being the input data (features) and the second element being the target output data (label).
        :param cost: The cost function to use when calculating the average cost of the model on the data.
        :return: A tuple containing the accuracy and average cost of the model on the data.
        """

        correct_predictions = 0
        average_cost = 0
        for inputs, expected_outputs in data:
            prediction = self.predict(inputs)
            if np.argmax(prediction) == np.argmax(expected_outputs):
                correct_predictions += 1
            average_cost += np.sum(cost.func(prediction, expected_outputs))
        accuracy = correct_predictions / len(data)
        average_cost /= len(data)
        return accuracy, average_cost